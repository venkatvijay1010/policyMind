"""Source-document provenance shared by ingestion, the API and maintenance scripts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import PurePath

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import PassageSource, SourceDocument
from app.infrastructure.storage.object_storage import (
    S3ObjectStorage,
    StorageError,
    build_object_key,
    get_object_storage,
)

logger = structlog.get_logger()

# Passages are whitespace-stripped slices of the source, so older chunkers (which recorded
# the raw chunk start) can sit a few characters before the true start of the passage text.
_MAX_OFFSET_DRIFT = 64
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def source_kind(filename: str, content_type: str) -> str:
    """Classify a stored source as "pdf", "text" or "other"."""
    suffix = PurePath(filename).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".md", ".txt"}:
        return "text"
    media_type = content_type.split(";")[0].strip().lower()
    if media_type == "application/pdf":
        return "pdf"
    if media_type.startswith("text/"):
        return "text"
    return "other"


def locate_texts(haystack: str, needles: Sequence[str]) -> list[tuple[int, int] | None]:
    """Find each needle in reading order.

    A needle matches only when nothing but whitespace separates it from the previous
    match, so a missing or reformatted element is reported as None instead of being
    matched to an unrelated repeat further down the page.
    """
    cursor = 0
    located: list[tuple[int, int] | None] = []
    for needle in needles:
        index = haystack.find(needle, cursor) if needle else -1
        if index < 0 or haystack[cursor:index].strip():
            located.append(None)
            continue
        cursor = index + len(needle)
        located.append((index, cursor))
    return located


def reconstruct_text(passages: Sequence[tuple[str, int]]) -> tuple[str, list[int]] | None:
    """Rebuild source text from ordered (content, recorded_start) passages.

    Returns the text and the exact start of every passage in it, or None when the
    overlapping passages disagree and the source cannot be recovered reliably.
    """
    buffer: list[str | None] = []
    starts: list[int] = []
    for content, recorded_start in passages:
        for drift in range(_MAX_OFFSET_DRIFT + 1):
            position = max(0, recorded_start) + drift
            overlap = buffer[position : position + len(content)]
            if all(known is None or known == char for known, char in zip(overlap, content)):
                break
        else:
            return None
        end = position + len(content)
        if end > len(buffer):
            buffer.extend([None] * (end - len(buffer)))
        buffer[position:end] = list(content)
        starts.append(position)
    return "".join(char if char is not None else "\n" for char in buffer), starts


def new_text_source_document(
    contract_id: int, text_content: str, filename: str, origin: str
) -> SourceDocument:
    """Build a source document for text that was ingested without an uploaded file."""
    data = text_content.encode("utf-8")
    safe_name = _UNSAFE_FILENAME_CHARS.sub("_", filename).strip(" .")[:255] or "source.txt"
    return SourceDocument(
        contract_id=contract_id,
        filename=safe_name,
        content_type="text/markdown" if safe_name.lower().endswith(".md") else "text/plain",
        content_sha256=hashlib.sha256(data).hexdigest(),
        original_bytes=data,
        page_count=None,
        upload_metadata={"source": origin},
        extraction_status="complete",
        needs_review=False,
    )


async def attach_text_source(
    session: AsyncSession,
    contract_id: int,
    text_content: str,
    filename: str,
    passage_ids: Sequence[int],
    origin: str,
) -> SourceDocument:
    """Create a text source document and link existing passages to it."""
    source = new_text_source_document(contract_id, text_content, filename, origin)
    session.add(source)
    await session.flush()
    session.add_all(
        PassageSource(
            passage_id=passage_id, source_document_id=source.id, page_number=None, element_ids=[]
        )
        for passage_id in passage_ids
    )
    storage = get_object_storage()
    if storage is not None:
        await store_original(source, storage)
    await session.flush()
    return source


async def store_original(source: SourceDocument, storage: S3ObjectStorage) -> None:
    """Upload the original bytes and record where they live on the row."""
    key = build_object_key(source.contract_id, source.id, source.content_sha256, source.filename)
    stored = await storage.put(key, source.original_bytes, source.content_type)
    source.storage_backend = "s3"
    source.storage_bucket = stored.bucket
    source.storage_key = stored.key
    source.storage_etag = stored.etag
    source.storage_uploaded_at = datetime.now(timezone.utc).replace(tzinfo=None)


async def read_original(source: SourceDocument) -> bytes:
    """Read the original from object storage, falling back to the database copy."""
    if source.storage_key and source.storage_bucket:
        try:
            storage = get_object_storage()
            if storage is not None:
                return await storage.get(source.storage_bucket, source.storage_key)
        except StorageError:
            logger.warning(
                "Object storage read failed; using the database copy",
                source_document_id=source.id,
            )
    original: bytes = source.original_bytes
    return original


async def delete_stored_objects(objects: Sequence[tuple[str, str]]) -> None:
    """Best-effort removal of objects whose database rows were deleted."""
    if not objects:
        return
    try:
        storage = get_object_storage()
    except StorageError:
        logger.warning("Object storage unavailable; stored objects were not deleted")
        return
    if storage is None:
        return
    for bucket, key in objects:
        try:
            await storage.delete(bucket, key)
        except StorageError:
            logger.warning("Stored object was not deleted", bucket=bucket, key=key)
