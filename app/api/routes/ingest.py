"""Document ingestion endpoints and safe chunking helpers."""

import asyncio
import hashlib
import re
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.schemas import KnowledgeSourceRequest, KnowledgeSourceResponse
from app.config import settings
from app.infrastructure.database.models import (
    ContractPassage,
    DocumentElement,
    PassageSource,
    Policy,
    SourceDocument,
)
from app.infrastructure.database.postgres import get_db_session
from app.infrastructure.documents.pdf_extractor import (
    EXTRACTION_VERSION,
    PDFExtractionError,
    PDFExtractionTimeoutError,
    extract_pdf_with_timeout,
)
from app.infrastructure.llm.embeddings import get_embedding_service
from app.infrastructure.rate_limit import limiter

logger = structlog.get_logger()

router = APIRouter(prefix="/knowledge/scopes", tags=["Knowledge"])


_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_UPPERCASE_HEADING = re.compile(r"^\s*([A-Z][A-Z0-9 &/(),.'-]{3,})\s*$")


def _section_at_offset(text_content: str, offset: int, chunk_end: int) -> str | None:
    """Return the active heading, preferring a section over a document title."""
    section: str | None = None
    section_is_document_title = False
    cursor = 0
    for line in text_content.splitlines(keepends=True):
        if cursor > offset:
            break
        stripped = line.strip()
        markdown_match = _MARKDOWN_HEADING.match(stripped)
        uppercase_match = _UPPERCASE_HEADING.match(stripped)
        if markdown_match:
            section = markdown_match.group(1).strip()
            section_is_document_title = stripped.startswith("# ")
        elif uppercase_match:
            section = uppercase_match.group(1).strip().title()
            section_is_document_title = False
        cursor += len(line)

    # A first chunk normally starts with the document's H1. Use the first
    # actual section in that chunk so a citation says "Coverage" rather than
    # merely repeating the document title.
    if section is None or section_is_document_title:
        cursor = 0
        for line in text_content.splitlines(keepends=True):
            stripped = line.strip()
            markdown_match = _MARKDOWN_HEADING.match(stripped)
            uppercase_match = _UPPERCASE_HEADING.match(stripped)
            if offset <= cursor < chunk_end:
                if markdown_match and not stripped.startswith("# "):
                    return markdown_match.group(1).strip()
                if uppercase_match:
                    return uppercase_match.group(1).strip().title()
            cursor += len(line)
    return section


def chunk_text(
    text_content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    page_starts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Split text into bounded chunks and retain section/page provenance."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    normalized_page_starts = sorted({start for start in (page_starts or []) if start >= 0})
    content_length = len(text_content)
    start = 0

    while start < content_length:
        maximum_end = min(start + chunk_size, content_length)
        end = maximum_end

        # Prefer paragraph or word boundaries without allowing a short boundary
        # to negate forward progress after overlap is applied.
        if maximum_end < content_length:
            minimum_boundary = start + chunk_overlap + 1
            paragraph_break = text_content.rfind("\n\n", minimum_boundary, maximum_end)
            word_break = text_content.rfind(" ", minimum_boundary, maximum_end)
            boundary = max(paragraph_break, word_break)
            if boundary > minimum_boundary:
                end = boundary

        raw_chunk = text_content[start:end]
        chunk_content = raw_chunk.strip()
        if chunk_content:
            leading_whitespace = len(raw_chunk) - len(raw_chunk.lstrip())
            source_offset_start = start + leading_whitespace
            chunks.append(
                {
                    "content": chunk_content,
                    "passage_order": len(chunks),
                    "source_offset_start": source_offset_start,
                    "source_offset_end": source_offset_start + len(chunk_content),
                    "topic_title": _section_at_offset(
                        text_content, source_offset_start, source_offset_start + len(chunk_content)
                    ),
                    # Text and Markdown do not have trustworthy physical pages. PDF extraction
                    # provides page starts; then this becomes the real 1-based source page.
                    "source_page": (
                        bisect_right(normalized_page_starts, source_offset_start)
                        if normalized_page_starts
                        else None
                    ),
                }
            )

        if end == content_length:
            break
        start = end - chunk_overlap

    return chunks


def _validate_source_uri(source_uri: str) -> None:
    """Allow URL ingestion only for explicitly trusted HTTPS hosts."""
    parsed_uri = urlparse(source_uri)
    hostname = (parsed_uri.hostname or "").lower().rstrip(".")
    allowed_hosts = settings.source_ingest_allowed_hosts_list

    if parsed_uri.scheme != "https" or not hostname:
        raise HTTPException(status_code=400, detail="source_uri must be an absolute HTTPS URL")
    if not allowed_hosts:
        raise HTTPException(
            status_code=403,
            detail="URL ingestion is disabled until SOURCE_INGEST_ALLOWED_HOSTS is configured",
        )
    if hostname not in allowed_hosts:
        raise HTTPException(status_code=403, detail="The source URI host is not allowlisted")


async def _fetch_source_text(source_uri: str) -> str:
    """Fetch a bounded trusted source without following redirects."""
    _validate_source_uri(source_uri)
    timeout = httpx.Timeout(settings.source_fetch_timeout_seconds)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            async with client.stream("GET", source_uri) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > settings.max_ingest_file_bytes:
                    raise HTTPException(
                        status_code=413, detail="Source document exceeds the size limit"
                    )

                content = bytearray()
                async for data in response.aiter_bytes():
                    content.extend(data)
                    if len(content) > settings.max_ingest_file_bytes:
                        raise HTTPException(
                            status_code=413, detail="Source document exceeds the size limit"
                        )
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Failed to fetch source URI", error=str(exc))
        raise HTTPException(status_code=422, detail="Unable to fetch source URI") from exc

    try:
        return bytes(content).decode(response.encoding or "utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Source document must be UTF-8 text") from exc


def _ensure_source_size(text_content: str) -> None:
    if len(text_content.encode("utf-8")) > settings.max_ingest_file_bytes:
        raise HTTPException(status_code=413, detail="Source document exceeds the size limit")


async def _source_request_from_upload(
    file: UploadFile,
    chunk_size: int,
    chunk_overlap: int,
) -> KnowledgeSourceRequest:
    """Read a supported uploaded file and prepare it for indexing."""
    content = await file.read(settings.max_ingest_file_bytes + 1)
    if len(content) > settings.max_ingest_file_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit")

    filename = file.filename or "uploaded-document"
    suffix = Path(filename).suffix.lower()

    page_starts: list[int] = []
    extraction_elements: list[dict[str, Any]] = []
    extraction_hash: str | None = None
    extraction_page_count: int | None = None
    extraction_needs_review = False
    extraction_version: str | None = None
    if suffix in {".txt", ".md"}:
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="Uploaded text must be UTF-8") from exc
    elif suffix == ".pdf":
        try:
            extraction = await asyncio.to_thread(
                extract_pdf_with_timeout,
                content,
                settings.pdf_extraction_timeout_seconds,
            )
            pages = []
            current_offset = 0
            for page in extraction.pages:
                page_starts.append(current_offset)
                page_text = page.text
                pages.append(page_text)
                current_offset += len(page_text) + 2
            text_content = "\n\n".join(pages)
            extraction_elements = [
                {
                    "page_number": element.page_number,
                    "element_type": element.element_type,
                    "text_content": element.text_content,
                    "structured_data": element.structured_data,
                    "bbox": element.bbox,
                    "confidence": element.confidence,
                    "parser_name": element.parser_name,
                    "parser_version": element.parser_version,
                    "extraction_version": EXTRACTION_VERSION,
                    "needs_review": element.needs_review,
                }
                for page in extraction.pages
                for element in page.elements
            ]
            extraction_hash = extraction.content_sha256
            extraction_page_count = extraction.page_count
            extraction_needs_review = extraction.needs_review
            extraction_version = EXTRACTION_VERSION
        except PDFExtractionTimeoutError as exc:
            raise HTTPException(
                status_code=504, detail="PDF extraction exceeded its time limit"
            ) from exc
        except PDFExtractionError as exc:
            error_detail = str(exc)
            if "timed out" in error_detail.lower():
                status_code = 504
            elif "OCR support requires" in error_detail or "verify that Tesseract" in error_detail:
                status_code = 503
            else:
                status_code = 422
            raise HTTPException(status_code=status_code, detail=error_detail) from exc
        except Exception as exc:
            logger.warning("Failed to extract PDF text", error_type=type(exc).__name__)
            raise HTTPException(
                status_code=422, detail="Unable to extract text from this PDF"
            ) from exc
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .txt, .md, or .pdf")

    if not text_content.strip():
        raise HTTPException(
            status_code=422, detail="The uploaded document does not contain readable text"
        )

    _ensure_source_size(text_content)
    return KnowledgeSourceRequest(
        source_text=text_content,
        segment_length=chunk_size,
        segment_overlap=chunk_overlap,
        source_page_starts=page_starts,
        original_filename=filename[:255],
        original_content_type=(file.content_type or "application/octet-stream")[:100],
        original_bytes=content,
        extracted_elements=extraction_elements,
        extraction_hash=extraction_hash or hashlib.sha256(content).hexdigest(),
        extraction_page_count=extraction_page_count,
        extraction_version=extraction_version,
        extraction_needs_review=extraction_needs_review,
    )


async def _index_contract_source(
    scope_key: int,
    source_request: KnowledgeSourceRequest,
    db: AsyncSession,
) -> KnowledgeSourceResponse:
    """Replace a knowledge scope's passages with a newly indexed source document."""
    start_time = time.monotonic()

    result = await db.execute(
        text("SELECT id, contract_title FROM benefit_contracts WHERE id = :id"),
        {"id": scope_key},
    )
    policy = result.fetchone()
    if not policy:
        raise HTTPException(status_code=404, detail=f"Knowledge scope {scope_key} not found")

    text_content = source_request.source_text or await _fetch_source_text(
        source_request.source_uri or ""
    )
    _ensure_source_size(text_content)

    logger.info(
        "Starting ingestion",
        scope_key=scope_key,
        text_length=len(text_content),
        segment_length=source_request.segment_length,
    )

    chunks = chunk_text(
        text_content,
        chunk_size=source_request.segment_length,
        chunk_overlap=source_request.segment_overlap,
        page_starts=source_request.source_page_starts,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created from document")

    try:
        embeddings = await get_embedding_service().embed_batch(
            [chunk["content"] for chunk in chunks]
        )
    except Exception as exc:
        logger.exception("Embedding failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Unable to create source embeddings") from exc

    if len(embeddings) != len(chunks):
        logger.error("Embedding count mismatch", chunks=len(chunks), embeddings=len(embeddings))
        raise HTTPException(
            status_code=502, detail="Embedding service returned an incomplete response"
        )

    # The request-scoped session commits only after the endpoint succeeds. Deleting
    # and inserting in one transaction makes PUT idempotent. Source documents are
    # append-only for auditability; replacing an index never deletes original bytes.
    await db.execute(
        text("DELETE FROM contract_passages WHERE contract_id = :contract_id"),
        {"contract_id": scope_key},
    )
    source_document: SourceDocument | None = None
    element_ids_by_page: dict[int, list[int]] = {}
    if source_request.original_bytes is not None:
        source_document = SourceDocument(
            contract_id=scope_key,
            filename=source_request.original_filename or "uploaded-document",
            content_type=source_request.original_content_type or "application/octet-stream",
            content_sha256=source_request.extraction_hash
            or hashlib.sha256(source_request.original_bytes).hexdigest(),
            original_bytes=source_request.original_bytes,
            page_count=source_request.extraction_page_count,
            upload_metadata={"source": "multipart_upload"},
            extraction_version=source_request.extraction_version,
            extraction_status="needs_review"
            if source_request.extraction_needs_review
            else "complete",
            needs_review=source_request.extraction_needs_review,
        )
        db.add(source_document)
        await db.flush()
        created_elements = []
        for item in source_request.extracted_elements:
            element = DocumentElement(source_document_id=source_document.id, **item)
            db.add(element)
            created_elements.append(element)
        await db.flush()
        for element in created_elements:
            element_ids_by_page.setdefault(element.page_number, []).append(element.id)

    for chunk, embedding in zip(chunks, embeddings):
        passage = ContractPassage(
            contract_id=scope_key,
            content=chunk["content"],
            passage_order=chunk["passage_order"],
            source_offset_start=chunk["source_offset_start"],
            source_offset_end=chunk["source_offset_end"],
            topic_title=chunk["topic_title"],
            source_page=chunk["source_page"],
            embedding=embedding,
        )
        db.add(passage)
        if source_document is not None:
            await db.flush()
            source_pages = [chunk["source_page"]]
            if source_request.source_page_starts:
                first_page = max(
                    1, bisect_right(source_request.source_page_starts, chunk["source_offset_start"])
                )
                last_page = max(
                    first_page,
                    bisect_right(
                        source_request.source_page_starts,
                        max(chunk["source_offset_start"], chunk["source_offset_end"] - 1),
                    ),
                )
                source_pages = list(range(first_page, last_page + 1))
            for source_page in source_pages:
                db.add(
                    PassageSource(
                        passage_id=passage.id,
                        source_document_id=source_document.id,
                        page_number=source_page,
                        element_ids=element_ids_by_page.get(source_page, []),
                    )
                )

    processing_time = int((time.monotonic() - start_time) * 1000)
    logger.info(
        "Ingestion complete",
        scope_key=scope_key,
        segments_created=len(chunks),
        processing_time_ms=processing_time,
    )

    return KnowledgeSourceResponse(
        scope_key=scope_key,
        segments_created=len(chunks),
        processing_time_ms=processing_time,
        message=f"Indexed {len(chunks)} segments for contract '{policy.contract_title}'",
        source_document_id=source_document.id if source_document else None,
        content_sha256=source_document.content_sha256 if source_document else None,
        extraction_status=source_document.extraction_status if source_document else None,
        needs_review=source_document.needs_review if source_document else False,
    )


@router.put("/{scope_key}/source", response_model=KnowledgeSourceResponse)
@limiter.limit(settings.rate_limit)
async def index_contract_source(
    scope_key: int,
    request: Request,
    source_request: KnowledgeSourceRequest,
    db: AsyncSession = Depends(get_db_session),
) -> KnowledgeSourceResponse:
    """Replace a knowledge scope's passages with an indexed source document."""
    return await _index_contract_source(scope_key, source_request, db)


@router.get("")
async def list_knowledge_scopes(
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, list[dict[str, Any]]]:
    """List sample and uploaded policy documents available to the chat UI."""
    result = await db.execute(
        text(
            """
            SELECT
                benefit_contracts.id AS scope_key,
                COALESCE(benefit_contracts.contract_title, benefit_contracts.contract_ref) AS title,
                benefit_contracts.plan_category AS plan_category,
                benefit_contracts.source_document_uri AS source_document_uri,
                COUNT(contract_passages.id) AS segment_count
            FROM benefit_contracts
            LEFT JOIN contract_passages ON contract_passages.contract_id = benefit_contracts.id
            GROUP BY benefit_contracts.id, benefit_contracts.contract_title,
                     benefit_contracts.contract_ref, benefit_contracts.plan_category,
                     benefit_contracts.source_document_uri
            ORDER BY benefit_contracts.id DESC
            """
        )
    )
    scopes = []
    for row in result.mappings():
        item = dict(row)
        item["segment_count"] = int(item["segment_count"] or 0)
        item["is_uploaded"] = item["plan_category"] == "UPLOADED_DOCUMENT"
        scopes.append(item)
    return {"scopes": scopes}


@router.post("/file", response_model=KnowledgeSourceResponse, status_code=201)
@limiter.limit(settings.rate_limit)
async def create_scope_from_file(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    chunk_size: int = Query(settings.chunk_size, ge=100, le=4000),
    chunk_overlap: int = Query(settings.chunk_overlap, ge=0, le=500),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new local policy scope and index an uploaded TXT, Markdown, or PDF document."""
    source_request = await _source_request_from_upload(file, chunk_size, chunk_overlap)
    filename = file.filename or "uploaded-document"
    document_title = (title or Path(filename).stem or "Uploaded policy document").strip()
    if not document_title:
        raise HTTPException(status_code=400, detail="Provide a title for the uploaded document")

    policy = Policy(
        contract_ref=f"LOCAL-{uuid4().hex[:12].upper()}",
        contract_title=document_title[:255],
        plan_category="UPLOADED_DOCUMENT",
        source_document_uri=f"upload://{filename}"[:500],
    )
    db.add(policy)
    await db.flush()

    return await _index_contract_source(policy.id, source_request, db)


@router.put("/{scope_key}/file", response_model=KnowledgeSourceResponse)
@limiter.limit(settings.rate_limit)
async def ingest_file(
    scope_key: int,
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = Query(settings.chunk_size, ge=100, le=4000),
    chunk_overlap: int = Query(settings.chunk_overlap, ge=0, le=500),
    db: AsyncSession = Depends(get_db_session),
):
    """Replace an existing scope with a TXT, Markdown, or PDF document."""
    source_request = await _source_request_from_upload(file, chunk_size, chunk_overlap)
    return await _index_contract_source(scope_key, source_request, db)


@router.delete("/{scope_key}/source")
@limiter.limit(settings.rate_limit)
async def delete_chunks(
    scope_key: int,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete all indexed passages for a knowledge scope."""
    result = await db.execute(
        text("DELETE FROM contract_passages WHERE contract_id = :contract_id"),
        {"contract_id": scope_key},
    )
    deleted = max(result.rowcount or 0, 0)
    await db.commit()

    return {
        "scope_key": scope_key,
        "segments_deleted": deleted,
        "message": f"Deleted {deleted} indexed segments for knowledge scope {scope_key}",
    }


@router.delete("/{scope_key}")
@limiter.limit(settings.rate_limit)
async def delete_uploaded_scope(
    scope_key: int,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a user-uploaded document and all of its indexed passages."""
    result = await db.execute(
        text(
            """
            SELECT contract_title, plan_category
            FROM benefit_contracts
            WHERE id = :scope_key
            """
        ),
        {"scope_key": scope_key},
    )
    scope = result.mappings().first()
    if not scope:
        raise HTTPException(status_code=404, detail="Knowledge scope not found")
    if scope["plan_category"] != "UPLOADED_DOCUMENT":
        raise HTTPException(status_code=403, detail="Only uploaded documents can be deleted here")

    await db.execute(
        text("DELETE FROM benefit_contracts WHERE id = :scope_key"),
        {"scope_key": scope_key},
    )
    return {
        "scope_key": scope_key,
        "message": f"Deleted uploaded document '{scope['contract_title'] or scope_key}'",
    }


@router.get("/{scope_key}/index-status")
async def ingestion_stats(
    scope_key: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Get ingestion statistics for a knowledge scope."""
    result = await db.execute(
        text(
            """
            SELECT
                COUNT(*) as chunk_count,
                AVG(LENGTH(content)) as avg_chunk_length,
                MIN(created_at) as first_ingested,
                MAX(created_at) as last_ingested
            FROM contract_passages
            WHERE contract_id = :contract_id
            """
        ),
        {"contract_id": scope_key},
    )
    stats = result.fetchone()

    return {
        "scope_key": scope_key,
        "segment_count": stats.chunk_count or 0,
        "avg_segment_length": round(stats.avg_chunk_length or 0, 2),
        "first_ingested": stats.first_ingested,
        "last_ingested": stats.last_ingested,
    }
