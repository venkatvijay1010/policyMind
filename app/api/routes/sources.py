"""Open the original source document behind an evidence citation."""

import asyncio
import re
from urllib.parse import quote

import structlog
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.api.schemas.schemas import SourceOpenResponse
from app.application.source_documents import read_original, source_kind
from app.config import settings
from app.infrastructure.database.models import (
    ContractPassage,
    DocumentElement,
    PassageSource,
    SourceDocument,
)
from app.infrastructure.database.postgres import get_db_session
from app.infrastructure.documents.pdf_renderer import (
    PageRenderError,
    is_valid_box,
    render_page_png,
)
from app.infrastructure.rate_limit import limiter

logger = structlog.get_logger()

router = APIRouter(prefix="/knowledge/sources", tags=["Knowledge"])

# Stored files are always served with a fixed, safe media type and never sniffed, so an
# uploaded document cannot run as HTML or script in this origin.
_MEDIA_TYPES = {"pdf": "application/pdf", "text": "text/plain; charset=utf-8"}
_FILE_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=300"}


def _utf16_length(value: str) -> int:
    """Browsers index strings in UTF-16 code units, not Unicode code points."""
    return len(value.encode("utf-16-le")) // 2


def _content_disposition(disposition: str, filename: str) -> str:
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename) or "document"
    return f"{disposition}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"


async def _load_source(
    db: AsyncSession, source_document_id: int, *, with_bytes: bool
) -> SourceDocument:
    statement = select(SourceDocument).where(SourceDocument.id == source_document_id)
    if not with_bytes:
        statement = statement.options(defer(SourceDocument.original_bytes))
    source: SourceDocument | None = (await db.execute(statement)).scalar_one_or_none()
    if source is None:
        raise HTTPException(status_code=404, detail="Source document not found")
    return source


async def _linked_passage(
    db: AsyncSession, source_document_id: int, passage_id: int
) -> tuple[ContractPassage, list[int]]:
    """Return a passage and the pages it spans, only if it belongs to this source."""
    statement = (
        select(ContractPassage, PassageSource.page_number)
        .join(PassageSource, PassageSource.passage_id == ContractPassage.id)
        .where(
            PassageSource.source_document_id == source_document_id,
            ContractPassage.id == passage_id,
        )
        .options(defer(ContractPassage.embedding))
    )
    rows = (await db.execute(statement)).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Passage not found in this source document")
    return rows[0][0], sorted({page for _, page in rows if page})


async def _highlight_boxes(
    db: AsyncSession, source_document_id: int, passage: ContractPassage, page: int
) -> list[list[float]]:
    """Bounding boxes of the page elements that make up the cited passage."""
    start, end = passage.source_offset_start, passage.source_offset_end
    if start is None or end is None:
        return []
    statement = select(DocumentElement.bbox).where(
        DocumentElement.source_document_id == source_document_id,
        DocumentElement.page_number == page,
        DocumentElement.text_offset_start < end,
        DocumentElement.text_offset_end > start,
    )
    return [
        [float(value) for value in bbox]
        for bbox in (await db.scalars(statement)).all()
        if is_valid_box(bbox)
    ]


@router.get("/{source_document_id}/open", response_model=SourceOpenResponse)
@limiter.limit(settings.rate_limit)
async def open_source(
    request: Request,
    source_document_id: int,
    passage_id: int | None = Query(None, ge=1),
    db: AsyncSession = Depends(get_db_session),
) -> SourceOpenResponse:
    """Describe how to display a source document with the cited passage marked."""
    source = await _load_source(db, source_document_id, with_bytes=False)
    kind = source_kind(source.filename, source.content_type)
    if kind == "other":
        raise HTTPException(status_code=415, detail="This source type cannot be previewed")

    passage: ContractPassage | None = None
    pages: list[int] = []
    if passage_id is not None:
        passage, pages = await _linked_passage(db, source.id, passage_id)

    opened = SourceOpenResponse(
        source_document_id=source.id,
        filename=source.filename,
        kind=kind,
        file_url=str(request.app.url_path_for("get_source_file", source_document_id=source.id)),
        passage_id=passage_id,
        section=passage.topic_title if passage else None,
    )

    if kind == "pdf":
        opened.page_count = source.page_count
        opened.page = pages[0] if pages else 1
        if source.page_count:
            opened.page = min(opened.page, source.page_count)
        if passage is not None:
            opened.highlighted = bool(await _highlight_boxes(db, source.id, passage, opened.page))
        return opened

    await db.refresh(source, ["original_bytes"])
    text = (await read_original(source)).decode("utf-8", errors="replace")
    opened.text = text
    if passage is not None and passage.source_offset_start is not None:
        start, end = passage.source_offset_start, passage.source_offset_end
        # Highlight only when the stored offsets still point at the cited text.
        if end is not None and text[start:end] == passage.content:
            opened.highlight_start = _utf16_length(text[:start])
            opened.highlight_end = opened.highlight_start + _utf16_length(passage.content)
            opened.highlighted = True
    return opened


@router.get("/{source_document_id}/file")
@limiter.limit(settings.rate_limit)
async def get_source_file(
    request: Request,
    source_document_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> Response:
    """Stream the original source document."""
    source = await _load_source(db, source_document_id, with_bytes=True)
    kind = source_kind(source.filename, source.content_type)
    headers = {
        **_FILE_HEADERS,
        "Content-Disposition": _content_disposition(
            "inline" if kind in _MEDIA_TYPES else "attachment", source.filename
        ),
    }
    return Response(
        content=await read_original(source),
        media_type=_MEDIA_TYPES.get(kind, "application/octet-stream"),
        headers=headers,
    )


@router.get("/{source_document_id}/pages/{page}/image")
@limiter.limit(settings.rate_limit)
async def get_source_page_image(
    request: Request,
    source_document_id: int,
    page: int = Path(..., ge=1),
    passage_id: int | None = Query(None, ge=1),
    scale: float = Query(1.6, ge=0.5, le=3.0),
    db: AsyncSession = Depends(get_db_session),
) -> Response:
    """Render one PDF page as an image, highlighting the cited passage when given."""
    source = await _load_source(db, source_document_id, with_bytes=True)
    if source_kind(source.filename, source.content_type) != "pdf":
        raise HTTPException(status_code=404, detail="Source document is not a PDF")
    if source.page_count and page > source.page_count:
        raise HTTPException(status_code=404, detail="Page not found")

    boxes: list[list[float]] = []
    if passage_id is not None:
        passage, _ = await _linked_passage(db, source.id, passage_id)
        boxes = await _highlight_boxes(db, source.id, passage, page)

    try:
        rendered = await asyncio.to_thread(
            render_page_png, await read_original(source), page, boxes, scale
        )
    except PageRenderError as exc:
        logger.warning("Page render failed", source_document_id=source.id, page=page)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    headers = {**_FILE_HEADERS, "X-Highlight-Count": str(len(boxes))}
    if rendered.highlight_top is not None:
        headers["X-Highlight-Top"] = f"{rendered.highlight_top:.4f}"
    return Response(content=rendered.png, media_type="image/png", headers=headers)
