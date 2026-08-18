"""Document ingestion endpoints and safe chunking helpers."""

import json
import time
from io import BytesIO
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
from app.infrastructure.database.models import Policy
from app.infrastructure.database.postgres import get_db_session
from app.infrastructure.llm.embeddings import get_embedding_service
from app.infrastructure.rate_limit import limiter

logger = structlog.get_logger()

router = APIRouter(prefix="/knowledge/scopes", tags=["Knowledge"])


def chunk_text(
    text_content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """Split text into bounded, overlapping chunks with correct source offsets."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
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

    if suffix in {".txt", ".md"}:
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="Uploaded text must be UTF-8") from exc
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            text_content = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError as exc:
            raise HTTPException(
                status_code=503, detail="PDF support is not installed on this server"
            ) from exc
        except Exception as exc:
            logger.warning("Failed to extract PDF text", filename=filename, error=str(exc))
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

    insert_parameters = [
        {
            "contract_id": scope_key,
            "content": chunk["content"],
            "passage_order": chunk["passage_order"],
            "source_offset_start": chunk["source_offset_start"],
            "source_offset_end": chunk["source_offset_end"],
            # SQLite persists local embedding vectors as JSON. JSON text also
            # remains portable if the database is later moved elsewhere.
            "embedding": json.dumps(embedding),
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    # The request-scoped session commits only after the endpoint succeeds. Deleting
    # and inserting in one transaction makes PUT idempotent and prevents partial indexes.
    await db.execute(
        text("DELETE FROM contract_passages WHERE contract_id = :contract_id"),
        {"contract_id": scope_key},
    )
    await db.execute(
        text(
            """
            INSERT INTO contract_passages
            (contract_id, content, passage_order, source_offset_start, source_offset_end, embedding)
            VALUES (:contract_id, :content, :passage_order, :source_offset_start,
                    :source_offset_end, :embedding)
            """
        ),
        insert_parameters,
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
