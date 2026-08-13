"""
Ingest endpoint - document ingestion and chunking.
"""
import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

from app.api.schemas.schemas import KnowledgeSourceRequest, KnowledgeSourceResponse
from app.infrastructure.database.postgres import get_db_session
from app.infrastructure.llm.embeddings import EmbeddingService
from app.config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/knowledge/scopes", tags=["Knowledge"])


def chunk_text(
    text_content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[dict]:
    """
    Split text into overlapping chunks.
    Returns list of dicts with content and metadata.
    """
    chunks = []
    
    # Split by paragraphs first
    paragraphs = text_content.split('\n\n')
    
    current_chunk = ""
    current_position = 0
    passage_order = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph exceeds chunk size
        if len(current_chunk) + len(para) + 2 > chunk_size:
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "passage_order": passage_order,
                    "source_offset_start": current_position,
                    "source_offset_end": current_position + len(current_chunk)
                })
                passage_order += 1
                
                # Keep overlap
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                current_position = current_position + len(current_chunk) - chunk_overlap
            else:
                # Paragraph itself is too long, force split
                while len(para) > chunk_size:
                    chunks.append({
                        "content": para[:chunk_size],
                        "passage_order": passage_order,
                        "source_offset_start": current_position,
                        "source_offset_end": current_position + chunk_size
                    })
                    passage_order += 1
                    para = para[chunk_size - chunk_overlap:]
                    current_position += chunk_size - chunk_overlap
                
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "passage_order": passage_order,
            "source_offset_start": current_position,
            "source_offset_end": current_position + len(current_chunk)
        })
    
    return chunks


@router.put("/{scope_key}/source", response_model=KnowledgeSourceResponse)
async def index_contract_source(
    scope_key: int,
    request: KnowledgeSourceRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Ingest a document into the vector store.
    
    The document will be:
    1. Chunked with configurable size and overlap
    2. Embedded using OpenAI embeddings
    3. Stored in PostgreSQL with pgvector
    
    Supply either `source_text` (raw content) or `source_uri` (a public content location).
    """
    start_time = time.time()
    
    # Validate policy exists
    result = await db.execute(
        text("SELECT id, contract_title FROM benefit_contracts WHERE id = :id"),
        {"id": scope_key}
    )
    policy = result.fetchone()
    if not policy:
        raise HTTPException(status_code=404, detail=f"Knowledge scope {scope_key} not found")
    
    # Get document text
    if request.source_text:
        text_content = request.source_text
    elif request.source_uri:
        # Fetch from URL
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(request.source_uri)
                response.raise_for_status()
                text_content = response.text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either source_text or source_uri is required")
    
    logger.info(
        "Starting ingestion",
        scope_key=scope_key,
        text_length=len(text_content),
        segment_length=request.segment_length
    )
    
    # Chunk the text
    chunks = chunk_text(
        text_content,
        chunk_size=request.segment_length,
        chunk_overlap=request.segment_overlap
    )
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created from document")
    
    # Generate embeddings
    embedding_service = EmbeddingService()
    chunk_texts = [c["content"] for c in chunks]
    
    try:
        embeddings = await embedding_service.embed_batch(chunk_texts)
    except Exception as e:
        logger.error("Embedding failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    
    # Store in database
    chunks_created = 0
    for chunk_data, embedding in zip(chunks, embeddings):
        try:
            await db.execute(
                text("""
                    INSERT INTO contract_passages
                    (contract_id, content, passage_order, source_offset_start, source_offset_end, embedding)
                    VALUES (:contract_id, :content, :passage_order, :source_offset_start, :source_offset_end, :embedding)
                """),
                {
                    "contract_id": scope_key,
                    "content": chunk_data["content"],
                    "passage_order": chunk_data["passage_order"],
                    "source_offset_start": chunk_data["source_offset_start"],
                    "source_offset_end": chunk_data["source_offset_end"],
                    "embedding": str(embedding)  # pgvector format
                }
            )
            chunks_created += 1
        except Exception as e:
            logger.error("Failed to insert chunk", error=str(e), passage_order=chunk_data["passage_order"])
    
    await db.commit()
    
    processing_time = int((time.time() - start_time) * 1000)
    
    logger.info(
        "Ingestion complete",
        scope_key=scope_key,
        segments_created=chunks_created,
        processing_time_ms=processing_time
    )
    
    return KnowledgeSourceResponse(
        scope_key=scope_key,
        segments_created=chunks_created,
        processing_time_ms=processing_time,
        message=f"Indexed {chunks_created} segments for contract '{policy.contract_title}'"
    )


@router.put("/{scope_key}/file")
async def ingest_file(
    scope_key: int,
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Ingest a document file (PDF, TXT, MD) into the vector store.
    """
    # Read file content
    content = await file.read()
    
    # Handle different file types
    if file.filename.endswith('.pdf'):
        # For PDF, you'd need pypdf or similar
        raise HTTPException(status_code=400, detail="PDF support not yet implemented. Use text endpoint.")
    elif file.filename.endswith(('.txt', '.md')):
        text_content = content.decode('utf-8')
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .txt or .md")
    
    # Delegate to main ingest
    request = KnowledgeSourceRequest(
        source_text=text_content,
        segment_length=chunk_size,
        segment_overlap=chunk_overlap
    )
    
    return await index_contract_source(scope_key, request, db)


@router.delete("/{scope_key}/source")
async def delete_chunks(
    scope_key: int,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Delete all chunks for a policy.
    Useful for re-ingestion.
    """
    result = await db.execute(
        text("DELETE FROM contract_passages WHERE contract_id = :contract_id RETURNING id"),
        {"contract_id": scope_key}
    )
    deleted = result.rowcount
    await db.commit()
    
    return {
        "scope_key": scope_key,
        "segments_deleted": deleted,
        "message": f"Deleted {deleted} indexed segments for knowledge scope {scope_key}"
    }


@router.get("/{scope_key}/index-status")
async def ingestion_stats(
    scope_key: int,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get ingestion statistics for a policy.
    """
    result = await db.execute(
        text("""
            SELECT 
                COUNT(*) as chunk_count,
                AVG(LENGTH(content)) as avg_chunk_length,
                MIN(created_at) as first_ingested,
                MAX(created_at) as last_ingested
            FROM contract_passages
            WHERE contract_id = :contract_id
        """),
        {"contract_id": scope_key}
    )
    stats = result.fetchone()
    
    return {
        "scope_key": scope_key,
        "segment_count": stats.chunk_count or 0,
        "avg_segment_length": round(stats.avg_chunk_length or 0, 2),
        "first_ingested": stats.first_ingested,
        "last_ingested": stats.last_ingested
    }
