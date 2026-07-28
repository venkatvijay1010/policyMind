"""
Ingest endpoint - document ingestion and chunking.
"""
import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

from app.api.schemas.schemas import IngestRequest, IngestResponse
from app.infrastructure.database.postgres import get_db_session
from app.infrastructure.llm.embeddings import EmbeddingService
from app.config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


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
    chunk_index = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph exceeds chunk size
        if len(current_chunk) + len(para) + 2 > chunk_size:
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "char_start": current_position,
                    "char_end": current_position + len(current_chunk)
                })
                chunk_index += 1
                
                # Keep overlap
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                current_position = current_position + len(current_chunk) - chunk_overlap
            else:
                # Paragraph itself is too long, force split
                while len(para) > chunk_size:
                    chunks.append({
                        "content": para[:chunk_size],
                        "chunk_index": chunk_index,
                        "char_start": current_position,
                        "char_end": current_position + chunk_size
                    })
                    chunk_index += 1
                    para = para[chunk_size - chunk_overlap:]
                    current_position += chunk_size - chunk_overlap
                
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "chunk_index": chunk_index,
            "char_start": current_position,
            "char_end": current_position + len(current_chunk)
        })
    
    return chunks


@router.post("", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Ingest a document into the vector store.
    
    The document will be:
    1. Chunked with configurable size and overlap
    2. Embedded using OpenAI embeddings
    3. Stored in PostgreSQL with pgvector
    
    You can provide either `document_text` (raw text) or `document_url` (URL to fetch).
    """
    start_time = time.time()
    
    # Validate policy exists
    result = await db.execute(
        text("SELECT id, policy_name FROM policies WHERE id = :id"),
        {"id": request.policy_id}
    )
    policy = result.fetchone()
    if not policy:
        raise HTTPException(status_code=404, detail=f"Policy {request.policy_id} not found")
    
    # Get document text
    if request.document_text:
        text_content = request.document_text
    elif request.document_url:
        # Fetch from URL
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(request.document_url)
                response.raise_for_status()
                text_content = response.text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either document_text or document_url required")
    
    logger.info(
        "Starting ingestion",
        policy_id=request.policy_id,
        text_length=len(text_content),
        chunk_size=request.chunk_size
    )
    
    # Chunk the text
    chunks = chunk_text(
        text_content,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
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
                    INSERT INTO policy_chunks 
                    (policy_id, content, chunk_index, char_start, char_end, embedding)
                    VALUES (:policy_id, :content, :chunk_index, :char_start, :char_end, :embedding)
                """),
                {
                    "policy_id": request.policy_id,
                    "content": chunk_data["content"],
                    "chunk_index": chunk_data["chunk_index"],
                    "char_start": chunk_data["char_start"],
                    "char_end": chunk_data["char_end"],
                    "embedding": str(embedding)  # pgvector format
                }
            )
            chunks_created += 1
        except Exception as e:
            logger.error("Failed to insert chunk", error=str(e), chunk_index=chunk_data["chunk_index"])
    
    await db.commit()
    
    processing_time = int((time.time() - start_time) * 1000)
    
    logger.info(
        "Ingestion complete",
        policy_id=request.policy_id,
        chunks_created=chunks_created,
        processing_time_ms=processing_time
    )
    
    return IngestResponse(
        policy_id=request.policy_id,
        chunks_created=chunks_created,
        processing_time_ms=processing_time,
        message=f"Successfully ingested {chunks_created} chunks for policy '{policy.policy_name}'"
    )


@router.post("/file")
async def ingest_file(
    policy_id: int,
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
    request = IngestRequest(
        policy_id=policy_id,
        document_text=text_content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return await ingest_document(request, db)


@router.delete("/{policy_id}")
async def delete_chunks(
    policy_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Delete all chunks for a policy.
    Useful for re-ingestion.
    """
    result = await db.execute(
        text("DELETE FROM policy_chunks WHERE policy_id = :policy_id RETURNING id"),
        {"policy_id": policy_id}
    )
    deleted = result.rowcount
    await db.commit()
    
    return {
        "policy_id": policy_id,
        "chunks_deleted": deleted,
        "message": f"Deleted {deleted} chunks for policy {policy_id}"
    }


@router.get("/stats/{policy_id}")
async def ingestion_stats(
    policy_id: int,
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
            FROM policy_chunks
            WHERE policy_id = :policy_id
        """),
        {"policy_id": policy_id}
    )
    stats = result.fetchone()
    
    return {
        "policy_id": policy_id,
        "chunk_count": stats.chunk_count or 0,
        "avg_chunk_length": round(stats.avg_chunk_length or 0, 2),
        "first_ingested": stats.first_ingested,
        "last_ingested": stats.last_ingested
    }
