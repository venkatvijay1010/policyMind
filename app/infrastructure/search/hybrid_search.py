"""
Hybrid search combining vector similarity and BM25 keyword search.
"""
from typing import List, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from rank_bm25 import BM25Okapi
import structlog

from app.infrastructure.database.models import PolicyChunk, Policy
from app.infrastructure.llm.embeddings import get_embedding_service
from app.domain.entities.models import DocumentChunk, SectionType
from app.config import settings

logger = structlog.get_logger()


@dataclass
class SearchResult:
    """A search result with score."""
    chunk: DocumentChunk
    score: float
    source: str  # 'vector', 'bm25', or 'hybrid'


class HybridSearch:
    """
    Hybrid search combining vector similarity and BM25.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.embedding_service = get_embedding_service()
    
    async def vector_search(
        self,
        query: str,
        policy_id: Optional[int] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Perform vector similarity search using pgvector.
        """
        # Get query embedding
        query_embedding = await self.embedding_service.embed_query(query)
        
        # Build query with optional policy filter
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        
        if policy_id:
            sql = text("""
                SELECT 
                    pc.id, pc.policy_id, pc.content, pc.section_type, 
                    pc.section_name, pc.page_number, pc.chunk_index, pc.token_count,
                    1 - (pc.embedding <=> :embedding::vector) as similarity,
                    p.policy_number
                FROM policy_chunks pc
                JOIN policies p ON pc.policy_id = p.id
                WHERE pc.policy_id = :policy_id
                ORDER BY pc.embedding <=> :embedding::vector
                LIMIT :limit
            """)
            result = await self.session.execute(
                sql, 
                {"embedding": embedding_str, "policy_id": policy_id, "limit": top_k}
            )
        else:
            sql = text("""
                SELECT 
                    pc.id, pc.policy_id, pc.content, pc.section_type, 
                    pc.section_name, pc.page_number, pc.chunk_index, pc.token_count,
                    1 - (pc.embedding <=> :embedding::vector) as similarity,
                    p.policy_number
                FROM policy_chunks pc
                JOIN policies p ON pc.policy_id = p.id
                ORDER BY pc.embedding <=> :embedding::vector
                LIMIT :limit
            """)
            result = await self.session.execute(
                sql, 
                {"embedding": embedding_str, "limit": top_k}
            )
        
        rows = result.fetchall()
        
        results = []
        for row in rows:
            chunk = DocumentChunk(
                id=row.id,
                policy_id=row.policy_id,
                content=row.content,
                section_type=SectionType(row.section_type) if row.section_type else None,
                section_name=row.section_name,
                page_number=row.page_number,
                chunk_index=row.chunk_index,
                token_count=row.token_count,
                score=float(row.similarity)
            )
            results.append(SearchResult(chunk=chunk, score=float(row.similarity), source='vector'))
        
        return results
    
    async def bm25_search(
        self,
        query: str,
        policy_id: Optional[int] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Perform BM25 keyword search.
        """
        # Fetch all chunks for BM25 (or filtered by policy)
        if policy_id:
            stmt = select(PolicyChunk).where(PolicyChunk.policy_id == policy_id)
        else:
            stmt = select(PolicyChunk).limit(1000)  # Limit for performance
        
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()
        
        if not chunks:
            return []
        
        # Tokenize documents for BM25
        tokenized_corpus = [chunk.content.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k results
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scored_chunks[:top_k]
        
        # Normalize scores to 0-1 range
        max_score = max(scores) if scores.any() else 1
        
        results = []
        for chunk, score in top_chunks:
            if score > 0:
                normalized_score = score / max_score if max_score > 0 else 0
                doc_chunk = DocumentChunk(
                    id=chunk.id,
                    policy_id=chunk.policy_id,
                    content=chunk.content,
                    section_type=SectionType(chunk.section_type) if chunk.section_type else None,
                    section_name=chunk.section_name,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    score=normalized_score
                )
                results.append(SearchResult(chunk=doc_chunk, score=normalized_score, source='bm25'))
        
        return results
    
    async def hybrid_search(
        self,
        query: str,
        policy_id: Optional[int] = None,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and BM25.
        Uses Reciprocal Rank Fusion (RRF) to merge results.
        """
        # Get results from both methods
        vector_results = await self.vector_search(query, policy_id, top_k=top_k * 2)
        bm25_results = await self.bm25_search(query, policy_id, top_k=top_k * 2)
        
        logger.debug(
            "Hybrid search results",
            vector_count=len(vector_results),
            bm25_count=len(bm25_results)
        )
        
        # Apply RRF fusion
        fused_results = self._rrf_fusion(
            vector_results, 
            bm25_results, 
            k=60,  # RRF constant
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        
        # Return top-k
        return fused_results[:top_k]
    
    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        k: int = 60,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF) to combine rankings.
        
        RRF score = sum(1 / (k + rank))
        """
        # Build rank dictionaries
        chunk_scores = {}  # chunk_id -> (chunk, total_score)
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result.chunk.id
            rrf_score = vector_weight * (1 / (k + rank))
            
            if chunk_id in chunk_scores:
                existing_chunk, existing_score = chunk_scores[chunk_id]
                chunk_scores[chunk_id] = (existing_chunk, existing_score + rrf_score)
            else:
                chunk_scores[chunk_id] = (result.chunk, rrf_score)
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.chunk.id
            rrf_score = bm25_weight * (1 / (k + rank))
            
            if chunk_id in chunk_scores:
                existing_chunk, existing_score = chunk_scores[chunk_id]
                chunk_scores[chunk_id] = (existing_chunk, existing_score + rrf_score)
            else:
                chunk_scores[chunk_id] = (result.chunk, rrf_score)
        
        # Sort by combined score
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build final results
        return [
            SearchResult(chunk=chunk, score=score, source='hybrid')
            for chunk, score in sorted_results
        ]
    
    async def search(
        self,
        query: str,
        policy_id: Optional[int] = None,
        top_k: int = 5,
        method: str = "hybrid"  # 'vector', 'bm25', or 'hybrid'
    ) -> List[SearchResult]:
        """
        Main search method - dispatches to appropriate search type.
        """
        if method == "vector":
            return await self.vector_search(query, policy_id, top_k)
        elif method == "bm25":
            return await self.bm25_search(query, policy_id, top_k)
        else:
            return await self.hybrid_search(query, policy_id, top_k)
