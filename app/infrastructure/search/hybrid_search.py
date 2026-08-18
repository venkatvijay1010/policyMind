"""Hybrid search combining local cosine similarity and BM25 keyword search."""

import json
import math
from dataclasses import dataclass
from typing import List, Optional

import structlog
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.domain.entities.models import DocumentChunk, SectionType
from app.infrastructure.database.models import ContractPassage
from app.infrastructure.llm.embeddings import get_embedding_service

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
        self, query: str, contract_id: Optional[int] = None, top_k: int = 10
    ) -> List[SearchResult]:
        """
        Perform vector similarity search using locally stored JSON embeddings.

        SQLite does not include pgvector, so this intentionally loads the small
        demo corpus and ranks it in Python. It keeps local setup dependency-free
        beyond SQLite and is not intended for millions of passages.
        """
        query_embedding = await self.embedding_service.embed_query(query)
        statement = select(ContractPassage).options(selectinload(ContractPassage.policy))
        if contract_id is not None:
            statement = statement.where(ContractPassage.contract_id == contract_id)
        else:
            statement = statement.limit(5000)

        result = await self.session.execute(statement)
        passages = result.scalars().all()

        ranked: list[SearchResult] = []
        for passage in passages:
            embedding = self._embedding_values(passage.embedding)
            score = self._cosine_similarity(query_embedding, embedding)
            if score is None:
                continue
            ranked.append(
                SearchResult(
                    chunk=self._to_document_chunk(passage, score),
                    score=score,
                    source="vector",
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _embedding_values(value: object) -> Optional[list[float]]:
        """Normalize JSON or legacy string embedding values into float vectors."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(value, list):
            return None
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _cosine_similarity(left: list[float], right: Optional[list[float]]) -> Optional[float]:
        """Return cosine similarity, skipping malformed or mismatched vectors."""
        if right is None or len(left) != len(right) or not left:
            return None
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return None
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    @staticmethod
    def _section_type(value: Optional[str]) -> Optional[SectionType]:
        if not value:
            return None
        try:
            return SectionType(value)
        except ValueError:
            return None

    def _to_document_chunk(self, passage: ContractPassage, score: float) -> DocumentChunk:
        """Map a persistence passage to the domain retrieval model."""
        return DocumentChunk(
            id=passage.id,
            contract_id=passage.contract_id,
            contract_title=passage.policy.contract_title if passage.policy else None,
            content=passage.content,
            topic_category=self._section_type(passage.topic_category),
            topic_title=passage.topic_title,
            source_page=passage.source_page,
            passage_order=passage.passage_order,
            token_count=passage.token_count,
            score=score,
        )

    async def bm25_search(
        self, query: str, contract_id: Optional[int] = None, top_k: int = 10
    ) -> List[SearchResult]:
        """
        Perform BM25 keyword search.
        """
        # Fetch all chunks for BM25 (or filtered by policy)
        if contract_id:
            stmt = (
                select(ContractPassage)
                .options(selectinload(ContractPassage.policy))
                .where(ContractPassage.contract_id == contract_id)
            )
        else:
            stmt = select(ContractPassage).options(selectinload(ContractPassage.policy)).limit(1000)

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
                doc_chunk = self._to_document_chunk(chunk, normalized_score)
                results.append(SearchResult(chunk=doc_chunk, score=normalized_score, source="bm25"))

        return results

    async def hybrid_search(
        self,
        query: str,
        contract_id: Optional[int] = None,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and BM25.
        Uses Reciprocal Rank Fusion (RRF) to merge results.
        """
        # Get results from both methods
        vector_results = await self.vector_search(query, contract_id, top_k=top_k * 2)
        bm25_results = await self.bm25_search(query, contract_id, top_k=top_k * 2)

        logger.debug(
            "Hybrid search results", vector_count=len(vector_results), bm25_count=len(bm25_results)
        )

        # Apply RRF fusion
        fused_results = self._rrf_fusion(
            vector_results,
            bm25_results,
            k=60,  # RRF constant
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        # Return top-k
        return fused_results[:top_k]

    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        k: int = 60,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF) to combine rankings.

        RRF score = sum(1 / (k + rank))
        """
        # Build rank dictionaries
        chunk_scores: dict[int, tuple[DocumentChunk, float]] = {}

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
        sorted_results = sorted(chunk_scores.values(), key=lambda x: x[1], reverse=True)

        # Build final results
        return [
            SearchResult(chunk=chunk, score=score, source="hybrid")
            for chunk, score in sorted_results
        ]

    async def search(
        self,
        query: str,
        contract_id: Optional[int] = None,
        top_k: int = 5,
        method: str = "hybrid",  # 'vector', 'bm25', or 'hybrid'
    ) -> List[SearchResult]:
        """
        Main search method - dispatches to appropriate search type.
        """
        if method == "vector":
            return await self.vector_search(query, contract_id, top_k)
        elif method == "bm25":
            return await self.bm25_search(query, contract_id, top_k)
        else:
            return await self.hybrid_search(query, contract_id, top_k)
