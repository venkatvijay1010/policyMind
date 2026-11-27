"""Hybrid search combining local cosine similarity and BM25 keyword search."""

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import structlog
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.domain.entities.models import DocumentChunk, SectionType
from app.infrastructure.database.models import ContractPassage, PassageSource, SourceDocument
from app.infrastructure.llm.embeddings import get_embedding_service
from app.infrastructure.search.keyword_store import ElasticsearchKeywordStore, get_keyword_store
from app.infrastructure.search.vector_store import QdrantVectorStore, get_vector_store

logger = structlog.get_logger()


def _passage_load_options():
    """Load passage provenance without pulling original document bytes into memory."""
    return (
        selectinload(ContractPassage.policy),
        selectinload(ContractPassage.source_links)
        .selectinload(PassageSource.source_document)
        .defer(SourceDocument.original_bytes),
    )


@dataclass
class SearchResult:
    """A search result with score."""

    chunk: DocumentChunk
    score: float
    source: str  # 'vector', 'bm25', or 'hybrid'


def rrf_merge(
    rankings: Sequence[tuple[Sequence[SearchResult], float]], k: int = 60
) -> List[SearchResult]:
    """Weighted RRF over ranked lists; a chunk's score is the sum of weight / (k + rank)."""
    scores: dict[int, tuple[DocumentChunk, float]] = {}
    for results, weight in rankings:
        for rank, result in enumerate(results, 1):
            chunk, total = scores.get(result.chunk.id, (result.chunk, 0.0))
            scores[result.chunk.id] = (chunk, total + weight / (k + rank))
    ordered = sorted(scores.values(), key=lambda item: item[1], reverse=True)
    return [SearchResult(chunk=chunk, score=score, source="hybrid") for chunk, score in ordered]


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
        store = get_vector_store()
        if store is not None:
            return await self._qdrant_search(store, query_embedding, contract_id, top_k)
        statement = select(ContractPassage).options(*_passage_load_options())
        if contract_id is not None:
            statement = statement.where(ContractPassage.contract_id == contract_id)
        else:
            statement = statement.limit(5000)

        result = await self.session.execute(statement)
        passages = result.scalars().all()

        ranked: list[SearchResult] = []
        stale = 0
        for passage in passages:
            if not self._embedding_is_current(passage):
                stale += 1
                continue
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
        self._warn_stale(stale)
        return ranked[:top_k]

    async def _qdrant_search(
        self,
        store: QdrantVectorStore,
        query_embedding: list[float],
        contract_id: Optional[int],
        top_k: int,
    ) -> List[SearchResult]:
        """Rank in Qdrant, then load the passages from the database."""
        hits = await store.search(query_embedding, contract_id, top_k)
        if not hits:
            return []
        rows = await self.session.execute(
            select(ContractPassage)
            .options(*_passage_load_options())
            .where(ContractPassage.id.in_([hit.passage_id for hit in hits]))
        )
        passages = {passage.id: passage for passage in rows.scalars().all()}

        results: list[SearchResult] = []
        stale = 0
        for hit in hits:
            passage = passages.get(hit.passage_id)
            # Skip points left behind by a replaced scope whose row id was reused elsewhere.
            if passage is None or passage.contract_id != hit.contract_id:
                continue
            if not self._embedding_is_current(passage):
                stale += 1
                continue
            results.append(
                SearchResult(
                    chunk=self._to_document_chunk(passage, hit.score),
                    score=hit.score,
                    source="vector",
                )
            )
        self._warn_stale(stale)
        return results

    @staticmethod
    def _embedding_is_current(passage: ContractPassage) -> bool:
        """Vectors from another model are not comparable; unrecorded (legacy) rows are trusted."""
        return passage.embedding_model in (None, settings.embedding_model)

    @staticmethod
    def _warn_stale(count: int) -> None:
        if count:
            logger.warning(
                "Skipped passages embedded with another model; run scripts/reembed_passages.py",
                skipped=count,
                current_model=settings.embedding_model,
            )

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
        source = passage.source_links[0].source_document if passage.source_links else None
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
            source_document_id=source.id if source else None,
            source_filename=source.filename if source else None,
        )

    async def _elasticsearch_bm25(
        self,
        store: ElasticsearchKeywordStore,
        query: str,
        contract_id: Optional[int],
        top_k: int,
    ) -> List[SearchResult]:
        """Rank in Elasticsearch, then load the passages from the database."""
        hits = await store.search(query, contract_id, top_k)
        if not hits:
            return []
        rows = await self.session.execute(
            select(ContractPassage)
            .options(*_passage_load_options())
            .where(ContractPassage.id.in_([hit.passage_id for hit in hits]))
        )
        passages = {passage.id: passage for passage in rows.scalars().all()}

        max_score = max(hit.score for hit in hits)
        results: list[SearchResult] = []
        for hit in hits:
            passage = passages.get(hit.passage_id)
            # Skip documents left behind by a replaced scope whose row id was reused elsewhere.
            if passage is None or passage.contract_id != hit.contract_id or hit.score <= 0:
                continue
            normalized = hit.score / max_score if max_score > 0 else 0.0
            results.append(
                SearchResult(
                    chunk=self._to_document_chunk(passage, normalized),
                    score=normalized,
                    source="bm25",
                )
            )
        return results

    async def bm25_search(
        self, query: str, contract_id: Optional[int] = None, top_k: int = 10
    ) -> List[SearchResult]:
        """
        Perform BM25 keyword search.
        """
        keyword_store = get_keyword_store()
        if keyword_store is not None:
            return await self._elasticsearch_bm25(keyword_store, query, contract_id, top_k)

        # Fetch all chunks for BM25 (or filtered by policy)
        if contract_id:
            stmt = (
                select(ContractPassage)
                .options(*_passage_load_options())
                .where(ContractPassage.contract_id == contract_id)
            )
        else:
            stmt = select(ContractPassage).options(*_passage_load_options()).limit(1000)

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
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and BM25.
        Uses weighted Reciprocal Rank Fusion (RRF) to merge results.
        """
        vector_weight = settings.hybrid_vector_weight if vector_weight is None else vector_weight
        bm25_weight = settings.hybrid_bm25_weight if bm25_weight is None else bm25_weight
        # Get results from both methods
        vector_results = await self.vector_search(query, contract_id, top_k=top_k * 2)
        vector_results = [
            result for result in vector_results if result.score >= settings.similarity_threshold
        ]
        bm25_results = await self.bm25_search(query, contract_id, top_k=top_k * 2)

        logger.debug(
            "Hybrid search results", vector_count=len(vector_results), bm25_count=len(bm25_results)
        )

        # Apply RRF fusion
        fused_results = self._rrf_fusion(
            vector_results,
            bm25_results,
            k=settings.rrf_k,
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

        RRF score = sum(weight / (k + rank))
        """
        return rrf_merge([(vector_results, vector_weight), (bm25_results, bm25_weight)], k)

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
