"""
RAG Agent - handles document Q&A with retrieval.
"""

import asyncio
from typing import List, Optional

import structlog
from openai import APIConnectionError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.agents.critic import find_unsupported_claims, revision_request
from app.application.agents.query_rewriter import rewrite_query
from app.config import settings
from app.domain.entities.models import (
    DocumentChunk,
    QueryResult,
    QueryType,
    RetrievalResult,
)
from app.domain.services.citation_builder import CitationBuilder
from app.domain.services.context_window import merge_passage_window
from app.infrastructure.database.models import ContractPassage
from app.infrastructure.llm.openai_client import get_openai_client
from app.infrastructure.search.hybrid_search import HybridSearch, SearchResult, rrf_merge
from app.infrastructure.search.reranker import get_reranker

logger = structlog.get_logger()


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) agent for document Q&A.

    Workflow:
    1. Retrieve relevant document chunks
    2. Build context from chunks
    3. Generate answer with citations
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.search = HybridSearch(session)
        self.llm = get_openai_client()
        self.citation_builder = CitationBuilder()
        self.reranker = get_reranker()

    async def retrieve(
        self,
        query: str,
        contract_id: Optional[int] = None,
        top_k: Optional[int] = None,
        search_method: str = "hybrid",
    ) -> RetrievalResult:
        """
        Retrieve relevant document chunks.
        """
        effective_top_k = top_k if top_k is not None else settings.top_k_retrieval
        # Over-fetch a small candidate set so reranking can improve on the
        # retriever's initial ordering. The absolute cap keeps extra work bounded.
        candidate_count = max(
            effective_top_k,
            min(effective_top_k * settings.rerank_candidate_multiplier, 50),
        )
        try:
            results = await self.search.search(
                query=query, contract_id=contract_id, top_k=candidate_count, method=search_method
            )
            if settings.query_rewrite_enabled and search_method == "hybrid":
                results = await self._merge_rewritten_results(
                    query, contract_id, candidate_count, results
                )
        except (APIConnectionError, ConnectionError, TimeoutError):
            if search_method == "bm25":
                raise
            logger.warning("Embedding search unavailable; using lexical fallback")
            search_method = "bm25"
            results = await self.search.bm25_search(query, contract_id, candidate_count)

        chunks = [
            result.chunk
            for result in results
            if result.chunk.content.strip()
            and (contract_id is None or result.chunk.contract_id == contract_id)
            and result.score > 0
            and (search_method != "vector" or result.score >= settings.similarity_threshold)
        ]
        chunks = await asyncio.to_thread(self.reranker.rerank, query, chunks, effective_top_k)
        chunks = await self._expand_with_neighbors(chunks)

        logger.info(
            "Retrieved and reranked chunks",
            query=query[:100],
            num_chunks=len(chunks),
            candidate_count=len(results),
            method=search_method,
        )

        return RetrievalResult(chunks=chunks, search_method=search_method)

    async def _merge_rewritten_results(
        self,
        query: str,
        contract_id: Optional[int],
        candidate_count: int,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Search the rewritten phrasing too; the original ranking keeps equal weight."""
        rewritten = await rewrite_query(query)
        if rewritten is None:
            return results
        extra = await self.search.search(
            query=rewritten, contract_id=contract_id, top_k=candidate_count, method="hybrid"
        )
        logger.info("Query rewritten", original=query[:100], rewritten=rewritten[:100])
        return rrf_merge([(results, 1.0), (extra, 1.0)], settings.rrf_k)[:candidate_count]

    async def _expand_with_neighbors(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Give the model the passages around each match; citations keep the matched text."""
        radius = settings.context_neighbor_passages
        if radius <= 0:
            return chunks
        covered: set[tuple[int, int]] = set()
        for chunk in chunks:
            if chunk.passage_order is None or (chunk.contract_id, chunk.passage_order) in covered:
                continue
            rows = (
                await self.session.execute(
                    select(
                        ContractPassage.passage_order,
                        ContractPassage.content,
                        ContractPassage.source_offset_start,
                        ContractPassage.source_offset_end,
                    )
                    .where(
                        ContractPassage.contract_id == chunk.contract_id,
                        ContractPassage.passage_order.between(
                            chunk.passage_order - radius, chunk.passage_order + radius
                        ),
                    )
                    .order_by(ContractPassage.passage_order)
                )
            ).all()
            if not rows:
                continue
            chunk.context_text = merge_passage_window([(r[1], r[2], r[3]) for r in rows])
            covered.update((chunk.contract_id, r[0]) for r in rows)
        return chunks

    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        """
        Build context string from retrieved chunks.
        """
        if not chunks:
            return "No relevant information found in policy documents."

        context_parts = [
            self._format_context_chunk(chunk, index) for index, chunk in enumerate(chunks, 1)
        ]

        return "\n\n---\n\n".join(context_parts)

    @staticmethod
    def _format_context_chunk(chunk: DocumentChunk, index: int) -> str:
        section_info = f" - Section: {chunk.topic_title}" if chunk.topic_title else ""
        if chunk.source_page:
            section_info += f" (Page {chunk.source_page})"
        return f"[Source {index}{section_info}]\n{chunk.context_text or chunk.content}"

    def _select_context_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Keep highest-ranked whole chunks within the prompt context budget."""
        selected: List[DocumentChunk] = []
        context_length = 0
        separator_length = len("\n\n---\n\n")
        for chunk in chunks:
            rendered_length = len(self._format_context_chunk(chunk, len(selected) + 1))
            added_length = rendered_length + (separator_length if selected else 0)
            if context_length + added_length <= settings.max_context_chars:
                selected.append(chunk)
                context_length += added_length
        return selected

    async def generate_answer(self, query: str, chunks: List[DocumentChunk]) -> str:
        """
        Generate an answer based on retrieved chunks.
        """
        chunks = self._select_context_chunks(chunks)
        context = self._build_context(chunks)

        system_prompt = """You are an insurance policy expert assistant.

Your task is to answer questions about insurance benefit_contracts based on the provided context.

IMPORTANT RULES:
1. ONLY use information from the provided context
2. If the answer is not in the context, say "I don't have information about this in the provided policy documents"
3. Be specific and cite the source numbers [Source 1], [Source 2], etc.
4. Use bullet points for lists
5. Include specific values, limits, and conditions when mentioned
6. Never make up information not in the context
7. Give the direct answer first and keep it to at most four short sentences unless a list is needed

Format your answer clearly and professionally."""

        answer = await self.llm.generate_answer(
            query=query, context=context, system_prompt=system_prompt
        )

        return answer

    async def _revise_if_unsupported(
        self, query: str, chunks: List[DocumentChunk], answer: str
    ) -> str:
        claims = await find_unsupported_claims(query, self._build_context(chunks), answer)
        if not claims:
            return answer
        logger.warning("Critic flagged unsupported claims", count=len(claims))
        try:
            revised = await self.generate_answer(revision_request(query, claims), chunks)
        except Exception as exc:
            logger.warning("Revision failed; keeping draft", error_type=type(exc).__name__)
            return answer
        return revised if revised.strip() else answer

    async def answer(
        self,
        query: str,
        contract_id: Optional[int] = None,
        top_k: Optional[int] = None,
        search_method: str = "hybrid",
    ) -> QueryResult:
        """
        Full RAG pipeline: retrieve + generate.
        """
        import time

        start_time = time.time()

        # 1. Retrieve relevant chunks
        retrieval_result = await self.retrieve(
            query=query, contract_id=contract_id, top_k=top_k, search_method=search_method
        )

        chunks = self._select_context_chunks(retrieval_result.chunks)

        if not chunks:
            return QueryResult(
                query=query,
                query_type=QueryType.DOCUMENT_QA,
                answer="I don't have sufficient relevant evidence in the provided policy documents to answer this question.",
                latency_ms=int((time.time() - start_time) * 1000),
                model_used=settings.chat_model,
            )

        # 2. Generate answer
        answer = await self.generate_answer(query, chunks)
        if settings.critic_enabled:
            answer = await self._revise_if_unsupported(query, chunks, answer)

        # 3. Build citations
        citations = self.citation_builder.build_citations(chunks)

        # 4. Add inline citations to answer
        answer_with_sources = self.citation_builder.add_inline_citations(answer, citations)
        if not self.citation_builder.validate_answer(answer_with_sources, citations):
            logger.warning("Invalid model citations; returning source excerpts")
            answer_with_sources = "\n\n".join(
                f"{chunk.content} [Source {index}]" for index, chunk in enumerate(chunks, 1)
            )

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "RAG answer generated",
            query=query[:100],
            num_sources=len(citations),
            latency_ms=latency_ms,
        )

        return QueryResult(
            query=query,
            query_type=QueryType.DOCUMENT_QA,
            answer=answer_with_sources,
            citations=citations,
            latency_ms=latency_ms,
            model_used=settings.chat_model,
        )
