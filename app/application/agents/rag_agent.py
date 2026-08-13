"""
RAG Agent - handles document Q&A with retrieval.
"""
from typing import List, Optional
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.models import (
    QueryResult, QueryType, DocumentChunk, Citation, RetrievalResult
)
from app.domain.services.citation_builder import CitationBuilder
from app.infrastructure.llm.openai_client import get_openai_client
from app.infrastructure.search.hybrid_search import HybridSearch
from app.config import settings

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
    
    async def retrieve(
        self,
        query: str,
        contract_id: Optional[int] = None,
        top_k: int = 5,
        search_method: str = "hybrid"
    ) -> RetrievalResult:
        """
        Retrieve relevant document chunks.
        """
        results = await self.search.search(
            query=query,
            contract_id=contract_id,
            top_k=top_k,
            method=search_method
        )
        
        chunks = [r.chunk for r in results]
        
        logger.info(
            "Retrieved chunks",
            query=query[:100],
            num_chunks=len(chunks),
            method=search_method
        )
        
        return RetrievalResult(
            chunks=chunks,
            search_method=search_method
        )
    
    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        """
        Build context string from retrieved chunks.
        """
        if not chunks:
            return "No relevant information found in policy documents."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            section_info = ""
            if chunk.topic_title:
                section_info = f" - Section: {chunk.topic_title}"
            if chunk.source_page:
                section_info += f" (Page {chunk.source_page})"
            
            context_parts.append(f"[Source {i}{section_info}]\n{chunk.content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    async def generate_answer(
        self,
        query: str,
        chunks: List[DocumentChunk]
    ) -> str:
        """
        Generate an answer based on retrieved chunks.
        """
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

Format your answer clearly and professionally."""

        answer = await self.llm.generate_answer(
            query=query,
            context=context,
            system_prompt=system_prompt
        )
        
        return answer
    
    async def answer(
        self,
        query: str,
        contract_id: Optional[int] = None,
        top_k: int = 5,
        search_method: str = "hybrid"
    ) -> QueryResult:
        """
        Full RAG pipeline: retrieve + generate.
        """
        import time
        start_time = time.time()
        
        # 1. Retrieve relevant chunks
        retrieval_result = await self.retrieve(
            query=query,
            contract_id=contract_id,
            top_k=top_k,
            search_method=search_method
        )
        
        chunks = retrieval_result.chunks
        
        # 2. Generate answer
        answer = await self.generate_answer(query, chunks)
        
        # 3. Build citations
        citations = self.citation_builder.build_citations(chunks)
        
        # 4. Add inline citations to answer
        answer_with_sources = self.citation_builder.add_inline_citations(answer, citations)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "RAG answer generated",
            query=query[:100],
            num_sources=len(citations),
            latency_ms=latency_ms
        )
        
        return QueryResult(
            query=query,
            query_type=QueryType.DOCUMENT_QA,
            answer=answer_with_sources,
            citations=citations,
            latency_ms=latency_ms,
            model_used=settings.chat_model
        )
