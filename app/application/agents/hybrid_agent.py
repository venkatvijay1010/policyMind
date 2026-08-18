"""
Hybrid Agent - combines RAG and SQL for comprehensive answers.
"""

from typing import Any, List, Optional, cast

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.agents.rag_agent import RAGAgent
from app.application.agents.sql_agent import SQLAgent
from app.config import settings
from app.domain.entities.models import Citation, QueryResult, QueryType
from app.infrastructure.llm.openai_client import get_openai_client

logger = structlog.get_logger()


class HybridAgent:
    """
    Hybrid Agent that combines document retrieval and SQL queries.

    Used for complex questions that need:
    - Policy context (what's covered)
    - Data analysis (claim statistics)

    Example: "How many service cases were declined for family-support coverage
             and what does the policy say about maternity benefits?"
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.rag_agent = RAGAgent(session)
        self.sql_agent = SQLAgent(session)
        self.llm = get_openai_client()

    async def _extract_sub_queries(self, query: str) -> dict[str, Any]:
        """
        Break down a hybrid query into document and SQL components.
        """
        prompt = f"""Analyze this insurance query and break it into components:

Query: {query}

Identify:
1. doc_query: What should we search in policy documents? (coverage, terms, exclusions)
2. sql_query: What data should we query from service_cases database? (statistics, counts, trends)

Return JSON:
{{
  "doc_query": "...",
  "sql_query": "...",
  "needs_doc": true/false,
  "needs_sql": true/false
}}"""

        try:
            result = await self.llm.generate_json(
                prompt=prompt, system_prompt="You are an expert at analyzing insurance queries."
            )
            return result
        except Exception as e:
            logger.error("Sub-query extraction failed", error=str(e))
            # Default: need both
            return {"doc_query": query, "sql_query": query, "needs_doc": True, "needs_sql": True}

    async def _synthesize_answer(
        self, original_query: str, doc_answer: Optional[str], sql_answer: Optional[str]
    ) -> str:
        """
        Combine document and SQL answers into a coherent response.
        """
        parts = []
        if doc_answer:
            parts.append(f"**Policy Information:**\n{doc_answer}")
        if sql_answer:
            parts.append(f"**Claims Data Analysis:**\n{sql_answer}")

        if len(parts) == 0:
            return "I couldn't find relevant information to answer your question."

        combined = "\n\n".join(parts)

        # If both answers exist, ask LLM to synthesize
        if doc_answer and sql_answer:
            prompt = f"""The user asked: {original_query}

Here's information from two sources:

{combined}

Please synthesize this into a single, coherent answer that:
1. Directly answers the user's question
2. Connects policy terms with actual service_cases data
3. Provides actionable insights if applicable
"""

            try:
                synthesized = await self.llm.generate(
                    prompt=prompt,
                    system_prompt="You are an insurance expert synthesizing policy and service_cases information.",
                    temperature=0.3,
                )
                return cast(str, synthesized)
            except Exception as e:
                logger.error("Answer synthesis failed", error=str(e))
                return combined

        return combined

    async def answer(
        self,
        query: str,
        contract_id: Optional[int] = None,
        search_method: str = "hybrid",
    ) -> QueryResult:
        """
        Execute hybrid RAG + SQL pipeline.
        """
        import time

        start_time = time.time()

        # 1. Break down the query
        sub_queries = await self._extract_sub_queries(query)

        doc_answer: Optional[str] = None
        sql_answer: Optional[str] = None
        all_citations: List[Citation] = []
        sql_query: Optional[str] = None
        sql_result: Optional[List[dict[str, Any]]] = None

        # 2. Execute document search if needed
        if bool(sub_queries.get("needs_doc", True)):
            doc_query = str(sub_queries.get("doc_query") or query)
            rag_result = await self.rag_agent.answer(
                query=doc_query,
                contract_id=contract_id,
                search_method=search_method,
            )
            doc_answer = rag_result.answer
            all_citations.extend(rag_result.citations)

        # 3. Execute SQL if needed
        if bool(sub_queries.get("needs_sql", True)):
            sql_query_text = str(sub_queries.get("sql_query") or query)
            sql_result_obj = await self.sql_agent.answer(sql_query_text)
            sql_answer = sql_result_obj.answer
            sql_query = sql_result_obj.sql_query
            sql_result = sql_result_obj.sql_result

        # 4. Synthesize final answer
        final_answer = await self._synthesize_answer(
            original_query=query, doc_answer=doc_answer, sql_answer=sql_answer
        )

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Hybrid answer generated",
            query=query[:100],
            had_doc=doc_answer is not None,
            had_sql=sql_answer is not None,
            latency_ms=latency_ms,
        )

        return QueryResult(
            query=query,
            query_type=QueryType.HYBRID,
            answer=final_answer,
            citations=all_citations,
            sql_query=sql_query,
            sql_result=sql_result,
            latency_ms=latency_ms,
            model_used=settings.chat_model,
        )
