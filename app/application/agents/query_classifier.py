"""
Query classifier agent - routes queries to appropriate handler.
"""
from typing import Optional
import structlog

from app.domain.entities.models import QueryType, ClassificationResult
from app.infrastructure.llm.openai_client import get_openai_client

logger = structlog.get_logger()


class QueryClassifier:
    """
    Classifies user queries to route them to the appropriate agent.
    
    - document_qa: Questions about policy terms, coverage, exclusions
    - records_sql: Questions requiring database queries on synthetic service-case data
    - hybrid: Questions needing both document and data context
    """
    
    def __init__(self):
        self.llm = get_openai_client()
        
        # Keywords that suggest SQL queries
        self.sql_keywords = {
            "how many", "total", "count", "sum", "average", "avg",
            "service case", "service_cases", "requested amount", "statistics", "trend",
            "in 2024", "in 2023", "last year", "this quarter",
            "by status", "by provider", "top 10", "breakdown"
        }
        
        # Keywords that suggest document queries
        self.doc_keywords = {
            "covered", "coverage", "exclusion", "excluded",
            "eligibility delay", "waiting period", "limit", "inner cap", "fixed share",
            "percentage share",
            "contract terms", "conditions", "eligibility",
            "pre-existing", "documents required", "service-case process"
        }
    
    async def classify(self, query: str) -> ClassificationResult:
        """
        Classify the query type.
        Uses a combination of keyword matching and LLM classification.
        """
        query_lower = query.lower()
        
        # Quick keyword-based classification
        sql_score = sum(1 for kw in self.sql_keywords if kw in query_lower)
        doc_score = sum(1 for kw in self.doc_keywords if kw in query_lower)
        
        logger.debug(
            "Keyword classification scores",
            query=query[:100],
            sql_score=sql_score,
            doc_score=doc_score
        )
        
        # If keywords are clear, skip LLM call
        if sql_score > 0 and doc_score == 0:
            return ClassificationResult(
                query_type=QueryType.RECORDS_SQL,
                confidence=0.9,
                reasoning="Query contains service_cases/statistics keywords"
            )
        
        if doc_score > 0 and sql_score == 0:
            return ClassificationResult(
                query_type=QueryType.DOCUMENT_QA,
                confidence=0.9,
                reasoning="Query contains policy/coverage keywords"
            )
        
        # Ambiguous - use LLM
        try:
            result = await self.llm.classify_query(query)
            
            query_type = QueryType(result.get("query_type", "document_qa"))
            confidence = float(result.get("confidence", 0.8))
            reasoning = result.get("reasoning", "")
            
            return ClassificationResult(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning
            )
        except Exception as e:
            logger.error("LLM classification failed", error=str(e))
            # Default to document_qa on failure
            return ClassificationResult(
                query_type=QueryType.DOCUMENT_QA,
                confidence=0.5,
                reasoning="Fallback due to classification error"
            )
    
    def is_sql_safe(self, query: str) -> bool:
        """
        Check if a query is safe (no SQL injection patterns).
        """
        dangerous_patterns = [
            "drop ", "delete ", "insert ", "update ", "alter ",
            "create ", "truncate ", "--", ";--", "/*", "*/"
        ]
        query_lower = query.lower()
        return not any(pattern in query_lower for pattern in dangerous_patterns)
