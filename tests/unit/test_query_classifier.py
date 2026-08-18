"""
Unit tests for query classifier.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.application.agents.query_classifier import QueryClassifier
from app.domain.entities.models import QueryType

pytestmark = pytest.mark.unit


class TestQueryClassifier:
    """Tests for QueryClassifier."""

    def setup_method(self):
        """Setup test fixtures."""
        self.classifier = QueryClassifier()

    @pytest.mark.asyncio
    async def test_classify_document_query_keywords(self):
        """Test classification of document queries by keywords."""
        queries = [
            "What is covered under family support?",
            "What are the exclusions?",
            "Is dental treatment covered?",
            "What is the waiting period for pre-existing diseases?",
        ]

        for query in queries:
            result = await self.classifier.classify(query)
            assert result.query_type == QueryType.DOCUMENT_QA
            assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_classify_sql_query_keywords(self):
        """Test classification of SQL queries by keywords."""
        queries = [
            "How many service cases were opened in 2024?",
            "What is the total requested amount by status?",
            "Show top 10 providers by service cases",
            "What is the average requested amount?",
        ]

        for query in queries:
            result = await self.classifier.classify(query)
            assert result.query_type == QueryType.RECORDS_SQL
            assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_sql_safety_check_safe(self):
        """Test SQL safety check allows safe queries."""
        safe_queries = [
            "How many service cases were declined?",
            "What is the average requested amount?",
            "Show service cases by provider",
        ]

        for query in safe_queries:
            assert self.classifier.is_sql_safe(query) is True

    @pytest.mark.asyncio
    async def test_sql_safety_check_unsafe(self):
        """Test SQL safety check blocks dangerous queries."""
        unsafe_queries = [
            "DROP TABLE service_cases",
            "DELETE FROM benefit_contracts",
            "INSERT INTO service_cases VALUES",
            "UPDATE participants SET status",
            "select * from service_cases; --",
        ]

        for query in unsafe_queries:
            assert self.classifier.is_sql_safe(query) is False

    @pytest.mark.asyncio
    async def test_ambiguous_query_uses_llm(self):
        """Test that ambiguous queries use LLM classification."""
        # Mock the LLM client
        with patch.object(self.classifier, "llm") as mock_llm:
            mock_llm.classify_query = AsyncMock(
                return_value={
                    "query_type": "hybrid",
                    "confidence": 0.85,
                    "reasoning": "Query needs both document and data context",
                }
            )

            # This query has both doc and sql keywords
            await self.classifier.classify(
                "What do the case statistics show and what do the contract terms say about exclusions?"
            )

            # Should use LLM for ambiguous queries
            mock_llm.classify_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_is_model_classified_as_chat(self):
        """Casual chat is decided by the model rather than a fixed greeting reply."""
        with patch.object(self.classifier, "llm") as mock_llm:
            mock_llm.classify_query = AsyncMock(
                return_value={
                    "query_type": "chat",
                    "confidence": 0.97,
                    "reasoning": "Casual social conversation",
                }
            )

            result = await self.classifier.classify("how r u?")

            assert result.query_type == QueryType.CHAT
            mock_llm.classify_query.assert_awaited_once_with("how r u?", [])

    @pytest.mark.asyncio
    async def test_classification_fallback_on_error(self):
        """Test fallback to document_qa on LLM error."""
        with patch.object(self.classifier, "llm") as mock_llm:
            mock_llm.classify_query = AsyncMock(side_effect=Exception("API error"))

            # Ambiguous query that would normally use LLM
            result = await self.classifier.classify("some ambiguous query here")

            # Should fallback to document_qa
            assert result.query_type == QueryType.DOCUMENT_QA
            assert result.confidence == 0.5


class TestQueryTypeKeywords:
    """Test keyword matching logic."""

    def setup_method(self):
        self.classifier = QueryClassifier()

    def test_sql_keywords_present(self):
        """Verify SQL keywords are defined."""
        assert "how many" in self.classifier.sql_keywords
        assert "total" in self.classifier.sql_keywords
        assert "average" in self.classifier.sql_keywords

    def test_doc_keywords_present(self):
        """Verify document keywords are defined."""
        assert "covered" in self.classifier.doc_keywords
        assert "exclusion" in self.classifier.doc_keywords
        assert "waiting period" in self.classifier.doc_keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
