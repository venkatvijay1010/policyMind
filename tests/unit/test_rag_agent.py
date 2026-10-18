from unittest.mock import AsyncMock

import pytest

from app.application.agents.rag_agent import RAGAgent
from app.domain.entities.models import DocumentChunk
from app.domain.services.citation_builder import CitationBuilder
from app.infrastructure.search.hybrid_search import SearchResult

pytestmark = pytest.mark.unit


def make_agent():
    agent = RAGAgent(AsyncMock())
    agent.search.search = AsyncMock(return_value=[])
    agent.generate_answer = AsyncMock(return_value="Covered [Source 99]")
    return agent


async def test_empty_evidence_abstains_without_llm():
    agent = make_agent()
    result = await agent.answer("What is covered?")
    assert "sufficient relevant evidence" in result.answer
    assert result.citations == []
    agent.generate_answer.assert_not_awaited()


async def test_weak_vector_and_wrong_scope_are_rejected():
    agent = make_agent()
    agent.search.search.return_value = [
        SearchResult(DocumentChunk(1, 1, "Weak"), 0.01, "vector"),
        SearchResult(DocumentChunk(2, 2, "Wrong scope"), 0.99, "vector"),
    ]
    result = await agent.retrieve("query", contract_id=1, search_method="vector")
    assert result.chunks == []


async def test_embedding_failure_uses_lexical_fallback():
    agent = make_agent()
    agent.search.search.side_effect = ConnectionError()
    agent.search.bm25_search = AsyncMock(
        return_value=[SearchResult(DocumentChunk(1, 1, "Dental care is covered."), 1.0, "bm25")]
    )
    result = await agent.retrieve("Dental")
    assert result.search_method == "bm25"
    assert len(result.chunks) == 1


async def test_invalid_citations_fall_back_to_excerpts():
    agent = make_agent()
    agent.search.search.return_value = [
        SearchResult(DocumentChunk(1, 1, "Dental care is covered."), 1.0, "bm25")
    ]
    result = await agent.answer("Dental coverage?", search_method="bm25")
    assert result.answer == "Dental care is covered. [Source 1]"
    assert CitationBuilder.validate_answer(result.answer, result.citations)
