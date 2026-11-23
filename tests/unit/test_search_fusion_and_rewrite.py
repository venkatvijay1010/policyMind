from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.agents import query_rewriter
from app.application.agents.rag_agent import RAGAgent
from app.config import settings
from app.domain.entities.models import DocumentChunk
from app.infrastructure.search.hybrid_search import HybridSearch, SearchResult, rrf_merge

pytestmark = pytest.mark.unit


def _hit(chunk_id: int) -> SearchResult:
    return SearchResult(DocumentChunk(id=chunk_id, contract_id=1, content="x"), 1.0, "vector")


def _llm(result=None, error=None):
    llm = MagicMock()
    llm.generate_json = AsyncMock(return_value=result, side_effect=error)
    return llm


def test_rrf_weights_change_the_winner():
    vector, bm25 = [_hit(1), _hit(2)], [_hit(2), _hit(1)]
    assert rrf_merge([(vector, 0.9), (bm25, 0.1)])[0].chunk.id == 1
    assert rrf_merge([(vector, 0.1), (bm25, 0.9)])[0].chunk.id == 2


def test_rrf_sums_scores_across_lists():
    merged = rrf_merge([([_hit(1)], 1.0), ([_hit(1)], 1.0)], k=60)
    assert merged[0].score == pytest.approx(2 / 61)


def test_hybrid_fusion_uses_given_weights():
    fused = HybridSearch._rrf_fusion(
        HybridSearch.__new__(HybridSearch),
        [_hit(1), _hit(2)],
        [_hit(2), _hit(1)],
        vector_weight=0.1,
        bm25_weight=0.9,
    )
    assert fused[0].chunk.id == 2


async def test_rewrite_returns_distinct_query(monkeypatch):
    monkeypatch.setattr(
        query_rewriter,
        "get_openai_client",
        lambda: _llm({"search_query": " hospitalisation limit "}),
    )
    assert await query_rewriter.rewrite_query("hospital stay cap") == "hospitalisation limit"


@pytest.mark.parametrize(
    "payload", [{}, {"search_query": ""}, {"search_query": "SAME"}, {"search_query": "y" * 400}]
)
async def test_rewrite_rejects_useless_output(monkeypatch, payload):
    monkeypatch.setattr(query_rewriter, "get_openai_client", lambda: _llm(payload))
    assert await query_rewriter.rewrite_query("same") is None


async def test_rewrite_fails_open(monkeypatch):
    monkeypatch.setattr(query_rewriter, "get_openai_client", lambda: _llm(error=RuntimeError("x")))
    assert await query_rewriter.rewrite_query("q") is None


async def test_rewritten_results_are_merged(monkeypatch):
    monkeypatch.setattr(
        "app.application.agents.rag_agent.rewrite_query", AsyncMock(return_value="better")
    )
    agent = RAGAgent.__new__(RAGAgent)
    agent.search = MagicMock()
    agent.search.search = AsyncMock(return_value=[_hit(2), _hit(3)])
    merged = await agent._merge_rewritten_results("q", None, 5, [_hit(1), _hit(2)])
    assert [r.chunk.id for r in merged][0] == 2
    assert {r.chunk.id for r in merged} == {1, 2, 3}
    agent.search.search.assert_awaited_once()
    assert agent.search.search.await_args.kwargs["query"] == "better"


async def test_no_rewrite_keeps_original_results(monkeypatch):
    monkeypatch.setattr(
        "app.application.agents.rag_agent.rewrite_query", AsyncMock(return_value=None)
    )
    agent = RAGAgent.__new__(RAGAgent)
    original = [_hit(1)]
    assert await agent._merge_rewritten_results("q", None, 5, original) is original


def test_defaults_keep_existing_behaviour():
    assert settings.hybrid_vector_weight == 0.7
    assert settings.hybrid_bm25_weight == 0.3
    assert settings.rrf_k == 60
    assert settings.query_rewrite_enabled is False
