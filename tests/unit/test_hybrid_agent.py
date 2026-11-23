import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.agents import hybrid_agent as module
from app.application.agents.hybrid_agent import HybridAgent
from app.domain.entities.models import QueryResult, QueryType

pytestmark = pytest.mark.unit


def _result(answer: str) -> QueryResult:
    return QueryResult(query="q", query_type=QueryType.HYBRID, answer=answer, citations=[])


def _agent(monkeypatch, doc, sql):
    monkeypatch.setattr(module.settings, "hybrid_parallel", True)
    agent = HybridAgent(MagicMock())
    agent._extract_sub_queries = AsyncMock(
        return_value={"doc_query": "d", "sql_query": "s", "needs_doc": True, "needs_sql": True}
    )
    agent._run_doc_branch = doc
    agent._run_sql_branch = sql
    agent._synthesize_answer = AsyncMock(
        side_effect=lambda **kw: "|".join(filter(None, [kw["doc_answer"], kw["sql_answer"]]))
    )
    return agent


async def test_branches_run_concurrently(monkeypatch):
    started = asyncio.Event()

    async def doc(*_):
        await asyncio.wait_for(started.wait(), 1)  # only returns if sql branch started
        return _result("doc")

    async def sql(*_):
        started.set()
        return _result("sql")

    result = await _agent(monkeypatch, doc, sql).answer("q")
    assert result.answer == "doc|sql"


async def test_one_failed_branch_degrades(monkeypatch):
    async def doc(*_):
        raise RuntimeError("boom")

    async def sql(*_):
        return _result("sql")

    result = await _agent(monkeypatch, doc, sql).answer("q")
    assert result.answer == "sql"


async def test_all_branches_failing_raises(monkeypatch):
    async def fail(*_):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await _agent(monkeypatch, fail, fail).answer("q")
