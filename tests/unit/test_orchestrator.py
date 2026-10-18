from unittest.mock import AsyncMock

import pytest

from app.application.graph.orchestrator import GraphExecutionError, PolicyMindGraph
from app.config import settings
from app.domain.entities.models import ClassificationResult, QueryResult, QueryType

pytestmark = pytest.mark.unit


def make_graph():
    graph = PolicyMindGraph(AsyncMock())
    graph.classifier.classify = AsyncMock(return_value=ClassificationResult(QueryType.CHAT, 1.0))
    graph.chat_agent.answer = AsyncMock(return_value=QueryResult("hi", QueryType.CHAT, "Hello"))
    return graph


async def test_transient_failure_retries_and_finalizes():
    graph = make_graph()
    graph.chat_agent.answer.side_effect = [
        ConnectionError(),
        QueryResult("hi", QueryType.CHAT, "Hello"),
    ]
    events = [event async for event in graph.stream("hi")]
    assert any("retry" in event for event in events)
    assert events[-1]["finalize"]["result"].answer == "Hello"
    assert graph.chat_agent.answer.await_count == 2


async def test_permanent_failure_uses_structured_error():
    graph = make_graph()
    graph.chat_agent.answer.side_effect = ValueError("private details")
    with pytest.raises(GraphExecutionError) as error:
        await graph.invoke("hi")
    assert error.value.code == "execution_failed"
    assert "private details" not in str(error.value)
    assert graph.chat_agent.answer.await_count == 1


async def test_node_timeout_is_bounded(monkeypatch):
    import asyncio

    graph = make_graph()
    monkeypatch.setattr(settings, "node_timeout_seconds", 0.01)
    monkeypatch.setattr(settings, "graph_max_retries", 0)

    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    graph.chat_agent.answer.side_effect = blocked
    with pytest.raises(GraphExecutionError) as error:
        await graph.invoke("hi")
    assert error.value.code == "node_timeout"


async def test_classification_failure_calls_no_agents():
    graph = make_graph()
    graph.classifier.classify.side_effect = ValueError()
    graph.rag_agent.answer = AsyncMock(
        return_value=QueryResult("hi", QueryType.DOCUMENT_QA, "No evidence.")
    )
    with pytest.raises(GraphExecutionError) as error:
        await graph.invoke("hi")
    assert error.value.code == "routing_unavailable"
    graph.rag_agent.answer.assert_not_awaited()
    graph.chat_agent.answer.assert_not_awaited()


async def test_clarification_calls_no_agents():
    graph = make_graph()
    graph.rag_agent.answer = AsyncMock()
    graph.sql_agent.answer = AsyncMock()
    graph.hybrid_agent.answer = AsyncMock()
    graph.classifier.classify.return_value = ClassificationResult(
        query_type=None, decision="clarify", clarification_question="Which benefit do you mean?"
    )
    result = await graph.invoke("What is its limit?")
    assert result.query_type == QueryType.CLARIFICATION
    assert result.answer == "Which benefit do you mean?"
    for agent in (graph.chat_agent, graph.rag_agent, graph.sql_agent, graph.hybrid_agent):
        agent.answer.assert_not_awaited()


async def test_followup_executes_resolved_query_but_returns_original_prompt():
    graph = make_graph()
    graph.classifier.classify.return_value = ClassificationResult(
        QueryType.RECORDS_SQL, resolved_query="Count declined cases in 2023."
    )
    graph.sql_agent.answer = AsyncMock(
        return_value=QueryResult(
            "Count declined cases in 2023.", QueryType.RECORDS_SQL, "Two cases."
        )
    )
    result = await graph.invoke("What about 2023?")
    graph.sql_agent.answer.assert_awaited_once_with("Count declined cases in 2023.")
    assert result.query == "What about 2023?"


async def test_invalid_citations_are_rejected():
    from app.domain.entities.models import Citation

    graph = make_graph()
    graph.chat_agent.answer.return_value = QueryResult(
        "hi",
        QueryType.DOCUMENT_QA,
        "Covered [Source 99]",
        citations=[Citation(1, "Policy", chunk_text="Evidence")],
    )
    with pytest.raises(GraphExecutionError) as error:
        await graph.invoke("hi")
    assert error.value.code == "invalid_citations"


async def test_total_query_timeout(monkeypatch):
    import asyncio

    graph = make_graph()
    monkeypatch.setattr(settings, "query_timeout_seconds", 0.01)

    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    graph.classifier.classify.side_effect = blocked
    with pytest.raises(GraphExecutionError) as error:
        await graph.invoke("hi")
    assert error.value.code == "query_timeout"


async def test_cancellation_does_not_retry():
    import asyncio

    graph = make_graph()
    started = asyncio.Event()

    async def blocked(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    graph.chat_agent.answer.side_effect = blocked
    task = asyncio.create_task(graph.invoke("hi"))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    assert graph.chat_agent.answer.await_count == 1
