from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.agents import critic
from app.application.agents.rag_agent import RAGAgent
from app.domain.entities.models import DocumentChunk

pytestmark = pytest.mark.unit


def _llm(result=None, error=None):
    llm = MagicMock()
    llm.generate_json = AsyncMock(return_value=result, side_effect=error)
    return llm


async def test_claims_are_cleaned_and_capped(monkeypatch):
    claims = ["a", " b ", "", 3, "c", "d", "e", "f"]
    monkeypatch.setattr(critic, "get_openai_client", lambda: _llm({"unsupported_claims": claims}))
    assert await critic.find_unsupported_claims("q", "ctx", "ans") == ["a", "b", "c", "d", "e"]


@pytest.mark.parametrize("payload", [{}, {"unsupported_claims": "oops"}])
async def test_malformed_output_is_ignored(monkeypatch, payload):
    monkeypatch.setattr(critic, "get_openai_client", lambda: _llm(payload))
    assert await critic.find_unsupported_claims("q", "ctx", "ans") == []


async def test_critic_failure_fails_open(monkeypatch):
    monkeypatch.setattr(critic, "get_openai_client", lambda: _llm(error=RuntimeError("down")))
    assert await critic.find_unsupported_claims("q", "ctx", "ans") == []


def _agent():
    agent = RAGAgent.__new__(RAGAgent)
    agent.generate_answer = AsyncMock(return_value="revised")
    return agent


CHUNKS = [DocumentChunk(id=1, contract_id=1, content="text")]


async def test_revision_happens_once_when_claims_found(monkeypatch):
    monkeypatch.setattr(
        "app.application.agents.rag_agent.find_unsupported_claims",
        AsyncMock(return_value=["bad claim"]),
    )
    agent = _agent()
    assert await agent._revise_if_unsupported("q", CHUNKS, "draft") == "revised"
    assert "bad claim" in agent.generate_answer.await_args.args[0]
    agent.generate_answer.assert_awaited_once()


async def test_clean_draft_is_untouched(monkeypatch):
    monkeypatch.setattr(
        "app.application.agents.rag_agent.find_unsupported_claims", AsyncMock(return_value=[])
    )
    agent = _agent()
    assert await agent._revise_if_unsupported("q", CHUNKS, "draft") == "draft"
    agent.generate_answer.assert_not_awaited()


async def test_failed_revision_keeps_draft(monkeypatch):
    monkeypatch.setattr(
        "app.application.agents.rag_agent.find_unsupported_claims", AsyncMock(return_value=["x"])
    )
    agent = _agent()
    agent.generate_answer = AsyncMock(side_effect=RuntimeError("down"))
    assert await agent._revise_if_unsupported("q", CHUNKS, "draft") == "draft"
