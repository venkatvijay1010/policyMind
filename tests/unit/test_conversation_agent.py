"""Unit tests for normal local chat without retrieval."""

from unittest.mock import AsyncMock, patch

import pytest

from app.application.agents.conversation_agent import ConversationAgent
from app.domain.entities.models import QueryType

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_conversation_agent_uses_local_model_without_citations():
    agent = ConversationAgent()
    history = [{"role": "user", "content": "Hello"}]

    with patch.object(agent, "llm") as mock_llm:
        mock_llm.generate_conversation = AsyncMock(return_value="I am doing well, thanks!")

        result = await agent.answer("how r u?", conversation=history)

    assert result.query_type == QueryType.CHAT
    assert result.answer == "I am doing well, thanks!"
    assert result.citations == []
    mock_llm.generate_conversation.assert_awaited_once()
    assert mock_llm.generate_conversation.await_args.kwargs["conversation"] == history
