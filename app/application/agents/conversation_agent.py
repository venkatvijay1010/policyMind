"""Normal local conversation agent that deliberately does not retrieve documents."""

import time
from typing import Sequence

from app.config import settings
from app.domain.entities.models import QueryResult, QueryType
from app.infrastructure.llm.openai_client import get_openai_client


class ConversationAgent:
    """Use the configured local model for ordinary, non-RAG conversation."""

    def __init__(self):
        self.llm = get_openai_client()

    async def answer(
        self,
        query: str,
        conversation: Sequence[dict[str, str]] = (),
    ) -> QueryResult:
        """Generate a concise chat response without searching policy documents."""
        started_at = time.monotonic()
        system_prompt = """You are PolicyMind, a friendly local AI assistant.

You are in normal conversation mode for this message. Reply naturally and helpfully,
using the recent conversation when it is useful. Keep the answer concise unless the
user asks for detail. Do not claim to have searched policy documents, uploaded files,
or service-case records in this mode. If the user asks about those sources, explain
that PolicyMind can search them when their question is routed as a document or data
request. Do not invent policy facts or database results."""

        answer = await self.llm.generate_conversation(
            prompt=query,
            conversation=conversation,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=min(settings.llm_max_tokens, 180),
        )

        return QueryResult(
            query=query,
            query_type=QueryType.CHAT,
            answer=answer,
            citations=[],
            latency_ms=int((time.monotonic() - started_at) * 1000),
            model_used=settings.chat_model,
        )
