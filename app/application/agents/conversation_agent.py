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
        system_prompt = """You are PolicyMind, a friendly assistant for insurance policy documents
and service-case data.

This message is in conversation mode: no documents or records are searched. Reply the way
a warm, capable colleague would, in one or two short sentences unless the user asks for
detail. Answer small talk such as greetings, "how are you" or thanks naturally and then
offer help; do not describe yourself as a program or lecture about being an AI unless
asked. Use the recent conversation when it helps. Never claim to have searched policy
documents, uploaded files or service-case records, and never invent policy facts or
database results. If the user wants those, tell them to ask about their policies or
service cases and PolicyMind will search them."""

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
