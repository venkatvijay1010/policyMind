"""OpenAI-compatible client wrapper for LLM operations.

Ollama implements the OpenAI chat-completions and embeddings contracts locally,
so the official OpenAI Python SDK can be used as a small compatibility client
without requiring an OpenAI API key.
"""

import json
from typing import Any, Optional, Sequence, cast

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = structlog.get_logger()


class LLMClient:
    """Async client for the configured OpenAI-compatible LLM provider."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.chat_model = settings.chat_model
        self.embedding_model = settings.embedding_model

    async def _complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[dict[str, Any]],
    ) -> str:
        """Send one OpenAI-compatible local chat completion request."""
        kwargs: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else settings.llm_max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        logger.debug(
            "Calling LLM",
            provider=settings.llm_provider,
            model=self.chat_model,
            prompt_length=sum(len(message["content"]) for message in messages),
        )

        response = await self.client.chat.completions.create(**kwargs)

        content = cast(Optional[str], response.choices[0].message.content)
        if content is None:
            raise ValueError("The configured LLM returned an empty response")

        logger.debug(
            "LLM response received",
            provider=settings.llm_provider,
            model=self.chat_model,
            tokens=response.usage.total_tokens if response.usage else None,
        )

        return content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Generate a response from the LLM.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return await self._complete(messages, temperature, max_tokens, response_format)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate_conversation(
        self,
        prompt: str,
        conversation: Sequence[dict[str, str]] = (),
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a normal chat reply without retrieval context."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for turn in conversation[-10:]:
            role = turn.get("role")
            content = str(turn.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content[:1600]})

        messages.append({"role": "user", "content": prompt})
        return await self._complete(messages, temperature, max_tokens, None)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Generate a structured JSON response from the LLM.
        """
        content = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", error=str(e), content=content[:500])
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        if not isinstance(parsed_content, dict):
            raise ValueError("The configured LLM returned a JSON value that is not an object")
        return cast(dict[str, Any], parsed_content)

    async def classify_query(
        self,
        query: str,
        conversation: Sequence[dict[str, str]] = (),
    ) -> dict[str, Any]:
        """
        Classify a message into chat, document_qa, records_sql, or hybrid.
        """
        system_prompt = """You are a query classifier for an insurance policy Q&A system.

Classify the latest user message into one of these categories:
- chat: Normal conversation, greetings, social questions, general assistance, or creative requests that do NOT need the policy documents or service-case data
- document_qa: Questions about policy coverage, exclusions, limits, terms, conditions, waiting periods
- records_sql: Questions about synthetic service-case data, statistics, amounts, counts, and trends
- hybrid: Questions that need both policy documents AND service_cases data

Examples:
- "how r u?" -> chat
- "Tell me a joke" -> chat
- "What is the waiting period?" -> document_qa
- "How many cases were declined in 2024?" -> records_sql

Return JSON with:
{
    "query_type": "chat" | "document_qa" | "records_sql" | "hybrid",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""

        recent_turns = []
        for turn in conversation[-6:]:
            role = turn.get("role")
            content = str(turn.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                recent_turns.append(f"{role.title()}: {content[:400]}")
        recent_context = "\n".join(recent_turns) or "(No earlier conversation.)"
        prompt = (
            f"Recent conversation for context only:\n{recent_context}\n\n"
            f"Latest user message to classify: {query}"
        )

        result = await self.generate_structured(
            prompt=prompt, system_prompt=system_prompt, temperature=0.0, max_tokens=96
        )

        return cast(dict[str, Any], result)

    async def generate_sql(
        self, query: str, schema_info: str, error_context: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Generate SQL from natural language query.
        """
        system_prompt = """You are a SQL generator for an insurance service_cases SQLite database.

Generate SQLite queries that are:
- Read-only (SELECT only, no INSERT/UPDATE/DELETE)
- Safe (no SQL injection, use parameterized patterns)
- Efficient (use appropriate indexes)
- Use SQLite date functions when needed. For example, use
  strftime('%Y', submitted_on) = '2024' rather than EXTRACT(YEAR FROM ...).

Available tables and columns:
{schema_info}

Return JSON with:
{{
    "sql": "SELECT ...",
    "explanation": "What this query does",
    "columns_returned": ["col1", "col2"]
}}"""

        prompt = f"Generate SQL for: {query}"

        if error_context:
            prompt += (
                f"\n\nPrevious attempt failed with error: {error_context}\nPlease fix the query."
            )

        result = await self.generate_structured(
            prompt=prompt,
            system_prompt=system_prompt.format(schema_info=schema_info),
            temperature=0.0,
        )

        return cast(dict[str, Any], result)

    async def generate_answer(
        self, query: str, context: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate an answer based on retrieved context.
        """
        if not system_prompt:
            system_prompt = """You are an insurance policy expert assistant.

Answer the user's question based ONLY on the provided context.
- Be accurate and specific
- Cite the source sections when possible
- If the information is not in the context, say "I don't have information about this in the provided policy documents"
- Never make up information
- Use clear, professional language

Format your answer clearly with bullet points or numbered lists when appropriate."""

        prompt = f"""Context from policy documents:
{context}

User Question: {query}

Answer:"""

        return cast(
            str,
            await self.generate(
                prompt=prompt, system_prompt=system_prompt, temperature=0.2
            ),
        )

    async def generate_json(
        self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1
    ) -> dict[str, Any]:
        """
        Alias for generate_structured for JSON output.
        """
        return cast(
            dict[str, Any],
            await self.generate_structured(
                prompt=prompt, system_prompt=system_prompt, temperature=temperature
            ),
        )


# Singleton instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the configured LLM client singleton."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


# Compatibility aliases avoid forcing every agent import to change at once.
OpenAIClient = LLMClient


def get_openai_client() -> LLMClient:
    """Backward-compatible alias for :func:`get_llm_client`."""
    return get_llm_client()
