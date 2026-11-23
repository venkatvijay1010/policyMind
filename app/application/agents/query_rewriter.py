"""Query rewriting: restates a question in the wording policy documents use."""

from typing import Optional

import structlog

from app.infrastructure.llm.openai_client import get_openai_client

logger = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You rewrite insurance questions into short search queries using the terms policy "
    "documents use (for example 'hospitalisation' for 'hospital stay'). Keep every number, "
    "plan name and condition from the question. Never answer the question."
)
_MAX_LENGTH = 300


async def rewrite_query(query: str) -> Optional[str]:
    """Return a distinct rewrite, or None if the call fails or adds nothing."""
    try:
        result = await get_openai_client().generate_json(
            prompt=f'Question: {query}\n\nReturn JSON: {{"search_query": "<rewritten query>"}}',
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("Query rewrite failed; using original", error_type=type(exc).__name__)
        return None
    rewritten = result.get("search_query")
    if not isinstance(rewritten, str):
        return None
    rewritten = rewritten.strip()
    if not rewritten or len(rewritten) > _MAX_LENGTH or rewritten.casefold() == query.casefold():
        return None
    return rewritten
