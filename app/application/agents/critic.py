"""Answer critic: flags claims in a draft that the retrieved context does not support."""

import structlog

from app.infrastructure.llm.openai_client import get_openai_client

logger = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You check answers against source text. Report only claims that the context "
    "does not state or directly imply. Do not judge style or completeness."
)


async def find_unsupported_claims(query: str, context: str, answer: str) -> list[str]:
    """Return unsupported claims; any critic failure returns [] so the draft is kept."""
    prompt = f"""Question: {query}

Context:
{context}

Draft answer:
{answer}

Return JSON: {{"unsupported_claims": ["<exact claim from the draft>", ...]}}
Use an empty list when every claim is supported."""
    try:
        result = await get_openai_client().generate_json(
            prompt=prompt, system_prompt=_SYSTEM_PROMPT, temperature=0.0
        )
        claims = result.get("unsupported_claims", [])
        if not isinstance(claims, list):
            return []
        return [claim.strip() for claim in claims if isinstance(claim, str) and claim.strip()][:5]
    except Exception as exc:
        logger.warning("Critic failed; keeping draft", error_type=type(exc).__name__)
        return []


def revision_request(query: str, claims: list[str]) -> str:
    listed = "\n".join(f"- {claim}" for claim in claims)
    return (
        f"{query}\n\nA reviewer found these statements unsupported by the context:\n{listed}\n"
        "Rewrite the answer without them, using only the context."
    )
