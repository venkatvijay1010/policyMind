"""Startup warm-up for the local model."""

import time

import structlog

from app.application.agents.conversation_agent import ConversationAgent
from app.application.agents.query_classifier import QueryClassifier

logger = structlog.get_logger()


async def warm_up_models() -> None:
    """Prime the model and the router/chat prompt caches so the first message is not cold."""
    started_at = time.monotonic()
    try:
        await QueryClassifier().classify("Hello")
        await ConversationAgent().answer("Hello")
    except Exception as exc:  # A failed warm-up must never affect startup.
        logger.warning("Model warm-up failed", error=str(exc))
        return
    logger.info("Model warm-up complete", duration_ms=int((time.monotonic() - started_at) * 1000))
