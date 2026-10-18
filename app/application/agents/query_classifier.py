"""
Query classifier agent - routes queries to appropriate handler.
"""

import asyncio
from time import perf_counter
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import settings
from app.domain.entities.models import ClassificationResult, QueryType
from app.infrastructure.llm.openai_client import ROUTING_POLICY_VERSION, get_openai_client

logger = structlog.get_logger()


class RoutingError(RuntimeError):
    def __init__(self, code: str):
        super().__init__("Unable to determine the required query tools.")
        self.code = code


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decision: Literal["route", "clarify"]
    query_type: Literal["chat", "document_qa", "records_sql", "hybrid"] | None
    needs_documents: bool
    needs_records: bool
    reason: str = Field(min_length=1, max_length=1000)
    resolved_query: str | None = Field(max_length=2000)
    clarification_question: str | None = Field(max_length=500)

    @field_validator("reason", "resolved_query", "clarification_question")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Routing text cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> "RoutingDecision":
        if self.decision == "clarify":
            if (
                self.query_type is not None
                or self.needs_documents
                or self.needs_records
                or self.resolved_query is not None
                or self.clarification_question is None
            ):
                raise ValueError("Clarification must ask a question without selecting tools")
        else:
            capabilities = {
                "chat": (False, False),
                "document_qa": (True, False),
                "records_sql": (False, True),
                "hybrid": (True, True),
            }
            if self.query_type is None or capabilities[self.query_type] != (
                self.needs_documents,
                self.needs_records,
            ):
                raise ValueError("Route does not match the required evidence")
            if self.resolved_query is None or self.clarification_question is not None:
                raise ValueError("Routing requires a standalone question, not a clarification")
        return self


class QueryClassifier:
    """
    Classifies user queries to route them to the appropriate agent.

    - chat: Normal conversation that does not need indexed sources
    - document_qa: Questions about policy terms, coverage, exclusions
    - records_sql: Questions requiring database queries on synthetic service-case data
    - hybrid: Questions needing both document and data context
    """

    def __init__(self):
        self.llm = get_openai_client()

    async def classify(
        self,
        query: str,
        conversation: list[dict[str, str]] | None = None,
    ) -> ClassificationResult:
        """Select evidence semantically and reject invalid or unavailable routing."""
        started = perf_counter()
        try:
            async with asyncio.timeout(settings.routing_timeout_seconds):
                for attempt in range(2):
                    try:
                        result = await self.llm.classify_query(
                            query,
                            conversation or [],
                            response_schema=RoutingDecision.model_json_schema(),
                            validation_retry=attempt > 0,
                        )
                        decision = RoutingDecision.model_validate(result)
                    except ValueError as exc:
                        logger.warning(
                            "Invalid routing decision",
                            attempt=attempt + 1,
                            error_type=type(exc).__name__,
                        )
                        if attempt == 1:
                            raise RoutingError("invalid_routing") from exc
                        continue
                    logger.info(
                        "Semantic routing decision",
                        policy_version=ROUTING_POLICY_VERSION,
                        decision=decision.decision,
                        query_type=decision.query_type,
                        needs_documents=decision.needs_documents,
                        needs_records=decision.needs_records,
                        duration_ms=int((perf_counter() - started) * 1000),
                        attempt=attempt + 1,
                    )
                    return ClassificationResult(
                        query_type=QueryType(decision.query_type) if decision.query_type else None,
                        reasoning=decision.reason,
                        decision=decision.decision,
                        needs_documents=decision.needs_documents,
                        needs_records=decision.needs_records,
                        resolved_query=decision.resolved_query,
                        clarification_question=decision.clarification_question,
                    )
        except RoutingError:
            raise
        except Exception as exc:
            logger.warning("Semantic routing unavailable", error_type=type(exc).__name__)
            raise RoutingError("routing_unavailable") from exc
        raise RoutingError("invalid_routing")

    def is_sql_safe(self, query: str) -> bool:
        """
        Check if a query is safe (no SQL injection patterns).
        """
        dangerous_patterns = [
            "drop ",
            "delete ",
            "insert ",
            "update ",
            "alter ",
            "create ",
            "truncate ",
            "--",
            ";--",
            "/*",
            "*/",
        ]
        query_lower = query.lower()
        return not any(pattern in query_lower for pattern in dangerous_patterns)
