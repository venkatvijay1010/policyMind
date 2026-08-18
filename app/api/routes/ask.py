"""Ask endpoints for standard and status-streamed chat requests."""

import asyncio
import json
from typing import Any, AsyncGenerator

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.schemas import (
    Citation,
    InsightQueryRequest,
    InsightQueryResponse,
    QueryTypeEnum,
)
from app.application.graph.orchestrator import PolicyMindGraph
from app.config import settings
from app.domain.entities.models import QueryType
from app.infrastructure.database.postgres import get_db_session
from app.infrastructure.rate_limit import limiter

logger = structlog.get_logger()

router = APIRouter(prefix="/insights", tags=["Insights"])

RETRIEVAL_METHODS = {
    "semantic": "vector",
    "lexical": "bm25",
    "blended": "hybrid",
}


def _search_method(retrieval_strategy: str) -> str:
    """Map the public retrieval strategy to the search implementation name."""
    return RETRIEVAL_METHODS[retrieval_strategy]


def _conversation_turns(insight_request: InsightQueryRequest) -> list[dict[str, str]]:
    """Convert validated API conversation turns into the graph's small payload shape."""
    return [{"role": turn.role, "content": turn.content} for turn in insight_request.conversation]


def _route_status(query_type: QueryType) -> tuple[str, str]:
    """Return concise, user-facing progress copy for the selected path."""
    if query_type == QueryType.CHAT:
        return "chatting", "Writing a response"
    if query_type == QueryType.RECORDS_SQL:
        return "retrieving", "Analyzing service-case data"
    if query_type == QueryType.HYBRID:
        return "retrieving", "Gathering relevant information"
    return "retrieving", "Searching your documents"


def _to_insight_response(result) -> InsightQueryResponse:
    """Convert the internal graph result to the public API response shape."""
    citations = [
        Citation(
            source_id=c.source_id,
            contract_title=c.contract_title,
            section=c.section,
            page=c.page,
            chunk_text=c.chunk_text[:500],
            relevance_score=c.relevance_score,
        )
        for c in result.citations
    ]
    query_type_map = {
        QueryType.CHAT: QueryTypeEnum.chat,
        QueryType.DOCUMENT_QA: QueryTypeEnum.document_qa,
        QueryType.RECORDS_SQL: QueryTypeEnum.records_sql,
        QueryType.HYBRID: QueryTypeEnum.hybrid,
    }
    return InsightQueryResponse(
        prompt=result.query,
        query_type=query_type_map.get(result.query_type, QueryTypeEnum.document_qa),
        answer=result.answer,
        citations=citations,
        sql_query=result.sql_query,
        sql_result=result.sql_result,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )


def _sse(event: str, payload: dict[str, Any]) -> str:
    """Encode a small server-sent event frame."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def _stream_insight(
    insight_request: InsightQueryRequest,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    """Run an insight query while publishing useful local-work status updates."""
    graph = PolicyMindGraph(db)
    conversation = _conversation_turns(insight_request)
    yield _sse(
        "status",
        {
            "stage": "understanding",
            "message": "Thinking",
            "elapsed_ms": 0,
        },
    )

    try:
        result = None
        async for event in graph.stream(
            query=insight_request.prompt,
            contract_id=insight_request.scope_key,
            search_method=_search_method(insight_request.retrieval_strategy.value),
            conversation=conversation,
        ):
            for node_name, update in event.items():
                if node_name == "classify":
                    query_type = update.get("query_type", QueryType.DOCUMENT_QA)
                    if not isinstance(query_type, QueryType):
                        query_type = QueryType.DOCUMENT_QA
                    stage, message = _route_status(query_type)
                    yield _sse(
                        "status",
                        {"stage": stage, "message": message, "elapsed_ms": 0},
                    )
                elif node_name in {"chat", "rag", "sql", "hybrid"}:
                    result = update.get("result")

        if result is None:
            raise RuntimeError("The local routing graph ended without a result.")

        response = _to_insight_response(result)
        asyncio.create_task(
            log_query_background(insight_request.prompt, result.query_type.value, result.latency_ms)
        )
        yield _sse("result", response.model_dump(mode="json"))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception(
            "Streaming query failed", error=str(exc), prompt=insight_request.prompt[:100]
        )
        yield _sse("error", {"detail": "Unable to process that question locally."})


async def log_query_background(query: str, query_type: str, latency_ms: int):
    """Background task to log query for analytics."""
    from sqlalchemy import text

    from app.infrastructure.database.postgres import async_session_factory

    try:
        async with async_session_factory() as db:
            await db.execute(
                text("""
                    INSERT INTO query_logs (query, query_type, latency_ms)
                    VALUES (:query, :query_type, :latency_ms)
                """),
                {"query": query, "query_type": query_type, "latency_ms": latency_ms},
            )
            await db.commit()
    except Exception as e:
        logger.error("Failed to log query", error=str(e))


@router.post("/query", response_model=InsightQueryResponse)
@limiter.limit(settings.rate_limit)
async def create_insight(
    request: Request,
    insight_request: InsightQueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Ask a question about insurance benefit_contracts or service_cases.

    The system will automatically:
    1. Classify your query type (chat, document Q&A, SQL, or hybrid)
    2. Route to the appropriate agent
    3. Return an answer with citations

    **Query Types:**
    - **chat**: Normal conversation with the local model, without document retrieval
    - **document_qa**: Questions about policy terms, coverage, exclusions
    - **records_sql**: Questions about synthetic service-case data, statistics, and trends
    - **hybrid**: Questions needing both policy context and data analysis

    **Examples:**
    - "What is the maternity coverage limit?" → document_qa
    - "How many service cases were declined in 2024?" → records_sql
    - "What's our rejection rate for pre-existing conditions and what does the policy say about them?" → hybrid
    """
    logger.info(
        "Processing query",
        prompt=insight_request.prompt[:100],
        scope_key=insight_request.scope_key,
    )

    try:
        # Create the graph and invoke
        graph = PolicyMindGraph(db)
        result = await graph.invoke(
            query=insight_request.prompt,
            contract_id=insight_request.scope_key,
            search_method=_search_method(insight_request.retrieval_strategy.value),
            conversation=_conversation_turns(insight_request),
        )

        # Log query in background
        background_tasks.add_task(
            log_query_background, insight_request.prompt, result.query_type.value, result.latency_ms
        )
        return _to_insight_response(result)

    except Exception as exc:
        logger.exception("Query failed", error=str(exc), prompt=insight_request.prompt[:100])
        raise HTTPException(
            status_code=500,
            detail="Unable to process the query",
        ) from exc


@router.post("/query/stream")
@limiter.limit(settings.rate_limit)
async def stream_insight(
    request: Request,
    insight_request: InsightQueryRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Return progress events followed by the final insight response as SSE."""
    del request
    return StreamingResponse(
        _stream_insight(insight_request, db),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/route-preview")
@limiter.limit(settings.rate_limit)
async def classify_only(
    request: Request,
    insight_request: InsightQueryRequest,
):
    """
    Classify a query without executing it.
    Useful for debugging or understanding query routing.
    """
    from app.application.agents.query_classifier import QueryClassifier

    classifier = QueryClassifier()
    result = await classifier.classify(insight_request.prompt, _conversation_turns(insight_request))

    return {
        "prompt": insight_request.prompt,
        "query_type": result.query_type.value,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
    }


@router.post("/document-only")
@limiter.limit(settings.rate_limit)
async def rag_only(
    request: Request,
    insight_request: InsightQueryRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Force RAG agent execution (skip classification).
    Useful for testing document retrieval specifically.
    """
    from app.application.agents.rag_agent import RAGAgent

    agent = RAGAgent(db)
    result = await agent.answer(
        query=insight_request.prompt,
        contract_id=insight_request.scope_key,
        search_method=_search_method(insight_request.retrieval_strategy.value),
    )

    citations = [
        Citation(
            source_id=c.source_id,
            contract_title=c.contract_title,
            section=c.section,
            page=c.page,
            chunk_text=c.chunk_text[:500],
            relevance_score=c.relevance_score,
        )
        for c in result.citations
    ]

    return InsightQueryResponse(
        prompt=result.query,
        query_type=QueryTypeEnum.document_qa,
        answer=result.answer,
        citations=citations,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )


@router.post("/records-only")
@limiter.limit(settings.rate_limit)
async def sql_only(
    request: Request,
    insight_request: InsightQueryRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Force SQL agent execution (skip classification).
    Useful for testing service_cases queries specifically.
    """
    from app.application.agents.sql_agent import SQLAgent

    agent = SQLAgent(db)
    result = await agent.answer(insight_request.prompt)

    return InsightQueryResponse(
        prompt=result.query,
        query_type=QueryTypeEnum.records_sql,
        answer=result.answer,
        citations=[],
        sql_query=result.sql_query,
        sql_result=result.sql_result,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )
