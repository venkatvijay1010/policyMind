"""
Ask endpoint - main query interface.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.api.schemas.schemas import InsightQueryRequest, InsightQueryResponse, Citation, QueryTypeEnum
from app.infrastructure.database.postgres import get_db_session
from app.application.graph.orchestrator import PolicyMindGraph
from app.domain.entities.models import QueryType

logger = structlog.get_logger()

router = APIRouter(prefix="/insights", tags=["Insights"])


async def log_query_background(
    query: str,
    query_type: str,
    latency_ms: int
):
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
                {"query": query, "query_type": query_type, "latency_ms": latency_ms}
            )
            await db.commit()
    except Exception as e:
        logger.error("Failed to log query", error=str(e))


@router.post("/query", response_model=InsightQueryResponse)
async def create_insight(
    request: InsightQueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Ask a question about insurance benefit_contracts or service_cases.
    
    The system will automatically:
    1. Classify your query type (document Q&A, SQL, or hybrid)
    2. Route to the appropriate agent
    3. Return an answer with citations
    
    **Query Types:**
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
        prompt=request.prompt[:100],
        scope_key=request.scope_key
    )
    
    try:
        # Create the graph and invoke
        graph = PolicyMindGraph(db)
        result = await graph.invoke(
            query=request.prompt,
            contract_id=request.scope_key
        )
        
        # Convert citations
        citations = [
            Citation(
                source_id=c.source_id,
                contract_title=c.contract_title,
                section=c.section,
                page=c.page,
                chunk_text=c.chunk_text[:500],  # Truncate for response
                relevance_score=c.relevance_score
            )
            for c in result.citations
        ]
        
        # Map internal QueryType to API enum
        query_type_map = {
            QueryType.DOCUMENT_QA: QueryTypeEnum.document_qa,
            QueryType.RECORDS_SQL: QueryTypeEnum.records_sql,
            QueryType.HYBRID: QueryTypeEnum.hybrid
        }
        
        # Log query in background
        background_tasks.add_task(
            log_query_background,
            request.prompt,
            result.query_type.value,
            result.latency_ms
        )
        
        return InsightQueryResponse(
            prompt=result.query,
            query_type=query_type_map.get(result.query_type, QueryTypeEnum.document_qa),
            answer=result.answer,
            citations=citations,
            sql_query=result.sql_query,
            sql_result=result.sql_result,
            latency_ms=result.latency_ms,
            model_used=result.model_used
        )
        
    except Exception as e:
        logger.error("Query failed", error=str(e), prompt=request.prompt[:100])
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@router.post("/route-preview")
async def classify_only(
    request: InsightQueryRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Classify a query without executing it.
    Useful for debugging or understanding query routing.
    """
    from app.application.agents.query_classifier import QueryClassifier
    
    classifier = QueryClassifier()
    result = await classifier.classify(request.prompt)
    
    return {
        "prompt": request.prompt,
        "query_type": result.query_type.value,
        "confidence": result.confidence,
        "reasoning": result.reasoning
    }


@router.post("/document-only")
async def rag_only(
    request: InsightQueryRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Force RAG agent execution (skip classification).
    Useful for testing document retrieval specifically.
    """
    from app.application.agents.rag_agent import RAGAgent
    
    agent = RAGAgent(db)
    result = await agent.answer(
        query=request.prompt,
        contract_id=request.scope_key,
        search_method={"semantic": "vector", "lexical": "bm25", "blended": "hybrid"}.get(
            request.retrieval_strategy.value, "hybrid"
        )
    )
    
    citations = [
        Citation(
            source_id=c.source_id,
            contract_title=c.contract_title,
            section=c.section,
            page=c.page,
            chunk_text=c.chunk_text[:500],
            relevance_score=c.relevance_score
        )
        for c in result.citations
    ]
    
    return InsightQueryResponse(
        prompt=result.query,
        query_type=QueryTypeEnum.document_qa,
        answer=result.answer,
        citations=citations,
        latency_ms=result.latency_ms,
        model_used=result.model_used
    )


@router.post("/records-only")
async def sql_only(
    request: InsightQueryRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Force SQL agent execution (skip classification).
    Useful for testing service_cases queries specifically.
    """
    from app.application.agents.sql_agent import SQLAgent
    
    agent = SQLAgent(db)
    result = await agent.answer(request.prompt)
    
    return InsightQueryResponse(
        prompt=result.query,
        query_type=QueryTypeEnum.records_sql,
        answer=result.answer,
        citations=[],
        sql_query=result.sql_query,
        sql_result=result.sql_result,
        latency_ms=result.latency_ms,
        model_used=result.model_used
    )
