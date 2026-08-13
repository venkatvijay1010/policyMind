"""
Health check endpoints.
"""
import time
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.api.schemas.schemas import HealthStatus, DetailedHealth
from app.infrastructure.database.postgres import get_db_session
from app.config import settings

router = APIRouter(prefix="/health", tags=["Health"])

# Track start time
START_TIME = time.time()


@router.get("", response_model=HealthStatus)
async def health_check(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Basic health check endpoint.
    
    Returns status of all critical components.
    """
    # Check database
    db_status = "connected"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"
    
    # Check LLM (simplified - just check if API key exists)
    llm_status = "available" if settings.openai_api_key else "not_configured"
    
    uptime = time.time() - START_TIME
    
    overall_status = "healthy" if db_status == "connected" and llm_status == "available" else "degraded"
    
    return HealthStatus(
        status=overall_status,
        version=settings.app_version,
        database=db_status,
        llm=llm_status,
        uptime_seconds=round(uptime, 2)
    )


@router.get("/detailed", response_model=DetailedHealth)
async def detailed_health(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Detailed health check with component status and metrics.
    """
    components = {}
    metrics = {}
    
    # Database check
    try:
        result = await db.execute(text("SELECT COUNT(*) FROM benefit_contracts"))
        policy_count = result.scalar()
        components["database"] = {
            "status": "healthy",
            "policy_count": policy_count
        }
    except Exception as e:
        components["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Vector store check
    try:
        result = await db.execute(text("SELECT COUNT(*) FROM contract_passages"))
        chunk_count = result.scalar()
        components["vector_store"] = {
            "status": "healthy",
            "chunk_count": chunk_count
        }
    except Exception as e:
        components["vector_store"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # LLM check
    components["llm"] = {
        "status": "configured" if settings.openai_api_key else "not_configured",
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model
    }
    
    # Metrics
    metrics["uptime_seconds"] = round(time.time() - START_TIME, 2)
    metrics["environment"] = settings.environment
    
    overall_status = "healthy"
    for comp in components.values():
        if comp.get("status") not in ["healthy", "configured"]:
            overall_status = "degraded"
            break
    
    return DetailedHealth(
        status=overall_status,
        version=settings.app_version,
        components=components,
        metrics=metrics
    )


@router.get("/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Kubernetes readiness probe.
    Returns 200 only when service is ready to accept traffic.
    """
    try:
        await db.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe.
    Returns 200 if the process is running.
    """
    return {"alive": True}
