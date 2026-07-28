"""
API Request/Response schemas.
"""
from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class QueryTypeEnum(str, Enum):
    document_qa = "document_qa"
    claims_sql = "claims_sql"
    hybrid = "hybrid"


# === Ask Endpoint Schemas ===

class AskRequest(BaseModel):
    """Request schema for /ask endpoint."""
    query: str = Field(..., min_length=3, max_length=1000, description="User's question")
    policy_id: Optional[int] = Field(None, description="Optional policy ID to scope search")
    search_method: str = Field("hybrid", description="Search method: vector, bm25, or hybrid")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the maternity coverage limit?",
                    "policy_id": 1,
                    "search_method": "hybrid"
                }
            ]
        }
    }


class Citation(BaseModel):
    """Citation reference for RAG answers."""
    source_id: int
    policy_name: str
    section: Optional[str]
    page: Optional[int]
    chunk_text: str
    relevance_score: float


class AskResponse(BaseModel):
    """Response schema for /ask endpoint."""
    query: str
    query_type: QueryTypeEnum
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    sql_query: Optional[str] = None
    sql_result: Optional[List[dict]] = None
    latency_ms: int
    model_used: str
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the maternity coverage limit?",
                    "query_type": "document_qa",
                    "answer": "The maternity coverage limit is ₹50,000 per pregnancy [Source 1].",
                    "citations": [
                        {
                            "source_id": 1,
                            "policy_name": "Group Health Policy",
                            "section": "Maternity Benefits",
                            "page": 12,
                            "chunk_text": "Maternity coverage: ₹50,000 per pregnancy...",
                            "relevance_score": 0.92
                        }
                    ],
                    "latency_ms": 450,
                    "model_used": "gpt-4-turbo-preview"
                }
            ]
        }
    }


# === Ingest Endpoint Schemas ===

class IngestRequest(BaseModel):
    """Request schema for /ingest endpoint."""
    policy_id: int = Field(..., description="Policy ID to ingest documents for")
    document_text: Optional[str] = Field(None, description="Raw text content to ingest")
    document_url: Optional[str] = Field(None, description="URL to fetch document from")
    chunk_size: int = Field(1000, ge=100, le=4000, description="Characters per chunk")
    chunk_overlap: int = Field(200, ge=0, le=500, description="Overlap between chunks")


class IngestResponse(BaseModel):
    """Response schema for /ingest endpoint."""
    policy_id: int
    chunks_created: int
    processing_time_ms: int
    message: str


# === Evaluation Endpoint Schemas ===

class EvalQuestion(BaseModel):
    """A single evaluation question."""
    question_id: int
    question: str
    expected_answer: str
    query_type: QueryTypeEnum
    difficulty: str = Field(..., description="easy, medium, hard")


class EvalRunRequest(BaseModel):
    """Request to run evaluation."""
    question_ids: Optional[List[int]] = Field(None, description="Specific questions to evaluate")
    query_types: Optional[List[QueryTypeEnum]] = Field(None, description="Filter by query type")
    sample_size: Optional[int] = Field(None, ge=1, le=100, description="Random sample size")


class EvalResult(BaseModel):
    """Result for a single evaluation question."""
    question_id: int
    question: str
    expected_answer: str
    actual_answer: str
    query_type: QueryTypeEnum
    is_correct: bool
    similarity_score: float
    latency_ms: int
    citations_count: int
    error: Optional[str] = None


class EvalSummary(BaseModel):
    """Summary of evaluation run."""
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_latency_ms: float
    avg_similarity: float
    by_query_type: dict
    by_difficulty: dict


class EvalRunResponse(BaseModel):
    """Response from evaluation run."""
    run_id: str
    started_at: datetime
    completed_at: datetime
    summary: EvalSummary
    results: List[EvalResult]


# === Health Endpoint Schemas ===

class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str
    llm: str
    uptime_seconds: float
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "1.0.0",
                    "database": "connected",
                    "llm": "available",
                    "uptime_seconds": 3600.5
                }
            ]
        }
    }


class DetailedHealth(BaseModel):
    """Detailed health status."""
    status: str
    version: str
    components: dict
    metrics: dict
