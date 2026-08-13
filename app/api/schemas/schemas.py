"""
API Request/Response schemas.
"""
from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from enum import Enum


class QueryTypeEnum(str, Enum):
    document_qa = "document_qa"
    records_sql = "records_sql"
    hybrid = "hybrid"


class RetrievalStrategyEnum(str, Enum):
    semantic = "semantic"
    lexical = "lexical"
    blended = "blended"


# === Ask Endpoint Schemas ===

class InsightQueryRequest(BaseModel):
    """Public request contract for the insight query endpoint."""
    prompt: str = Field(..., min_length=3, max_length=1000, description="Question to answer")
    scope_key: Optional[int] = Field(None, description="Optional synthetic knowledge-scope key")
    retrieval_strategy: RetrievalStrategyEnum = Field(
        RetrievalStrategyEnum.blended,
        description="Retrieval strategy: semantic, lexical, or blended"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "What is the family-support benefit cap?",
                    "scope_key": 1,
                    "retrieval_strategy": "blended"
                }
            ]
        }
    }


class Citation(BaseModel):
    """Citation reference for RAG answers."""
    source_id: int
    contract_title: str
    section: Optional[str]
    page: Optional[int]
    chunk_text: str
    relevance_score: float


class InsightQueryResponse(BaseModel):
    """Public response contract for an insight query."""
    prompt: str
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
                    "prompt": "What is the family-support benefit cap?",
                    "query_type": "document_qa",
                    "answer": "The family-support benefit cap is CU 50,000 per event [Source 1].",
                    "citations": [
                        {
                            "source_id": 1,
                            "contract_title": "Group Health Policy",
                            "section": "Maternity Benefits",
                            "page": 12,
                            "chunk_text": "Family-support benefit: CU 50,000 per event...",
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

class KnowledgeSourceRequest(BaseModel):
    """Content supplied for a synthetic benefit contract."""
    source_text: Optional[str] = Field(None, description="Raw source content")
    source_uri: Optional[str] = Field(None, description="Public URI from which to fetch content")
    segment_length: int = Field(1000, ge=100, le=4000, description="Characters per segment")
    segment_overlap: int = Field(200, ge=0, le=500, description="Overlap between segments")

    @model_validator(mode="after")
    def require_one_source(self):
        if bool(self.source_text) == bool(self.source_uri):
            raise ValueError("Provide exactly one of source_text or source_uri")
        return self


class KnowledgeSourceResponse(BaseModel):
    """Result of indexing a contract source."""
    scope_key: int
    segments_created: int
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
