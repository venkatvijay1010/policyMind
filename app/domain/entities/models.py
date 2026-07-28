"""
Domain entities for PolicyMind.
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    DOCUMENT_QA = "document_qa"
    CLAIMS_SQL = "claims_sql"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class ClaimStatus(str, Enum):
    """Claim processing status."""
    REGISTERED = "REGISTERED"
    UNDER_PROCESS = "UNDER_PROCESS"
    APPROVED = "APPROVED"
    SETTLED = "SETTLED"
    REJECTED = "REJECTED"


class ClaimType(str, Enum):
    """Type of claim."""
    CASHLESS = "CASHLESS"
    REIMBURSEMENT = "REIMBURSEMENT"


class SectionType(str, Enum):
    """Types of policy document sections."""
    COVERAGE = "COVERAGE"
    EXCLUSION = "EXCLUSION"
    LIMIT = "LIMIT"
    DEDUCTIBLE = "DEDUCTIBLE"
    WAITING_PERIOD = "WAITING_PERIOD"
    COPAY = "COPAY"
    NETWORK = "NETWORK"
    CLAIM_PROCESS = "CLAIM_PROCESS"
    DEFINITION = "DEFINITION"
    GENERAL = "GENERAL"


@dataclass
class PolicyDocument:
    """Represents a policy document."""
    id: int
    policy_number: str
    policy_name: Optional[str] = None
    product_type: str = "GROUP_HEALTH"
    insured_name: Optional[str] = None
    policy_start_date: Optional[date] = None
    policy_end_date: Optional[date] = None
    total_lives: Optional[int] = None
    total_sum_insured: Optional[Decimal] = None


@dataclass
class DocumentChunk:
    """A chunk of policy document with embedding."""
    id: int
    policy_id: int
    content: str
    section_type: Optional[SectionType] = None
    section_name: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None
    score: Optional[float] = None  # Similarity score when retrieved


@dataclass
class Citation:
    """Citation information for an answer."""
    policy_number: str
    section_type: Optional[str] = None
    section_name: Optional[str] = None
    page_number: Optional[int] = None
    snippet: Optional[str] = None
    relevance_score: Optional[float] = None


@dataclass
class QueryResult:
    """Result of a query to the RAG system."""
    query: str
    query_type: QueryType
    answer: str
    citations: List[Citation] = field(default_factory=list)
    sql_query: Optional[str] = None
    sql_result: Optional[List[dict]] = None
    confidence: Optional[float] = None
    latency_ms: Optional[int] = None
    token_count: Optional[int] = None
    model_used: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result of document retrieval."""
    chunks: List[DocumentChunk]
    query_embedding: Optional[List[float]] = None
    search_method: str = "vector"  # vector, bm25, hybrid


@dataclass
class ClassificationResult:
    """Result of query classification."""
    query_type: QueryType
    confidence: float
    reasoning: Optional[str] = None


@dataclass
class CoverageInfo:
    """Coverage calculation result."""
    coverage_name: str
    is_covered: bool
    coverage_limit: Optional[Decimal] = None
    deductible: Optional[Decimal] = None
    copay_percentage: Optional[Decimal] = None
    waiting_period_days: Optional[int] = None
    sub_limit: Optional[str] = None
    exclusions: Optional[List[str]] = None
    notes: Optional[str] = None


@dataclass
class ClaimSummary:
    """Summary of claims for SQL queries."""
    total_claims: int
    total_amount: Decimal
    approved_amount: Decimal
    average_claim: Decimal
    by_status: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
