"""
Domain entities for PolicyMind.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import List, Optional


class QueryType(str, Enum):
    """Types of queries the system can handle."""

    CHAT = "chat"
    DOCUMENT_QA = "document_qa"
    RECORDS_SQL = "records_sql"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class CaseStatus(str, Enum):
    """Claim processing status."""

    OPENED = "OPENED"
    IN_REVIEW = "IN_REVIEW"
    ELIGIBLE = "ELIGIBLE"
    RESOLVED = "RESOLVED"
    DECLINED = "DECLINED"


class FundingMode(str, Enum):
    """Funding mode for a service case."""

    DIRECT_BILLING = "DIRECT_BILLING"
    MEMBER_PAID = "MEMBER_PAID"


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
class BenefitContract:
    """Represents a policy document."""

    id: int
    contract_ref: str
    contract_title: Optional[str] = None
    plan_category: str = "EMPLOYEE_BENEFITS"
    sponsor_label: Optional[str] = None
    effective_from: Optional[date] = None
    effective_until: Optional[date] = None
    participant_count: Optional[int] = None
    aggregate_benefit_cap: Optional[Decimal] = None


@dataclass
class DocumentChunk:
    """A chunk of policy document with embedding."""

    id: int
    contract_id: int
    content: str
    contract_title: Optional[str] = None
    topic_category: Optional[SectionType] = None
    topic_title: Optional[str] = None
    source_page: Optional[int] = None
    passage_order: Optional[int] = None
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None
    score: Optional[float] = None  # Similarity score when retrieved


@dataclass
class Citation:
    """Citation information for an answer."""

    source_id: int
    contract_title: str
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_text: str = ""
    relevance_score: float = 0.0


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

    benefit_title: str
    is_covered: bool
    benefit_cap: Optional[Decimal] = None
    deductible: Optional[Decimal] = None
    percentage_share: Optional[Decimal] = None
    eligibility_delay_days: Optional[int] = None
    sub_limit: Optional[str] = None
    exclusions: Optional[List[str]] = None
    notes: Optional[str] = None


@dataclass
class CaseSummary:
    """Summary of synthetic service cases for SQL queries."""

    case_count: int
    requested_total: Decimal
    eligible_amount: Decimal
    average_requested: Decimal
    by_status: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
