"""
SQLAlchemy ORM models for the database.
"""
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional
from sqlalchemy import (
    Boolean, Date, DateTime, ForeignKey, Integer, 
    Numeric, String, Text, ARRAY, JSON, func
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.infrastructure.database.postgres import Base
from app.config import settings


class Policy(Base):
    """Insurance policy model."""
    __tablename__ = "benefit_contracts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    contract_ref: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    contract_title: Mapped[Optional[str]] = mapped_column(String(255))
    plan_category: Mapped[str] = mapped_column(String(50), default="EMPLOYEE_BENEFITS")
    plan_ref: Mapped[Optional[str]] = mapped_column(String(50))
    sponsor_label: Mapped[Optional[str]] = mapped_column(String(255))
    sponsor_sector: Mapped[Optional[str]] = mapped_column(String(100))
    effective_from: Mapped[Optional[date]] = mapped_column(Date)
    effective_until: Mapped[Optional[date]] = mapped_column(Date)
    participant_count: Mapped[Optional[int]] = mapped_column(Integer)
    aggregate_benefit_cap: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    contribution_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    service_partner_label: Mapped[Optional[str]] = mapped_column(String(255))
    source_document_uri: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    chunks: Mapped[List["ContractPassage"]] = relationship("ContractPassage", back_populates="policy", cascade="all, delete-orphan")
    plan_benefits: Mapped[List["Coverage"]] = relationship("Coverage", back_populates="policy", cascade="all, delete-orphan")
    participants: Mapped[List["Member"]] = relationship("Member", back_populates="policy", cascade="all, delete-orphan")
    service_cases: Mapped[List["Claim"]] = relationship("Claim", back_populates="policy", cascade="all, delete-orphan")


class ContractPassage(Base):
    """Policy document chunks for RAG."""
    __tablename__ = "contract_passages"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    contract_id: Mapped[int] = mapped_column(Integer, ForeignKey("benefit_contracts.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    topic_category: Mapped[Optional[str]] = mapped_column(String(100))
    topic_title: Mapped[Optional[str]] = mapped_column(String(255))
    source_page: Mapped[Optional[int]] = mapped_column(Integer)
    passage_order: Mapped[Optional[int]] = mapped_column(Integer)
    source_offset_start: Mapped[Optional[int]] = mapped_column(Integer)
    source_offset_end: Mapped[Optional[int]] = mapped_column(Integer)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(settings.embedding_dimension))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    policy: Mapped["Policy"] = relationship("Policy", back_populates="chunks")


class Coverage(Base):
    """Coverage details for a policy."""
    __tablename__ = "plan_benefits"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    contract_id: Mapped[int] = mapped_column(Integer, ForeignKey("benefit_contracts.id", ondelete="CASCADE"))
    benefit_ref: Mapped[str] = mapped_column(String(50), nullable=False)
    benefit_title: Mapped[str] = mapped_column(String(255), nullable=False)
    benefit_tier: Mapped[str] = mapped_column(String(50), default="BASE")
    limit_basis: Mapped[Optional[str]] = mapped_column(String(50))
    benefit_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    benefit_cap: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    inner_cap_percent: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    fixed_share_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    percentage_share: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    eligibility_delay_days: Mapped[Optional[int]] = mapped_column(Integer)
    is_core_benefit: Mapped[bool] = mapped_column(Boolean, default=True)
    eligibility_rules: Mapped[Optional[str]] = mapped_column(Text)
    non_covered_notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    policy: Mapped["Policy"] = relationship("Policy", back_populates="plan_benefits")


class Member(Base):
    """Member/insured person details."""
    __tablename__ = "participants"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    contract_id: Mapped[int] = mapped_column(Integer, ForeignKey("benefit_contracts.id", ondelete="CASCADE"))
    participant_ref: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    sponsor_person_ref: Mapped[Optional[str]] = mapped_column(String(50))
    participant_label: Mapped[str] = mapped_column(String(255), nullable=False)
    enrolment_role: Mapped[str] = mapped_column(String(50), default="SELF")
    gender: Mapped[Optional[str]] = mapped_column(String(10))
    birth_date: Mapped[Optional[date]] = mapped_column(Date)
    age: Mapped[Optional[int]] = mapped_column(Integer)
    benefit_ceiling: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    contribution_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    plan_variant: Mapped[Optional[str]] = mapped_column(String(100))
    enrolled_on: Mapped[Optional[date]] = mapped_column(Date)
    eligibility_from: Mapped[Optional[date]] = mapped_column(Date)
    eligibility_until: Mapped[Optional[date]] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(50), default="ACTIVE")
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(100))
    postal_code: Mapped[Optional[str]] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    policy: Mapped["Policy"] = relationship("Policy", back_populates="participants")
    service_cases: Mapped[List["Claim"]] = relationship("Claim", back_populates="member", cascade="all, delete-orphan")


class Claim(Base):
    """Insurance claim records."""
    __tablename__ = "service_cases"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_ref: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    contract_id: Mapped[int] = mapped_column(Integer, ForeignKey("benefit_contracts.id", ondelete="CASCADE"))
    participant_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("participants.id", ondelete="CASCADE"))
    funding_mode: Mapped[str] = mapped_column(String(50), default="DIRECT_BILLING")
    care_setting: Mapped[str] = mapped_column(String(50), default="FACILITY_STAY")
    condition_code: Mapped[Optional[str]] = mapped_column(String(20))
    condition_label: Mapped[Optional[str]] = mapped_column(String(500))
    service_category: Mapped[Optional[str]] = mapped_column(String(255))
    provider_label: Mapped[Optional[str]] = mapped_column(String(255))
    provider_city: Mapped[Optional[str]] = mapped_column(String(100))
    provider_region: Mapped[Optional[str]] = mapped_column(String(100))
    provider_registry_ref: Mapped[Optional[str]] = mapped_column(String(50))
    service_started_on: Mapped[Optional[date]] = mapped_column(Date)
    service_ended_on: Mapped[Optional[date]] = mapped_column(Date)
    requested_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    eligible_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    fixed_share_applied: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    percentage_share_applied: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    payable_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    case_status: Mapped[str] = mapped_column(String(50), default="OPENED")
    submitted_on: Mapped[Optional[date]] = mapped_column(Date)
    resolved_on: Mapped[Optional[date]] = mapped_column(Date)
    decision_reason: Mapped[Optional[str]] = mapped_column(Text)
    reviewer_notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    policy: Mapped["Policy"] = relationship("Policy", back_populates="service_cases")
    member: Mapped[Optional["Member"]] = relationship("Member", back_populates="service_cases")


class ICDCode(Base):
    """ICD diagnostic codes reference."""
    __tablename__ = "icd_codes"
    
    code: Mapped[str] = mapped_column(String(20), primary_key=True)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(255))
    is_chronic: Mapped[bool] = mapped_column(Boolean, default=False)
    is_pre_existing: Mapped[bool] = mapped_column(Boolean, default=False)
    typical_hospitalization_days: Mapped[Optional[int]] = mapped_column(Integer)
    typical_treatment_cost: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))


class CareProvider(Base):
    """CareProvider network details."""
    __tablename__ = "care_providers"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    registry_ref: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    provider_label: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_kind: Mapped[str] = mapped_column(String(50), default="PARTICIPATING")
    address: Mapped[Optional[str]] = mapped_column(Text)
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(100))
    postal_code: Mapped[Optional[str]] = mapped_column(String(10))
    tier: Mapped[Optional[str]] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_participating: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvalQuestion(Base):
    """Evaluation questions for RAG testing."""
    __tablename__ = "eval_questions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth_answer: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(50), default="DOCUMENT_QA")
    difficulty: Mapped[str] = mapped_column(String(20), default="MEDIUM")
    relevant_contract_ids: Mapped[Optional[List[int]]] = mapped_column(ARRAY(Integer))
    relevant_chunk_ids: Mapped[Optional[List[int]]] = mapped_column(ARRAY(Integer))
    expected_sql: Mapped[Optional[str]] = mapped_column(Text)
    question_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvalResult(Base):
    """Evaluation run results."""
    __tablename__ = "eval_results"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(100), nullable=False)
    question_id: Mapped[int] = mapped_column(Integer, ForeignKey("eval_questions.id"))
    generated_answer: Mapped[Optional[str]] = mapped_column(Text)
    actual_answer: Mapped[Optional[str]] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[Optional[List[int]]] = mapped_column(ARRAY(Integer))
    faithfulness_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    relevance_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    similarity_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    context_precision: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class QueryLog(Base):
    """Query logging for observability."""
    __tablename__ = "query_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[Optional[str]] = mapped_column(String(50))
    response: Mapped[Optional[str]] = mapped_column(Text)
    retrieved_chunks: Mapped[Optional[dict]] = mapped_column(JSON)
    citations: Mapped[Optional[dict]] = mapped_column(JSON)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    model_used: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
