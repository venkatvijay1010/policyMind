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
    __tablename__ = "policies"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_number: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    policy_name: Mapped[Optional[str]] = mapped_column(String(255))
    product_type: Mapped[str] = mapped_column(String(50), default="GROUP_HEALTH")
    product_code: Mapped[Optional[str]] = mapped_column(String(50))
    insured_name: Mapped[Optional[str]] = mapped_column(String(255))
    industry_type: Mapped[Optional[str]] = mapped_column(String(100))
    policy_start_date: Mapped[Optional[date]] = mapped_column(Date)
    policy_end_date: Mapped[Optional[date]] = mapped_column(Date)
    total_lives: Mapped[Optional[int]] = mapped_column(Integer)
    total_sum_insured: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    premium_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    tpa_name: Mapped[Optional[str]] = mapped_column(String(255))
    document_s3_link: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    chunks: Mapped[List["PolicyChunk"]] = relationship("PolicyChunk", back_populates="policy", cascade="all, delete-orphan")
    coverages: Mapped[List["Coverage"]] = relationship("Coverage", back_populates="policy", cascade="all, delete-orphan")
    members: Mapped[List["Member"]] = relationship("Member", back_populates="policy", cascade="all, delete-orphan")
    claims: Mapped[List["Claim"]] = relationship("Claim", back_populates="policy", cascade="all, delete-orphan")


class PolicyChunk(Base):
    """Policy document chunks for RAG."""
    __tablename__ = "policy_chunks"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_id: Mapped[int] = mapped_column(Integer, ForeignKey("policies.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    section_type: Mapped[Optional[str]] = mapped_column(String(100))
    section_name: Mapped[Optional[str]] = mapped_column(String(255))
    page_number: Mapped[Optional[int]] = mapped_column(Integer)
    chunk_index: Mapped[Optional[int]] = mapped_column(Integer)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(settings.embedding_dimension))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    policy: Mapped["Policy"] = relationship("Policy", back_populates="chunks")


class Coverage(Base):
    """Coverage details for a policy."""
    __tablename__ = "coverages"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_id: Mapped[int] = mapped_column(Integer, ForeignKey("policies.id", ondelete="CASCADE"))
    coverage_code: Mapped[str] = mapped_column(String(50), nullable=False)
    coverage_name: Mapped[str] = mapped_column(String(255), nullable=False)
    coverage_type: Mapped[str] = mapped_column(String(50), default="BASE")
    sum_type: Mapped[Optional[str]] = mapped_column(String(50))
    coverage_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    coverage_limit: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    sub_limit_percentage: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    deductible_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    copay_percentage: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    waiting_period_days: Mapped[Optional[int]] = mapped_column(Integer)
    is_mandatory: Mapped[bool] = mapped_column(Boolean, default=True)
    benefit_criteria: Mapped[Optional[str]] = mapped_column(Text)
    exclusion_notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    policy: Mapped["Policy"] = relationship("Policy", back_populates="coverages")


class Member(Base):
    """Member/insured person details."""
    __tablename__ = "members"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_id: Mapped[int] = mapped_column(Integer, ForeignKey("policies.id", ondelete="CASCADE"))
    member_id: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    employee_code: Mapped[Optional[str]] = mapped_column(String(50))
    member_name: Mapped[str] = mapped_column(String(255), nullable=False)
    relationship: Mapped[str] = mapped_column(String(50), default="SELF")
    gender: Mapped[Optional[str]] = mapped_column(String(10))
    date_of_birth: Mapped[Optional[date]] = mapped_column(Date)
    age: Mapped[Optional[int]] = mapped_column(Integer)
    sum_insured: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    premium_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    package_name: Mapped[Optional[str]] = mapped_column(String(100))
    enrollment_date: Mapped[Optional[date]] = mapped_column(Date)
    risk_inception_date: Mapped[Optional[date]] = mapped_column(Date)
    risk_expiry_date: Mapped[Optional[date]] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(50), default="ACTIVE")
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(100))
    pincode: Mapped[Optional[str]] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    policy: Mapped["Policy"] = relationship("Policy", back_populates="members")
    claims: Mapped[List["Claim"]] = relationship("Claim", back_populates="member", cascade="all, delete-orphan")


class Claim(Base):
    """Insurance claim records."""
    __tablename__ = "claims"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_number: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    policy_id: Mapped[int] = mapped_column(Integer, ForeignKey("policies.id", ondelete="CASCADE"))
    member_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("members.id", ondelete="CASCADE"))
    claim_type: Mapped[str] = mapped_column(String(50), default="CASHLESS")
    claim_category: Mapped[str] = mapped_column(String(50), default="IPD")
    diagnosis_code: Mapped[Optional[str]] = mapped_column(String(20))
    diagnosis_description: Mapped[Optional[str]] = mapped_column(String(500))
    treatment_type: Mapped[Optional[str]] = mapped_column(String(255))
    hospital_name: Mapped[Optional[str]] = mapped_column(String(255))
    hospital_city: Mapped[Optional[str]] = mapped_column(String(100))
    hospital_state: Mapped[Optional[str]] = mapped_column(String(100))
    hospital_rohini_code: Mapped[Optional[str]] = mapped_column(String(50))
    admission_date: Mapped[Optional[date]] = mapped_column(Date)
    discharge_date: Mapped[Optional[date]] = mapped_column(Date)
    claim_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    approved_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    deductible_applied: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    copay_applied: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    net_payable: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2))
    claim_status: Mapped[str] = mapped_column(String(50), default="REGISTERED")
    registration_date: Mapped[Optional[date]] = mapped_column(Date)
    settlement_date: Mapped[Optional[date]] = mapped_column(Date)
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text)
    adjuster_notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    policy: Mapped["Policy"] = relationship("Policy", back_populates="claims")
    member: Mapped[Optional["Member"]] = relationship("Member", back_populates="claims")


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


class Hospital(Base):
    """Hospital network details."""
    __tablename__ = "hospitals"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rohini_code: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    hospital_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hospital_type: Mapped[str] = mapped_column(String(50), default="NETWORK")
    address: Mapped[Optional[str]] = mapped_column(Text)
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(100))
    pincode: Mapped[Optional[str]] = mapped_column(String(10))
    tier: Mapped[Optional[str]] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvalQuestion(Base):
    """Evaluation questions for RAG testing."""
    __tablename__ = "eval_questions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth_answer: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(50), default="DOCUMENT_QA")
    difficulty: Mapped[str] = mapped_column(String(20), default="MEDIUM")
    relevant_policy_ids: Mapped[Optional[List[int]]] = mapped_column(ARRAY(Integer))
    relevant_chunk_ids: Mapped[Optional[List[int]]] = mapped_column(ARRAY(Integer))
    expected_sql: Mapped[Optional[str]] = mapped_column(Text)
    metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvalResult(Base):
    """Evaluation run results."""
    __tablename__ = "eval_results"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(100), nullable=False)
    question_id: Mapped[int] = mapped_column(Integer, ForeignKey("eval_questions.id"))
    generated_answer: Mapped[Optional[str]] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[Optional[List[int]]] = mapped_column(ARRAY(Integer))
    faithfulness_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    relevance_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
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
