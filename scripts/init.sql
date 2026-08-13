-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Policies table
CREATE TABLE IF NOT EXISTS benefit_contracts (
    id SERIAL PRIMARY KEY,
    contract_ref VARCHAR(100) UNIQUE NOT NULL,
    contract_title VARCHAR(255),
    plan_category VARCHAR(50) NOT NULL DEFAULT 'EMPLOYEE_BENEFITS',
    plan_ref VARCHAR(50),
    sponsor_label VARCHAR(255),
    sponsor_sector VARCHAR(100),
    effective_from DATE,
    effective_until DATE,
    participant_count INTEGER,
    aggregate_benefit_cap DECIMAL(15,2),
    contribution_amount DECIMAL(15,2),
    service_partner_label VARCHAR(255),
    source_document_uri VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Policy chunks table (for RAG)
CREATE TABLE IF NOT EXISTS contract_passages (
    id SERIAL PRIMARY KEY,
    contract_id INTEGER REFERENCES benefit_contracts(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    topic_category VARCHAR(100),
    topic_title VARCHAR(255),
    source_page INTEGER,
    passage_order INTEGER,
    source_offset_start INTEGER,
    source_offset_end INTEGER,
    token_count INTEGER,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON contract_passages
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Coverages table
CREATE TABLE IF NOT EXISTS plan_benefits (
    id SERIAL PRIMARY KEY,
    contract_id INTEGER REFERENCES benefit_contracts(id) ON DELETE CASCADE,
    benefit_ref VARCHAR(50) NOT NULL,
    benefit_title VARCHAR(255) NOT NULL,
    benefit_tier VARCHAR(50) DEFAULT 'BASE',
    limit_basis VARCHAR(50),
    benefit_value DECIMAL(15,2),
    benefit_cap DECIMAL(15,2),
    inner_cap_percent DECIMAL(5,2),
    fixed_share_amount DECIMAL(15,2),
    percentage_share DECIMAL(5,2),
    eligibility_delay_days INTEGER,
    is_core_benefit BOOLEAN DEFAULT true,
    eligibility_rules TEXT,
    non_covered_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Members table
CREATE TABLE IF NOT EXISTS participants (
    id SERIAL PRIMARY KEY,
    contract_id INTEGER REFERENCES benefit_contracts(id) ON DELETE CASCADE,
    participant_ref VARCHAR(50) UNIQUE,
    sponsor_person_ref VARCHAR(50),
    participant_label VARCHAR(255) NOT NULL,
    enrolment_role VARCHAR(50) NOT NULL DEFAULT 'SELF',
    gender VARCHAR(10),
    birth_date DATE,
    age INTEGER,
    benefit_ceiling DECIMAL(15,2),
    contribution_amount DECIMAL(15,2),
    plan_variant VARCHAR(100),
    enrolled_on DATE,
    eligibility_from DATE,
    eligibility_until DATE,
    status VARCHAR(50) DEFAULT 'ACTIVE',
    city VARCHAR(100),
    state VARCHAR(100),
    postal_code VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Claims table
CREATE TABLE IF NOT EXISTS service_cases (
    id SERIAL PRIMARY KEY,
    case_ref VARCHAR(100) UNIQUE NOT NULL,
    contract_id INTEGER REFERENCES benefit_contracts(id) ON DELETE CASCADE,
    participant_id INTEGER REFERENCES participants(id) ON DELETE CASCADE,
    funding_mode VARCHAR(50) NOT NULL DEFAULT 'DIRECT_BILLING',
    care_setting VARCHAR(50) DEFAULT 'FACILITY_STAY',
    condition_code VARCHAR(20),
    condition_label VARCHAR(500),
    service_category VARCHAR(255),
    provider_label VARCHAR(255),
    provider_city VARCHAR(100),
    provider_region VARCHAR(100),
    provider_registry_ref VARCHAR(50),
    service_started_on DATE,
    service_ended_on DATE,
    requested_amount DECIMAL(15,2),
    eligible_amount DECIMAL(15,2),
    fixed_share_applied DECIMAL(15,2),
    percentage_share_applied DECIMAL(15,2),
    payable_amount DECIMAL(15,2),
    case_status VARCHAR(50) DEFAULT 'OPENED',
    submitted_on DATE,
    resolved_on DATE,
    decision_reason TEXT,
    reviewer_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ICD codes reference table
CREATE TABLE IF NOT EXISTS icd_codes (
    code VARCHAR(20) PRIMARY KEY,
    description VARCHAR(500) NOT NULL,
    category VARCHAR(255),
    is_chronic BOOLEAN DEFAULT false,
    is_pre_existing BOOLEAN DEFAULT false,
    typical_hospitalization_days INTEGER,
    typical_treatment_cost DECIMAL(15,2)
);

-- CareProviders table
CREATE TABLE IF NOT EXISTS care_providers (
    id SERIAL PRIMARY KEY,
    registry_ref VARCHAR(50) UNIQUE,
    provider_label VARCHAR(255) NOT NULL,
    provider_kind VARCHAR(50) DEFAULT 'PARTICIPATING',
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(100),
    postal_code VARCHAR(10),
    tier VARCHAR(20),
    is_active BOOLEAN DEFAULT true,
    is_participating BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation questions table
CREATE TABLE IF NOT EXISTS eval_questions (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    ground_truth_answer TEXT NOT NULL,
    query_type VARCHAR(50) NOT NULL DEFAULT 'DOCUMENT_QA',
    difficulty VARCHAR(20) DEFAULT 'MEDIUM',
    relevant_contract_ids INTEGER[],
    relevant_chunk_ids INTEGER[],
    expected_sql TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation results table
CREATE TABLE IF NOT EXISTS eval_results (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) NOT NULL,
    question_id INTEGER REFERENCES eval_questions(id),
    generated_answer TEXT,
    actual_answer TEXT,
    retrieved_chunk_ids INTEGER[],
    faithfulness_score DECIMAL(5,4),
    relevance_score DECIMAL(5,4),
    similarity_score DECIMAL(5,4),
    context_precision DECIMAL(5,4),
    is_correct BOOLEAN,
    latency_ms INTEGER,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query logs table
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    query_type VARCHAR(50),
    response TEXT,
    retrieved_chunks JSONB,
    citations JSONB,
    latency_ms INTEGER,
    token_count INTEGER,
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_service_cases_policy ON service_cases(contract_id);
CREATE INDEX IF NOT EXISTS idx_service_cases_participant ON service_cases(participant_id);
CREATE INDEX IF NOT EXISTS idx_service_cases_status ON service_cases(case_status);
CREATE INDEX IF NOT EXISTS idx_service_cases_date ON service_cases(submitted_on);
CREATE INDEX IF NOT EXISTS idx_participants_policy ON participants(contract_id);
CREATE INDEX IF NOT EXISTS idx_chunks_policy ON contract_passages(contract_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON contract_passages(topic_category);

-- Insert sample ICD codes
INSERT INTO icd_codes (code, description, category, is_chronic, typical_hospitalization_days, typical_treatment_cost) VALUES
('A90', 'Dengue fever', 'Infectious Diseases', false, 5, 50000),
('B34.9', 'Viral infection, unspecified', 'Infectious Diseases', false, 3, 25000),
('E11.9', 'Type 2 diabetes mellitus without complications', 'Metabolic Disorders', true, 0, 15000),
('I25.10', 'Atherosclerotic heart disease of native coronary artery', 'Cardiovascular', true, 7, 300000),
('J18.9', 'Pneumonia, unspecified organism', 'Respiratory', false, 5, 75000),
('K35.80', 'Unspecified acute appendicitis', 'Digestive', false, 3, 80000),
('K80.20', 'Calculus of gallbladder without cholecystitis', 'Digestive', false, 2, 90000),
('N20.0', 'Calculus of kidney', 'Genitourinary', false, 2, 60000),
('O80', 'Encounter for full-term uncomplicated delivery', 'Pregnancy', false, 3, 50000),
('S72.001A', 'Fracture of unspecified part of neck of femur', 'Injury', false, 10, 200000)
ON CONFLICT (code) DO NOTHING;

COMMENT ON TABLE benefit_contracts IS 'Master table for insurance benefit_contracts';
COMMENT ON TABLE contract_passages IS 'Document chunks with vector embeddings for RAG';
COMMENT ON TABLE service_cases IS 'Insurance service_cases records';
COMMENT ON TABLE eval_questions IS 'Evaluation dataset for RAG system testing';
