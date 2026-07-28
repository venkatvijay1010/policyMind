-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Policies table
CREATE TABLE IF NOT EXISTS policies (
    id SERIAL PRIMARY KEY,
    policy_number VARCHAR(100) UNIQUE NOT NULL,
    policy_name VARCHAR(255),
    product_type VARCHAR(50) NOT NULL DEFAULT 'GROUP_HEALTH',
    product_code VARCHAR(50),
    insured_name VARCHAR(255),
    industry_type VARCHAR(100),
    policy_start_date DATE,
    policy_end_date DATE,
    total_lives INTEGER,
    total_sum_insured DECIMAL(15,2),
    premium_amount DECIMAL(15,2),
    tpa_name VARCHAR(255),
    document_s3_link VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Policy chunks table (for RAG)
CREATE TABLE IF NOT EXISTS policy_chunks (
    id SERIAL PRIMARY KEY,
    policy_id INTEGER REFERENCES policies(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    section_type VARCHAR(100),
    section_name VARCHAR(255),
    page_number INTEGER,
    chunk_index INTEGER,
    token_count INTEGER,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON policy_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Coverages table
CREATE TABLE IF NOT EXISTS coverages (
    id SERIAL PRIMARY KEY,
    policy_id INTEGER REFERENCES policies(id) ON DELETE CASCADE,
    coverage_code VARCHAR(50) NOT NULL,
    coverage_name VARCHAR(255) NOT NULL,
    coverage_type VARCHAR(50) DEFAULT 'BASE',
    sum_type VARCHAR(50),
    coverage_value DECIMAL(15,2),
    coverage_limit DECIMAL(15,2),
    sub_limit_percentage DECIMAL(5,2),
    deductible_amount DECIMAL(15,2),
    copay_percentage DECIMAL(5,2),
    waiting_period_days INTEGER,
    is_mandatory BOOLEAN DEFAULT true,
    benefit_criteria TEXT,
    exclusion_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Members table
CREATE TABLE IF NOT EXISTS members (
    id SERIAL PRIMARY KEY,
    policy_id INTEGER REFERENCES policies(id) ON DELETE CASCADE,
    member_id VARCHAR(50) UNIQUE,
    employee_code VARCHAR(50),
    member_name VARCHAR(255) NOT NULL,
    relationship VARCHAR(50) NOT NULL DEFAULT 'SELF',
    gender VARCHAR(10),
    date_of_birth DATE,
    age INTEGER,
    sum_insured DECIMAL(15,2),
    premium_amount DECIMAL(15,2),
    package_name VARCHAR(100),
    enrollment_date DATE,
    risk_inception_date DATE,
    risk_expiry_date DATE,
    status VARCHAR(50) DEFAULT 'ACTIVE',
    city VARCHAR(100),
    state VARCHAR(100),
    pincode VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Claims table
CREATE TABLE IF NOT EXISTS claims (
    id SERIAL PRIMARY KEY,
    claim_number VARCHAR(100) UNIQUE NOT NULL,
    policy_id INTEGER REFERENCES policies(id) ON DELETE CASCADE,
    member_id INTEGER REFERENCES members(id) ON DELETE CASCADE,
    claim_type VARCHAR(50) NOT NULL DEFAULT 'CASHLESS',
    claim_category VARCHAR(50) DEFAULT 'IPD',
    diagnosis_code VARCHAR(20),
    diagnosis_description VARCHAR(500),
    treatment_type VARCHAR(255),
    hospital_name VARCHAR(255),
    hospital_city VARCHAR(100),
    hospital_state VARCHAR(100),
    hospital_rohini_code VARCHAR(50),
    admission_date DATE,
    discharge_date DATE,
    claim_amount DECIMAL(15,2),
    approved_amount DECIMAL(15,2),
    deductible_applied DECIMAL(15,2),
    copay_applied DECIMAL(15,2),
    net_payable DECIMAL(15,2),
    claim_status VARCHAR(50) DEFAULT 'REGISTERED',
    registration_date DATE,
    settlement_date DATE,
    rejection_reason TEXT,
    adjuster_notes TEXT,
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

-- Hospitals table
CREATE TABLE IF NOT EXISTS hospitals (
    id SERIAL PRIMARY KEY,
    rohini_code VARCHAR(50) UNIQUE,
    hospital_name VARCHAR(255) NOT NULL,
    hospital_type VARCHAR(50) DEFAULT 'NETWORK',
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(100),
    pincode VARCHAR(10),
    tier VARCHAR(20),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation questions table
CREATE TABLE IF NOT EXISTS eval_questions (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    ground_truth_answer TEXT NOT NULL,
    query_type VARCHAR(50) NOT NULL DEFAULT 'DOCUMENT_QA',
    difficulty VARCHAR(20) DEFAULT 'MEDIUM',
    relevant_policy_ids INTEGER[],
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
    retrieved_chunk_ids INTEGER[],
    faithfulness_score DECIMAL(5,4),
    relevance_score DECIMAL(5,4),
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
CREATE INDEX IF NOT EXISTS idx_claims_policy ON claims(policy_id);
CREATE INDEX IF NOT EXISTS idx_claims_member ON claims(member_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(claim_status);
CREATE INDEX IF NOT EXISTS idx_claims_date ON claims(registration_date);
CREATE INDEX IF NOT EXISTS idx_members_policy ON members(policy_id);
CREATE INDEX IF NOT EXISTS idx_chunks_policy ON policy_chunks(policy_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON policy_chunks(section_type);

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

COMMENT ON TABLE policies IS 'Master table for insurance policies';
COMMENT ON TABLE policy_chunks IS 'Document chunks with vector embeddings for RAG';
COMMENT ON TABLE claims IS 'Insurance claims records';
COMMENT ON TABLE eval_questions IS 'Evaluation dataset for RAG system testing';
