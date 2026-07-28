# PolicyMind Post-Run Guide

> **Complete guide for running, testing, validating, and demoing PolicyMind**  
> Assumes the project has been successfully started with `docker-compose up --build`

---

## Table of Contents

1. [Verifying Application Startup](#1-verifying-application-startup)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Health Checks](#3-health-checks)
4. [Seeding Initial Data](#4-seeding-initial-data)
5. [Feature Guide & Workflows](#5-feature-guide--workflows)
6. [Testing Each Subsystem](#6-testing-each-subsystem)
7. [Data Ingestion](#7-data-ingestion)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Troubleshooting](#9-troubleshooting)
10. [Verification Checklist](#10-verification-checklist)
11. [Interview Demo Flow](#11-interview-demo-flow)

---

## 1. Verifying Application Startup

### 1.1 Check Container Status

```bash
docker ps
```

**Expected Output:**
```
CONTAINER ID   IMAGE                    STATUS                   PORTS                    NAMES
abc123...      policymind-app           Up 2 minutes             0.0.0.0:8000->8000/tcp   policymind-app
def456...      pgvector/pgvector:pg16   Up 2 minutes (healthy)   0.0.0.0:5432->5432/tcp   policymind-db
```

**Verify both containers are `Up` and `policymind-db` shows `(healthy)`.**

### 1.2 Check Application Logs

```bash
docker logs policymind-app --tail 50
```

**Expected startup messages:**
```
INFO: Starting PolicyMind application env=development
INFO: Database initialized
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 1.3 Access Swagger UI

Open in browser: **http://localhost:8000/docs**

You should see the FastAPI Swagger interface with these endpoint groups:
- **Health** - System status endpoints
- **Ingestion** - Document upload endpoints
- **Query** - Main Q&A endpoints
- **Evaluation** - Benchmarking endpoints

### 1.4 Quick Smoke Test

```bash
curl http://localhost:8000/health/live
```

**Expected Response:**
```json
{"alive": true}
```

---

## 2. System Architecture Overview

### 2.1 Runtime Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Request                            │
│                     POST /api/v1/ask                             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   LangGraph Orchestrator                  │   │
│  │                                                           │   │
│  │   ┌──────────┐    ┌─────────────────────────────────┐    │   │
│  │   │ CLASSIFY │───▶│         ROUTE DECISION          │    │   │
│  │   └──────────┘    └─────────────────────────────────┘    │   │
│  │                              │                            │   │
│  │         ┌────────────────────┼────────────────────┐      │   │
│  │         ▼                    ▼                    ▼      │   │
│  │   ┌──────────┐        ┌──────────┐        ┌──────────┐   │   │
│  │   │ RAG Node │        │ SQL Node │        │ Hybrid   │   │   │
│  │   │          │        │          │        │ Node     │   │   │
│  │   └────┬─────┘        └────┬─────┘        └────┬─────┘   │   │
│  │        │                   │                   │         │   │
│  │        └───────────────────┴───────────────────┘         │   │
│  │                            │                              │   │
│  │                            ▼                              │   │
│  │                    ┌──────────────┐                       │   │
│  │                    │   RESPONSE   │                       │   │
│  │                    └──────────────┘                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     OpenAI      │    │   Embedding     │
│   + pgvector    │    │   GPT-4 Turbo   │    │   Service       │
│                 │    │                 │    │                 │
│ • Policies      │    │ • Classification│    │ • text-embed-   │
│ • Chunks        │    │ • SQL Gen       │    │   ding-3-small  │
│ • Claims        │    │ • Answers       │    │ • 1536 dims     │
│ • Members       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 Query Classification Logic

| Query Contains | Classification | Agent Used |
|----------------|----------------|------------|
| "covered", "exclusion", "limit", "waiting period" | `document_qa` | RAG Agent |
| "how many", "total", "count", "claims", "statistics" | `claims_sql` | SQL Agent |
| Both document + data keywords | `hybrid` | Hybrid Agent |
| Ambiguous | LLM decides | Based on LLM classification |

### 2.3 Data Flow for Each Query Type

**Document Q&A (RAG):**
```
Query → Embed Query → Vector Search (pgvector) → BM25 Search → 
RRF Fusion → Top-K Chunks → Build Context → GPT-4 Generate → 
Add Citations → Response
```

**Claims SQL:**
```
Query → Safety Check → GPT-4 Generate SQL → Execute SQL → 
Retry on Error (max 2) → GPT-4 Explain Results → Response
```

**Hybrid:**
```
Query → Extract Sub-queries → Parallel Execute (RAG + SQL) → 
GPT-4 Synthesize → Combined Response
```

---

## 3. Health Checks

### 3.1 Basic Health Check

**Endpoint:** `GET /health`

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected",
  "llm": "available",
  "uptime_seconds": 125.5
}
```

**Status Meanings:**
- `healthy` - All systems operational
- `degraded` - Some components have issues
- `unhealthy` - Critical failure

### 3.2 Detailed Health Check

**Endpoint:** `GET /health/detailed`

```bash
curl http://localhost:8000/health/detailed
```

**Expected Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "database": {
      "status": "healthy",
      "policy_count": 3
    },
    "vector_store": {
      "status": "healthy",
      "chunk_count": 45
    },
    "llm": {
      "status": "configured",
      "chat_model": "gpt-4-turbo-preview",
      "embedding_model": "text-embedding-3-small"
    }
  },
  "metrics": {
    "uptime_seconds": 125.5,
    "environment": "development"
  }
}
```

### 3.3 Kubernetes Probes

**Liveness Probe:** `GET /health/live`
```bash
curl http://localhost:8000/health/live
# {"alive": true}
```

**Readiness Probe:** `GET /health/ready`
```bash
curl http://localhost:8000/health/ready
# {"ready": true}  (or 503 if DB not ready)
```

---

## 4. Seeding Initial Data

### 4.1 Run the Seed Script

```bash
docker exec -it policymind-app python -m data.seed.seed_database
```

**Expected Output:**
```
Starting database seeding...

1. Seeding ICD codes...
   Seeded 10 ICD codes

2. Seeding hospitals...
   Seeded 5 hospitals

3. Seeding policies...
   Seeded 3 policies

4. Seeding policy chunks with embeddings...
   Seeded 45 policy chunks

5. Seeding members...
   Seeded 45 members

6. Seeding claims...
   Seeded 150 claims

7. Seeding evaluation questions...
   Seeded 20 evaluation questions

Database seeding complete!
```

### 4.2 Verify Seeded Data

```bash
curl http://localhost:8000/health/detailed
```

Check that:
- `policy_count` > 0
- `chunk_count` > 0

### 4.3 What Gets Seeded

| Table | Records | Description |
|-------|---------|-------------|
| `policies` | 3 | Sample insurance policies |
| `policy_chunks` | ~45 | Document chunks with embeddings |
| `members` | 45 | Insured members (15 per policy) |
| `claims` | ~150 | Sample claims with various statuses |
| `hospitals` | 5 | Network and non-network hospitals |
| `icd_codes` | 10 | Common diagnosis codes |
| `eval_questions` | 20 | Ground truth Q&A pairs |

---

## 5. Feature Guide & Workflows

### 5.1 Main Q&A Endpoint

**Endpoint:** `POST /api/v1/ask`

This is the primary feature - ask any insurance-related question.

#### Request Format

```json
{
  "query": "Your question here",
  "policy_id": 1,           // Optional: scope to specific policy
  "search_method": "hybrid" // Options: "vector", "bm25", "hybrid"
}
```

#### Workflow: Document Q&A

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the maternity coverage limit?",
    "policy_id": 1
  }'
```

**Internal Flow:**
1. Query received by `/ask` endpoint
2. LangGraph orchestrator invokes `classify` node
3. Keywords detected: "coverage", "limit" → `document_qa`
4. Routes to RAG agent
5. RAG agent:
   - Generates query embedding (OpenAI)
   - Performs vector search (pgvector cosine similarity)
   - Performs BM25 keyword search
   - Fuses results using Reciprocal Rank Fusion
   - Builds context from top-5 chunks
   - Calls GPT-4 with context + question
   - Builds citations from source chunks
6. Returns answer with sources

**Expected Response:**
```json
{
  "query": "What is the maternity coverage limit?",
  "query_type": "document_qa",
  "answer": "The maternity coverage limit is **₹50,000 per pregnancy**. [Source 1]\n\nKey details:\n- Normal Delivery: ₹25,000\n- Cesarean Section: ₹50,000\n- Waiting Period: 36 months from policy inception\n\n**Sources:**\n[1] Policy-1, Maternity Benefits (Page 2)",
  "citations": [
    {
      "source_id": 1,
      "policy_name": "Policy-1",
      "section": "Maternity Benefits",
      "page": 2,
      "chunk_text": "MATERNITY BENEFITS\n\nCoverage Amount: Up to ₹50,000 per pregnancy\n\nWaiting Period: 36 months...",
      "relevance_score": 0.92
    }
  ],
  "sql_query": null,
  "sql_result": null,
  "latency_ms": 1250,
  "model_used": "gpt-4-turbo-preview"
}
```

#### Workflow: Claims SQL Query

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many claims were rejected and what was the total amount?"
  }'
```

**Internal Flow:**
1. Query classified as `claims_sql` (keywords: "how many", "claims")
2. Routes to SQL agent
3. SQL agent:
   - Validates query is safe (no injection patterns)
   - Generates SQL using GPT-4 with schema context
   - Executes SQL against PostgreSQL
   - If error: regenerates SQL (up to 2 retries)
   - Generates natural language explanation
4. Returns answer with SQL query and results

**Expected Response:**
```json
{
  "query": "How many claims were rejected and what was the total amount?",
  "query_type": "claims_sql",
  "answer": "Based on the claims data:\n\n• **Total Rejected Claims:** 15\n• **Total Rejected Amount:** ₹12,45,000\n\nThe most common rejection reasons were:\n1. Pre-existing condition - waiting period not completed (6 claims)\n2. Non-disclosure of medical history (4 claims)\n3. Treatment not covered under policy (3 claims)",
  "citations": [],
  "sql_query": "SELECT COUNT(*) as rejected_count, SUM(claim_amount) as total_amount FROM claims WHERE claim_status = 'REJECTED'",
  "sql_result": [
    {
      "rejected_count": 15,
      "total_amount": 1245000
    }
  ],
  "latency_ms": 890,
  "model_used": "gpt-4-turbo-preview"
}
```

#### Workflow: Hybrid Query

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is our rejection rate for pre-existing conditions and what does the policy say about the waiting period?"
  }'
```

**Internal Flow:**
1. Query classified as `hybrid` (both data + document keywords)
2. Routes to Hybrid agent
3. Hybrid agent:
   - Extracts sub-queries using GPT-4
   - Parallel execution:
     - RAG: searches for "pre-existing waiting period"
     - SQL: queries rejection statistics
   - Synthesizes combined answer using GPT-4
4. Returns unified response

**Expected Response:**
```json
{
  "query": "What is our rejection rate for pre-existing conditions...",
  "query_type": "hybrid",
  "answer": "**Policy Information:**\nPre-existing diseases have a waiting period of **48 months** from policy inception. After this period, they are covered at par with other claims with no additional deductible.\n\n**Claims Data Analysis:**\nOut of 150 total claims, **6 were rejected** due to pre-existing condition waiting period not being completed, representing a **4% rejection rate** for this specific reason.\n\n**Sources:**\n[1] Policy-1, Pre-existing Diseases",
  "citations": [...],
  "sql_query": "SELECT COUNT(*) FROM claims WHERE rejection_reason LIKE '%pre-existing%'",
  "sql_result": [{"count": 6}],
  "latency_ms": 2100,
  "model_used": "gpt-4-turbo-preview"
}
```

### 5.2 Query Classification Endpoint

**Endpoint:** `POST /api/v1/ask/classify`

Test classification without executing the query.

```bash
curl -X POST http://localhost:8000/api/v1/ask/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the room rent limit?"}'
```

**Response:**
```json
{
  "query": "What is the room rent limit?",
  "query_type": "document_qa",
  "confidence": 0.9,
  "reasoning": "Query contains policy/coverage keywords"
}
```

### 5.3 Force RAG Endpoint

**Endpoint:** `POST /api/v1/ask/rag`

Bypass classification and force document search.

```bash
curl -X POST http://localhost:8000/api/v1/ask/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about exclusions"}'
```

### 5.4 Force SQL Endpoint

**Endpoint:** `POST /api/v1/ask/sql`

Bypass classification and force SQL query.

```bash
curl -X POST http://localhost:8000/api/v1/ask/sql \
  -H "Content-Type: application/json" \
  -d '{"query": "Show claim statistics"}'
```

---

## 6. Testing Each Subsystem

### 6.1 Database Connectivity

```bash
# Test direct DB connection
docker exec -it policymind-db psql -U policymind -d policymind -c "SELECT COUNT(*) FROM policies;"
```

**Expected:** Returns count of policies (3 after seeding)

### 6.2 Vector Search (pgvector)

```bash
# Check vector index exists
docker exec -it policymind-db psql -U policymind -d policymind -c "\d policy_chunks"
```

**Look for:** `embedding vector(1536)` column

**Test vector search via API:**
```bash
curl -X POST http://localhost:8000/api/v1/ask/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "maternity benefits", "search_method": "vector"}'
```

### 6.3 BM25 Search

```bash
curl -X POST http://localhost:8000/api/v1/ask/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "maternity benefits", "search_method": "bm25"}'
```

### 6.4 Hybrid Search (RRF)

```bash
curl -X POST http://localhost:8000/api/v1/ask/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "maternity benefits", "search_method": "hybrid"}'
```

### 6.5 LLM (OpenAI)

```bash
# Classification uses LLM
curl -X POST http://localhost:8000/api/v1/ask/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "This is a test query about coverage and claims"}'
```

**If LLM fails:** Response will have `"confidence": 0.5` and `"reasoning": "Fallback due to classification error"`

### 6.6 SQL Generation & Execution

```bash
curl -X POST http://localhost:8000/api/v1/ask/sql \
  -H "Content-Type: application/json" \
  -d '{"query": "Show top 5 hospitals by number of claims"}'
```

**Verify:** 
- `sql_query` field contains valid SELECT statement
- `sql_result` field contains data rows

### 6.7 Citation Builder

Any RAG query should include citations. Verify:
- `citations` array is not empty
- Each citation has `source_id`, `policy_name`, `chunk_text`, `relevance_score`

---

## 7. Data Ingestion

### 7.1 Ingest Text Content

**Endpoint:** `POST /api/v1/ingest`

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": 1,
    "document_text": "DENTAL COVERAGE\n\nDental treatments are covered only when they require hospitalization for more than 24 hours.\n\nCoverage Limit: ₹25,000 per year\n\nExclusions:\n- Routine dental checkups\n- Cosmetic dental procedures\n- Dentures and implants",
    "chunk_size": 500,
    "chunk_overlap": 50
  }'
```

**Expected Response:**
```json
{
  "policy_id": 1,
  "chunks_created": 1,
  "processing_time_ms": 1250,
  "message": "Successfully ingested 1 chunks for policy 'Group Health Shield Premium'"
}
```

**What Happens Internally:**
1. Text is split into chunks (configurable size/overlap)
2. Each chunk is embedded via OpenAI
3. Chunks + embeddings stored in `policy_chunks` table
4. Vector index updated for similarity search

### 7.2 Ingest from URL

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": 1,
    "document_url": "https://example.com/policy.txt"
  }'
```

### 7.3 Upload File

```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "policy_id=1" \
  -F "file=@/path/to/document.txt"
```

**Supported formats:** `.txt`, `.md`

### 7.4 Check Ingestion Stats

```bash
curl http://localhost:8000/api/v1/ingest/stats/1
```

**Response:**
```json
{
  "policy_id": 1,
  "chunk_count": 46,
  "avg_chunk_length": 450.5,
  "first_ingested": "2024-01-15T10:30:00",
  "last_ingested": "2024-01-15T10:35:00"
}
```

### 7.5 Re-ingest (Delete + Add)

```bash
# Delete existing chunks
curl -X DELETE http://localhost:8000/api/v1/ingest/1

# Re-ingest
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"policy_id": 1, "document_text": "..."}'
```

---

## 8. Evaluation Framework

### 8.1 List Evaluation Questions

```bash
curl "http://localhost:8000/api/v1/eval/questions?limit=5"
```

**Response:**
```json
[
  {
    "question_id": 1,
    "question": "What is the maternity coverage limit?",
    "expected_answer": "₹50,000 per pregnancy",
    "query_type": "document_qa",
    "difficulty": "easy"
  },
  ...
]
```

### 8.2 Run Evaluation

```bash
curl -X POST http://localhost:8000/api/v1/eval/run \
  -H "Content-Type: application/json" \
  -d '{
    "sample_size": 10
  }'
```

**Response:**
```json
{
  "run_id": "abc123-...",
  "started_at": "2024-01-15T10:30:00",
  "completed_at": "2024-01-15T10:32:00",
  "summary": {
    "total_questions": 10,
    "correct_answers": 8,
    "accuracy": 0.8,
    "avg_latency_ms": 1150.5,
    "avg_similarity": 0.82,
    "by_query_type": {
      "document_qa": {"total": 6, "correct": 5, "accuracy": 0.83},
      "claims_sql": {"total": 4, "correct": 3, "accuracy": 0.75}
    },
    "by_difficulty": {
      "easy": {"total": 4, "correct": 4, "accuracy": 1.0},
      "medium": {"total": 4, "correct": 3, "accuracy": 0.75},
      "hard": {"total": 2, "correct": 1, "accuracy": 0.5}
    }
  },
  "results": [...]
}
```

### 8.3 Filter Evaluations

```bash
# By query type
curl -X POST http://localhost:8000/api/v1/eval/run \
  -H "Content-Type: application/json" \
  -d '{"query_types": ["document_qa"]}'

# By specific questions
curl -X POST http://localhost:8000/api/v1/eval/run \
  -H "Content-Type: application/json" \
  -d '{"question_ids": [1, 2, 3]}'
```

### 8.4 View Evaluation History

```bash
curl http://localhost:8000/api/v1/eval/history
```

### 8.5 View Run Details

```bash
curl http://localhost:8000/api/v1/eval/run/abc123-...
```

---

## 9. Troubleshooting

### 9.1 Container Issues

| Problem | Diagnosis | Fix |
|---------|-----------|-----|
| App container not starting | `docker logs policymind-app` | Check for import errors |
| DB container not healthy | `docker logs policymind-db` | Check PostgreSQL errors |
| Can't connect to port 8000 | `docker ps` | Ensure container is running |

### 9.2 API Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `500 Internal Server Error` | Check logs | `docker logs policymind-app --tail 100` |
| `422 Validation Error` | Invalid request body | Check request format in Swagger |
| `404 Not Found` | Wrong endpoint | Check endpoint URL |
| `503 Service Unavailable` | DB not ready | Wait for DB health check |
| `429 Too Many Requests` | Rate limited | Wait 1 minute (100 req/min limit) |

### 9.3 OpenAI Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Failed to generate embedding` | Invalid API key | Check `OPENAI_API_KEY` in `.env` |
| `Rate limit exceeded` | Too many requests | Implement backoff, check OpenAI dashboard |
| `Context length exceeded` | Query too long | Reduce chunk size or query length |

### 9.4 Database Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `relation does not exist` | Tables not created | Re-run `docker-compose down -v && docker-compose up` |
| `connection refused` | DB not running | Check DB container status |
| `column does not exist` | Schema mismatch | Check init.sql matches models.py |

### 9.5 Search Quality Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Irrelevant results | Poor embeddings | Re-ingest with smaller chunks |
| Missing results | Query mismatch | Try different search_method |
| No citations | No chunks in DB | Seed data or ingest documents |

### 9.6 SQL Generation Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Incorrect SQL | Schema not understood | Check SCHEMA_INFO in sql_agent.py |
| SQL execution error | Bad query generated | Agent retries up to 2 times |
| "Query rejected" | Dangerous SQL detected | Safety check blocked injection attempt |

---

## 10. Verification Checklist

### Pre-Demo Checklist

- [ ] Both containers running (`docker ps`)
- [ ] Health check returns `healthy` 
- [ ] Database has seeded data (`chunk_count > 0`)
- [ ] OpenAI API key is valid (classification works)
- [ ] Vector search returns results
- [ ] SQL queries execute successfully

### Functional Checklist

| Feature | Test Command | Expected Result |
|---------|--------------|-----------------|
| Health | `curl /health` | `status: healthy` |
| RAG Query | `POST /ask` with coverage question | Answer + citations |
| SQL Query | `POST /ask` with claims question | Answer + sql_result |
| Hybrid | `POST /ask` with mixed question | Combined answer |
| Classification | `POST /ask/classify` | query_type + confidence |
| Ingestion | `POST /ingest` | chunks_created > 0 |
| Evaluation | `POST /eval/run` | accuracy score |

### Quick Validation Script

```bash
#!/bin/bash
echo "1. Health Check..."
curl -s http://localhost:8000/health | jq .status

echo "2. Document Q&A..."
curl -s -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is covered?"}' | jq .query_type

echo "3. SQL Query..."
curl -s -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How many claims?"}' | jq .query_type

echo "4. Classification..."
curl -s -X POST http://localhost:8000/api/v1/ask/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' | jq .confidence

echo "All checks complete!"
```

---

## 11. Interview Demo Flow

### Recommended 10-Minute Demo Script

#### Part 1: System Overview (2 min)

1. **Show Swagger UI** (http://localhost:8000/docs)
   - "This is PolicyMind - an Agentic RAG system for Insurance Q&A"
   - Point out endpoint groups

2. **Health Check**
   ```
   GET /health/detailed
   ```
   - "Shows all components are operational"
   - "We have 3 policies, 45 document chunks in our vector store"

#### Part 2: Document Q&A - RAG (2 min)

3. **Ask a coverage question**
   ```json
   POST /api/v1/ask
   {"query": "What is the maternity coverage limit?"}
   ```
   
   **Explain:**
   - "Query gets classified as document_qa based on keywords"
   - "RAG agent performs hybrid search - combining vector similarity and BM25"
   - "Top chunks are used as context for GPT-4"
   - "Answer includes citations with source, section, and page"

#### Part 3: Claims SQL (2 min)

4. **Ask a data question**
   ```json
   POST /api/v1/ask
   {"query": "How many claims were rejected last year and what were the top reasons?"}
   ```
   
   **Explain:**
   - "Classified as claims_sql - needs database query"
   - "System generates SQL using GPT-4 with schema awareness"
   - "Safety checks prevent SQL injection"
   - "Self-correction: if SQL fails, it regenerates up to 2 times"
   - "Results are explained in natural language"

#### Part 4: Hybrid Query (2 min)

5. **Ask a complex question**
   ```json
   POST /api/v1/ask
   {"query": "What's our rejection rate for pre-existing conditions and what does the policy say about waiting periods?"}
   ```
   
   **Explain:**
   - "Hybrid agent breaks this into sub-queries"
   - "Runs RAG and SQL in parallel"
   - "Synthesizes a unified response"

#### Part 5: Architecture Highlight (2 min)

6. **Show classification endpoint**
   ```json
   POST /api/v1/ask/classify
   {"query": "Some test query"}
   ```
   
   **Explain:**
   - "LangGraph state machine orchestrates the flow"
   - "Classify → Route → Execute → Return"
   - "Each agent is specialized for its task"

7. **Mention key features:**
   - Hybrid search (vector + BM25 + RRF fusion)
   - Self-correcting SQL with retry logic
   - Citations for every claim
   - Deterministic coverage calculations (never trust LLM for math)
   - Evaluation framework for benchmarking

### Technical Deep-Dive Questions & Answers

**Q: Why hybrid search instead of just vector search?**
> Vector search captures semantic similarity but can miss exact keyword matches. BM25 handles exact terms well. Reciprocal Rank Fusion (RRF) combines both rankings, giving us best of both worlds. Our k=60 parameter balances the contribution.

**Q: How do you prevent SQL injection?**
> Three layers: (1) Query classifier rejects dangerous keywords before routing, (2) SQL agent validates generated SQL starts with SELECT/WITH only, (3) Banned keywords list blocks DROP, DELETE, INSERT, etc. Plus we use parameterized queries.

**Q: Why not use LLM for coverage calculations?**
> LLMs are unreliable for arithmetic. We have a deterministic CoverageCalculator class that handles room rent limits, copay, deductibles with exact Decimal math. The LLM explains results, but never computes them.

**Q: How do you handle LLM failures?**
> Retry with exponential backoff (3 attempts). For classification, we fall back to document_qa with low confidence. For SQL, we provide error context to the LLM for self-correction.

**Q: What's the embedding model and dimension?**
> text-embedding-3-small with 1536 dimensions. Stored in PostgreSQL using pgvector extension with IVFFlat index for efficient similarity search.

---

## Appendix: Sample Queries by Category

### Document Q&A Queries
- "What is the maternity coverage limit?"
- "Is dental treatment covered?"
- "What are the permanent exclusions?"
- "What's the waiting period for pre-existing diseases?"
- "What documents are required for claim submission?"
- "What's the room rent limit for premium plan?"
- "Is Ayurveda treatment covered?"
- "What is the copay percentage?"

### Claims SQL Queries
- "How many claims were filed this year?"
- "What is the total approved amount?"
- "Show top 10 hospitals by claim count"
- "What's the average claim amount by diagnosis?"
- "How many claims are pending?"
- "What's the claim approval rate?"
- "Show claims by status breakdown"
- "Which member has the highest claims?"

### Hybrid Queries
- "What's our rejection rate and what does the policy say about exclusions?"
- "How do maternity claims compare to the policy limit?"
- "Show claims for pre-existing conditions and explain the waiting period"
- "What's the copay impact on our high-value claims?"

---

*Last updated: July 2026*
