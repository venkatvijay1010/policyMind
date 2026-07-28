# PolicyMind 🧠

**Agentic RAG System for Insurance Policy Q&A**

An intelligent question-answering system that combines Retrieval-Augmented Generation (RAG) with LangGraph-based agent orchestration to answer complex insurance policy questions.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-purple.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)

## 🎯 Overview

PolicyMind is an AI-powered assistant that helps users understand insurance policies and analyze claims data. It uses:

- **Hybrid Search**: Combines vector similarity (pgvector) with BM25 for optimal retrieval
- **Query Classification**: Automatically routes queries to the right agent
- **LangGraph Orchestration**: State machine-based agent coordination
- **Self-Correcting SQL**: Generates and validates SQL queries for claims analysis

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Classifier                              │
│              (document_qa / claims_sql / hybrid)                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    RAG Agent    │ │    SQL Agent    │ │  Hybrid Agent   │
│  (Policy Docs)  │ │ (Claims Data)   │ │   (Combined)    │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph Orchestrator                       │
│              (State Machine + Answer Synthesis)                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Structured Response                           │
│            (Answer + Citations + SQL Results)                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### Setup

1. **Clone and navigate to project**
   ```bash
   cd policymind
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Start with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Seed the database (optional)**
   ```bash
   docker-compose exec app python data/seed/seed_database.py
   ```

5. **Access the API**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Local Development

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Start PostgreSQL (with pgvector)
docker-compose up -d db

# Run the application
uvicorn app.main:app --reload
```

## 📖 API Endpoints

### Query Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ask` | POST | Main query endpoint (auto-routes to appropriate agent) |
| `/api/v1/ask/classify` | POST | Classify query type without execution |
| `/api/v1/ask/rag` | POST | Force RAG agent (document search) |
| `/api/v1/ask/sql` | POST | Force SQL agent (claims data) |

### Example Query

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the maternity coverage limit?"}'
```

**Response:**
```json
{
  "query": "What is the maternity coverage limit?",
  "query_type": "document_qa",
  "answer": "The maternity coverage limit is up to ₹50,000 per pregnancy. Normal delivery is covered up to ₹25,000 and Cesarean Section up to ₹50,000. [Source 1]",
  "citations": [
    {
      "source_id": 1,
      "policy_name": "Group Health Shield Premium",
      "section": "Maternity Benefits",
      "page": 2,
      "chunk_text": "MATERNITY BENEFITS\n\nCoverage Amount: Up to ₹50,000...",
      "relevance_score": 0.92
    }
  ],
  "latency_ms": 450,
  "model_used": "gpt-4-turbo-preview"
}
```

### Ingestion Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ingest` | POST | Ingest document text |
| `/api/v1/ingest/file` | POST | Upload and ingest file |
| `/api/v1/ingest/{policy_id}` | DELETE | Delete chunks for policy |
| `/api/v1/ingest/stats/{policy_id}` | GET | Get ingestion statistics |

### Evaluation Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/eval/questions` | GET | List evaluation questions |
| `/api/v1/eval/run` | POST | Run evaluation |
| `/api/v1/eval/history` | GET | View evaluation history |

### Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/detailed` | GET | Detailed component status |
| `/health/ready` | GET | Kubernetes readiness probe |
| `/health/live` | GET | Kubernetes liveness probe |

## 📁 Project Structure

```
policymind/
├── app/
│   ├── api/
│   │   ├── routes/           # API endpoints
│   │   └── schemas/          # Pydantic models
│   ├── application/
│   │   ├── agents/           # Query agents
│   │   └── graph/            # LangGraph orchestrator
│   ├── domain/
│   │   ├── entities/         # Domain models
│   │   └── services/         # Business logic
│   ├── infrastructure/
│   │   ├── database/         # PostgreSQL + pgvector
│   │   ├── llm/              # OpenAI client + embeddings
│   │   └── search/           # Hybrid search
│   ├── evaluation/           # Evaluation metrics
│   ├── config.py             # Settings
│   └── main.py               # FastAPI app
├── data/
│   ├── generators/           # Synthetic data generators
│   └── seed/                 # Database seeding
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── scripts/
│   └── init.sql              # Database schema
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_coverage_calculator.py -v
```

## 🔧 Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` |
| `CHAT_MODEL` | OpenAI chat model | `gpt-4-turbo-preview` |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `APP_ENV` | Environment (development/production) | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |

## 📊 Evaluation

PolicyMind includes a built-in evaluation framework:

```bash
# Run evaluation on all questions
curl -X POST "http://localhost:8000/api/v1/eval/run" \
  -H "Content-Type: application/json" \
  -d '{}'

# Run evaluation on specific query types
curl -X POST "http://localhost:8000/api/v1/eval/run" \
  -H "Content-Type: application/json" \
  -d '{"query_types": ["document_qa"]}'
```

Metrics tracked:
- **Accuracy**: Percentage of correct answers (>70% similarity)
- **Latency**: p50, p95, p99 response times
- **By Query Type**: Breakdown for document_qa, claims_sql, hybrid

## 🏆 Key Features

1. **Intelligent Query Routing**
   - Automatic classification into document Q&A, SQL, or hybrid
   - Keyword-based fast path + LLM fallback

2. **Hybrid Search**
   - Vector similarity using pgvector (HNSW index)
   - BM25 for keyword matching
   - Reciprocal Rank Fusion (RRF) combining both

3. **Self-Correcting SQL Agent**
   - Natural language to SQL generation
   - Automatic retry on execution errors
   - Read-only query enforcement

4. **Deterministic Coverage Calculations**
   - Room rent limits by plan type
   - Copay and deductible calculations
   - Network/non-network hospital logic

5. **Built-in Evaluation**
   - Pre-defined test questions
   - Answer similarity scoring
   - Historical tracking of evaluation runs

## 📝 Example Queries

**Document Q&A:**
- "What is the maternity coverage limit?"
- "What are the permanent exclusions?"
- "What is the waiting period for pre-existing diseases?"

**Claims SQL:**
- "How many claims were rejected in 2024?"
- "What is the average claim amount by hospital?"
- "Show top 5 cities by total approved amount"

**Hybrid:**
- "What's our rejection rate for pre-existing conditions and what does the policy say about them?"
- "How many maternity claims were filed and what are the benefits?"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ using FastAPI, LangGraph, and OpenAI
