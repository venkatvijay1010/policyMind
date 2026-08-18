# PolicyMind 🧠

**Agentic RAG System for Insurance Policy Q&A**

An intelligent question-answering system that combines Retrieval-Augmented Generation (RAG) with LangGraph-based agent orchestration to answer complex insurance policy questions.

> All organizations, participants, providers, identifiers, amounts, locations, contract terms,
> workflows, and records in this repository are fictional synthetic examples. They are not copied
> from, compatible with, or representative of any employer or insurer production system.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-purple.svg)
![SQLite](https://img.shields.io/badge/SQLite-local-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-local-green.svg)

## 🎯 Overview

PolicyMind is an AI-powered assistant that helps users understand insurance benefit_contracts and analyze service_cases data. It uses:

- **Hybrid Search**: Combines local cosine similarity with BM25 for document retrieval
- **Query Classification**: Automatically routes queries to the right agent
- **LangGraph Orchestration**: State machine-based agent coordination
- **Self-Correcting SQL**: Generates and validates SQL queries for service_cases analysis

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Classifier                              │
│              (document_qa / records_sql / hybrid)                 │
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
- [Ollama](https://ollama.com/) running locally

### Setup

1. **Clone and navigate to project**
   ```bash
   cd policymind
   ```

2. **Pull the local models**
   ```bash
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```

3. **Create environment file**
   ```bash
   # Windows PowerShell
   Copy-Item .env.example .env.local
   # macOS/Linux
   cp .env.example .env.local
   ```
   `.env.local` overrides any legacy values in `.env`, so you do not need to
   delete an old OpenAI key or PostgreSQL configuration.

4. **Seed the local SQLite database**
   ```bash
   python -m data.seed.seed_database
   ```

5. **Run the application and open the chat UI**
   ```bash
   uvicorn app.main:app --reload
   ```
   - Chat UI: http://localhost:8000
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

# Pull local models once
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# Copy local settings and create the SQLite database
Copy-Item .env.example .env.local  # Windows PowerShell
python -m data.seed.seed_database

# Run the application; SQLite is created in data/policymind.db
uvicorn app.main:app --reload
```

### Optional Docker run

Run Ollama on your host machine first, then start the app container. Docker
Desktop resolves `host.docker.internal` to the host Ollama service.

```bash
docker compose up --build
docker compose exec app python -m data.seed.seed_database
```

## 📖 API Endpoints

### Query Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/insights/query` | POST | Main query endpoint (auto-routes to appropriate agent) |
| `/api/v2/insights/route-preview` | POST | Classify query type without execution |
| `/api/v2/insights/document-only` | POST | Force RAG agent (document search) |
| `/api/v2/insights/records-only` | POST | Force SQL agent (service_cases data) |

### Example Query

```bash
curl -X POST "http://localhost:8000/api/v2/insights/query" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the family-support coverage limit?"}'
```

**Response:**
```json
{
  "prompt": "What is the family-support coverage limit?",
  "query_type": "document_qa",
  "answer": "The family-support benefit is capped at CU 50,000 per event. [Source 1]",
  "citations": [
    {
      "source_id": 1,
      "contract_title": "Northstar Benefits Plus",
      "section": "Family Support Benefits",
      "page": 2,
      "chunk_text": "FAMILY-SUPPORT BENEFITS\n\nBenefit Amount: Up to CU 50,000...",
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
| `/api/v2/knowledge/scopes/{scope_key}/source` | PUT | Index source text or a public URI |
| `/api/v2/knowledge/scopes/{scope_key}/file` | PUT | Upload and index a text file |
| `/api/v2/knowledge/scopes/{scope_key}/source` | DELETE | Delete indexed segments |
| `/api/v2/knowledge/scopes/{scope_key}/index-status` | GET | Get indexing statistics |

### Evaluation Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/eval/questions` | GET | List evaluation questions |
| `/api/v2/eval/run` | POST | Run evaluation |
| `/api/v2/eval/history` | GET | View evaluation history |

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
│   │   ├── database/         # SQLite + SQLAlchemy
│   │   ├── llm/              # Ollama-compatible chat + embeddings
│   │   └── search/           # Local cosine + BM25 search
│   ├── evaluation/           # Evaluation metrics
│   ├── web/                  # Browser chat UI (HTML/CSS/JS)
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
| `LLM_PROVIDER` | `ollama` by default; `openai` remains optional | `ollama` |
| `LLM_BASE_URL` | OpenAI-compatible LLM endpoint | `http://localhost:11434/v1` |
| `LLM_API_KEY` | Ignored by Ollama; required by the compatibility client | `ollama` |
| `DATABASE_URL` | Local SQLite database connection | `sqlite+aiosqlite:///./data/policymind.db` |
| `CHAT_MODEL` | Local Ollama chat model | `qwen2.5:7b` |
| `EMBEDDING_MODEL` | Local Ollama embedding model | `nomic-embed-text` |
| `EMBEDDING_DIMENSION` | Dimension produced by the embedding model | `768` |
| `APP_ENV` | Environment (development/production) | `development` |
| `APP_DEBUG` | Enables SQLAlchemy debug logging | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `RATE_LIMIT` | Per-client request limit | `60/minute` |
| `RATE_LIMIT_STORAGE_URI` | Shared rate-limit backend URI; use Redis with multiple API replicas | `memory://` |
| `SOURCE_INGEST_ALLOWED_HOSTS` | Comma-separated HTTPS hosts allowed for URL ingestion | Disabled by default |

URL ingestion is intentionally disabled until `SOURCE_INGEST_ALLOWED_HOSTS` is configured. Text and Markdown file uploads remain available, with a 10 MB default size limit.

## 📊 Evaluation

PolicyMind includes a built-in evaluation framework:

```bash
# Run evaluation on all questions
curl -X POST "http://localhost:8000/api/v2/eval/run" \
  -H "Content-Type: application/json" \
  -d '{}'

# Run evaluation on specific query types
curl -X POST "http://localhost:8000/api/v2/eval/run" \
  -H "Content-Type: application/json" \
  -d '{"query_types": ["document_qa"]}'
```

Metrics tracked:
- **Accuracy**: Percentage of correct answers (>70% similarity)
- **Latency**: p50, p95, p99 response times
- **By Query Type**: Breakdown for document_qa, records_sql, hybrid

## 🏆 Key Features

1. **Intelligent Query Routing**
   - Automatic classification into document Q&A, SQL, or hybrid
   - Keyword-based fast path + LLM fallback

2. **Hybrid Search**
   - Local cosine similarity over SQLite JSON embeddings
   - BM25 for keyword matching
   - Reciprocal Rank Fusion (RRF) combining both

3. **Self-Correcting SQL Agent**
   - Natural language to SQL generation
   - Automatic retry on execution errors
   - Read-only query enforcement

4. **Deterministic Coverage Calculations**
   - Room rent limits by plan type
   - Percentage Share and fixed share calculations
   - Network/non-participating provider logic

5. **Built-in Evaluation**
   - Pre-defined test questions
   - Answer similarity scoring
   - Historical tracking of evaluation runs

## 📝 Example Queries

**Document Q&A:**
- "What is the family-support coverage limit?"
- "What are the permanent exclusions?"
- "What is the waiting period for pre-existing diseases?"

**Claims SQL:**
- "How many service_cases were rejected in 2024?"
- "What is the average claim amount by hospital?"
- "Show top 5 cities by total approved amount"

**Hybrid:**
- "What's our rejection rate for pre-existing conditions and what does the policy say about them?"
- "How many family-support service_cases were filed and what are the benefits?"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ using FastAPI, LangGraph, SQLite, and Ollama
