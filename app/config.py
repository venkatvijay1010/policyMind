"""
Application configuration using Pydantic Settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        # .env.local is intentionally loaded after .env so a local-first setup
        # can override legacy OpenAI/PostgreSQL values without exposing or
        # overwriting a developer's private .env file.
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "PolicyMind"
    app_version: str = "1.0.0"
    app_env: str = "development"
    log_level: str = "INFO"
    # Do not read the generic DEBUG variable. It is commonly set by unrelated tools.
    debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("APP_DEBUG", "POLICY_MIND_DEBUG"),
    )

    # Database. SQLite is deliberately the default for a no-Docker local setup.
    database_url: str = "sqlite+aiosqlite:///./data/policymind.db"
    database_sync_url: str = "sqlite:///./data/policymind.db"

    # LLM provider. Ollama exposes an OpenAI-compatible local API, which lets the
    # application keep one small client abstraction while remaining offline-first.
    llm_provider: Literal["ollama", "openai"] = "ollama"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = Field(
        default="ollama",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "qwen2.5:7b"
    embedding_dimension: int = Field(default=768, ge=1)
    # Keep local CPU inference responsive. Override with LLM_MAX_TOKENS when
    # running a larger/faster model or when longer answers are required.
    llm_max_tokens: int = Field(default=220, ge=64, le=4000)
    # Prime the local model and prompt caches at startup so the first message is not cold.
    warm_up_on_startup: bool = True
    # USD per 1k tokens; 0 for local models. Set for hosted models to track spend.
    llm_cost_per_1k_input: float = Field(default=0.0, ge=0)
    llm_cost_per_1k_output: float = Field(default=0.0, ge=0)
    # /metrics is unauthenticated; restrict it at the gateway or disable it when exposed.
    metrics_enabled: bool = True

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Search
    top_k_retrieval: int = 3
    similarity_threshold: float = 0.7
    rerank_candidate_multiplier: int = Field(default=3, ge=1, le=10)
    reranker: Literal["local", "cross_encoder"] = "local"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Below this many candidates the model cost outweighs any reordering gain.
    cross_encoder_min_candidates: int = Field(default=10, ge=2, le=100)
    max_context_chars: int = Field(default=6000, ge=500, le=50000)
    # Passages added on each side of a match as prompt context; 0 keeps matched passages only.
    context_neighbor_passages: int = Field(default=0, ge=0, le=3)
    # Runs hybrid document and SQL branches concurrently, each on its own DB session.
    hybrid_parallel: bool = False
    # Splits a hybrid question into document and SQL parts; costs one extra LLM call per query.
    hybrid_subquery_split: bool = True
    # In-process caches that skip repeated embedding and routing calls for identical input.
    embedding_cache_size: int = Field(default=512, ge=0, le=10000)
    routing_cache_size: int = Field(default=256, ge=0, le=10000)
    routing_cache_ttl_seconds: float = Field(default=300.0, gt=0, le=86400)
    # One extra LLM call per document answer; a single revision, never blocks the answer.
    critic_enabled: bool = False
    # Fusion weights favour embeddings for paraphrases; raise BM25 for clause numbers and exact terms.
    hybrid_vector_weight: float = Field(default=0.7, ge=0, le=1)
    hybrid_bm25_weight: float = Field(default=0.3, ge=0, le=1)
    rrf_k: int = Field(default=60, ge=1, le=200)
    # Adds one LLM call per search; results from both phrasings are merged.
    query_rewrite_enabled: bool = False

    # Rate limiting
    query_timeout_seconds: float = Field(default=180.0, gt=0, le=900)
    routing_timeout_seconds: float = Field(default=60.0, gt=0, le=120)
    routing_output_mode: Literal["json_object", "json_schema"] = "json_object"
    node_timeout_seconds: float = Field(default=90.0, gt=0, le=600)
    graph_max_retries: int = Field(default=1, ge=0, le=3)
    sql_timeout_seconds: float = Field(default=10.0, gt=0, le=60)

    rate_limit: str = "60/minute"
    rate_limit_storage_uri: str = "memory://"

    # Ingestion
    source_ingest_allowed_hosts: str = ""
    max_ingest_file_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024)
    source_fetch_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    pdf_max_pages: int = Field(default=200, ge=1, le=2000)
    pdf_render_dpi: int = Field(default=200, ge=72, le=400)
    pdf_max_render_pixels: int = Field(default=20_000_000, ge=1_000_000, le=100_000_000)
    pdf_ocr_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    pdf_extraction_timeout_seconds: float = Field(default=180.0, gt=0, le=900)
    pdf_max_extracted_chars: int = Field(default=2_000_000, ge=1000, le=20_000_000)
    pdf_min_native_chars_per_page: int = Field(default=40, ge=0, le=10000)
    pdf_ocr_review_confidence: float = Field(default=60.0, ge=0, le=100)

    # Source document storage. "database" keeps originals in the database only; "s3" also
    # writes each original to a private bucket. AWS credentials come from the standard
    # credential chain (environment, shared profile or IAM role), never from these settings.
    storage_backend: Literal["database", "s3"] = "database"
    s3_bucket: str = ""
    s3_region: str = ""
    s3_prefix: str = "source-documents"
    s3_endpoint_url: str = ""

    # Vector index. "database" ranks the JSON embeddings stored with each passage; "qdrant"
    # searches a Qdrant collection (the database stays the source of truth for passages).
    # QDRANT_URL is a server URL such as http://localhost:6333, or ":memory:" for tests.
    vector_backend: Literal["database", "qdrant"] = "database"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "policymind_passages"
    # HNSW m/ef_construct apply only when the collection is created; ef trades recall for latency.
    qdrant_hnsw_m: int = Field(default=16, ge=4, le=64)
    qdrant_hnsw_ef_construct: int = Field(default=128, ge=16, le=512)
    qdrant_hnsw_ef: int = Field(default=64, ge=8, le=512)

    # Keyword index. "memory" runs rank_bm25 in-process per query; "elasticsearch" searches an
    # Elasticsearch index (the database stays the source of truth for passages).
    bm25_backend: Literal["memory", "elasticsearch"] = "memory"
    elasticsearch_url: str = ""
    elasticsearch_api_key: str = ""
    elasticsearch_index: str = "policymind_passages"

    # CORS
    cors_origins: str = "*"  # Comma-separated list of origins, or * for all

    @model_validator(mode="after")
    def require_bucket_for_s3(self) -> "Settings":
        if self.storage_backend == "s3" and not self.s3_bucket.strip():
            raise ValueError("S3_BUCKET is required when STORAGE_BACKEND=s3")
        if self.vector_backend == "qdrant" and not self.qdrant_url.strip():
            raise ValueError("QDRANT_URL is required when VECTOR_BACKEND=qdrant")
        if self.bm25_backend == "elasticsearch" and not self.elasticsearch_url.strip():
            raise ValueError("ELASTICSEARCH_URL is required when BM25_BACKEND=elasticsearch")
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def environment(self) -> str:
        """Backward-compatible read-only name for the application environment."""
        return self.app_env

    @property
    def is_ollama(self) -> bool:
        """Whether requests are routed to a locally running Ollama server."""
        return self.llm_provider == "ollama"

    @property
    def is_llm_configured(self) -> bool:
        """Return whether the selected provider has enough connection settings."""
        if self.is_ollama:
            return bool(self.llm_base_url)
        return bool(self.llm_api_key)

    @property
    def source_ingest_allowed_hosts_list(self) -> set[str]:
        """Return normalized hostnames explicitly allowed for URL ingestion."""
        return {
            host.strip().lower().rstrip(".")
            for host in self.source_ingest_allowed_hosts.split(",")
            if host.strip()
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
