"""
Application configuration using Pydantic Settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
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
    chat_model: str = "qwen2.5:3b"
    embedding_dimension: int = Field(default=768, ge=1)
    # Keep local CPU inference responsive. Override with LLM_MAX_TOKENS when
    # running a larger/faster model or when longer answers are required.
    llm_max_tokens: int = Field(default=220, ge=64, le=4000)

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Search
    top_k_retrieval: int = 3
    similarity_threshold: float = 0.7

    # Rate limiting
    rate_limit: str = "60/minute"
    rate_limit_storage_uri: str = "memory://"

    # Ingestion
    source_ingest_allowed_hosts: str = ""
    max_ingest_file_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024)
    source_fetch_timeout_seconds: float = Field(default=10.0, gt=0, le=60)

    # CORS
    cors_origins: str = "*"  # Comma-separated list of origins, or * for all

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
