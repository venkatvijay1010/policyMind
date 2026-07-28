"""
Application configuration using Pydantic Settings.
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # App
    app_name: str = "PolicyMind"
    app_version: str = "1.0.0"
    app_env: str = "development"
    environment: str = "development"  # Alias for app_env
    log_level: str = "INFO"
    debug: bool = False
    
    # Database
    database_url: str = "postgresql+asyncpg://policymind:policymind@localhost:5432/policymind"
    database_sync_url: str = "postgresql://policymind:policymind@localhost:5432/policymind"
    
    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4-turbo-preview"
    embedding_dimension: int = 1536
    
    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "policymind"
    
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Search
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7
    
    # Rate limiting
    max_tokens_per_request: int = 4000
    
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
