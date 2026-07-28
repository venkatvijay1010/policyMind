"""
Pytest configuration and fixtures.
"""
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest_asyncio


# Configure pytest-asyncio
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def mock_db_session() -> AsyncGenerator[AsyncMock, None]:
    """
    Mock database session for unit tests.
    Provides a mock that simulates SQLAlchemy async session behavior.
    """
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()
    
    # Mock scalar result
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1
    mock_result.scalars.return_value.all.return_value = []
    mock_result.fetchall.return_value = []
    mock_result.fetchone.return_value = None
    mock_session.execute.return_value = mock_result
    
    yield mock_session


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client for unit tests."""
    mock_client = MagicMock()
    
    # Mock chat completions
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage.total_tokens = 100
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Mock embeddings
    mock_embed_response = MagicMock()
    mock_embed_response.data = [MagicMock()]
    mock_embed_response.data[0].embedding = [0.1] * 1536
    mock_embed_response.data[0].index = 0
    
    mock_client.embeddings.create = AsyncMock(return_value=mock_embed_response)
    
    return mock_client


@pytest.fixture
def mock_settings() -> MagicMock:
    """Mock application settings."""
    settings = MagicMock()
    settings.openai_api_key = "test-key"
    settings.chat_model = "gpt-4-turbo-preview"
    settings.embedding_model = "text-embedding-3-small"
    settings.embedding_dimension = 1536
    settings.database_url = "postgresql+asyncpg://test:test@localhost/test"
    settings.app_env = "test"
    settings.debug = True
    settings.top_k_retrieval = 5
    settings.similarity_threshold = 0.7
    return settings


@pytest.fixture(autouse=True)
def mock_env_settings(mock_settings: MagicMock):
    """Auto-use fixture to mock settings for all tests."""
    with patch("app.config.settings", mock_settings):
        yield


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
