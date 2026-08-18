"""
Integration tests for API endpoints.
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_db():
    """Mock database session."""
    mock = AsyncMock()
    mock.execute = AsyncMock()
    mock.commit = AsyncMock()
    return mock


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, mock_db):
        """Test basic health check endpoint."""
        from app.infrastructure.database.postgres import get_db_session
        from app.main import app

        async def override_get_db_session():
            yield mock_db

        app.dependency_overrides[get_db_session] = override_get_db_session
        try:
            mock_db.execute.return_value.scalar.return_value = 1

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")

                assert response.status_code == 200
                data = response.json()
                assert "status" in data
                assert "version" in data
        finally:
            app.dependency_overrides.pop(get_db_session, None)

    @pytest.mark.asyncio
    async def test_liveness_probe(self):
        """Test Kubernetes liveness probe."""
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/live")

            assert response.status_code == 200
            assert response.json()["alive"] is True

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excessive_requests(self):
        """Verify an endpoint limit is actually enforced."""
        from app.main import app

        transport = ASGITransport(app=app, client=("198.51.100.10", 12345))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            statuses = [(await client.get("/")).status_code for _ in range(61)]

        assert statuses[-1] == 429


class TestAskEndpoints:
    """Tests for ask/query endpoints."""

    @pytest.mark.asyncio
    async def test_ask_endpoint_validation(self):
        """Test request validation on ask endpoint."""
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Empty query should fail
            response = await client.post("/api/v2/insights/query", json={"prompt": ""})
            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_ask_endpoint_accepts_short_greeting(self, mock_db):
        """Short messages can be routed as normal chat rather than rejected."""
        from app.domain.entities.models import QueryResult, QueryType
        from app.infrastructure.database.postgres import get_db_session
        from app.main import app

        async def override_get_db_session():
            yield mock_db

        app.dependency_overrides[get_db_session] = override_get_db_session
        try:
            with (
                patch("app.api.routes.ask.PolicyMindGraph") as mock_graph_class,
                patch("app.api.routes.ask.log_query_background", new_callable=AsyncMock),
            ):
                mock_graph_class.return_value.invoke = AsyncMock(
                    return_value=QueryResult(
                        query="hi",
                        query_type=QueryType.CHAT,
                        answer="Hello from the local model.",
                        latency_ms=12,
                        model_used="qwen2.5:3b",
                    )
                )
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/api/v2/insights/query",
                        json={"prompt": "hi"},
                    )

            assert response.status_code == 200
            assert response.json()["query_type"] == "chat"
        finally:
            app.dependency_overrides.pop(get_db_session, None)


class TestIngestEndpoints:
    """Tests for ingestion endpoints."""

    @pytest.mark.asyncio
    async def test_ingest_requires_content(self):
        """Test that ingest requires document content."""
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put(
                "/api/v2/knowledge/scopes/1/source",
                json={},  # Missing source_text and source_uri
            )
            # Should fail validation or return 400
            assert response.status_code in [400, 422]


class TestChatEndpoint:
    """Tests for the browser chat entrypoint and API discovery endpoint."""

    @pytest.mark.asyncio
    async def test_root_serves_chat_application(self):
        """The root URL should serve the browser chat client, not Swagger JSON."""
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/")

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            assert "PolicyMind" in response.text

    @pytest.mark.asyncio
    async def test_api_status_returns_app_info(self):
        """Programmatic clients can still discover API metadata at /api."""
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api")

            assert response.status_code == 200
            data = response.json()
            assert "app" in data
            assert "version" in data
            assert "status" in data
            assert data["status"] == "running"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
