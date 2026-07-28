"""
Integration tests for API endpoints.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock


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
        from app.main import app
        
        with patch('app.api.routes.health.get_db_session', return_value=mock_db):
            mock_db.execute.return_value.scalar.return_value = 1
            
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as client:
                response = await client.get("/health")
                
                assert response.status_code == 200
                data = response.json()
                assert "status" in data
                assert "version" in data
    
    @pytest.mark.asyncio
    async def test_liveness_probe(self):
        """Test Kubernetes liveness probe."""
        from app.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/health/live")
            
            assert response.status_code == 200
            assert response.json()["alive"] is True


class TestAskEndpoints:
    """Tests for ask/query endpoints."""
    
    @pytest.mark.asyncio
    async def test_ask_endpoint_validation(self):
        """Test request validation on ask endpoint."""
        from app.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            # Empty query should fail
            response = await client.post(
                "/api/v1/ask",
                json={"query": ""}
            )
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_ask_endpoint_short_query(self):
        """Test that short queries are rejected."""
        from app.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/ask",
                json={"query": "hi"}  # Too short (min 3 chars)
            )
            assert response.status_code == 422


class TestIngestEndpoints:
    """Tests for ingestion endpoints."""
    
    @pytest.mark.asyncio
    async def test_ingest_requires_content(self):
        """Test that ingest requires document content."""
        from app.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/ingest",
                json={"policy_id": 1}  # Missing document_text and document_url
            )
            # Should fail validation or return 400
            assert response.status_code in [400, 422]


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    @pytest.mark.asyncio
    async def test_root_returns_app_info(self):
        """Test root endpoint returns app information."""
        from app.main import app
        
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/")
            
            assert response.status_code == 200
            data = response.json()
            assert "app" in data
            assert "version" in data
            assert "status" in data
            assert data["status"] == "running"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
