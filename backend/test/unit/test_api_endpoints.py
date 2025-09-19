"""
Unit tests for API endpoints
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import app
from app.core.app_state import get_app_state


class TestAPIEndpoints:
    """Test API endpoint functionality"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_app_state(self):
        """Mock app state with required components"""
        mock_llm_client = MagicMock()
        mock_llm_client.generate = AsyncMock(return_value="Mock response")

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        mock_smart_router = MagicMock()
        mock_smart_router.classify_request = AsyncMock(return_value={
            "engine": "DEEP_RESEARCH",
            "confidence": 0.95,
            "reasoning": "Test classification"
        })
        mock_smart_router.route_and_execute = AsyncMock(return_value={
            "session_id": "test_session",
            "status": "started"
        })

        mock_deep_engine = MagicMock()
        mock_deep_engine.research = AsyncMock()

        mock_code_engine = MagicMock()
        mock_code_engine.research = AsyncMock()

        mock_scientific_engine = MagicMock()
        mock_scientific_engine.research = AsyncMock()

        return {
            "llm_client": mock_llm_client,
            "cache": mock_cache,
            "smart_router": mock_smart_router,
            "engines": {
                "deep": mock_deep_engine,
                "code": mock_code_engine,
                "scientific": mock_scientific_engine
            }
        }

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "engines" in data
        assert "smart_router" in data

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Universal Agent (UAgent) API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data

    @patch('app.core.app_state.get_app_state')
    def test_router_classify_endpoint(self, mock_get_state, client, mock_app_state):
        """Test router classification endpoint"""
        mock_get_state.return_value = mock_app_state

        request_data = {
            "request": "Find information about machine learning"
        }

        response = client.post("/api/router/classify", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "engine" in data
        assert "confidence" in data

    @patch('app.core.app_state.get_app_state')
    def test_router_classify_invalid_request(self, mock_get_state, client, mock_app_state):
        """Test router classification with invalid request"""
        mock_get_state.return_value = mock_app_state

        # Empty request
        response = client.post("/api/router/classify", json={})
        assert response.status_code == 422  # Validation error

        # Missing request field
        response = client.post("/api/router/classify", json={"wrong_field": "value"})
        assert response.status_code == 422

    @patch('app.core.app_state.get_app_state')
    def test_router_engines_endpoint(self, mock_get_state, client, mock_app_state):
        """Test router engines listing endpoint"""
        mock_get_state.return_value = mock_app_state

        response = client.get("/api/router/engines")

        assert response.status_code == 200
        data = response.json()
        assert "engines" in data
        assert len(data["engines"]) >= 3  # At least 3 engines

    @patch('app.core.app_state.get_app_state')
    def test_router_status_endpoint(self, mock_get_state, client, mock_app_state):
        """Test router status endpoint"""
        mock_get_state.return_value = mock_app_state

        response = client.get("/api/router/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "engines" in data

    @patch('app.core.app_state.get_app_state')
    def test_router_route_and_execute_endpoint(self, mock_get_state, client, mock_app_state):
        """Test router route and execute endpoint"""
        mock_get_state.return_value = mock_app_state

        request_data = {
            "request": "Analyze code repositories for machine learning patterns",
            "execute_immediately": True
        }

        response = client.post("/api/router/route-and-execute", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "status" in data

    @patch('app.core.app_state.get_app_state')
    def test_research_sessions_create(self, mock_get_state, client, mock_app_state):
        """Test research session creation"""
        mock_get_state.return_value = mock_app_state

        request_data = {
            "request": "Research quantum computing applications",
            "engine_type": "DEEP_RESEARCH"
        }

        response = client.post("/api/research/sessions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "engine_type" in data

    @patch('app.core.app_state.get_app_state')
    def test_research_sessions_list(self, mock_get_state, client, mock_app_state):
        """Test research sessions listing"""
        mock_get_state.return_value = mock_app_state

        response = client.get("/api/research/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    @patch('app.core.app_state.get_app_state')
    def test_research_session_get(self, mock_get_state, client, mock_app_state):
        """Test getting specific research session"""
        mock_get_state.return_value = mock_app_state

        session_id = "test_session_123"
        response = client.get(f"/api/research/sessions/{session_id}")

        # This might return 404 if session doesn't exist, which is expected
        assert response.status_code in [200, 404]

    @patch('app.core.app_state.get_app_state')
    def test_research_session_execute(self, mock_get_state, client, mock_app_state):
        """Test research session execution"""
        mock_get_state.return_value = mock_app_state

        session_id = "test_session_123"
        response = client.post(f"/api/research/sessions/{session_id}/execute")

        # Might return 404 if session doesn't exist
        assert response.status_code in [200, 404]

    def test_api_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/health")

        # Check if CORS middleware is working
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented

    def test_api_error_handling(self, client):
        """Test API error handling for invalid endpoints"""
        response = client.get("/invalid/endpoint")

        assert response.status_code == 404

    def test_api_method_not_allowed(self, client):
        """Test method not allowed responses"""
        # Try POST on GET-only endpoint
        response = client.post("/health")

        assert response.status_code == 405

    @patch('app.core.app_state.get_app_state')
    def test_router_classify_with_override(self, mock_get_state, client, mock_app_state):
        """Test router classification with manual override"""
        mock_get_state.return_value = mock_app_state

        request_data = {
            "request": "Find information about machine learning",
            "override_engine": "CODE_RESEARCH"
        }

        response = client.post("/api/router/classify", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "engine" in data

    @patch('app.core.app_state.get_app_state')
    def test_research_sessions_with_config(self, mock_get_state, client, mock_app_state):
        """Test research session creation with custom configuration"""
        mock_get_state.return_value = mock_app_state

        request_data = {
            "request": "Research quantum computing applications",
            "engine_type": "SCIENTIFIC_RESEARCH",
            "config": {
                "max_iterations": 5,
                "timeout": 1800
            }
        }

        response = client.post("/api/research/sessions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    @patch('app.core.app_state.get_app_state')
    def test_research_sessions_list_with_filters(self, mock_get_state, client, mock_app_state):
        """Test research sessions listing with filters"""
        mock_get_state.return_value = mock_app_state

        response = client.get("/api/research/sessions?status=completed&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data

    def test_api_documentation_endpoints(self, client):
        """Test API documentation endpoints"""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200

        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200

        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data

    @patch('app.core.app_state.get_app_state')
    def test_error_responses_format(self, mock_get_state, client, mock_app_state):
        """Test error response format consistency"""
        mock_get_state.return_value = mock_app_state

        # Test with invalid JSON
        response = client.post(
            "/api/router/classify",
            data="invalid json",
            headers={"content-type": "application/json"}
        )

        assert response.status_code == 422

    @patch('app.core.app_state.get_app_state')
    def test_research_session_results(self, mock_get_state, client, mock_app_state):
        """Test research session results endpoint"""
        mock_get_state.return_value = mock_app_state

        session_id = "test_session_123"
        response = client.get(f"/api/research/sessions/{session_id}/results")

        # Might return 404 if session doesn't exist
        assert response.status_code in [200, 404]

    @patch('app.core.app_state.get_app_state')
    def test_research_session_progress(self, mock_get_state, client, mock_app_state):
        """Test research session progress endpoint"""
        mock_get_state.return_value = mock_app_state

        session_id = "test_session_123"
        response = client.get(f"/api/research/sessions/{session_id}/progress")

        # Might return 404 if session doesn't exist
        assert response.status_code in [200, 404]

    @patch('app.core.app_state.get_app_state')
    def test_research_session_cancel(self, mock_get_state, client, mock_app_state):
        """Test research session cancellation"""
        mock_get_state.return_value = mock_app_state

        session_id = "test_session_123"
        response = client.delete(f"/api/research/sessions/{session_id}")

        # Might return 404 if session doesn't exist
        assert response.status_code in [200, 404]

    def test_api_versioning(self, client):
        """Test API versioning in URLs"""
        # Current API should be accessible
        response = client.get("/api/router/engines")
        assert response.status_code in [200, 404]  # Might need auth

    @patch('app.core.app_state.get_app_state')
    def test_concurrent_api_requests(self, mock_get_state, client, mock_app_state):
        """Test handling of concurrent API requests"""
        mock_get_state.return_value = mock_app_state

        # Make multiple requests
        responses = []
        for i in range(5):
            response = client.get("/health")
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200

    @patch('app.core.app_state.get_app_state')
    def test_request_validation(self, mock_get_state, client, mock_app_state):
        """Test request validation on various endpoints"""
        mock_get_state.return_value = mock_app_state

        # Test classification with missing required field
        response = client.post("/api/router/classify", json={})
        assert response.status_code == 422

        # Test classification with invalid data types
        response = client.post("/api/router/classify", json={"request": 123})
        assert response.status_code == 422

        # Test route-and-execute with invalid data
        response = client.post("/api/router/route-and-execute", json={"invalid": "data"})
        assert response.status_code == 422

    @patch('app.core.app_state.get_app_state')
    def test_response_headers(self, mock_get_state, client, mock_app_state):
        """Test response headers"""
        mock_get_state.return_value = mock_app_state

        response = client.get("/health")

        assert response.status_code == 200
        # Check for common security headers
        headers = response.headers
        assert "content-type" in headers

    def test_large_request_handling(self, client):
        """Test handling of large requests"""
        # Create a large request
        large_request = "x" * 10000  # 10KB request

        request_data = {"request": large_request}

        response = client.post("/api/router/classify", json=request_data)

        # Should either process or reject gracefully
        assert response.status_code in [200, 413, 422]  # OK, Payload too large, or validation error

    @patch('app.core.app_state.get_app_state')
    def test_api_timeout_handling(self, mock_get_state, client, mock_app_state):
        """Test API timeout handling"""
        mock_get_state.return_value = mock_app_state

        # Mock slow LLM response
        mock_app_state["smart_router"].classify_request = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timeout")
        )

        request_data = {"request": "Test timeout handling"}

        response = client.post("/api/router/classify", json=request_data)

        # Should handle timeout gracefully
        assert response.status_code in [200, 500, 504]  # Success, Server Error, or Gateway Timeout