"""
Integration Tests for Research API Endpoints

Real tests using FastAPI TestClient with dependency overrides for mocked components.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import FastAPI app and dependencies
try:
    from extensions.uagent_research.uagent_research.api.research_routes import router
    from extensions.uagent_research.uagent_research.models.base import Base, get_session
    from extensions.uagent_research.uagent_research.models import Experiment, ExperimentStatus
    from extensions.uagent_research.services.research_session_manager import ResearchSessionManager
    from extensions.uagent_research.orchestrator.event_bus import EventBus
    from extensions.uagent_research.control.control_bus import ControlMessage
    from fastapi import FastAPI
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"Required imports not available: {e}", allow_module_level=True)


# Create test database engine
@pytest.fixture
def test_db_engine():
    """Create in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_db_session(test_db_engine):
    """Create test database session."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_experiment(test_db_session):
    """Create a sample experiment in the database."""
    experiment = Experiment(
        id="exp_test_123",
        goal="Test research goal",
        status=ExperimentStatus.RUNNING,
        created_at=datetime.utcnow(),
        session_id="session_test_456"
    )
    test_db_session.add(experiment)
    test_db_session.commit()
    test_db_session.refresh(experiment)
    return experiment


@pytest.fixture
def mock_session_manager():
    """Create mock ResearchSessionManager."""
    mock = Mock(spec=ResearchSessionManager)
    mock.get_status = Mock(return_value={
        "experiment_id": "exp_test_123",
        "status": "running",
        "stats": {
            "total_nodes": 10,
            "completed": 5,
            "failed": 1,
            "running": 4,
            "pending": 0,
            "total_cost": 0.25,
            "total_tokens": 5000
        },
        "adapters": {
            "codeact": {
                "status": "running",
                "current_step": "Testing implementation",
                "last_event": "2025-01-06T10:30:00",
                "cost": 0.10,
                "tokens": 1200
            }
        },
        "active_branches": [],
        "created_at": "2025-01-06T10:00:00",
        "last_update": "2025-01-06T10:30:00"
    })
    mock.send_control = AsyncMock()
    return mock


@pytest.fixture
def mock_event_bus():
    """Create mock EventBus."""
    mock = Mock(spec=EventBus)
    mock.get_events = Mock(return_value={
        "events": [
            {
                "version": 1,
                "timestamp": "2025-01-06T10:25:00",
                "type": "STEP",
                "branch_id": "exp_test_123-idea-0",
                "data": {"action": "Analyzing code", "reasoning": "Test reasoning"}
            },
            {
                "version": 2,
                "timestamp": "2025-01-06T10:26:00",
                "type": "OBSERVATION",
                "branch_id": "exp_test_123-idea-0",
                "data": {"observation": "Found implementation"}
            }
        ],
        "current_version": 2,
        "has_more": False
    })
    return mock


@pytest.fixture
def test_app(test_db_engine, mock_session_manager, mock_event_bus):
    """Create FastAPI test app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)
    
    # Override database session
    def override_get_session():
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    app.dependency_overrides[get_session] = override_get_session
    
    return app


@pytest.fixture
def test_client(test_app, mock_session_manager, mock_event_bus):
    """Create TestClient with dependency mocks."""
    # Patch global functions to return mocks
    with patch('extensions.uagent_research.uagent_research.api.research_routes.get_session_manager', return_value=mock_session_manager):
        with patch('extensions.uagent_research.uagent_research.api.research_routes.get_event_bus', return_value=mock_event_bus):
            client = TestClient(test_app)
            yield client


class TestStatusEndpoint:
    """Tests for GET /api/research/experiments/{id}/status endpoint."""
    
    def test_get_status_with_session_manager(self, test_client, sample_experiment, mock_session_manager):
        """Verify returns detailed status from session manager."""
        response = test_client.get(f"/api/research/experiments/{sample_experiment.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["experiment_id"] == sample_experiment.id
        assert data["status"] == "running"
        assert data["stats"]["total_nodes"] == 10
        assert data["stats"]["completed"] == 5
        assert "codeact" in data["adapters"]
        
        # Verify session manager was called
        mock_session_manager.get_status.assert_called_once_with(sample_experiment.id)
    
    def test_get_status_fallback_to_db(self, test_client, sample_experiment, mock_session_manager):
        """Verify fallback when experiment not in session manager."""
        # Configure mock to raise KeyError (not registered)
        mock_session_manager.get_status.side_effect = KeyError("Not registered")
        
        response = test_client.get(f"/api/research/experiments/{sample_experiment.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return fallback status from database
        assert data["experiment_id"] == sample_experiment.id
        assert "message" in data  # Fallback includes message
        assert "not available" in data["message"].lower()
    
    def test_get_status_experiment_not_found(self, test_client):
        """Verify 404 for non-existent experiment."""
        response = test_client.get("/api/research/experiments/nonexistent_exp/status")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestEventsEndpoint:
    """Tests for GET /api/research/experiments/{id}/events endpoint."""
    
    def test_get_events_success(self, test_client, sample_experiment, mock_event_bus):
        """Verify returns events from EventBus."""
        response = test_client.get(f"/api/research/experiments/{sample_experiment.id}/events")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["experiment_id"] == sample_experiment.id
        assert data["current_version"] == 2
        assert len(data["events"]) == 2
        assert data["events"][0]["version"] == 1
        assert data["events"][0]["type"] == "STEP"
        assert data["has_more"] == False
        
        # Verify EventBus was called with correct params
        mock_event_bus.get_events.assert_called_once_with(sample_experiment.id, 0, 100)
    
    def test_get_events_with_pagination(self, test_client, sample_experiment, mock_event_bus):
        """Verify since_version and limit work."""
        # Configure mock for pagination
        mock_event_bus.get_events.return_value = {
            "events": [{"version": 11, "timestamp": "2025-01-06T10:30:00", "type": "STEP", "branch_id": "test", "data": {}}],
            "current_version": 25,
            "has_more": True
        }
        
        response = test_client.get(
            f"/api/research/experiments/{sample_experiment.id}/events",
            params={"since_version": 10, "limit": 50}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["since_version"] == 10
        assert data["current_version"] == 25
        assert len(data["events"]) == 1
        assert data["events"][0]["version"] == 11
        
        # Verify correct params passed to EventBus
        mock_event_bus.get_events.assert_called_once_with(sample_experiment.id, 10, 50)
    
    def test_get_events_has_more_flag(self, test_client, sample_experiment, mock_event_bus):
        """Verify has_more is true when more events available."""
        mock_event_bus.get_events.return_value = {
            "events": [{"version": i, "timestamp": "2025-01-06T10:30:00", "type": "STEP", "branch_id": "test", "data": {}} for i in range(1, 11)],
            "current_version": 20,
            "has_more": True
        }
        
        response = test_client.get(
            f"/api/research/experiments/{sample_experiment.id}/events",
            params={"limit": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["has_more"] == True
        assert len(data["events"]) == 10
    
    def test_get_events_experiment_not_found(self, test_client):
        """Verify 404 for non-existent experiment."""
        response = test_client.get("/api/research/experiments/nonexistent_exp/events")
        
        assert response.status_code == 404


class TestControlEndpoint:
    """Tests for PATCH /api/research/experiments/{id} endpoint."""
    
    def test_control_pause(self, test_client, sample_experiment, mock_session_manager):
        """Verify pause command sent to ControlBus."""
        response = test_client.patch(
            f"/api/research/experiments/{sample_experiment.id}",
            json={"action": "pause"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "acknowledged"
        assert data["action"] == "pause"
        assert data["experiment_id"] == sample_experiment.id
        
        # Verify send_control was called with correct ControlMessage
        mock_session_manager.send_control.assert_called_once()
        call_args = mock_session_manager.send_control.call_args
        assert call_args[0][0] == sample_experiment.id  # experiment_id
        control_msg = call_args[0][1]  # ControlMessage
        assert control_msg.action == "pause"
    
    def test_control_cancel_node(self, test_client, sample_experiment, mock_session_manager):
        """Verify cancel_node with valid target."""
        response = test_client.patch(
            f"/api/research/experiments/{sample_experiment.id}",
            json={"action": "cancel_node", "target": {"node_id": "idea-2"}}
        )
        
        assert response.status_code == 200
        
        # Verify correct control message
        mock_session_manager.send_control.assert_called_once()
        call_args = mock_session_manager.send_control.call_args
        control_msg = call_args[0][1]
        assert control_msg.action == "cancel_node"
        assert control_msg.target["node_id"] == "idea-2"
    
    def test_control_cancel_node_missing_target(self, test_client, sample_experiment):
        """Verify 400 when node_id missing."""
        response = test_client.patch(
            f"/api/research/experiments/{sample_experiment.id}",
            json={"action": "cancel_node", "target": {}}
        )
        
        assert response.status_code == 400
        assert "node_id" in response.json()["detail"].lower()
    
    def test_control_reprioritize(self, test_client, sample_experiment, mock_session_manager):
        """Verify reprioritize with valid payload."""
        response = test_client.patch(
            f"/api/research/experiments/{sample_experiment.id}",
            json={
                "action": "reprioritize",
                "target": {"adapter": "codeact"},
                "payload": {"delta": 0.2}
            }
        )
        
        assert response.status_code == 200
        
        # Verify correct control message
        call_args = mock_session_manager.send_control.call_args
        control_msg = call_args[0][1]
        assert control_msg.action == "reprioritize"
        assert control_msg.payload["delta"] == 0.2
    
    def test_control_reprioritize_invalid_delta(self, test_client, sample_experiment):
        """Verify 400 when delta invalid."""
        response = test_client.patch(
            f"/api/research/experiments/{sample_experiment.id}",
            json={
                "action": "reprioritize",
                "target": {"adapter": "codeact"},
                "payload": {"delta": "invalid"}
            }
        )
        
        assert response.status_code == 400
        assert "delta" in response.json()["detail"].lower()
    
    def test_control_steer(self, test_client, sample_experiment, mock_session_manager):
        """Verify steer with valid target and text."""
        response = test_client.patch(
            f"/api/research/experiments/{sample_experiment.id}",
            json={
                "action": "steer",
                "target": {"adapter": "codeact"},
                "payload": {"text": "Focus on performance"}
            }
        )
        
        assert response.status_code == 200
        
        # Verify correct control message
        call_args = mock_session_manager.send_control.call_args
        control_msg = call_args[0][1]
        assert control_msg.action == "steer"
        assert control_msg.payload["text"] == "Focus on performance"
    
    def test_control_steer_missing_text(self, test_client, sample_experiment):
        """Verify 400 when text missing."""
        response = test_client.patch(
            f"/api/research/experiments/{sample_experiment.id}",
            json={
                "action": "steer",
                "target": {"adapter": "codeact"},
                "payload": {}
            }
        )
        
        assert response.status_code == 400
        assert "text" in response.json()["detail"].lower()
    
    def test_control_add_node(self, test_client, sample_experiment, mock_session_manager):
        """Verify add_node with valid payload."""
        response = test_client.patch(
            f"/api/research/experiments/{sample_experiment.id}",
            json={
                "action": "add_node",
                "payload": {
                    "parent_id": "root",
                    "node": {
                        "type": "IDEA",
                        "title": "Test DuckDB",
                        "content": "Explore DuckDB vector extension"
                    }
                }
            }
        )
        
        assert response.status_code == 200
    
    def test_control_invalid_action(self, test_client, sample_experiment):
        """Verify 400 for unknown action."""
        response = test_client.patch(
            f"/api/research/experiments/{sample_experiment.id}",
            json={"action": "invalid_action"}
        )
        
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()
    
    def test_control_experiment_not_found(self, test_client):
        """Verify 404 for non-existent experiment."""
        response = test_client.patch(
            "/api/research/experiments/nonexistent_exp",
            json={"action": "pause"}
        )
        
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
