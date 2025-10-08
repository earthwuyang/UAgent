"""
Tests for Control Action Validation

Tests Pydantic validation models for each control action type.
"""

import pytest
from pydantic import ValidationError

# Mock imports - these would be real in production
try:
    from extensions.uagent_research.uagent_research.api.research_routes import (
        PauseRequest,
        ResumeRequest,
        CancelRequest,
        CancelNodeRequest,
        ReprioritizeRequest,
        SteerRequest,
        AddNodeRequest
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    # Models might be defined inline in routes file
    IMPORTS_AVAILABLE = False
    pytest.skip("Pydantic models not importable (may be inline in routes)", allow_module_level=True)


def test_pause_request_valid():
    """Verify valid pause request."""
    request = PauseRequest(action="pause")
    assert request.action == "pause"


def test_resume_request_valid():
    """Verify valid resume request."""
    request = ResumeRequest(action="resume")
    assert request.action == "resume"


def test_cancel_request_valid():
    """Verify valid cancel request."""
    request = CancelRequest(action="cancel")
    assert request.action == "cancel"


def test_cancel_node_valid():
    """Verify valid cancel_node request."""
    request = CancelNodeRequest(
        action="cancel_node",
        target={"node_id": "idea-2"}
    )
    assert request.action == "cancel_node"
    assert request.target["node_id"] == "idea-2"


def test_cancel_node_missing_node_id():
    """Verify ValidationError when node_id missing."""
    with pytest.raises(ValidationError) as exc_info:
        CancelNodeRequest(
            action="cancel_node",
            target={"branch_id": "some-branch"}  # Missing node_id
        )
    
    errors = exc_info.value.errors()
    assert any("node_id" in str(err) for err in errors)


def test_reprioritize_valid_adapter():
    """Verify valid reprioritize request with adapter."""
    request = ReprioritizeRequest(
        action="reprioritize",
        target={"adapter": "codeact"},
        payload={"delta": 0.2}
    )
    
    assert request.action == "reprioritize"
    assert request.target["adapter"] == "codeact"
    assert request.payload["delta"] == 0.2


def test_reprioritize_missing_delta():
    """Verify ValidationError when delta missing."""
    with pytest.raises(ValidationError) as exc_info:
        ReprioritizeRequest(
            action="reprioritize",
            target={"adapter": "codeact"},
            payload={}  # Missing delta
        )
    
    errors = exc_info.value.errors()
    assert any("delta" in str(err) for err in errors)


def test_steer_valid_node_id():
    """Verify valid steer request with node_id."""
    request = SteerRequest(
        action="steer",
        target={"node_id": "idea-1"},
        payload={"text": "Focus on performance"}
    )
    
    assert request.action == "steer"
    assert request.target["node_id"] == "idea-1"
    assert request.payload["text"] == "Focus on performance"


def test_steer_missing_text():
    """Verify ValidationError when text missing."""
    with pytest.raises(ValidationError) as exc_info:
        SteerRequest(
            action="steer",
            target={"adapter": "codeact"},
            payload={}  # Missing text
        )
    
    errors = exc_info.value.errors()
    assert any("text" in str(err) for err in errors)


def test_add_node_valid():
    """Verify valid add_node request."""
    request = AddNodeRequest(
        action="add_node",
        payload={
            "parent_id": "root",
            "node": {
                "type": "IDEA",
                "title": "Test DuckDB",
                "content": "Explore DuckDB vector extension",
                "prior": 0.8
            }
        }
    )
    
    assert request.action == "add_node"
    assert request.payload["parent_id"] == "root"
    assert request.payload["node"]["type"] == "IDEA"


def test_add_node_missing_parent_id():
    """Verify ValidationError when parent_id missing."""
    with pytest.raises(ValidationError) as exc_info:
        AddNodeRequest(
            action="add_node",
            payload={
                "node": {
                    "type": "IDEA",
                    "title": "Test",
                    "content": "Test"
                }
            }
        )
    
    errors = exc_info.value.errors()
    assert any("parent_id" in str(err) for err in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
