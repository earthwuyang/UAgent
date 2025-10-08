"""
Unit Tests for EventBus Event Storage

Tests event log functionality in EventBus including versioning, retrieval, and cleanup.
"""

import pytest
import asyncio
from collections import deque
from datetime import datetime

# Mock imports for testing
try:
    from extensions.uagent_research.orchestrator.event_bus import EventBus, get_event_bus
    from extensions.uagent_research.uagent_research.models.events import StepEvent, EventType
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pytest.skip("EventBus not available", allow_module_level=True)


@pytest.fixture
def event_bus():
    """Create EventBus instance for testing."""
    return EventBus()


@pytest.fixture
def sample_events():
    """Generate list of test events."""
    events = []
    for i in range(10):
        events.append(StepEvent(
            branch_id=f"test-exp-branch-{i % 3}",
            node_id=f"node-{i}",
            action=f"Step {i}",
            reasoning="Test reasoning"
        ))
    return events


@pytest.mark.asyncio
async def test_event_storage_on_publish(event_bus, sample_events):
    """Verify events are stored when published."""
    experiment_id = "test-exp"
    
    # Publish events
    for event in sample_events[:5]:
        await event_bus.publish(event)
    
    # Check storage
    result = event_bus.get_events(experiment_id, since_version=0, limit=100)
    assert len(result["events"]) == 5
    assert result["current_version"] == 5


@pytest.mark.asyncio
async def test_event_versioning(event_bus):
    """Verify version numbers increment correctly."""
    experiment_id = "test-exp"
    
    for i in range(5):
        event = StepEvent(
            branch_id=f"{experiment_id}-branch",
            node_id=f"node-{i}",
            action=f"Step {i}",
            reasoning="Test"
        )
        await event_bus.publish(event)
    
    result = event_bus.get_events(experiment_id, since_version=0, limit=100)
    versions = [e["version"] for e in result["events"]]
    assert versions == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_get_events_since_version(event_bus):
    """Verify filtering by version works."""
    experiment_id = "test-exp"
    
    # Publish 10 events
    for i in range(10):
        event = StepEvent(
            branch_id=f"{experiment_id}-branch",
            node_id=f"node-{i}",
            action=f"Step {i}",
            reasoning="Test"
        )
        await event_bus.publish(event)
    
    # Get events since version 5
    result = event_bus.get_events(experiment_id, since_version=5, limit=100)
    
    assert result["current_version"] == 10
    assert len(result["events"]) == 5  # versions 6-10
    assert result["events"][0]["version"] == 6
    assert result["events"][-1]["version"] == 10
    assert result["has_more"] == False


@pytest.mark.asyncio
async def test_get_events_with_limit(event_bus):
    """Verify limit parameter works."""
    experiment_id = "test-exp"
    
    # Publish 20 events
    for i in range(20):
        event = StepEvent(
            branch_id=f"{experiment_id}-branch",
            node_id=f"node-{i}",
            action=f"Step {i}",
            reasoning="Test"
        )
        await event_bus.publish(event)
    
    # Get with limit
    result = event_bus.get_events(experiment_id, since_version=0, limit=10)
    
    assert len(result["events"]) == 10
    assert result["has_more"] == True


@pytest.mark.asyncio
async def test_clear_event_log(event_bus):
    """Verify clearing event log."""
    experiment_id = "test-exp"
    
    # Publish events
    for i in range(5):
        event = StepEvent(
            branch_id=f"{experiment_id}-branch",
            node_id=f"node-{i}",
            action=f"Step {i}",
            reasoning="Test"
        )
        await event_bus.publish(event)
    
    # Verify events exist
    result = event_bus.get_events(experiment_id, since_version=0, limit=100)
    assert len(result["events"]) == 5
    
    # Clear log
    event_bus.clear_event_log(experiment_id)
    
    # Verify cleared
    result = event_bus.get_events(experiment_id, since_version=0, limit=100)
    assert len(result["events"]) == 0


@pytest.mark.asyncio
async def test_get_events_nonexistent_experiment(event_bus):
    """Verify returns empty for unknown experiment."""
    result = event_bus.get_events("nonexistent", since_version=0, limit=100)
    
    assert result["events"] == []
    assert result["current_version"] == 0
    assert result["has_more"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
