"""
Comprehensive Integration Tests for Heartbeat System

Tests the complete heartbeat flow from EventBus through WebSocket delivery,
including configuration, multiple branches, and orchestrator integration.
"""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add extension to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from uagent_research.orchestrator.event_bus import EventBus
from uagent_research.orchestrator.ws_publisher import WebSocketPublisher
from uagent_research.api.websocket_routes import ConnectionManager
from uagent_research.uagent_research.models.events import StepEvent, CompleteEvent


class MockWebSocket:
    """Mock WebSocket for testing"""
    def __init__(self, ws_id="test"):
        self.id = ws_id
        self.messages = []
        self.closed = False
    
    async def accept(self):
        pass
    
    async def send_json(self, data):
        if not self.closed:
            self.messages.append(data)
    
    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_heartbeat_configuration():
    """Test EventBus with different heartbeat configurations"""
    
    # Test 1: Disabled heartbeats (interval=0)
    event_bus_disabled = EventBus(heartbeat_interval=0)
    ws = MockWebSocket("test-disabled")
    
    event = StepEvent(branch_id="test-branch", action="test", reasoning="test")
    await event_bus_disabled.publish(event)
    
    # Wait longer than any possible heartbeat
    await asyncio.sleep(2)
    
    # Should not have any heartbeat supervisors
    assert len(event_bus_disabled._heartbeat_tasks) == 0, "No heartbeat tasks should exist when disabled"
    
    await event_bus_disabled.close()
    
    # Test 2: Short interval (2 seconds)
    event_bus_short = EventBus(heartbeat_interval=2)
    manager = ConnectionManager()
    ws = MockWebSocket("test-short")
    await manager.connect(ws, experiment_id="test_exp")
    
    publisher = WebSocketPublisher(event_bus_short, manager)
    await publisher.start()
    
    try:
        event = StepEvent(branch_id="test-branch-2", action="test", reasoning="test")
        await event_bus_short.publish(event)
        
        # Wait for ~2 seconds (one heartbeat cycle)
        await asyncio.sleep(2.5)
        
        heartbeats = [msg for msg in ws.messages if msg.get("action") == "heartbeat"]
        assert len(heartbeats) >= 1, "Should receive heartbeat with 2s interval"
        
    finally:
        await publisher.stop()
        await event_bus_short.close()
    
    # Test 3: Longer interval (5 seconds) - default
    event_bus_long = EventBus(heartbeat_interval=5)
    manager2 = ConnectionManager()
    ws2 = MockWebSocket("test-long")
    await manager2.connect(ws2, experiment_id="test_exp2")
    
    publisher2 = WebSocketPublisher(event_bus_long, manager2)
    await publisher2.start()
    
    try:
        event = StepEvent(branch_id="test-branch-3", action="test", reasoning="test")
        await event_bus_long.publish(event)
        
        # Wait less than interval
        await asyncio.sleep(3)
        
        heartbeats = [msg for msg in ws2.messages if msg.get("action") == "heartbeat"]
        # Should not have received heartbeat yet
        assert len(heartbeats) == 0, "Should not receive heartbeat before interval expires"
        
        # Wait for interval to expire
        await asyncio.sleep(3)
        
        heartbeats = [msg for msg in ws2.messages if msg.get("action") == "heartbeat"]
        assert len(heartbeats) >= 1, "Should receive heartbeat after 5s interval"
        
    finally:
        await publisher2.stop()
        await event_bus_long.close()
    
    print("✅ Heartbeat configuration test passed")


@pytest.mark.asyncio
async def test_heartbeat_multiple_branches():
    """Test independent heartbeat supervisors for multiple branches"""
    event_bus = EventBus(heartbeat_interval=1)
    manager = ConnectionManager()
    ws = MockWebSocket("test-multi")
    await manager.connect(ws, experiment_id="test_exp")
    
    publisher = WebSocketPublisher(event_bus, manager)
    await publisher.start()
    
    try:
        # Start three branches
        branch_ids = ["branch-1", "branch-2", "branch-3"]
        for branch_id in branch_ids:
            event = StepEvent(branch_id=branch_id, action=f"start {branch_id}", reasoning="test")
            await event_bus.publish(event)
        
        # Wait for heartbeats to start
        await asyncio.sleep(0.5)
        
        # Verify all branches have supervisors
        assert len(event_bus._heartbeat_tasks) == 3, "Should have 3 heartbeat supervisors"
        assert all(bid in event_bus._active_branches for bid in branch_ids), "All branches should be active"
        
        # Stop one branch
        event_bus.stop_branch_heartbeat("branch-1")
        await asyncio.sleep(0.2)
        
        # Verify only 2 supervisors remain
        assert len(event_bus._heartbeat_tasks) == 2, "Should have 2 heartbeat supervisors after stopping one"
        assert "branch-1" not in event_bus._heartbeat_tasks, "branch-1 should be stopped"
        assert "branch-2" in event_bus._heartbeat_tasks, "branch-2 should still be active"
        assert "branch-3" in event_bus._heartbeat_tasks, "branch-3 should still be active"
        
        # Wait for heartbeats
        await asyncio.sleep(1.5)
        
        # Count heartbeats per branch
        branch1_heartbeats = [msg for msg in ws.messages if msg.get("branch_id") == "branch-1" and msg.get("action") == "heartbeat"]
        branch2_heartbeats = [msg for msg in ws.messages if msg.get("branch_id") == "branch-2" and msg.get("action") == "heartbeat"]
        branch3_heartbeats = [msg for msg in ws.messages if msg.get("branch_id") == "branch-3" and msg.get("action") == "heartbeat"]
        
        # branch-1 should have no heartbeats (or very few if stopped late)
        assert len(branch1_heartbeats) <= 1, "branch-1 should have minimal heartbeats after stopping"
        # branch-2 and branch-3 should have heartbeats
        assert len(branch2_heartbeats) >= 1, "branch-2 should have heartbeats"
        assert len(branch3_heartbeats) >= 1, "branch-3 should have heartbeats"
        
    finally:
        await publisher.stop()
        await event_bus.close()
    
    print("✅ Multiple branches heartbeat test passed")


@pytest.mark.asyncio
async def test_heartbeat_websocket_delivery():
    """Test end-to-end heartbeat delivery through WebSocket"""
    event_bus = EventBus(heartbeat_interval=1)
    manager = ConnectionManager()
    ws = MockWebSocket("test-delivery")
    await manager.connect(ws, experiment_id="test_exp")
    
    publisher = WebSocketPublisher(event_bus, manager)
    await publisher.start()
    
    try:
        # Publish event to start heartbeat
        event = StepEvent(
            branch_id="delivery-test",
            action="test_action",
            reasoning="Testing delivery"
        )
        await event_bus.publish(event)
        
        # Wait for heartbeat
        await asyncio.sleep(1.5)
        
        # Find heartbeat message
        heartbeat_msgs = [msg for msg in ws.messages if msg.get("action") == "heartbeat"]
        assert len(heartbeat_msgs) >= 1, "Should receive heartbeat message"
        
        # Verify heartbeat message structure
        heartbeat = heartbeat_msgs[0]
        assert heartbeat.get("branch_id") == "delivery-test", "Should have correct branch_id"
        assert heartbeat.get("action") == "heartbeat", "Should have action='heartbeat'"
        assert "timestamp" in heartbeat or "reasoning" in heartbeat, "Should have timestamp or reasoning"
        
        # Verify event_type matches StepEvent structure
        # (The message format should match the event format used in WebSocketPublisher)
        
    finally:
        await publisher.stop()
        await event_bus.close()
    
    print("✅ WebSocket delivery test passed")


@pytest.mark.asyncio
async def test_heartbeat_backpressure():
    """Test that heartbeats don't cause queue overflow under backpressure"""
    # Create EventBus with small buffer
    event_bus = EventBus(max_buffer_size=5, heartbeat_interval=1)
    
    # Create slow subscriber that doesn't consume events
    received_events = []
    
    async def slow_subscriber():
        async for event in event_bus.subscribe("slow-sub"):
            received_events.append(event)
            # Don't consume - simulate slow client
            await asyncio.sleep(10)  # Very slow
    
    # Start slow subscriber in background
    subscriber_task = asyncio.create_task(slow_subscriber())
    
    try:
        # Publish many events rapidly to trigger backpressure
        for i in range(20):
            event = StepEvent(
                branch_id="backpressure-test",
                action=f"event_{i}",
                reasoning=f"Event {i}"
            )
            await event_bus.publish(event)
            await asyncio.sleep(0.1)
        
        # Wait for heartbeat
        await asyncio.sleep(2)
        
        # Check stats
        stats = event_bus.get_stats()
        
        # Some events should be dropped due to backpressure
        assert stats["events_dropped"] > 0, "Some events should be dropped under backpressure"
        
        # Heartbeats should still be sent
        assert stats["heartbeats_sent"] >= 1, "Heartbeats should still be sent"
        
        # System should not crash
        assert event_bus is not None
        
    finally:
        subscriber_task.cancel()
        await event_bus.close()
    
    print("✅ Backpressure test passed")


@pytest.mark.asyncio
async def test_heartbeat_stats():
    """Test heartbeat statistics tracking"""
    event_bus = EventBus(heartbeat_interval=1)
    
    # Publish event to start heartbeat
    event = StepEvent(branch_id="stats-test", action="test", reasoning="test")
    await event_bus.publish(event)
    
    # Wait for heartbeat
    await asyncio.sleep(1.5)
    
    # Get stats
    stats = event_bus.get_stats()
    
    # Verify heartbeat stats
    assert "heartbeats_sent" in stats, "Should have heartbeats_sent stat"
    assert "active_heartbeats" in stats, "Should have active_heartbeats stat"
    assert "heartbeat_branches" in stats, "Should have heartbeat_branches stat"
    
    assert stats["heartbeats_sent"] >= 1, "Should have sent heartbeats"
    assert stats["active_heartbeats"] == 1, "Should have 1 active heartbeat"
    assert "stats-test" in stats["heartbeat_branches"], "Should list stats-test branch"
    
    # Stop heartbeat
    event_bus.stop_branch_heartbeat("stats-test")
    await asyncio.sleep(0.2)
    
    # Get updated stats
    stats = event_bus.get_stats()
    assert stats["active_heartbeats"] == 0, "Should have 0 active heartbeats after stopping"
    
    await event_bus.close()
    
    print("✅ Heartbeat stats test passed")


if __name__ == "__main__":
    print("Running comprehensive heartbeat integration tests...\n")
    
    asyncio.run(test_heartbeat_configuration())
    asyncio.run(test_heartbeat_multiple_branches())
    asyncio.run(test_heartbeat_websocket_delivery())
    asyncio.run(test_heartbeat_backpressure())
    asyncio.run(test_heartbeat_stats())
    
    print("\n✅ All heartbeat integration tests passed!")
