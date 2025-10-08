"""
Test WebSocket functionality
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add extension to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from uagent_research.api.websocket_routes import ConnectionManager
from uagent_research.orchestrator.event_bus import EventBus
from uagent_research.orchestrator.ws_publisher import WebSocketPublisher
from uagent_research.uagent_research.models.events import StepEvent


@pytest.mark.asyncio
async def test_connection_manager():
    """Test connection manager basic functionality"""
    manager = ConnectionManager()

    # Mock WebSocket class
    class MockWebSocket:
        def __init__(self, id):
            self.id = id
            self.messages = []
            self.closed = False

        async def accept(self):
            pass

        async def send_json(self, data):
            if not self.closed:
                self.messages.append(data)

        async def close(self):
            self.closed = True

    # Test experiment connection
    ws1 = MockWebSocket("ws1")
    await manager.connect(ws1, experiment_id="exp_1")

    # Verify connection
    assert "exp_1" in manager.active_connections
    assert ws1 in manager.active_connections["exp_1"]

    # Test broadcasting - use broadcast method instead of send_experiment_update
    await manager.broadcast({"type": "test", "data": "hello"}, "exp_1")

    # Verify message received - check at index 1 (index 0 is connection confirmation)
    assert len(ws1.messages) >= 1
    # Find the test message
    test_msg = next((msg for msg in ws1.messages if msg.get("type") == "test"), None)
    assert test_msg is not None
    assert test_msg["data"] == "hello"

    # Test disconnect
    manager.disconnect(ws1, experiment_id="exp_1")
    assert "exp_1" not in manager.active_connections

    print("✅ Connection manager test passed")


@pytest.mark.asyncio
async def test_multiple_connections():
    """Test multiple WebSocket connections to same experiment"""
    manager = ConnectionManager()

    class MockWebSocket:
        def __init__(self, id):
            self.id = id
            self.messages = []

        async def accept(self):
            pass

        async def send_json(self, data):
            self.messages.append(data)

    # Connect multiple clients
    ws1 = MockWebSocket("ws1")
    ws2 = MockWebSocket("ws2")
    ws3 = MockWebSocket("ws3")

    await manager.connect(ws1, experiment_id="exp_1")
    await manager.connect(ws2, experiment_id="exp_1")
    await manager.connect(ws3, experiment_id="exp_2")

    # Send update to exp_1 using broadcast
    await manager.broadcast({"type": "progress", "percentage": 50}, "exp_1")

    # Verify only exp_1 clients received update (accounting for connection confirmation)
    assert len(ws1.messages) >= 1
    assert len(ws2.messages) >= 1
    # ws3 should only have connection confirmation for exp_2
    assert all(msg.get("type") != "progress" for msg in ws3.messages)

    # Find progress messages
    ws1_progress = next((msg for msg in ws1.messages if msg.get("type") == "progress"), None)
    ws2_progress = next((msg for msg in ws2.messages if msg.get("type") == "progress"), None)
    
    assert ws1_progress is not None
    assert ws2_progress is not None
    assert ws1_progress["percentage"] == 50
    assert ws2_progress["percentage"] == 50

    print("✅ Multiple connections test passed")


@pytest.mark.asyncio
async def test_heartbeat_delivery():
    """Test that heartbeats are delivered through WebSocket"""
    # Create EventBus with short heartbeat interval for testing
    event_bus = EventBus(heartbeat_interval=1)
    
    # Create ConnectionManager
    manager = ConnectionManager()
    
    # Mock WebSocket
    class MockWebSocket:
        def __init__(self):
            self.messages = []
            self.closed = False
            
        async def accept(self):
            pass
            
        async def send_json(self, data):
            if not self.closed:
                self.messages.append(data)
                
        async def close(self):
            self.closed = True
    
    # Connect WebSocket
    ws = MockWebSocket()
    await manager.connect(ws, experiment_id="test_exp")
    
    # Create and start WebSocketPublisher
    publisher = WebSocketPublisher(event_bus, manager)
    await publisher.start()
    
    try:
        # Publish a StepEvent to trigger heartbeat supervisor
        event = StepEvent(
            branch_id="test-branch",
            action="test_step",
            reasoning="Testing heartbeat"
        )
        await event_bus.publish(event)
        
        # Wait for heartbeat interval + buffer
        await asyncio.sleep(2.5)
        
        # Check for heartbeat events in messages
        heartbeat_msgs = [msg for msg in ws.messages if msg.get("action") == "heartbeat"]
        assert len(heartbeat_msgs) >= 1, "Should receive at least one heartbeat event"
        
        print(f"✅ Heartbeat delivery test passed ({len(heartbeat_msgs)} heartbeats received)")
        
    finally:
        # Cleanup
        await publisher.stop()
        await event_bus.close()


@pytest.mark.asyncio
async def test_heartbeat_stops_on_completion():
    """Test that heartbeats stop when branch completes"""
    # Create EventBus with short heartbeat interval
    event_bus = EventBus(heartbeat_interval=1)
    
    # Create ConnectionManager
    manager = ConnectionManager()
    
    # Mock WebSocket
    class MockWebSocket:
        def __init__(self):
            self.messages = []
            
        async def accept(self):
            pass
            
        async def send_json(self, data):
            self.messages.append(data)
    
    # Connect WebSocket
    ws = MockWebSocket()
    await manager.connect(ws, experiment_id="test_exp")
    
    # Create and start WebSocketPublisher
    publisher = WebSocketPublisher(event_bus, manager)
    await publisher.start()
    
    try:
        # Publish event to start heartbeat
        event = StepEvent(
            branch_id="test-branch",
            action="test_step",
            reasoning="Testing"
        )
        await event_bus.publish(event)
        
        # Wait for first heartbeat
        await asyncio.sleep(1.5)
        msg_count_before = len([msg for msg in ws.messages if msg.get("action") == "heartbeat"])
        
        # Stop heartbeat
        event_bus.stop_branch_heartbeat("test-branch")
        
        # Wait another heartbeat period
        await asyncio.sleep(2)
        msg_count_after = len([msg for msg in ws.messages if msg.get("action") == "heartbeat"])
        
        # Should not receive new heartbeats after stopping
        assert msg_count_after == msg_count_before, \
            f"Should not receive new heartbeats after stopping (before: {msg_count_before}, after: {msg_count_after})"
        
        print("✅ Heartbeat stop test passed")
        
    finally:
        # Cleanup
        await publisher.stop()
        await event_bus.close()


if __name__ == "__main__":
    asyncio.run(test_connection_manager())
    asyncio.run(test_multiple_connections())
    asyncio.run(test_heartbeat_delivery())
    asyncio.run(test_heartbeat_stops_on_completion())

    print("\n✅ All WebSocket tests passed!")
