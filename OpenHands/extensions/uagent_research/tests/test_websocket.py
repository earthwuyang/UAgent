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

    # Test sending update
    await manager.send_experiment_update("exp_1", {"type": "test", "data": "hello"})

    # Verify message received
    assert len(ws1.messages) == 1
    assert ws1.messages[0]["type"] == "test"
    assert ws1.messages[0]["data"] == "hello"
    assert "timestamp" in ws1.messages[0]

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

    # Send update to exp_1
    await manager.send_experiment_update("exp_1", {"type": "progress", "percentage": 50})

    # Verify only exp_1 clients received update
    assert len(ws1.messages) == 1
    assert len(ws2.messages) == 1
    assert len(ws3.messages) == 0

    # Verify message content
    assert ws1.messages[0]["type"] == "progress"
    assert ws2.messages[0]["percentage"] == 50

    print("✅ Multiple connections test passed")


@pytest.mark.asyncio
async def test_session_connections():
    """Test session-level WebSocket connections"""
    manager = ConnectionManager()

    class MockWebSocket:
        def __init__(self, id):
            self.id = id
            self.messages = []

        async def accept(self):
            pass

        async def send_json(self, data):
            self.messages.append(data)

    # Connect to session
    ws1 = MockWebSocket("ws1")
    await manager.connect(ws1, session_id="session_1")

    # Send session update
    await manager.send_session_update("session_1", {
        "type": "experiment_started",
        "experiment_id": "exp_1"
    })

    # Verify message received
    assert len(ws1.messages) == 1
    assert ws1.messages[0]["type"] == "experiment_started"

    print("✅ Session connections test passed")


if __name__ == "__main__":
    asyncio.run(test_connection_manager())
    asyncio.run(test_multiple_connections())
    asyncio.run(test_session_connections())

    print("\n✅ All WebSocket tests passed!")
