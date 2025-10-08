"""
Unit tests for MessageBus
"""

import asyncio
import pytest
from openhands.server.session.message_bus import MessageBus
from openhands.events.agent_event import (
    AgentMessage,
    AgentSpawnedEvent,
    RequestMessage,
    ResponseMessage,
    NotificationMessage,
    CommandEvent,
)


@pytest.mark.asyncio
async def test_register_agent():
    """Test agent registration."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", ["orchestration"])
    
    assert "agent_a" in bus._agents
    assert bus._agents["agent_a"].agent_type == "coordinator"
    assert "orchestration" in bus._agents["agent_a"].capabilities
    
    await bus.close()


@pytest.mark.asyncio
async def test_unregister_agent():
    """Test agent unregistration."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", [])
    bus.unregister_agent("agent_a")
    
    assert "agent_a" not in bus._agents
    
    await bus.close()


@pytest.mark.asyncio
async def test_duplicate_registration():
    """Test handling duplicate agent registration."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", [])
    bus.register_agent("agent_a", "research", ["tree_search"])  # Duplicate
    
    # Should update the registration
    assert bus._agents["agent_a"].agent_type == "research"
    
    await bus.close()


@pytest.mark.asyncio
async def test_send_message():
    """Test sending direct message from agent A to agent B."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", [])
    bus.register_agent("agent_b", "research", [])
    
    # Create subscriber task for agent_b
    messages_received = []
    
    async def subscriber():
        async for msg in bus.subscribe("agent_b"):
            messages_received.append(msg)
            break
    
    subscriber_task = asyncio.create_task(subscriber())
    
    # Send message
    await bus.send_message(
        from_agent_id="agent_a",
        to_agent_id="agent_b",
        message=NotificationMessage(
            from_agent_id="agent_a",
            to_agent_id="agent_b",
            content="Hello agent_b"
        )
    )
    
    # Wait for message to be received
    await asyncio.wait_for(subscriber_task, timeout=2.0)
    
    assert len(messages_received) == 1
    assert messages_received[0].content == "Hello agent_b"
    
    await bus.close()


@pytest.mark.asyncio
async def test_broadcast_message():
    """Test broadcasting message to all agents."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", [])
    bus.register_agent("agent_b", "research", [])
    bus.register_agent("agent_c", "code", [])
    
    messages_b = []
    messages_c = []
    
    async def subscriber_b():
        async for msg in bus.subscribe("agent_b"):
            messages_b.append(msg)
            break
    
    async def subscriber_c():
        async for msg in bus.subscribe("agent_c"):
            messages_c.append(msg)
            break
    
    task_b = asyncio.create_task(subscriber_b())
    task_c = asyncio.create_task(subscriber_c())
    
    # Broadcast message
    await bus.send_message(
        from_agent_id="agent_a",
        to_agent_id=None,  # Broadcast
        message=NotificationMessage(
            from_agent_id="agent_a",
            content="Broadcast to all"
        )
    )
    
    await asyncio.wait_for(asyncio.gather(task_b, task_c), timeout=2.0)
    
    assert len(messages_b) == 1
    assert len(messages_c) == 1
    assert messages_b[0].content == "Broadcast to all"
    assert messages_c[0].content == "Broadcast to all"
    
    await bus.close()


@pytest.mark.asyncio
async def test_message_to_nonexistent_agent():
    """Test sending message to non-existent agent raises error."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", [])
    
    with pytest.raises(ValueError, match="not registered"):
        await bus.send_message(
            from_agent_id="agent_a",
            to_agent_id="nonexistent",
            message=NotificationMessage(
                from_agent_id="agent_a",
                content="Test"
            )
        )
    
    await bus.close()


@pytest.mark.asyncio
async def test_request_response_success():
    """Test request/response RPC pattern."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", [])
    bus.register_agent("agent_b", "research", [])
    
    # Agent B responds to requests
    async def responder():
        async for msg in bus.subscribe("agent_b"):
            if isinstance(msg, RequestMessage):
                await bus.send_response(
                    to_agent_id=msg.from_agent_id,
                    correlation_id=msg.correlation_id,
                    status="success",
                    result={"status": "ok"}
                )
                break
    
    responder_task = asyncio.create_task(responder())
    
    # Agent A sends request
    response = await bus.request(
        from_agent_id="agent_a",
        to_agent_id="agent_b",
        message=RequestMessage(
            from_agent_id="agent_a",
            to_agent_id="agent_b",
            request_type="status",
            payload={}
        ),
        timeout=5.0
    )
    
    assert response.status == "success"
    assert response.result["status"] == "ok"
    
    await responder_task
    await bus.close()


@pytest.mark.asyncio
async def test_request_timeout():
    """Test request timeout when no response received."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", [])
    bus.register_agent("agent_b", "research", [])
    
    # Agent B does not respond
    
    with pytest.raises(asyncio.TimeoutError):
        await bus.request(
            from_agent_id="agent_a",
            to_agent_id="agent_b",
            message=RequestMessage(
                from_agent_id="agent_a",
                to_agent_id="agent_b",
                request_type="status",
                payload={}
            ),
            timeout=1.0  # Short timeout
        )
    
    await bus.close()


@pytest.mark.asyncio
async def test_route_by_capability():
    """Test routing message to agent with specific capability."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", ["orchestration"])
    bus.register_agent("agent_b", "research", ["tree_search", "code_execution"])
    bus.register_agent("agent_c", "code", ["code_execution"])
    
    # Route to agent with code_execution capability
    agent_id = bus.route_message("code_execution")
    
    # Should route to first matching agent (agent_b)
    assert agent_id == "agent_b"
    
    await bus.close()


@pytest.mark.asyncio
async def test_route_no_matching_agent():
    """Test routing when no agent has required capability."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", ["orchestration"])
    
    agent_id = bus.route_message("nonexistent_capability")
    
    assert agent_id is None
    
    await bus.close()


@pytest.mark.asyncio
async def test_close_message_bus():
    """Test MessageBus cleanup."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", [])
    bus.register_agent("agent_b", "research", [])
    
    await bus.close()
    
    assert bus._closed
    assert len(bus._agents) == 0
    assert len(bus._message_queues) == 0
    assert len(bus._pending_requests) == 0


@pytest.mark.asyncio
async def test_close_with_pending_requests():
    """Test closing MessageBus with pending requests cancels them."""
    bus = MessageBus()
    
    bus.register_agent("agent_a", "coordinator", [])
    bus.register_agent("agent_b", "research", [])
    
    # Start a request that won't be answered
    request_task = asyncio.create_task(
        bus.request(
            from_agent_id="agent_a",
            to_agent_id="agent_b",
            message=RequestMessage(
                from_agent_id="agent_a",
                to_agent_id="agent_b",
                request_type="status"
            ),
            timeout=10.0
        )
    )
    
    # Give request time to start
    await asyncio.sleep(0.1)
    
    # Close bus
    await bus.close()
    
    # Request should be cancelled
    with pytest.raises(asyncio.TimeoutError):
        await request_task
