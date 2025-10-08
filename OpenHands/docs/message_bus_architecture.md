# MessageBus Architecture

## Overview

MessageBus provides higher-level inter-agent communication patterns beyond the existing EventBus and ControlBus.

**Comparison:**
- **EventBus** (in `extensions/uagent_research/orchestrator/event_bus.py`): One-way event streaming (orchestrator → subscribers)
- **ControlBus** (in `extensions/uagent_research/control/control_bus.py`): One-way command routing (UI/agent → orchestrator)
- **MessageBus** (in `openhands/server/session/message_bus.py`): Bidirectional agent-to-agent communication with request/response, routing, and bridging

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  MessageBus (Coordinator)                │
│  - Request/Response RPC                                 │
│  - Direct agent-to-agent messaging                      │
│  - Capability-based routing                             │
│  - EventBus/ControlBus bridging                         │
└────────────────┬────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    ↓            ↓            ↓
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Main     │ │ Research │ │ Code     │
│ Agent    │ │ Sub-Agent│ │ Sub-Agent│
│          │ │          │ │          │
│ (Session)│ │(Orchestr)│ │(CodeAct) │
└──────────┘ └──────────┘ └──────────┘
```

## Communication Patterns

### 1. Request/Response (RPC)

```python
# Main agent requests status from research sub-agent
response = await message_bus.request(
    from_agent_id="session_123",
    to_agent_id="research_abc",
    message=RequestMessage(
        request_type="get_status",
        payload={}
    ),
    timeout=10.0
)
```

### 2. Direct Messaging

```python
# Send notification to specific agent
await message_bus.send_message(
    from_agent_id="coordinator",
    to_agent_id="research_abc",
    message=NotificationMessage(
        content="Budget threshold reached"
    )
)
```

### 3. Broadcast

```python
# Broadcast to all agents
await message_bus.send_message(
    from_agent_id="coordinator",
    to_agent_id=None,  # Broadcast
    message=AgentSpawnedEvent(...)
)
```

### 4. Capability-based Routing

```python
# Register agent with capabilities
message_bus.register_agent(
    agent_id="research_abc",
    agent_type="research",
    capabilities=["tree_search", "code_execution"]
)

# Route message to agent with capability
target_agent = message_bus.route_message(
    required_capability="code_execution"
)
```

## Integration with Existing Buses

### EventBus Bridge

EventBus events are automatically converted to MessageBus messages:

```python
# EventBus: StepEvent → MessageBus: ProgressUpdateEvent
# EventBus: CompleteEvent → MessageBus: NodeCompleteEvent
```

### ControlBus Bridge

ControlBus commands are automatically converted to MessageBus messages:

```python
# ControlBus: ControlMessage(action="pause") → MessageBus: CommandEvent
```

## Usage Examples

### Coordinator spawning sub-agent

```python
coordinator = MultiAgentCoordinator(session_id="sess_123")
experiment_id = await coordinator.spawn_research_agent(
    goal="Research neural architecture search"
)

# MessageBus automatically emits AgentSpawnedEvent
```

### Sub-agent reporting progress

```python
# In TreeSearchOrchestrator
await self.message_bus.send_message(
    from_agent_id=self.tree.research_id,
    to_agent_id=None,  # Broadcast
    message=ProgressUpdateEvent(
        progress=0.45,
        current_task="Executing node-5",
        stats=self.stats
    )
)
```

### Main agent sending command

```python
# User: "pause research"
await coordinator.send_command(
    sub_agent_id="research_abc",
    command="pause"
)

# MessageBus routes CommandEvent to research sub-agent
```

## Message Types

### AgentMessage (Base Class)

Base class for all messages with common fields:
- `event_id`: Unique message identifier
- `from_agent_id`: Sender agent ID
- `to_agent_id`: Receiver agent ID (None for broadcast)
- `timestamp`: ISO timestamp
- `correlation_id`: For request/response correlation

### Specific Message Types

1. **AgentSpawnedEvent**: Emitted when sub-agent is spawned
2. **NodeCompleteEvent**: Emitted when research tree node completes
3. **CommandEvent**: Control commands (pause, resume, cancel, steer)
4. **RequestMessage**: RPC request expecting response
5. **ResponseMessage**: RPC response with status and result
6. **ProgressUpdateEvent**: Periodic progress updates
7. **NotificationMessage**: Fire-and-forget notifications

## Implementation Details

### MessageBus Class

```python
class MessageBus:
    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._message_queues: Dict[str, asyncio.Queue] = {}
        self._event_bus: Optional[EventBus] = None
        self._control_bus: Optional[ControlBus] = None
```

### Key Methods

- `register_agent(agent_id, agent_type, capabilities)`: Register agent
- `unregister_agent(agent_id)`: Unregister agent
- `send_message(from_agent_id, to_agent_id, message)`: Send message
- `request(from_agent_id, to_agent_id, message, timeout)`: RPC request
- `send_response(to_agent_id, correlation_id, status, result)`: RPC response
- `subscribe(agent_id)`: Subscribe to messages for agent
- `route_message(required_capability)`: Route by capability
- `bridge_event_bus(event_bus)`: Bridge EventBus
- `bridge_control_bus(control_bus)`: Bridge ControlBus

## Benefits

1. **Type Safety**: Pydantic models for all messages
2. **Request/Response**: Async RPC between agents
3. **Routing**: Intelligent message delivery based on capabilities
4. **Bridging**: Seamless integration with EventBus and ControlBus
5. **Scalability**: Foundation for complex multi-agent workflows
6. **Decoupling**: Agents don't need direct references to each other

## Future Enhancements

- Message persistence for replay and debugging
- Message priority queues for urgent commands
- Agent discovery service for dynamic registration
- Message encryption for sensitive data
- Distributed MessageBus across multiple servers
- Message filtering and subscription patterns (pub/sub)
- Dead letter queue for failed message delivery
- Metrics and monitoring (message throughput, latency, etc.)

## Testing

Comprehensive unit tests are available in `tests/unit/test_message_bus.py`:

```bash
# Run MessageBus tests
pytest tests/unit/test_message_bus.py -v

# Run specific test
pytest tests/unit/test_message_bus.py::test_request_response_success -v
```

## Integration Checklist

When integrating a new agent with MessageBus:

1. ✅ Register agent with MessageBus on initialization
2. ✅ Define agent capabilities
3. ✅ Subscribe to MessageBus for incoming messages
4. ✅ Handle RequestMessage and send ResponseMessage
5. ✅ Emit ProgressUpdateEvent periodically
6. ✅ Unregister agent on cleanup
7. ✅ Handle message bus closure gracefully

## Example: Full Integration

```python
class MyCustomAgent:
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        
        # Register with MessageBus
        self.message_bus.register_agent(
            agent_id=agent_id,
            agent_type="custom",
            capabilities=["custom_task"]
        )
    
    async def run(self):
        # Subscribe to messages
        async for message in self.message_bus.subscribe(self.agent_id):
            if isinstance(message, RequestMessage):
                await self._handle_request(message)
            elif isinstance(message, CommandEvent):
                await self._handle_command(message)
    
    async def _handle_request(self, request: RequestMessage):
        # Process request
        result = {"status": "completed"}
        
        # Send response
        await self.message_bus.send_response(
            to_agent_id=request.from_agent_id,
            correlation_id=request.correlation_id,
            status="success",
            result=result
        )
    
    async def cleanup(self):
        # Unregister from MessageBus
        self.message_bus.unregister_agent(self.agent_id)
```

## Troubleshooting

### Message not received

- Check agent is registered: `agent_id in message_bus._agents`
- Verify subscriber is running: Check for active `subscribe()` task
- Check message routing: Ensure `to_agent_id` is correct

### Request timeout

- Increase timeout parameter
- Check responder is handling RequestMessage
- Verify correlation_id matches in response

### Agent not found error

- Ensure agent registered before sending messages
- Check agent_id spelling
- Verify agent hasn't been unregistered

## Related Documentation

- [EventBus Documentation](../extensions/uagent_research/orchestrator/event_bus.py)
- [ControlBus Documentation](../extensions/uagent_research/control/control_bus.py)
- [MultiAgentCoordinator](../openhands/server/session/multi_agent_coordinator.py)
- [AgentEvent Types](../openhands/events/agent_event.py)
