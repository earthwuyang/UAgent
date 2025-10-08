"""
MessageBus - Higher-level inter-agent communication patterns

Provides request/response, direct messaging, and routing beyond EventBus/ControlBus.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    queue: asyncio.Queue
    registered_at: datetime = field(default_factory=datetime.utcnow)


class MessageBus:
    """
    Higher-level inter-agent communication bus providing:
    - Request/response pattern (async RPC)
    - Direct agent-to-agent messaging
    - Message routing based on capabilities
    - Integration with EventBus and ControlBus
    """
    
    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}  # agent_id -> AgentInfo
        self._pending_requests: Dict[str, asyncio.Future] = {}  # correlation_id -> Future
        self._message_queues: Dict[str, asyncio.Queue] = {}  # agent_id -> Queue
        self._event_bus: Optional[Any] = None
        self._control_bus: Optional[Any] = None
        self._bridge_tasks: List[asyncio.Task] = []
        self._closed = False
        logger.info("MessageBus initialized")
    
    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str]
    ) -> None:
        """
        Register an agent with the MessageBus.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (coordinator, research, code, etc.)
            capabilities: List of capabilities (tree_search, code_execution, etc.)
        """
        if agent_id in self._agents:
            logger.warning(f"Agent {agent_id} already registered, updating info")
        
        queue = asyncio.Queue()
        self._agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities,
            queue=queue
        )
        self._message_queues[agent_id] = queue
        logger.info(f"Registered agent {agent_id} (type={agent_type}, capabilities={capabilities})")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the MessageBus.
        
        Args:
            agent_id: Agent identifier to unregister
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            del self._message_queues[agent_id]
            logger.info(f"Unregistered agent {agent_id}")
        else:
            logger.warning(f"Attempted to unregister unknown agent {agent_id}")
    
    async def send_message(
        self,
        from_agent_id: str,
        to_agent_id: Optional[str],
        message: 'AgentMessage'
    ) -> None:
        """
        Send a message from one agent to another (or broadcast).
        
        Args:
            from_agent_id: Sender agent ID
            to_agent_id: Receiver agent ID (None for broadcast)
            message: The message to send
        """
        if self._closed:
            logger.warning("MessageBus is closed, message not sent")
            return
        
        message.from_agent_id = from_agent_id
        message.to_agent_id = to_agent_id
        
        if to_agent_id is None:
            # Broadcast to all agents
            for agent_id, queue in self._message_queues.items():
                if agent_id != from_agent_id:
                    await queue.put(message)
            logger.debug(f"Broadcast message from {from_agent_id} to all agents")
        else:
            # Direct message
            if to_agent_id not in self._message_queues:
                logger.error(f"Cannot send message to unknown agent {to_agent_id}")
                raise ValueError(f"Agent {to_agent_id} not registered")
            
            await self._message_queues[to_agent_id].put(message)
            logger.debug(f"Sent message from {from_agent_id} to {to_agent_id}")
    
    async def request(
        self,
        from_agent_id: str,
        to_agent_id: str,
        message: 'RequestMessage',
        timeout: float = 30.0
    ) -> 'ResponseMessage':
        """
        Send a request and wait for response (RPC pattern).
        
        Args:
            from_agent_id: Requester agent ID
            to_agent_id: Target agent ID
            message: Request message
            timeout: Timeout in seconds
            
        Returns:
            ResponseMessage from target agent
            
        Raises:
            asyncio.TimeoutError: If request times out
            ValueError: If target agent not registered
        """
        if self._closed:
            raise RuntimeError("MessageBus is closed")
        
        correlation_id = message.correlation_id or uuid.uuid4().hex
        message.correlation_id = correlation_id
        
        # Create future for response
        future = asyncio.Future()
        self._pending_requests[correlation_id] = future
        
        try:
            # Send request
            await self.send_message(from_agent_id, to_agent_id, message)
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        
        except asyncio.TimeoutError:
            logger.error(f"Request {correlation_id} from {from_agent_id} to {to_agent_id} timed out")
            raise
        
        finally:
            # Cleanup pending request
            self._pending_requests.pop(correlation_id, None)
    
    async def send_response(
        self,
        to_agent_id: str,
        correlation_id: str,
        status: str = "success",
        result: Optional[Any] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Send a response to a previous request.
        
        Args:
            to_agent_id: Original requester agent ID
            correlation_id: Correlation ID from original request
            status: Response status (success/error)
            result: Response result data
            error_message: Error message if status is error
        """
        if correlation_id not in self._pending_requests:
            logger.warning(f"No pending request for correlation_id {correlation_id}")
            return
        
        from openhands.events.agent_event import ResponseMessage
        
        response = ResponseMessage(
            from_agent_id="",  # Will be set by send_message
            to_agent_id=to_agent_id,
            correlation_id=correlation_id,
            status=status,
            result=result,
            error_message=error_message
        )
        
        # Resolve the future
        future = self._pending_requests.get(correlation_id)
        if future and not future.done():
            future.set_result(response)
    
    async def subscribe(self, agent_id: str) -> AsyncIterator['AgentMessage']:
        """
        Subscribe to messages for a specific agent.
        
        Args:
            agent_id: Agent ID to subscribe for
            
        Yields:
            Messages sent to this agent
        """
        if agent_id not in self._message_queues:
            logger.error(f"Cannot subscribe for unknown agent {agent_id}")
            raise ValueError(f"Agent {agent_id} not registered")
        
        queue = self._message_queues[agent_id]
        
        while not self._closed:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # Handle ResponseMessage
                from openhands.events.agent_event import ResponseMessage
                if isinstance(message, ResponseMessage) and message.correlation_id:
                    # Deliver response to pending request
                    future = self._pending_requests.get(message.correlation_id)
                    if future and not future.done():
                        future.set_result(message)
                    continue
                
                yield message
            
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    

    
    async def publish(self, event: 'AgentMessage') -> None:
        """
        Publish an event to all subscribed handlers by event type.
        
        Args:
            event: The event to publish
        """
        # Broadcast to all agents (similar to send_message with to_agent_id=None)
        for agent_id, queue in self._message_queues.items():
            await queue.put(event)
        
        logger.debug(f"Published {type(event).__name__} to all agents")
    
    def subscribe_event(
        self,
        event_type: type['AgentMessage'],
        handler: Callable[['AgentMessage'], None]
    ) -> None:
        """
        Register a handler for a specific event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Callback function to handle events
            
        Note: This creates a background task that filters messages by type.
        """
        if not hasattr(self, '_event_handlers'):
            self._event_handlers: Dict[type, List[Callable]] = {}
        
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.__name__}")
    def route_message(self, required_capability: str) -> Optional[str]:
        """
        Route message to an agent with required capability.
        
        Args:
            required_capability: Capability needed (e.g., "code_execution")
            
        Returns:
            Agent ID with matching capability, or None if not found
        """
        for agent_id, agent_info in self._agents.items():
            if required_capability in agent_info.capabilities:
                logger.debug(f"Routed to agent {agent_id} for capability {required_capability}")
                return agent_id
        
        logger.warning(f"No agent found with capability {required_capability}")
        return None
    
    def bridge_event_bus(self, event_bus: Any) -> None:
        """
        Bridge EventBus events to MessageBus.
        
        Args:
            event_bus: EventBus instance to bridge
        """
        self._event_bus = event_bus
        
        # Start background task to bridge events
        task = asyncio.create_task(self._event_bus_bridge_loop())
        self._bridge_tasks.append(task)
        logger.info("EventBus bridge established")
    
    def bridge_control_bus(self, control_bus: Any) -> None:
        """
        Bridge ControlBus commands to MessageBus.
        
        Args:
            control_bus: ControlBus instance to bridge
        """
        self._control_bus = control_bus
        
        # Start background task to bridge commands
        task = asyncio.create_task(self._control_bus_bridge_loop())
        self._bridge_tasks.append(task)
        logger.info("ControlBus bridge established")
    
    async def _event_bus_bridge_loop(self) -> None:
        """Background task bridging EventBus to MessageBus."""
        if not self._event_bus:
            return
        
        try:
            logger.info("EventBus bridge loop started")
            
            # Subscribe to all events from EventBus
            async for event in self._event_bus.subscribe(
                subscriber_id="messagebus_bridge",
                event_types=None,  # All event types
                branch_ids=None    # All branches
            ):
                if self._closed:
                    break
                
                # Import here to avoid circular dependency
                from openhands.events.agent_event import ProgressUpdateEvent, NodeCompleteEvent
                
                # Map EventBus events to MessageBus messages
                try:
                    if hasattr(event, 'type'):
                        # Map CompleteEvent to NodeCompleteEvent
                        if event.type.value == "complete":
                            message = NodeCompleteEvent(
                                from_agent_id=getattr(event, 'experiment_id', 'unknown'),
                                node_id=getattr(event, 'branch_id', 'unknown'),
                                branch_id=getattr(event, 'branch_id', 'unknown'),
                                experiment_id=getattr(event, 'experiment_id', 'unknown'),
                                result=getattr(event, 'result', {}),
                                cost=getattr(event, 'cost', 0.0),
                                artifacts=getattr(event, 'artifacts', None)
                            )
                            # Broadcast to all agents
                            for queue in self._message_queues.values():
                                await queue.put(message)
                        
                        # Map Step/Progress events to ProgressUpdateEvent
                        elif event.type.value in ("step", "progress"):
                            message = ProgressUpdateEvent(
                                from_agent_id=getattr(event, 'experiment_id', 'unknown'),
                                progress=getattr(event, 'progress', 0.0),
                                current_task=getattr(event, 'step', 'In progress'),
                                stats=getattr(event, 'stats', {})
                            )
                            # Broadcast to all agents
                            for queue in self._message_queues.values():
                                await queue.put(message)
                
                except Exception as e:
                    logger.debug(f"Failed to bridge EventBus event: {e}")
        
        except asyncio.CancelledError:
            logger.info("EventBus bridge loop cancelled")
        except Exception as e:
            logger.error(f"EventBus bridge error: {e}", exc_info=True)
    
    async def _control_bus_bridge_loop(self) -> None:
        """Background task bridging ControlBus to MessageBus."""
        if not self._control_bus:
            return
        
        try:
            logger.info("ControlBus bridge loop started")
            
            # Subscribe to all control messages (using wildcard or iterate through agents)
            # Note: ControlBus requires experiment_id, so we need to subscribe for each agent
            # For now, we'll create subscriptions dynamically as agents are registered
            
            # Keep track of active subscriptions
            active_subscriptions = {}
            
            while not self._closed:
                # Check for new agents to subscribe to
                for agent_id in list(self._agents.keys()):
                    if agent_id not in active_subscriptions and agent_id != "coordinator":
                        # Create subscription task for this agent
                        task = asyncio.create_task(
                            self._subscribe_control_bus_for_agent(agent_id)
                        )
                        active_subscriptions[agent_id] = task
                
                # Clean up completed subscriptions
                for agent_id in list(active_subscriptions.keys()):
                    if agent_id not in self._agents or active_subscriptions[agent_id].done():
                        if agent_id in active_subscriptions:
                            active_subscriptions[agent_id].cancel()
                            del active_subscriptions[agent_id]
                
                await asyncio.sleep(1.0)
        
        except asyncio.CancelledError:
            logger.info("ControlBus bridge loop cancelled")
        except Exception as e:
            logger.error(f"ControlBus bridge error: {e}", exc_info=True)
    
    async def _subscribe_control_bus_for_agent(self, agent_id: str) -> None:
        """Subscribe to ControlBus messages for a specific agent."""
        try:
            from openhands.events.agent_event import CommandEvent
            
            async for control_msg in self._control_bus.subscribe(agent_id):
                if self._closed or agent_id not in self._agents:
                    break
                
                # Convert ControlMessage to CommandEvent
                try:
                    command_event = CommandEvent(
                        from_agent_id="control_bus",
                        to_agent_id=agent_id,
                        command_type=control_msg.action,
                        target_agent_id=agent_id,
                        payload=control_msg.payload
                    )
                    
                    # Send to agent's queue
                    if agent_id in self._message_queues:
                        await self._message_queues[agent_id].put(command_event)
                
                except Exception as e:
                    logger.debug(f"Failed to bridge ControlBus message: {e}")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"ControlBus subscription error for {agent_id}: {e}")
    
    async def close(self) -> None:
        """Close the MessageBus and cleanup resources."""
        if self._closed:
            return
        
        self._closed = True
        logger.info("Closing MessageBus")
        
        # Cancel pending requests
        for correlation_id, future in self._pending_requests.items():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
        
        # Cancel bridge tasks
        for task in self._bridge_tasks:
            task.cancel()
        
        # Wait for bridge tasks to complete
        await asyncio.gather(*self._bridge_tasks, return_exceptions=True)
        self._bridge_tasks.clear()
        
        # Clear queues
        self._agents.clear()
        self._message_queues.clear()
        
        logger.info("MessageBus closed")


# Import AgentMessage types at runtime to avoid circular imports
def __getattr__(name):
    if name == "AgentMessage":
        from openhands.events.agent_event import AgentMessage
        return AgentMessage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
