"""
AgentEvent - Event types for inter-agent communication via MessageBus

Defines typed messages for request/response, notifications, and agent lifecycle events.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """Base class for all agent messages in MessageBus."""
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    from_agent_id: str
    to_agent_id: Optional[str] = None  # None for broadcast
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    correlation_id: Optional[str] = None  # For request/response correlation


class AgentSpawnedEvent(AgentMessage):
    """
    Emitted when a sub-agent is spawned.
    
    Used by coordinator to notify main agent of new sub-agents.
    """
    sub_agent_id: str
    agent_type: str  # coordinator, research, code, analysis
    goal: str
    parent_agent_id: str
    capabilities: List[str]  # tree_search, code_execution, etc.


class NodeCompleteEvent(AgentMessage):
    """
    Emitted when a research tree node completes.
    
    Bridges from EventBus ResearchEvent to MessageBus.
    """
    node_id: str
    branch_id: str
    experiment_id: str
    result: Any
    cost: float
    artifacts: Optional[Dict[str, Any]] = None


class CommandEvent(AgentMessage):
    """
    Emitted when a control command is issued.
    
    Bridges from ControlBus ControlMessage to MessageBus.
    """
    command_type: str  # pause, resume, cancel, steer
    target_agent_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class RequestMessage(AgentMessage):
    """
    For request/response RPC pattern.
    
    Expects ResponseMessage with matching correlation_id.
    """
    request_type: str  # status, pause, resume, get_stats, etc.
    payload: Dict[str, Any] = Field(default_factory=dict)
    timeout: float = 30.0


class ResponseMessage(AgentMessage):
    """
    Response to RequestMessage.
    
    Must have correlation_id matching original request.
    """
    status: str  # success, error
    result: Optional[Any] = None
    error_message: Optional[str] = None


class ProgressUpdateEvent(AgentMessage):
    """
    Periodic progress updates from sub-agents.
    
    Used for UI updates.
    """
    progress: float  # 0.0-1.0
    current_task: str
    stats: Dict[str, Any] = Field(default_factory=dict)


class NotificationMessage(AgentMessage):
    """
    Fire-and-forget notification message.
    
    Does not expect a response.
    """
    content: str
    notification_type: str = "info"  # info, warning, error
    metadata: Dict[str, Any] = Field(default_factory=dict)
