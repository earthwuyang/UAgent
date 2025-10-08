"""
Sub-Agent Observation Events

Represents WebSocket events for sub-agent lifecycle and progress tracking.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class SubAgentSpawnedObservation(Observation):
    """Emitted when a new sub-agent is spawned.
    
    Attributes:
        content: Human-readable message about sub-agent spawn
        sub_agent_id: Unique identifier for the sub-agent
        sub_agent_type: Type of sub-agent (research, code, analysis)
        goal: Research goal/objective
        session_id: Parent session ID
    """
    
    sub_agent_id: str = ""
    sub_agent_type: str = "research"
    goal: str = ""
    session_id: str = ""
    observation: str = ObservationType.SUB_AGENT_SPAWNED
    
    @property
    def message(self) -> str:
        return self.content or f"Sub-agent {self.sub_agent_id} spawned: {self.goal}"


@dataclass
class SubAgentProgressObservation(Observation):
    """Emitted periodically with sub-agent progress updates.
    
    Attributes:
        content: Human-readable progress message
        sub_agent_id: Unique identifier for the sub-agent
        status: Current status (running, paused, complete, failed, cancelled)
        progress: Progress percentage (0.0 to 1.0)
        current_task: Description of current task
        tree_stats: Optional tree statistics dict
    """
    
    sub_agent_id: str = ""
    status: str = "running"
    progress: float = 0.0
    current_task: str = ""
    tree_stats: Optional[Dict[str, Any]] = None
    observation: str = ObservationType.SUB_AGENT_PROGRESS
    
    @property
    def message(self) -> str:
        return self.content or f"Sub-agent {self.sub_agent_id} progress: {self.progress:.1%}"


@dataclass
class SubAgentCompletedObservation(Observation):
    """Emitted when a sub-agent completes (success, failure, or cancellation).
    
    Attributes:
        content: Human-readable completion message
        sub_agent_id: Unique identifier for the sub-agent
        status: Final status (complete, failed, cancelled)
        result: Optional result data or error message
        tree_stats: Optional final tree statistics
    """
    
    sub_agent_id: str = ""
    status: str = "complete"
    result: Optional[str] = None
    tree_stats: Optional[Dict[str, Any]] = None
    observation: str = ObservationType.SUB_AGENT_COMPLETED
    
    @property
    def message(self) -> str:
        return self.content or f"Sub-agent {self.sub_agent_id} {self.status}"
