"""
Event Models for Research Tree

These events are emitted by agent adapters during execution and consumed
by the TreeSearchOrchestrator to build the research tree.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events in research execution"""
    PLAN = "plan"
    STEP = "step"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    SUMMARY = "summary"
    CRITIQUE = "critique"
    COMPLETE = "complete"
    ERROR = "error"


class ArtifactType(str, Enum):
    """Types of artifacts produced during research"""
    URL = "url"
    FILE = "file"
    CODE = "code"
    SNIPPET = "snippet"
    PLOT = "plot"
    DATASET = "dataset"
    SUMMARY = "summary"


class Artifact(BaseModel):
    """Research artifact (output/evidence)"""
    kind: ArtifactType
    locator: str  # URL, file path, or identifier
    content: Optional[str] = None  # Optional content
    summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    hash: Optional[str] = None  # Content hash for deduplication


class Event(BaseModel):
    """Base event emitted by agent adapters"""
    type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    branch_id: str  # Research branch identifier
    node_id: Optional[str] = None  # Associated tree node


class PlanEvent(Event):
    """Planning event - agent decides on approach"""
    type: EventType = EventType.PLAN
    steps: List[str]  # Planned steps
    reasoning: str  # Why this approach


class StepEvent(Event):
    """Step execution event"""
    type: EventType = EventType.STEP
    action: str  # Action being taken
    reasoning: Optional[str] = None  # Why this action


class ToolCallEvent(Event):
    """Tool invocation event"""
    type: EventType = EventType.TOOL_CALL
    tool: str  # Tool name
    args: Dict[str, Any]  # Tool arguments
    cost: float = 0.0  # Estimated cost


class ObservationEvent(Event):
    """Observation from tool/action execution"""
    type: EventType = EventType.OBSERVATION
    result: Dict[str, Any]  # Execution result
    success: bool = True
    error: Optional[str] = None


class SummaryEvent(Event):
    """Summary of findings"""
    type: EventType = EventType.SUMMARY
    content: str  # Summary content
    citations: List[str] = Field(default_factory=list)  # Sources
    confidence: float = 0.0  # 0-1 confidence score
    artifacts: List[Artifact] = Field(default_factory=list)


class CritiqueEvent(Event):
    """Self-critique or reflection"""
    type: EventType = EventType.CRITIQUE
    content: str  # Critique content
    improvements: List[str] = Field(default_factory=list)


class CompleteEvent(Event):
    """Completion event"""
    type: EventType = EventType.COMPLETE
    artifacts: List[Artifact]  # Final artifacts
    summary: str  # Final summary
    success: bool = True


class ErrorEvent(Event):
    """Error event"""
    type: EventType = EventType.ERROR
    message: str
    traceback: Optional[str] = None
    recoverable: bool = False


# Union type for all events
ResearchEvent = (
    PlanEvent
    | StepEvent
    | ToolCallEvent
    | ObservationEvent
    | SummaryEvent
    | CritiqueEvent
    | CompleteEvent
    | ErrorEvent
)
