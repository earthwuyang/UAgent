"""
Sub-Agent Status Data Model

Represents the status of a research sub-agent for API responses and UI display.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class SubAgentStatus:
    """Status information for a research sub-agent."""
    
    id: str  # Sub-agent unique identifier
    type: str  # Sub-agent type ("research", "code", "analysis")
    goal: str  # Research goal/objective
    status: str  # Current status ("running", "paused", "complete", "failed", "cancelled")
    progress: float  # Progress percentage (0.0 to 1.0)
    current_task: str  # Description of current task being executed
    session_id: str  # Parent session ID
    created_at: float  # Unix timestamp of creation
    completed_at: Optional[float] = None  # Unix timestamp of completion
    max_iterations: int = 50  # Max iterations configured
    tree_stats: Dict[str, Any] = field(default_factory=dict)  # Tree statistics
    is_running: bool = True  # Whether task is still running
