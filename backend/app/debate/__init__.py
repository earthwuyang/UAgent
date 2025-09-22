"""Multi-agent debate utilities."""

from .debate_manager import DebateManager, DebateConfig, DebaterConfig
from .policy import DebatePolicy, should_debate

__all__ = [
    "DebateManager",
    "DebateConfig",
    "DebaterConfig",
    "DebatePolicy",
    "should_debate",
]
