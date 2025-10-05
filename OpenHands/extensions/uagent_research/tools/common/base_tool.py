"""
Base Tool Interface

All research tools implement this interface for unified access.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any
    cost: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RateLimit:
    """Rate limiting configuration"""

    def __init__(self, rate: float, capacity: int):
        """
        Initialize rate limiter.

        Args:
            rate: Requests per second
            capacity: Burst capacity
        """
        self.rate = rate
        self.capacity = capacity


class Tool(ABC):
    """
    Base interface for research tools.

    All tools (web search, browsing, code execution, etc.) implement this interface.
    """

    # Tool metadata (override in subclasses)
    name: str = "base_tool"
    description: str = "Base tool"
    cost_per_call: float = 0.0  # USD per invocation

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tool.

        Args:
            config: Tool-specific configuration (API keys, etc.)
        """
        self.config = config or {}
        self._call_count = 0
        self._total_cost = 0.0

    @abstractmethod
    async def invoke(self, **kwargs) -> ToolResult:
        """
        Execute tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with success status and data

        Example:
            result = await tool.invoke(query="neural networks")
            if result.success:
                print(result.data)
        """
        pass

    def cache_key(self, **kwargs) -> str:
        """
        Generate cache key for deduplication.

        Args:
            **kwargs: Tool arguments

        Returns:
            Hash string for caching

        Example:
            key = tool.cache_key(query="test", num=10)
            # Returns: "a3b4c5d6..."
        """
        # Sort keys for consistent hashing
        sorted_args = json.dumps(kwargs, sort_keys=True)
        hash_obj = hashlib.sha256(sorted_args.encode())
        return hash_obj.hexdigest()[:16]

    async def validate_args(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate tool arguments.

        Args:
            **kwargs: Tool arguments

        Returns:
            (is_valid, error_message)
        """
        return True, None

    def record_call(self, cost: float = None):
        """Record tool invocation for analytics"""
        self._call_count += 1
        call_cost = cost if cost is not None else self.cost_per_call
        self._total_cost += call_cost

    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        return {
            "name": self.name,
            "call_count": self._call_count,
            "total_cost": self._total_cost,
            "avg_cost": self._total_cost / max(1, self._call_count)
        }


class ToolRegistry:
    """Registry of available tools"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool"""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self._tools.get(name)

    def list(self) -> list[str]:
        """List all registered tools"""
        return list(self._tools.keys())

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tools"""
        return {
            name: tool.get_stats()
            for name, tool in self._tools.items()
        }


# Global tool registry
tool_registry = ToolRegistry()
