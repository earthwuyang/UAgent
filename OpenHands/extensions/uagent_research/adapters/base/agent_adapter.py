"""
Agent Adapter Base Interface

All research agents (DeepResearch, RepoMaster, CodeAct) implement this interface
to provide a unified execution model for the TreeSearchOrchestrator.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import logging

from ...uagent_research.models.research_tree import Task, Context
from ...uagent_research.models.events import ResearchEvent


logger = logging.getLogger(__name__)


class AgentAdapter(ABC):
    """
    Base interface for research agent adapters.

    Adapters wrap different agent frameworks (DeepResearch, Autogen, CodeAct)
    and provide a unified event stream for the orchestrator.
    """

    def __init__(self, config: Optional[dict] = None, name: Optional[str] = None):
        """
        Initialize adapter.

        Args:
            config: Adapter-specific configuration
            name: Adapter name (e.g., "deepresearch", "repomaster", "codeact")
                  If not provided, uses the class's `name` attribute
        """
        # Use provided name or class attribute
        if name:
            self.name = name
        elif not hasattr(self, 'name'):
            raise ValueError("Adapter must have either a 'name' parameter or class attribute")

        self.config = config or {}
        self._cancelled = False

    @abstractmethod
    async def run(self, task: Task, context: Context) -> AsyncIterator[ResearchEvent]:
        """
        Execute task and yield events.

        This is the main entry point for the adapter. It should:
        1. Execute the task using the underlying agent framework
        2. Yield events (Plan, Step, ToolCall, Observation, Summary, Complete)
        3. Handle cancellation gracefully
        4. Track costs and budgets

        Args:
            task: Task with goal, constraints, and budget
            context: Execution context (parent nodes, tools, secrets)

        Yields:
            ResearchEvent: Events during execution

        Example:
            async for event in adapter.run(task, context):
                if isinstance(event, ToolCallEvent):
                    print(f"Calling tool: {event.tool}")
                elif isinstance(event, CompleteEvent):
                    print(f"Complete! Summary: {event.summary}")
        """
        pass

    @abstractmethod
    async def cancel(self):
        """
        Cancel ongoing execution.

        Should gracefully stop the agent and clean up resources.
        """
        self._cancelled = True

    async def send_message(self, message: str) -> None:
        """
        Optionally deliver a runtime message to the adapter.

        Adapters that support runtime steering can override this method to
        forward guidance to the underlying agent session (e.g. via chat).
        The default implementation is a no-op so adapters that do not support
        this capability do not need to override it.

        Args:
            message: Message text to deliver to the running agent.
        """

        logger.debug(
            "Adapter '%s' does not implement runtime messaging. Ignoring message: %s",
            getattr(self, "name", self.__class__.__name__),
            message[:100],
        )

    def is_cancelled(self) -> bool:
        """Check if adapter has been cancelled"""
        return self._cancelled

    async def estimate_cost(self, task: Task, context: Context) -> float:
        """
        Estimate cost of executing task (in USD).

        Optional method for adapters to implement cost estimation.

        Args:
            task: Task to estimate
            context: Execution context

        Returns:
            Estimated cost in USD
        """
        return 0.0

    def supports_task(self, task: Task, context: Context) -> float:
        """
        Score how well this adapter can handle the task.

        Used by SkillRouter to select the best adapter.

        Args:
            task: Task to evaluate
            context: Execution context

        Returns:
            Score 0-1 (higher is better match)
        """
        return 0.5  # Default: neutral score


class AdapterRegistry:
    """Registry of available agent adapters"""

    def __init__(self):
        self._adapters: dict[str, AgentAdapter] = {}

    def register(self, adapter: AgentAdapter):
        """Register an adapter"""
        self._adapters[adapter.name] = adapter
        logger.info(f"Registered adapter: {adapter.name}")

    def get(self, name: str) -> Optional[AgentAdapter]:
        """Get adapter by name"""
        return self._adapters.get(name)

    def list(self) -> list[str]:
        """List all registered adapters"""
        return list(self._adapters.keys())

    def get_all_adapters(self) -> list:
        """Get all registered adapter instances"""
        return list(self._adapters.values())

    def select_best(self, task: Task, context: Context) -> Optional[AgentAdapter]:
        """
        Select best adapter for task.

        Args:
            task: Task to execute
            context: Execution context

        Returns:
            Best matching adapter or None
        """
        if not self._adapters:
            return None

        # Score all adapters
        scored = [
            (adapter.supports_task(task, context), adapter)
            for adapter in self._adapters.values()
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return best
        best_score, best_adapter = scored[0]

        if best_score > 0:
            logger.info(f"Selected adapter {best_adapter.name} (score: {best_score:.2f}) for task")
            return best_adapter

        return None


# Global registry instance
adapter_registry = AdapterRegistry()
