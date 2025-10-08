"""
CodeAct Adapter - Real OpenHands CodeActAgent Integration

Integrates OpenHands CodeActAgent for real code execution via HeadlessAgentSession.
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, Any, Optional

from ...adapters.base.agent_adapter import AgentAdapter
from ...uagent_research.models.research_tree import Task, Context
from ...uagent_research.models.events import (
    ResearchEvent,
    PlanEvent,
    StepEvent,
    ErrorEvent,
    CompleteEvent,
)
from .session_runner import HeadlessAgentSession
from ...bridges.openhands_bridge import OpenHandsEventBridge

logger = logging.getLogger(__name__)


class CodeActAdapter(AgentAdapter):
    """
    Adapter for real OpenHands CodeActAgent integration.

    Uses HeadlessAgentSession to run actual CodeActAgent in embedded mode,
    with OpenHandsEventBridge converting events to ResearchEvents.

    Capabilities:
    - Code execution (Python, Bash, IPython)
    - File operations (read, write, edit)
    - Testing and benchmarking
    - Browser interactions
    - General programming tasks
    - Debugging and refactoring

    Best for:
    - Running experiments
    - Testing code
    - Implementing algorithms
    - Data processing
    - General coding tasks
    """

    name = "codeact"
    description = "Real OpenHands CodeActAgent for programming tasks"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CodeAct adapter.

        Args:
            config: Adapter configuration
                - llm_config: LLM configuration for the agent
                - max_iterations: Max execution iterations (default: 30)
                - agent_config: Optional AgentConfig
        """
        super().__init__(name="codeact", config=config or {})

        # Extract configuration
        self.llm_config = config.get("llm_config") if config else None
        self.max_iterations = config.get("max_iterations", 30) if config else 30
        self.agent_config = config.get("agent_config") if config else None

        # Agent state
        self._current_session: Optional[HeadlessAgentSession] = None
        self._cancelled = False

        # Import CodeActAgent lazily to avoid circular imports
        self._codeact_agent_class = None

    def _get_codeact_agent_class(self):
        """Lazy load CodeActAgent class"""
        if self._codeact_agent_class is None:
            try:
                from openhands.agenthub.codeact_agent import CodeActAgent
                self._codeact_agent_class = CodeActAgent
                logger.info("CodeActAgent loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import CodeActAgent: {e}")
                # DIAGNOSTIC: Log import failure
                logger.error(f"[DIAGNOSTIC] CodeActAgent import failed:")
                logger.error(f"[DIAGNOSTIC]   Error: {str(e)}", exc_info=True)
                # Fallback: try alternative import path
                try:
                    from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
                # DIAGNOSTIC: Log successful import
                logger.info(f"[DIAGNOSTIC] Successfully loaded CodeActAgent")
                    self._codeact_agent_class = CodeActAgent
                    logger.info("CodeActAgent loaded successfully (alternative path)")
                except ImportError:
                    logger.error("CodeActAgent not found in any expected location")
                    raise

        return self._codeact_agent_class

    def _get_llm_config(self):
        """Get LLM config, using default if not provided"""
        if self.llm_config:
            return self.llm_config

        # Create default LLM config
        try:
            from openhands.core.config import LLMConfig
            return LLMConfig(
                model="gpt-4o",
                api_key=None,  # Will use env var
            )
        except Exception as e:
            logger.error(f"Failed to create default LLM config: {e}")
            raise

    async def run(self, task: Task, context: Context) -> AsyncIterator[ResearchEvent]:
        """
        Execute code task using real OpenHands CodeActAgent.

        Creates a HeadlessAgentSession, starts CodeActAgent, and bridges
        OpenHands events to ResearchEvents.

        Args:
            task: Research task
            context: Execution context

        Yields:
            Research events from CodeActAgent execution

        Example:
            async for event in adapter.run(
                task=Task(goal="Implement quicksort in Python"),
                context=Context(branch_id="idea-0-hyp-0")
            ):
                await event_bus.publish(event)
        """
        
        logger.info(f"[CODEACT] run() called for task {task.id}")
        logger.info(f"[CODEACT] Task goal: {task.goal[:100] if task.goal else 'N/A'}")
        try:
            self._cancelled = False

            # Emit initial plan event
            yield PlanEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                steps=[
                    "Initialize CodeActAgent",
                    "Execute task in sandboxed environment",
                    "Stream execution events",
                    "Collect results and artifacts"
                ],
                reasoning=f"Using OpenHands CodeActAgent for: {task.goal}"
            )

            # Get CodeActAgent class
            try:
                agent_class = self._get_codeact_agent_class()
            except Exception as e:
                logger.error(f"Failed to load CodeActAgent: {e}")
                yield ErrorEvent(
                    branch_id=context.branch_id,
                    node_id=task.id,
                    message=f"CodeActAgent not available: {str(e)}"
                )
                return

            # Get LLM config
            try:
                llm_config = self._get_llm_config()
            except Exception as e:
                logger.error(f"Failed to get LLM config: {e}")
                yield ErrorEvent(
                    branch_id=context.branch_id,
                    node_id=task.id,
                    message=f"LLM configuration error: {str(e)}"
                )
                return

            # Create HeadlessAgentSession
        logger.info(f"[CODEACT] Creating HeadlessAgentSession for task {task.id}")
            experiment_id = f"{context.branch_id}-{task.id}-codeact"

            yield StepEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                action="Creating CodeActAgent session",
                reasoning=f"Session ID: {experiment_id}"
            )

            try:
                self._current_session = HeadlessAgentSession(
                    agent_class=agent_class,
                    llm_config=llm_config,
                    agent_config=self.agent_config,
                    experiment_id=experiment_id,
                    max_iterations=self.max_iterations
                )
            except Exception as e:
                logger.error(f"Failed to create HeadlessAgentSession: {e}", exc_info=True)
                yield ErrorEvent(
                    branch_id=context.branch_id,
                    node_id=task.id,
                    message=f"Session creation failed: {str(e)}"
                )
                return

            # Create event bridge
            event_bridge = OpenHandsEventBridge(
                event_stream=self._current_session.event_stream,
                branch_id=context.branch_id,
                node_id=task.id
            )

            # Start agent execution
            yield StepEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                action="Starting CodeActAgent execution",
                reasoning=f"Task: {task.goal}"
            )

            try:
                await self._current_session.start(initial_message=task.goal)
            except Exception as e:
                logger.error(f"Failed to start session: {e}", exc_info=True)
                yield ErrorEvent(
                    branch_id=context.branch_id,
                    node_id=task.id,
                    message=f"Failed to start execution: {str(e)}"
                )
                await self._cleanup_session()
                return

            # Stream bridged events
            event_count = 0
            try:
                async for research_event in event_bridge.stream():
                    # Check cancellation
                    if self._cancelled:
                        logger.info(f"CodeAct task cancelled for {task.id}")
                        await self._current_session.cancel()
                        yield ErrorEvent(
                            branch_id=context.branch_id,
                            node_id=task.id,
                            message="Task cancelled by user"
                        )
                        break

                    event_count += 1
                
                # DIAGNOSTIC: Log total event count
                logger.info(f"[DIAGNOSTIC] Event streaming complete. Total events: {event_count}")
                    yield research_event

                    # Check if this was a completion event
                    if isinstance(research_event, CompleteEvent):
                        logger.info(f"CodeAct task completed for {task.id}")
                        break

            except Exception as e:
                logger.error(f"Error streaming events: {e}", exc_info=True)
                yield ErrorEvent(
                    branch_id=context.branch_id,
                    node_id=task.id,
                    message=f"Event streaming error: {str(e)}"
                )

            # Cleanup session
            await self._cleanup_session()

            logger.info(
                f"CodeAct execution finished for {task.id}: "
                f"{event_count} events emitted"
            )

        except Exception as e:
            logger.error(f"CodeAct execution failed: {e}", exc_info=True)
            yield ErrorEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                message=f"Unexpected error: {str(e)}"
            )
            await self._cleanup_session()

    async def _cleanup_session(self):
        """Cleanup current session"""
        if self._current_session:
            try:
                await self._current_session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")
            finally:
                self._current_session = None

    async def cancel(self):
        """Cancel ongoing execution"""
        self._cancelled = True

        if self._current_session:
            await self._current_session.cancel()

        logger.info("CodeAct task cancellation requested")

    async def send_message(self, message: str) -> None:
        """Forward steering directives to the active CodeAct session."""

        if not self._current_session:
            logger.warning(
                "Cannot deliver steering message to CodeAct adapter: no active session"
            )
            return

        try:
            await self._current_session.send_user_message(message)
            logger.info(
                "Delivered steering message to CodeAct session: %s", message[:100]
            )
        except Exception as exc:
            logger.error(
                "Failed to deliver steering message to CodeAct session: %s", exc,
                exc_info=True,
            )

    def supports_task(self, task: Task, context: Context) -> float:
        """
        Score task suitability for CodeAct.

        Returns:
            0.0-1.0 score (higher = better match)
        """
        goal_lower = task.goal.lower()

        # Strong indicators for code execution
        code_keywords = [
            "run",
            "execute",
            "test",
            "benchmark",
            "implement",
            "write code",
            "debug",
            "fix",
            "refactor",
            "experiment",
            "build",
            "compile",
            "analyze code",
        ]

        score = 0.0

        for keyword in code_keywords:
            if keyword in goal_lower:
                score += 0.15

        # Boost for explicit programming languages
        languages = [
            "python", "bash", "shell", "javascript", "java",
            "c++", "rust", "go", "typescript"
        ]
        for lang in languages:
            if lang in goal_lower:
                score += 0.2

        # File operation indicators
        file_ops = ["file", "directory", "folder", "edit", "modify"]
        for op in file_ops:
            if op in goal_lower:
                score += 0.1

        # Default score for general coding tasks
        if score == 0.0:
            score = 0.3  # CodeAct can handle most tasks

        return min(1.0, score)

    async def estimate_cost(self, task: Task, context: Context) -> float:
        """
        Estimate execution cost.

        Returns:
            Estimated cost in USD
        """
        # Real CodeActAgent estimation:
        # - Typical task: 10-50 iterations
        # - Average ~1000 tokens per iteration (input + output)
        # - GPT-4: ~$0.03/1K tokens input, ~$0.06/1K tokens output
        # - Average: ~$0.05-$0.25 per task

        # Conservative estimate
        return 0.10


# Example usage
async def test_codeact_adapter():
    """Test real CodeAct adapter"""
    logger.info("Testing CodeActAdapter with real OpenHands integration")

    # Note: This test requires proper OpenHands setup
    # For basic testing, just verify the adapter loads

    adapter = CodeActAdapter()

    logger.info(f"✓ CodeActAdapter initialized: {adapter.name}")
    logger.info(f"  Description: {adapter.description}")
    logger.info(f"  Max iterations: {adapter.max_iterations}")

    # Test task scoring
    from ...uagent_research.models.research_tree import Task, Context

    test_tasks = [
        Task(id="1", goal="Implement quicksort in Python"),
        Task(id="2", goal="Run benchmark comparing sorting algorithms"),
        Task(id="3", goal="Search for neural architecture papers"),
    ]

    for task in test_tasks:
        score = adapter.supports_task(task, Context(branch_id="test"))
        logger.info(f"  Task '{task.goal[:40]}...': score={score:.2f}")

    logger.info("✓ CodeActAdapter test completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_codeact_adapter())
