"""
CodeAct Adapter

Wraps OpenHands CodeActAgent for code execution and general programming tasks.
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime

from ...adapters.base.agent_adapter import AgentAdapter
from ...uagent_research.models.research_tree import Task, Context
from ...uagent_research.models.events import (
    ResearchEvent,
    PlanEvent,
    StepEvent,
    ToolCallEvent,
    ObservationEvent,
    SummaryEvent,
    CompleteEvent,
    ErrorEvent,
    Artifact,
)

logger = logging.getLogger(__name__)


class CodeActAdapter(AgentAdapter):
    """
    Adapter for OpenHands CodeActAgent.

    Capabilities:
    - Code execution (Python, Bash)
    - File operations (read, write, edit)
    - Testing and benchmarking
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
    description = "Code execution agent for programming tasks"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CodeAct adapter.

        Args:
            config: Adapter configuration
                - model: LLM model name
                - max_iterations: Max execution iterations (default: 30)
                - work_dir: Working directory for execution
        """
        super().__init__(config)

        self.model = config.get("model", "gpt-4o") if config else "gpt-4o"
        self.max_iterations = config.get("max_iterations", 30) if config else 30
        self.work_dir = config.get("work_dir", "/tmp/codeact") if config else "/tmp/codeact"

        # Agent state
        self._agent = None
        self._current_task = None
        self._cancelled = False

    async def run(self, task: Task, context: Context) -> AsyncIterator[ResearchEvent]:
        """
        Execute code task using CodeActAgent.

        Args:
            task: Research task
            context: Execution context

        Yields:
            Research events (Plan, Step, ToolCall, Observation, Summary, Complete)

        Example:
            async for event in adapter.run(
                task=Task(goal="Run benchmark comparing sorting algorithms"),
                context=Context()
            ):
                print(f"{event.type}: {event}")
        """
        try:
            self._current_task = task
            self._cancelled = False

            # Emit plan event
            yield PlanEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                plan=f"Code execution: {task.goal}",
                steps=[
                    "Understand task requirements",
                    "Write/modify code as needed",
                    "Execute code and collect results",
                    "Analyze output and generate summary",
                ],
            )

            # Execute task
            async for event in self._execute_code_task(task, context):
                if self._cancelled:
                    yield ErrorEvent(
                        branch_id=context.branch_id,
                        node_id=task.id,
                        error="Task cancelled by user",
                    )
                    return

                yield event

        except Exception as e:
            logger.error(f"CodeAct execution failed: {e}", exc_info=True)
            yield ErrorEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                error=str(e),
            )

    async def _execute_code_task(
        self, task: Task, context: Context
    ) -> AsyncIterator[ResearchEvent]:
        """
        Execute code task.

        This is a simplified implementation for demonstration.
        In production, this would integrate with the actual CodeActAgent.
        """
        # For now, emit placeholder events
        # In production, this would:
        # 1. Initialize CodeActAgent
        # 2. Stream events from agent execution
        # 3. Convert OpenHands events to ResearchEvents

        yield StepEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            step_number=1,
            description=f"Preparing to execute: {task.goal}",
            artifacts=[],
        )

        # Simulate code execution
        await asyncio.sleep(0.5)

        yield ObservationEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            observation="Task analysis complete",
            artifacts=[
                Artifact(
                    type="snippet",
                    content=f"Task: {task.goal}\nContext: {task.context or 'None'}",
                    metadata={"language": "text"},
                )
            ],
        )

        yield StepEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            step_number=2,
            description="Executing code",
            artifacts=[],
        )

        # Simulate execution
        await asyncio.sleep(1.0)

        # Generate mock result
        result_code = f"""
# {task.goal}

# This is a placeholder implementation
# In production, this would be the actual CodeActAgent execution

def main():
    print("Task: {task.goal}")
    print("Execution completed successfully")

if __name__ == "__main__":
    main()
"""

        yield ObservationEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            observation="Code execution completed",
            artifacts=[
                Artifact(
                    type="code",
                    content=result_code.strip(),
                    metadata={"language": "python", "executed": True},
                )
            ],
        )

        # Generate summary
        summary = f"""# Execution Summary: {task.goal}

## Task
{task.goal}

## Context
{task.context or 'None'}

## Result
Code execution completed successfully.

## Notes
This is a simplified placeholder implementation.
In production, this adapter will:
1. Initialize OpenHands CodeActAgent
2. Execute the task in a sandboxed environment
3. Stream real-time events and observations
4. Collect and return actual execution results
"""

        yield SummaryEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            summary=summary,
            artifacts=[
                Artifact(
                    type="snippet",
                    content=summary,
                    metadata={"format": "markdown"},
                )
            ],
        )

        # Complete
        yield CompleteEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            summary="Code execution completed (placeholder implementation)",
            artifacts=[
                Artifact(
                    type="code",
                    content=result_code.strip(),
                    metadata={"language": "python"},
                )
            ],
        )

    async def cancel(self):
        """Cancel ongoing execution"""
        self._cancelled = True
        logger.info("CodeAct task cancelled")

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
        ]

        score = 0.0

        for keyword in code_keywords:
            if keyword in goal_lower:
                score += 0.2

        # Boost for explicit programming languages
        languages = ["python", "bash", "shell", "javascript", "java", "c++"]
        for lang in languages:
            if lang in goal_lower:
                score += 0.2

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
        # Estimate: ~10-30 LLM calls for typical task
        # Average ~500 tokens per call
        # Rough estimate: $0.02-$0.05
        return 0.03


# Example usage
async def test_codeact_adapter():
    """Test CodeAct adapter"""
    adapter = CodeActAdapter()

    task = Task(
        goal="Run a simple Python script to calculate factorial of 10",
        context="Use iterative approach",
    )

    context = Context(branch_id="test-branch")

    print(f"Running CodeAct adapter for: {task.goal}\n")

    async for event in adapter.run(task, context):
        print(f"[{event.type}] {event.timestamp}")

        if hasattr(event, "description"):
            print(f"  {event.description}")
        elif hasattr(event, "observation"):
            print(f"  {event.observation}")
        elif hasattr(event, "summary"):
            print(f"  Summary: {event.summary[:100]}...")

        print()


if __name__ == "__main__":
    asyncio.run(test_codeact_adapter())
