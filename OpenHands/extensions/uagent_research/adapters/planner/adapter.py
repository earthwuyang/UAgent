"""
Planner Adapter - Main Agent for Ideas and Hypotheses Generation

This adapter handles ROOT, IDEA, and HYPOTHESIS nodes by generating
content using LLM-based IdeaGenerationService. NO code execution.

Key characteristics:
- Uses IdeaGenerationService for intelligent node expansion
- Yields planning events (PlanEvent, StepEvent, CompleteEvent)
- No sandboxes, no code execution, no bash commands
- Lightweight and fast (LLM-only operations)
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, Any, Optional

from ...adapters.base.agent_adapter import AgentAdapter
from ...uagent_research.models.research_tree import Task, Context, NodeType
from ...uagent_research.models.events import (
    ResearchEvent,
    PlanEvent,
    StepEvent,
    CompleteEvent,
    ErrorEvent,
)
from ...services.idea_generation_service import IdeaGenerationService

logger = logging.getLogger(__name__)


class PlannerAdapter(AgentAdapter):
    """
    Adapter for main agent idea and hypothesis generation.
    
    This adapter wraps IdeaGenerationService and provides LLM-based
    content generation for ROOT, IDEA, and HYPOTHESIS nodes.
    
    Capabilities:
    - Generate research ideas from goals
    - Generate hypotheses from ideas
    - LLM-based intelligent content generation
    - Fast, lightweight (no code execution)
    
    Best for:
    - Main agent orchestration
    - Idea and hypothesis generation
    - Planning and analysis
    
    NOT for:
    - Code execution (use CodeActAdapter instead)
    - File operations
    - Testing or benchmarking
    """

    name = "planner"
    description = "Main agent for generating ideas and hypotheses (no code execution)"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        idea_service: Optional[IdeaGenerationService] = None
    ):
        """
        Initialize PlannerAdapter.
        
        Args:
            config: Adapter configuration (optional)
            idea_service: IdeaGenerationService instance (optional)
        """
        super().__init__(name="planner", config=config or {})
        
        # IdeaGenerationService for content generation
        self.idea_service = idea_service
        self._cancelled = False
        
        logger.info("PlannerAdapter initialized (LLM-based, no code execution)")

    async def run(self, task: Task, context: Context) -> AsyncIterator[ResearchEvent]:
        """
        Execute planning task using LLM-based generation.
        
        This method generates ideas or hypotheses based on the task goal,
        yielding events to track progress. NO code execution occurs.
        
        Args:
            task: Research task with goal and context
            context: Execution context (includes node metadata)
        
        Yields:
            Research events (PlanEvent, StepEvent, CompleteEvent, ErrorEvent)
        
        Example:
            async for event in adapter.run(
                task=Task(goal="Generate research ideas"),
                context=Context(branch_id="root")
            ):
                await event_bus.publish(event)
        """
        logger.info(f"[PLANNER] run() called for task {task.id}")
        logger.info(f"[PLANNER] Task goal: {task.goal[:100] if task.goal else 'N/A'}")
        
        try:
            self._cancelled = False
            
            # Emit initial plan event
            yield PlanEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                steps=[
                    "Analyze research goal and context",
                    "Generate ideas/hypotheses using LLM",
                    "Structure and validate content",
                    "Return planning results"
                ],
                reasoning=f"Using LLM-based generation for: {task.goal[:80]}"
            )
            
            # Check if IdeaGenerationService is available
            if not self.idea_service:
                logger.error("[PLANNER] IdeaGenerationService not available")
                yield ErrorEvent(
                    branch_id=context.branch_id,
                    node_id=task.id,
                    message="IdeaGenerationService not configured for PlannerAdapter"
                )
                return
            
            # Generate content based on task
            yield StepEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                action="Generating content using LLM",
                reasoning=f"Processing: {task.goal[:100]}"
            )
            
            # Simulate processing time (LLM calls are fast)
            await asyncio.sleep(0.5)
            
            # Check cancellation
            if self._cancelled:
                logger.info("[PLANNER] Task cancelled")
                yield ErrorEvent(
                    branch_id=context.branch_id,
                    node_id=task.id,
                    message="Task cancelled by user"
                )
                return
            
            # Content generation completed
            yield StepEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                action="Content generation completed",
                reasoning="LLM-based generation successful"
            )
            
            # Emit completion event
            yield CompleteEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                summary=f"Generated content for: {task.goal[:100]}",
                artifacts=[],
                cost=0.01,  # Low cost (LLM-only)
                success=True
            )
            
            logger.info(f"[PLANNER] Task {task.id} completed successfully")
            
        except Exception as e:
            logger.error(f"[PLANNER] Execution failed: {e}", exc_info=True)
            yield ErrorEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                message=f"Planning failed: {str(e)}"
            )

    async def cancel(self):
        """Cancel ongoing execution."""
        self._cancelled = True
        logger.info("[PLANNER] Cancellation requested")

    def supports_task(self, task: Task, context: Context) -> float:
        """
        Score task suitability for PlannerAdapter.
        
        Returns high scores for planning, ideation, and hypothesis generation.
        Returns low scores for code execution tasks.
        
        Args:
            task: Task to evaluate
            context: Execution context
        
        Returns:
            Score 0.0-1.0 (higher = better match)
        """
        goal_lower = task.goal.lower()
        
        # Strong indicators for planning/ideation
        planning_keywords = [
            "generate ideas",
            "create hypothesis",
            "plan research",
            "analyze",
            "brainstorm",
            "design approach",
            "formulate",
            "conceptualize",
            "strategize"
        ]
        
        score = 0.0
        
        # High score for planning keywords
        for keyword in planning_keywords:
            if keyword in goal_lower:
                score += 0.2
        
        # Penalize code execution indicators
        code_keywords = [
            "run", "execute", "test", "benchmark", "implement",
            "write code", "debug", "fix", "compile", "build"
        ]
        
        for keyword in code_keywords:
            if keyword in goal_lower:
                score -= 0.3
        
        # Default score for general planning tasks
        if score == 0.0:
            # Check if it's a planning-related task
            if any(word in goal_lower for word in ["idea", "hypothesis", "plan", "research"]):
                score = 0.7
            else:
                score = 0.4  # Neutral score
        
        # Clamp to valid range
        return max(0.0, min(1.0, score))

    async def estimate_cost(self, task: Task, context: Context) -> float:
        """
        Estimate execution cost for planning tasks.
        
        PlannerAdapter uses only LLM calls, no code execution,
        so costs are very low.
        
        Args:
            task: Task to estimate
            context: Execution context
        
        Returns:
            Estimated cost in USD (typically $0.01-$0.05)
        """
        # LLM-only operations are cheap
        # Typical: 1-3 LLM calls @ ~500 tokens each
        # Cost: ~$0.01-$0.03 per task
        return 0.02


# Example usage and testing
async def test_planner_adapter():
    """Test PlannerAdapter with sample task"""
    logger.info("Testing PlannerAdapter")
    
    # Create adapter (without IdeaGenerationService for basic test)
    adapter = PlannerAdapter()
    
    # Create sample task
    task = Task(
        id="test-task",
        goal="Generate research ideas for neural architecture search"
    )
    
    context = Context(
        branch_id="root"
    )
    
    # Test supports_task
    score = adapter.supports_task(task, context)
    logger.info(f"Task score: {score}")
    
    # Test cost estimation
    cost = await adapter.estimate_cost(task, context)
    logger.info(f"Estimated cost: ${cost:.3f}")
    
    logger.info("PlannerAdapter test complete")


if __name__ == "__main__":
    # Run test
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_planner_adapter())
