"""
DeepResearch Adapter

Wraps DeepResearch WebSailor ReAct agent for web research tasks.
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime
import sys
import os

# Add vendor path to sys.path
vendor_path = os.path.join(os.path.dirname(__file__), "../../vendor/deepresearch")
if vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

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


class DeepResearchAdapter(AgentAdapter):
    """
    Adapter for DeepResearch WebSailor agent.

    Capabilities:
    - Web search via Bing/Google
    - Multi-turn ReAct reasoning
    - Web page browsing and content extraction
    - Evidence collection and synthesis

    Best for:
    - Literature reviews
    - Fact-checking
    - Background research
    - Web-based information gathering
    """

    name = "deepresearch"
    description = "Web research agent using ReAct reasoning"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DeepResearch adapter.

        Args:
            config: Adapter configuration
                - model: LLM model name
                - max_turns: Max ReAct turns (default: 40)
                - search_tool: "bing" or "google" (default: "bing")
        """
        super().__init__(config)

        self.model = config.get("model", "gpt-4o") if config else "gpt-4o"
        self.max_turns = config.get("max_turns", 40) if config else 40
        self.search_tool = config.get("search_tool", "bing") if config else "bing"

        # Agent state
        self._agent = None
        self._current_task = None
        self._cancelled = False

    async def run(self, task: Task, context: Context) -> AsyncIterator[ResearchEvent]:
        """
        Execute web research task using DeepResearch.

        Args:
            task: Research task
            context: Execution context

        Yields:
            Research events (Plan, Step, ToolCall, Observation, Summary, Complete)

        Example:
            async for event in adapter.run(
                task=Task(goal="Find papers on neural architecture search"),
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
                plan=f"Web research: {task.goal}",
                steps=[
                    "Search web for relevant sources",
                    "Browse top results",
                    "Extract and synthesize information",
                    "Generate summary",
                ],
            )

            # Execute research
            async for event in self._execute_research(task, context):
                if self._cancelled:
                    yield ErrorEvent(
                        branch_id=context.branch_id,
                        node_id=task.id,
                        error="Task cancelled by user",
                    )
                    return

                yield event

        except Exception as e:
            logger.error(f"DeepResearch execution failed: {e}", exc_info=True)
            yield ErrorEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                error=str(e),
            )

    async def _execute_research(
        self, task: Task, context: Context
    ) -> AsyncIterator[ResearchEvent]:
        """
        Execute research using DeepResearch agent.

        This is a simplified implementation that uses our own tools
        instead of the original DeepResearch implementation.
        """
        from ...tools.search.bing_search_tool import BingSearchTool
        from ...tools.browse.web_browse_tool import WebBrowseTool

        search_tool = BingSearchTool()
        browse_tool = WebBrowseTool()

        # Step 1: Search for information
        yield StepEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            step_number=1,
            description=f"Searching web for: {task.goal}",
            artifacts=[],
        )

        search_result = await search_tool.invoke(query=task.goal, num_results=5)

        if not search_result.success:
            yield ErrorEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                error=f"Search failed: {search_result.error}",
            )
            return

        # Emit search results
        search_artifacts = [
            Artifact(
                type="url",
                content=result["url"],
                metadata={"title": result["title"], "snippet": result["snippet"]},
            )
            for result in search_result.data
        ]

        yield ObservationEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            observation=f"Found {len(search_result.data)} relevant sources",
            artifacts=search_artifacts,
        )

        # Step 2: Browse top results
        browsed_content = []

        for i, result in enumerate(search_result.data[:3], 1):  # Browse top 3
            yield StepEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                step_number=i + 1,
                description=f"Browsing: {result['title']}",
                artifacts=[],
            )

            browse_result = await browse_tool.invoke(
                url=result["url"], extract_main_content=True
            )

            if browse_result.success:
                content = browse_result.data["content"][:2000]  # Limit content
                browsed_content.append(
                    {
                        "url": result["url"],
                        "title": result["title"],
                        "content": content,
                    }
                )

                yield ObservationEvent(
                    branch_id=context.branch_id,
                    node_id=task.id,
                    observation=f"Extracted {browse_result.data['word_count']} words",
                    artifacts=[
                        Artifact(
                            type="snippet",
                            content=content,
                            metadata={"source": result["url"]},
                        )
                    ],
                )

        # Step 3: Synthesize findings
        yield StepEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            step_number=5,
            description="Synthesizing research findings",
            artifacts=[],
        )

        # Generate summary from browsed content
        summary_lines = [f"# Research Summary: {task.goal}\n"]

        for item in browsed_content:
            summary_lines.append(f"\n## {item['title']}")
            summary_lines.append(f"Source: {item['url']}")
            summary_lines.append(f"\n{item['content'][:500]}...\n")

        summary = "\n".join(summary_lines)

        yield SummaryEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            summary=summary,
            artifacts=[
                Artifact(
                    type="snippet",
                    content=summary,
                    metadata={"sources": len(browsed_content)},
                )
            ],
        )

        # Complete
        yield CompleteEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            summary=f"Web research completed. Analyzed {len(browsed_content)} sources.",
            artifacts=search_artifacts + [
                Artifact(type="snippet", content=summary, metadata={})
            ],
        )

    async def cancel(self):
        """Cancel ongoing research"""
        self._cancelled = True
        logger.info("DeepResearch task cancelled")

    def supports_task(self, task: Task, context: Context) -> float:
        """
        Score task suitability for DeepResearch.

        Returns:
            0.0-1.0 score (higher = better match)
        """
        goal_lower = task.goal.lower()

        # Strong indicators for web research
        web_keywords = [
            "search",
            "find",
            "research",
            "literature review",
            "papers",
            "articles",
            "information",
            "what is",
            "how does",
            "explain",
        ]

        score = 0.0

        for keyword in web_keywords:
            if keyword in goal_lower:
                score += 0.2

        # Boost for URLs
        if "http://" in task.goal or "https://" in task.goal:
            score += 0.3

        return min(1.0, score)

    async def estimate_cost(self, task: Task, context: Context) -> float:
        """
        Estimate execution cost.

        Returns:
            Estimated cost in USD
        """
        # Estimate: 5 searches + 3 browses + LLM calls
        # Our tools are free (Playwright-based)
        # Only LLM cost for synthesis
        return 0.01  # Minimal cost


# Example usage
async def test_deepresearch_adapter():
    """Test DeepResearch adapter"""
    adapter = DeepResearchAdapter()

    task = Task(
        goal="Find recent papers on neural architecture search",
        context="Looking for papers from 2023-2024",
    )

    context = Context(branch_id="test-branch")

    print(f"Running DeepResearch adapter for: {task.goal}\n")

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
    asyncio.run(test_deepresearch_adapter())
