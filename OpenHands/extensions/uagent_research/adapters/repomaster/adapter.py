"""
RepoMaster Adapter

Wraps RepoMaster agent for GitHub repository search and code research.
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime
import sys
import os

# Add vendor path to sys.path
vendor_path = os.path.join(os.path.dirname(__file__), "../../vendor/repomaster")
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


class RepoMasterAdapter(AgentAdapter):
    """
    Adapter for RepoMaster agent.

    Capabilities:
    - GitHub repository search
    - Repository analysis and exploration
    - Code understanding via hierarchical navigation
    - Finding existing implementations and solutions

    Best for:
    - Finding code examples
    - Discovering libraries and tools
    - Understanding existing implementations
    - Code-based research
    """

    name = "repomaster"
    description = "Code research agent for GitHub repository discovery"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RepoMaster adapter.

        Args:
            config: Adapter configuration
                - model: LLM model name
                - max_repos: Max repositories to search (default: 5)
        """
        super().__init__(config)

        self.model = config.get("model", "gpt-4o") if config else "gpt-4o"
        self.max_repos = config.get("max_repos", 5) if config else 5

        # Agent state
        self._agent = None
        self._current_task = None
        self._cancelled = False

    async def run(self, task: Task, context: Context) -> AsyncIterator[ResearchEvent]:
        """
        Execute code research task using RepoMaster.

        Args:
            task: Research task
            context: Execution context

        Yields:
            Research events (Plan, Step, ToolCall, Observation, Summary, Complete)

        Example:
            async for event in adapter.run(
                task=Task(goal="Find Python implementations of neural architecture search"),
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
                plan=f"Code research: {task.goal}",
                steps=[
                    "Search GitHub for relevant repositories",
                    "Analyze top repositories",
                    "Extract key implementations",
                    "Generate summary with examples",
                ],
            )

            # Execute research
            async for event in self._execute_code_research(task, context):
                if self._cancelled:
                    yield ErrorEvent(
                        branch_id=context.branch_id,
                        node_id=task.id,
                        message="Task cancelled by user",
                    )
                    return

                yield event

        except Exception as e:
            logger.error(f"RepoMaster execution failed: {e}", exc_info=True)
            yield ErrorEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                message=str(e),
            )

    async def _execute_code_research(
        self, task: Task, context: Context
    ) -> AsyncIterator[ResearchEvent]:
        """
        Execute code research using RepoMaster agent.

        This is a simplified implementation that searches GitHub via web search
        and analyzes repositories.
        """
        from ...tools.search.bing_search_tool import BingSearchTool
        from ...tools.browse.web_browse_tool import WebBrowseTool

        search_tool = BingSearchTool()
        browse_tool = WebBrowseTool()

        # Step 1: Search GitHub
        yield StepEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            step_number=1,
            description=f"Searching GitHub for: {task.goal}",
            artifacts=[],
        )

        # Construct GitHub search query
        github_query = f"site:github.com {task.goal}"

        search_result = await search_tool.invoke(query=github_query, num_results=self.max_repos)

        if not search_result.success:
            yield ErrorEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                message=f"GitHub search failed: {search_result.error}",
            )
            return

        # Filter for actual GitHub repositories
        repo_results = [
            r for r in search_result.data
            if "github.com" in r["url"] and "/blob/" not in r["url"] and "/issues/" not in r["url"]
        ]

        if not repo_results:
            yield ErrorEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                message="No GitHub repositories found",
            )
            return

        # Emit search results
        repo_artifacts = [
            Artifact(
                type="url",
                content=result["url"],
                metadata={"title": result["title"], "snippet": result["snippet"]},
            )
            for result in repo_results
        ]

        yield ObservationEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            observation=f"Found {len(repo_results)} GitHub repositories",
            artifacts=repo_artifacts,
        )

        # Step 2: Analyze top repositories
        analyzed_repos = []

        for i, repo in enumerate(repo_results[:3], 1):  # Analyze top 3
            yield StepEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                step_number=i + 1,
                description=f"Analyzing repository: {repo['title']}",
                artifacts=[],
            )

            # Browse repository page
            browse_result = await browse_tool.invoke(
                url=repo["url"],
                extract_main_content=True,
                include_links=True
            )

            if browse_result.success:
                content = browse_result.data["content"][:1500]  # Limit content

                # Extract README content and key files
                analyzed_repos.append(
                    {
                        "url": repo["url"],
                        "title": repo["title"],
                        "description": repo["snippet"],
                        "content": content,
                        "links": browse_result.data.get("links", [])[:10],
                    }
                )

                yield ObservationEvent(
                    branch_id=context.branch_id,
                    node_id=task.id,
                    observation=f"Analyzed repository: {repo['title']}",
                    artifacts=[
                        Artifact(
                            type="code",
                            content=content,
                            metadata={
                                "source": repo["url"],
                                "language": "python",  # Could be detected
                            },
                        )
                    ],
                )

        # Step 3: Synthesize findings
        yield StepEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            step_number=5,
            description="Synthesizing code research findings",
            artifacts=[],
        )

        # Generate summary from analyzed repos
        summary_lines = [f"# Code Research Summary: {task.goal}\n"]
        summary_lines.append(f"\nFound {len(repo_results)} repositories, analyzed top {len(analyzed_repos)}:\n")

        for i, repo in enumerate(analyzed_repos, 1):
            summary_lines.append(f"\n## {i}. {repo['title']}")
            summary_lines.append(f"URL: {repo['url']}")
            summary_lines.append(f"Description: {repo['description']}")
            summary_lines.append(f"\n### Key Content:")
            summary_lines.append(f"{repo['content'][:400]}...\n")

        summary = "\n".join(summary_lines)

        yield SummaryEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            summary=summary,
            artifacts=[
                Artifact(
                    type="snippet",
                    content=summary,
                    metadata={"repositories": len(analyzed_repos)},
                )
            ],
        )

        # Complete
        yield CompleteEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            summary=f"Code research completed. Analyzed {len(analyzed_repos)} repositories.",
            artifacts=repo_artifacts + [
                Artifact(type="snippet", content=summary, metadata={})
            ],
        )

    async def cancel(self):
        """Cancel ongoing research"""
        self._cancelled = True
        logger.info("RepoMaster task cancelled")

    def supports_task(self, task: Task, context: Context) -> float:
        """
        Score task suitability for RepoMaster.

        Returns:
            0.0-1.0 score (higher = better match)
        """
        goal_lower = task.goal.lower()

        # Strong indicators for code research
        code_keywords = [
            "github",
            "repository",
            "implementation",
            "code",
            "library",
            "framework",
            "package",
            "find code",
            "existing solution",
            "open source",
        ]

        score = 0.0

        for keyword in code_keywords:
            if keyword in goal_lower:
                score += 0.2

        # Boost for explicit GitHub mentions
        if "github.com" in task.goal:
            score += 0.4

        return min(1.0, score)

    async def estimate_cost(self, task: Task, context: Context) -> float:
        """
        Estimate execution cost.

        Returns:
            Estimated cost in USD
        """
        # Estimate: 1 search + 3 repository browses
        # Our tools are free (Playwright-based)
        return 0.0  # Free


# Example usage
async def test_repomaster_adapter():
    """Test RepoMaster adapter"""
    adapter = RepoMasterAdapter()

    task = Task(
        goal="Find Python implementations of neural architecture search algorithms",
        context="Looking for well-documented repositories with examples",
    )

    context = Context(branch_id="test-branch")

    print(f"Running RepoMaster adapter for: {task.goal}\n")

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
    asyncio.run(test_repomaster_adapter())
