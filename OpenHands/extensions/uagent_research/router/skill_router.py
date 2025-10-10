"""
Skill Router - Intelligent Task Routing

Routes research tasks to the appropriate agent adapter based on task type.
Uses heuristic rules (upgradeable to learned routing later).
"""

import logging
import re
from typing import Optional, Dict, Any, List
from enum import Enum

from extensions.uagent_research.uagent_research.models.research_tree import Task, Context, ResearchNode, NodeType

logger = logging.getLogger(__name__)


class SkillType(str, Enum):
    """Types of research skills"""
    WEB_RESEARCH = "web_research"  # Web search and browsing
    CODE_RESEARCH = "code_research"  # GitHub repo discovery and analysis
    CODE_EXECUTION = "code_execution"  # Running experiments
    GENERAL_CODING = "general_coding"  # General programming tasks


class SkillRouter:
    """
    Routes tasks to appropriate adapters based on task requirements.

    Routing Strategy:
    1. Web Research (DeepResearch):
       - Contains: "search", "research", "find information", "literature review"
       - Has URLs to visit
       - Needs web data

    2. Code Research (RepoMaster):
       - Contains: "github", "repository", "find implementation", "existing code"
       - Needs code examples
       - Looking for libraries/tools

    3. Code Execution (CodeAct):
       - Contains: "run", "execute", "test", "benchmark", "experiment"
       - Has code to run
       - Needs results

    4. General Coding (CodeAct):
       - Default for coding tasks
       - Write/debug code
       - Refactoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize router.

        Args:
            config: Router configuration
        """
        self.config = config or {}

        # Keyword patterns for each skill type
        self.patterns = {
            SkillType.WEB_RESEARCH: [
                r'\b(search|google|bing|find information|look up|research)\b',
                r'\b(literature review|papers|articles|read about)\b',
                r'\b(what is|how does|explain|understand)\b',
                r'https?://[^\s]+',  # URLs
            ],
            SkillType.CODE_RESEARCH: [
                r'\b(github|repository|repo|find (implementation|code|library))\b',
                r'\b(existing (solution|implementation|code|tool))\b',
                r'\b(open source|package|library|framework)\b',
                r'\b(clone|download repo)\b',
            ],
            SkillType.CODE_EXECUTION: [
                r'\b(run|execute|test|benchmark|experiment|measure)\b',
                r'\b(performance|accuracy|speed|evaluate)\b',
                r'\b(compare|benchmark)\b',
            ],
        }

        # Compile patterns
        self.compiled_patterns = {
            skill: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for skill, patterns in self.patterns.items()
        }

    def route(self, task: Task, context: Context) -> str:
        """
        Route task to appropriate adapter.

        Args:
            task: Task to route
            context: Execution context

        Returns:
            Adapter name ("deepresearch", "repomaster", or "codeact")

        Example:
            router = SkillRouter()
            adapter_name = router.route(
                task=Task(goal="Search for neural architecture search papers"),
                context=Context()
            )
            # Returns: "deepresearch"
        """
        # Score each skill type
        scores = self._score_skills(task, context)

        # Log scores
        logger.info(f"Skill scores for task '{task.goal[:50]}...': {scores}")

        # Select best skill
        best_skill = max(scores.items(), key=lambda x: x[1])[0]

        # Map skill to adapter
        adapter_name = self._skill_to_adapter(best_skill)

        logger.info(f"Routed task to adapter: {adapter_name} (skill: {best_skill})")

        return adapter_name

    def _score_skills(self, task: Task, context: Context) -> Dict[SkillType, float]:
        """
        Score each skill type for the task.

        Args:
            task: Task to score
            context: Execution context

        Returns:
            Dictionary of skill type -> score (0-1)
        """
        scores = {skill: 0.0 for skill in SkillType}

        # Combine goal and context for analysis
        text = task.goal.lower()
        if task.context:
            text += " " + task.context.lower()

        # Add parent node context
        for parent in context.parent_nodes:
            text += " " + parent.content.lower()

        # Score based on pattern matching
        for skill, patterns in self.compiled_patterns.items():
            matches = sum(1 for pattern in patterns if pattern.search(text))
            if matches > 0:
                # Normalize score: more matches = higher score
                scores[skill] = min(1.0, matches * 0.3)  # 0.3 per match, max 1.0

        # Boost based on parent node types
        parent_types = [p.type for p in context.parent_nodes]

        if NodeType.WEB_SEARCH in parent_types:
            scores[SkillType.WEB_RESEARCH] += 0.2

        if NodeType.CODE_SEARCH in parent_types:
            scores[SkillType.CODE_RESEARCH] += 0.2

        if NodeType.EXPERIMENT in parent_types:
            scores[SkillType.CODE_EXECUTION] += 0.2

        # Normalize scores to 0-1
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        # Default to general coding if no clear winner
        if all(score < 0.3 for score in scores.values()):
            scores[SkillType.GENERAL_CODING] = 0.5

        return scores

    def _skill_to_adapter(self, skill: SkillType) -> str:
        """
        Map skill type to adapter name.

        Args:
            skill: Skill type

        Returns:
            Adapter name
        """
        mapping = {
            SkillType.WEB_RESEARCH: "deepresearch",
            SkillType.CODE_RESEARCH: "repomaster",
            SkillType.CODE_EXECUTION: "codeact",
            SkillType.GENERAL_CODING: "codeact",
        }

        return mapping.get(skill, "codeact")

    def suggest_next_actions(self, node: ResearchNode, context: Context) -> List[str]:
        """
        Suggest next actions based on current node.

        Args:
            node: Current research node
            context: Execution context

        Returns:
            List of suggested next actions

        Example:
            suggestions = router.suggest_next_actions(idea_node, context)
            # Returns: ["Search web for papers", "Find GitHub implementations", "Run experiment"]
        """
        suggestions = []

        if node.type == NodeType.IDEA:
            suggestions.append("Search web for related research")
            suggestions.append("Find existing implementations on GitHub")
            suggestions.append("Generate testable hypotheses")

        elif node.type == NodeType.HYPOTHESIS:
            suggestions.append("Design experiment to test hypothesis")
            suggestions.append("Find code/data to support testing")
            suggestions.append("Run preliminary tests")

        elif node.type == NodeType.WEB_SEARCH:
            suggestions.append("Browse top results for details")
            suggestions.append("Extract key findings")
            suggestions.append("Summarize evidence")

        elif node.type == NodeType.CODE_SEARCH:
            suggestions.append("Analyze repository structure")
            suggestions.append("Find entry points and examples")
            suggestions.append("Test code execution")

        elif node.type == NodeType.EXPERIMENT:
            suggestions.append("Analyze results")
            suggestions.append("Compare with baselines")
            suggestions.append("Formulate conclusions")

        return suggestions


# Example usage
def test_skill_router():
    """Test skill router"""
    router = SkillRouter()

    # Test cases
    test_tasks = [
        Task(goal="Search for neural architecture search papers"),
        Task(goal="Find GitHub implementations of NAS algorithms"),
        Task(goal="Run benchmark to compare DARTS vs ENAS"),
        Task(goal="Implement a new attention mechanism"),
        Task(goal="Research the latest developments in transformers"),
    ]

    for task in test_tasks:
        adapter = router.route(task, Context())
        print(f"Task: {task.goal}")
        print(f"  → Routed to: {adapter}\n")


if __name__ == "__main__":
    test_skill_router()
