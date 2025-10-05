"""
Tree Search Orchestrator

Manages parallel research tree exploration using PUCT (AlphaZero-style) scoring.
Coordinates multiple adapters to explore different branches concurrently.
"""

import asyncio
import logging
import math
from typing import Dict, List, Optional, AsyncIterator, Set
from datetime import datetime
from collections import defaultdict
import uuid

from ..uagent_research.models.research_tree import (
    ResearchTree,
    ResearchNode,
    NodeType,
    NodeStatus,
    Task,
    Context,
    Budget,
)
from ..uagent_research.models.events import ResearchEvent, EventType, CompleteEvent
from ..adapters.base.agent_adapter import AgentAdapter, adapter_registry
from ..router.skill_router import SkillRouter
from .event_bus import EventBus

logger = logging.getLogger(__name__)


class TreeSearchOrchestrator:
    """
    Orchestrates parallel research tree exploration.

    Features:
    - PUCT-based node selection (AlphaZero-style)
    - Parallel branch execution with bounded concurrency
    - Budget enforcement (cost, tokens, time)
    - Real-time event streaming via EventBus
    - Graceful cancellation and cleanup

    Example:
        orchestrator = TreeSearchOrchestrator(
            max_parallel=3,
            budget=Budget(max_cost=1.0, max_iterations=10)
        )

        tree = await orchestrator.run(
            goal="Research and implement neural architecture search"
        )
    """

    def __init__(
        self,
        max_parallel: int = 3,
        budget: Optional[Budget] = None,
        router: Optional[SkillRouter] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize tree search orchestrator.

        Args:
            max_parallel: Max parallel branches to execute
            budget: Budget constraints
            router: Skill router for adapter selection
            event_bus: Event bus for streaming
        """
        self.max_parallel = max_parallel
        self.budget = budget or Budget()
        self.router = router or SkillRouter()
        self.event_bus = event_bus or EventBus()

        # Execution state
        self.tree: Optional[ResearchTree] = None
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._cancelled = False
        self._semaphore = asyncio.Semaphore(max_parallel)

        # Statistics
        self.stats = {
            "total_nodes": 0,
            "completed_nodes": 0,
            "failed_nodes": 0,
            "total_cost": 0.0,
            "total_tokens": 0,
            "iterations": 0,
        }

    async def run(
        self,
        goal: str,
        context: Optional[str] = None,
        max_iterations: int = 10,
    ) -> ResearchTree:
        """
        Run research tree exploration.

        Args:
            goal: Research goal
            context: Additional context
            max_iterations: Max PUCT iterations

        Returns:
            Completed research tree

        Example:
            tree = await orchestrator.run(
                goal="Find and test neural architecture search implementations",
                max_iterations=5
            )
        """
        try:
            # Initialize tree
            self.tree = ResearchTree(
                root=ResearchNode(
                    id="root",
                    type=NodeType.ROOT,
                    title="Research Root",
                    content=goal,
                    status=NodeStatus.COMPLETE,
                )
            )

            self._cancelled = False

            logger.info(f"Starting tree search for: {goal}")

            # Main PUCT loop
            for iteration in range(max_iterations):
                if self._cancelled:
                    logger.info("Tree search cancelled")
                    break

                if not await self._check_budget():
                    logger.info("Budget exhausted")
                    break

                self.stats["iterations"] = iteration + 1

                logger.info(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")

                # Select best node using PUCT
                node = self._select_best_node()

                if node is None:
                    logger.info("No more nodes to explore")
                    break

                # Expand node (generate children)
                children = await self._expand_node(node, goal, context)

                if not children:
                    logger.info(f"No children generated for node {node.id}")
                    continue

                # Execute children in parallel
                await self._execute_children_parallel(children)

                # Update tree statistics
                self._update_tree_stats()

            logger.info(f"\nTree search completed: {self.stats}")

            return self.tree

        except Exception as e:
            logger.error(f"Tree search failed: {e}", exc_info=True)
            raise

    def _select_best_node(self) -> Optional[ResearchNode]:
        """
        Select best node to expand using PUCT scoring.

        PUCT Formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Where:
        - Q(s,a): Average value of node
        - P(s,a): Prior probability (confidence)
        - N(s): Parent visits
        - N(s,a): Node visits
        - c: Exploration constant

        Returns:
            Best node to expand, or None
        """
        if not self.tree:
            return None

        # Find expandable nodes (completed but not fully explored)
        expandable_nodes = []

        for node in self.tree.nodes.values():
            if node.type == NodeType.ROOT:
                # Always expandable
                expandable_nodes.append(node)
            elif node.status == NodeStatus.COMPLETE:
                # Check if has unexplored action space
                child_count = len(self.tree.get_children(node.id))
                max_children = self._get_max_children(node.type)

                if child_count < max_children:
                    expandable_nodes.append(node)

        if not expandable_nodes:
            return None

        # Calculate PUCT scores
        best_node = None
        best_score = -float("inf")

        exploration_constant = 1.414  # sqrt(2)

        for node in expandable_nodes:
            # Get parent visits
            parent_id = self.tree.get_parent(node.id)
            parent_visits = 1.0  # Root has 1 visit

            if parent_id:
                parent = self.tree.nodes.get(parent_id)
                parent_visits = parent.visits if parent else 1.0

            # PUCT score
            q_value = node.avg_value
            prior = node.prior
            visits = node.visits

            exploration_term = (
                exploration_constant * prior * math.sqrt(parent_visits) / (1.0 + visits)
            )

            puct_score = q_value + exploration_term

            logger.debug(
                f"Node {node.id}: PUCT={puct_score:.3f} (Q={q_value:.3f}, "
                f"U={exploration_term:.3f}, visits={visits})"
            )

            if puct_score > best_score:
                best_score = puct_score
                best_node = node

        logger.info(
            f"Selected node {best_node.id} (score={best_score:.3f}, type={best_node.type})"
        )

        return best_node

    def _get_max_children(self, node_type: NodeType) -> int:
        """Get maximum children for node type"""
        max_children_map = {
            NodeType.ROOT: 3,  # Generate 3 ideas
            NodeType.IDEA: 2,  # Generate 2 hypotheses per idea
            NodeType.HYPOTHESIS: 1,  # Generate 1 experiment per hypothesis
            NodeType.WEB_SEARCH: 0,  # Leaf node
            NodeType.CODE_SEARCH: 0,  # Leaf node
            NodeType.EXPERIMENT: 0,  # Leaf node
        }

        return max_children_map.get(node_type, 0)

    async def _expand_node(
        self, node: ResearchNode, goal: str, context: Optional[str]
    ) -> List[ResearchNode]:
        """
        Expand node by generating children.

        Args:
            node: Node to expand
            goal: Research goal
            context: Additional context

        Returns:
            List of child nodes
        """
        logger.info(f"Expanding node {node.id} (type={node.type})")

        children = []

        if node.type == NodeType.ROOT:
            # Generate research ideas
            children = [
                ResearchNode(
                    id=f"idea-{i}",
                    type=NodeType.IDEA,
                    title=f"Idea {i+1}: Web Research",
                    content=f"Search web for: {goal}",
                    status=NodeStatus.PENDING,
                    prior=0.8,
                )
                for i in range(2)
            ] + [
                ResearchNode(
                    id=f"idea-2",
                    type=NodeType.IDEA,
                    title="Idea 3: Code Research",
                    content=f"Search GitHub for implementations of: {goal}",
                    status=NodeStatus.PENDING,
                    prior=0.7,
                )
            ]

        elif node.type == NodeType.IDEA:
            # Generate hypotheses
            children = [
                ResearchNode(
                    id=f"{node.id}-hyp-{i}",
                    type=NodeType.HYPOTHESIS,
                    title=f"Hypothesis {i+1}",
                    content=f"Test hypothesis for: {node.content}",
                    status=NodeStatus.PENDING,
                    prior=0.6,
                )
                for i in range(2)
            ]

        elif node.type == NodeType.HYPOTHESIS:
            # Generate experiment
            children = [
                ResearchNode(
                    id=f"{node.id}-exp",
                    type=NodeType.EXPERIMENT,
                    title="Run Experiment",
                    content=f"Execute experiment for: {node.content}",
                    status=NodeStatus.PENDING,
                    prior=0.5,
                )
            ]

        # Add children to tree
        for child in children:
            self.tree.add_node(child, parent_id=node.id)
            self.stats["total_nodes"] += 1

        logger.info(f"Generated {len(children)} children for node {node.id}")

        return children

    async def _execute_children_parallel(self, children: List[ResearchNode]):
        """
        Execute children nodes in parallel with bounded concurrency.

        Args:
            children: Child nodes to execute
        """
        tasks = []

        for child in children:
            task = asyncio.create_task(self._execute_node(child))
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Child {children[i].id} failed: {result}")

    async def _execute_node(self, node: ResearchNode):
        """
        Execute single node using appropriate adapter.

        Args:
            node: Node to execute
        """
        async with self._semaphore:
            try:
                logger.info(f"Executing node {node.id} (type={node.type})")

                node.status = NodeStatus.RUNNING

                # Create task for node
                task = Task(
                    id=node.id,
                    goal=node.content,
                    context=node.title,
                )

                context = Context(
                    branch_id=node.id,
                    parent_nodes=[],
                )

                # Route to adapter
                adapter_name = self.router.route(task, context)
                adapter = adapter_registry.get(adapter_name)

                if not adapter:
                    raise Exception(f"Adapter '{adapter_name}' not found")

                logger.info(f"Routed node {node.id} to adapter: {adapter_name}")

                # Execute via adapter
                events_received = 0

                async for event in adapter.run(task, context):
                    events_received += 1

                    # Publish event to bus
                    await self.event_bus.publish(event)

                    # Update node on completion
                    if event.type == EventType.COMPLETE:
                        node.status = NodeStatus.COMPLETE
                        node.visits += 1
                        node.avg_value = 0.8  # Success value

                        self.stats["completed_nodes"] += 1

                        # Update costs
                        cost = await adapter.estimate_cost(task, context)
                        node.cost = cost
                        self.stats["total_cost"] += cost

                logger.info(
                    f"Node {node.id} completed ({events_received} events received)"
                )

            except Exception as e:
                logger.error(f"Node {node.id} execution failed: {e}", exc_info=True)

                node.status = NodeStatus.FAILED
                node.visits += 1
                node.avg_value = 0.0  # Failure value

                self.stats["failed_nodes"] += 1

    async def _check_budget(self) -> bool:
        """Check if budget allows continuation"""
        if self.budget.max_cost and self.stats["total_cost"] >= self.budget.max_cost:
            logger.warning(f"Cost budget exceeded: {self.stats['total_cost']}")
            return False

        if (
            self.budget.max_iterations
            and self.stats["iterations"] >= self.budget.max_iterations
        ):
            logger.warning(f"Iteration budget exceeded: {self.stats['iterations']}")
            return False

        return True

    def _update_tree_stats(self):
        """Update tree-level statistics"""
        if not self.tree:
            return

        self.tree.stats = {
            "total_nodes": len(self.tree.nodes),
            "total_edges": len(self.tree.edges),
            "max_depth": self.tree.calculate_max_depth(),
            "total_cost": self.stats["total_cost"],
            "total_tokens": self.stats["total_tokens"],
        }

    async def cancel(self):
        """Cancel ongoing tree search"""
        self._cancelled = True

        # Cancel all running tasks
        for task_id, task in self._running_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled task {task_id}")

        logger.info("Tree search cancellation requested")


# Example usage
async def test_tree_orchestrator():
    """Test tree search orchestrator"""
    from ..adapters.deepresearch.adapter import DeepResearchAdapter
    from ..adapters.repomaster.adapter import RepoMasterAdapter
    from ..adapters.codeact.adapter import CodeActAdapter

    # Register adapters
    adapter_registry.register(DeepResearchAdapter())
    adapter_registry.register(RepoMasterAdapter())
    adapter_registry.register(CodeActAdapter())

    # Create orchestrator
    orchestrator = TreeSearchOrchestrator(
        max_parallel=2,
        budget=Budget(max_cost=0.5, max_iterations=3),
    )

    # Run tree search
    tree = await orchestrator.run(
        goal="Research neural architecture search and find implementations",
        max_iterations=2,
    )

    print(f"\n=== Tree Search Results ===")
    print(f"Total nodes: {len(tree.nodes)}")
    print(f"Total edges: {len(tree.edges)}")
    print(f"Max depth: {tree.calculate_max_depth()}")
    print(f"Total cost: ${orchestrator.stats['total_cost']:.3f}")
    print(f"\nTree structure:")

    for node_id, node in tree.nodes.items():
        parent_id = tree.get_parent(node_id)
        indent = "  " * tree.get_node_depth(node_id)
        print(
            f"{indent}- {node.id} ({node.type}): {node.title} [{node.status}]"
        )


if __name__ == "__main__":
    asyncio.run(test_tree_orchestrator())
