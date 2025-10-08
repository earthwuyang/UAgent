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
from ..control.control_bus import ControlBus, ControlMessage

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
        control_bus: Optional[ControlBus] = None,
    ):
        """
        Initialize tree search orchestrator.

        Args:
            max_parallel: Max parallel branches to execute
            budget: Budget constraints
            router: Skill router for adapter selection
            event_bus: Event bus for streaming
            control_bus: Control bus for runtime commands
        """
        self.max_parallel = max_parallel
        self.budget = budget or Budget()
        self.router = router or SkillRouter()
        self.event_bus = event_bus or EventBus()
        self.control_bus = control_bus or ControlBus()

        # Execution state
        self.tree: Optional[ResearchTree] = None
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._cancelled = False
        self._paused = False
        self._semaphore = asyncio.Semaphore(max_parallel)

        # Control state
        self._steer_map: Dict[str, str] = {}  # node_id/branch_id -> steer text
        self._control_task: Optional[asyncio.Task] = None

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
        research_id: Optional[str] = None,
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
            # Initialize tree with research_id; allow caller to override for UI mapping
            if research_id is None:
                import uuid
                research_id = f"research_{uuid.uuid4().hex[:8]}"

            # Create root node
            root_node = ResearchNode(
                id="root",
                type=NodeType.ROOT,
                title="Research Root",
                content=goal,
                status=NodeStatus.COMPLETE,
            )

            # Initialize tree
            self.tree = ResearchTree(research_id=research_id)

            # Add root node to tree
            self.tree.add_node(root_node)

            # Publish initial tree state so UI reflects activity immediately
            self._update_tree_stats()

            self._cancelled = False
            self._paused = False

            logger.info(f"Starting tree search for: {goal} (research_id: {research_id})")

            # Start control loop in background (concurrent with PUCT loop)
            self._control_task = asyncio.create_task(self._control_loop())

            try:
                # Main PUCT loop
                for iteration in range(max_iterations):
                    # Check cancellation
                    if self._cancelled:
                        logger.info("Tree search cancelled")
                        break

                    # Check pause state
                    if self._paused:
                        logger.info("Research paused, waiting for resume...")
                        await asyncio.sleep(1)  # Wait for resume command
                        continue

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

            finally:
                # Cleanup: Cancel control loop
                if self._control_task and not self._control_task.done():
                    self._control_task.cancel()
                    try:
                        await self._control_task
                    except asyncio.CancelledError:
                        pass

                # Unsubscribe from control bus
                self.control_bus.unsubscribe_all(research_id)

                # Clear event log for this experiment
                if self.event_bus and self.tree:
                    try:
                        self.event_bus.clear_event_log(self.tree.research_id)
                        logger.info(f"Cleared event log for {self.tree.research_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clear event log: {e}")

                logger.info("Control loop cleanup complete")

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

        # Publish snapshot immediately after expansion so UI reflects new nodes
        try:
            self._update_tree_stats()
        except Exception:
            logger.debug("Failed to publish tree after expansion", exc_info=True)

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
            self._running_tasks[child.id] = task
            task.add_done_callback(
                lambda t, node_id=child.id: self._running_tasks.pop(node_id, None)
            )
            tasks.append(task)

        try:
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Child {children[i].id} failed: {result}")
        finally:
            for child in children:
                self._running_tasks.pop(child.id, None)

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

                node.adapter = adapter_name
                logger.info(f"Routed node {node.id} to adapter: {adapter_name}")

                await self._deliver_pending_steer(node.id, adapter_name)

                # Execute via adapter
                events_received = 0

                async for event in adapter.run(task, context):
                    events_received += 1

                    # Attach experiment_id to event before publishing
                    if self.tree and self.tree.research_id:
                        # Create shallow copy or update event with experiment_id
                        if hasattr(event, 'copy'):
                            # If Pydantic model with copy method
                            try:
                                event = event.copy(update={'experiment_id': self.tree.research_id})
                            except:
                                # Fallback: set attribute directly
                                event.experiment_id = self.tree.research_id
                        else:
                            # Set attribute directly
                            event.experiment_id = self.tree.research_id

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
            finally:
                self._running_tasks.pop(node.id, None)
                self._steer_map.pop(node.id, None)
                self.event_bus.stop_branch_heartbeat(node.id)

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

    async def _deliver_pending_steer(self, node_id: str, adapter_name: str) -> None:
        """Send any queued steering directives to the resolved adapter."""

        for key in (node_id, adapter_name):
            steer_text = self._steer_map.pop(key, None)
            if not steer_text:
                continue

            delivered = await self._send_adapter_message(adapter_name, steer_text)
            if delivered:
                logger.info(
                    f"Delivered steering message to adapter {adapter_name}: {steer_text[:100]}"
                )
                break
            # If delivery failed, restore directive for future attempts
            self._steer_map[key] = steer_text

    async def _send_adapter_message(self, adapter_name: str, message: str) -> bool:
        """Forward steering message to adapter if supported."""

        adapter = adapter_registry.get(adapter_name)
        if not adapter:
            logger.warning(f"Adapter '{adapter_name}' not found for steering message")
            return False

        if not hasattr(adapter, "send_message"):
            logger.debug(f"Adapter '{adapter_name}' does not support runtime messaging")
            return False

        try:
            await adapter.send_message(message)
            return True
        except Exception as exc:
            logger.error(
                f"Failed to deliver steering message to adapter {adapter_name}: {exc}",
                exc_info=True,
            )
            return False

    def _publish_tree_to_api(self):
        """Publish tree state to API endpoint for frontend consumption"""
        if not self.tree:
            return

        try:
            # Import the server-exposed API route that backs the frontend
            # Use absolute import to ensure we hit the package used by app.py
            from uagent_research.api.research_routes import update_tree_state

            # Serialize tree to dict
            tree_snapshot = {
                "version": self.stats.get("iterations", 0),
                "timestamp": datetime.now().isoformat(),
                "experiment_id": self.tree.research_id,
                "data": {
                    "nodes": [
                        {
                            "id": node.id,
                            "type": node.type.value,
                    "title": node.title,
                    "content": node.content,
                    "parent_id": node.parent_id,
                    "status": node.status.value,
                    "prior": node.prior,
                    "visits": node.visits,
                    "avg_value": node.avg_value,
                    "cost": node.cost,
                        }
                        for node in self.tree.nodes.values()
                    ],
                    "edges": [
                        {
                            "source": edge.parent_id,
                            "target": edge.child_id,
                            "type": edge.relation,
                        }
                        for edge in self.tree.edges
                    ],
                    "stats": self.tree.stats if hasattr(self.tree, "stats") else {},
                }
            }

            # Update API endpoint
            update_tree_state(self.tree.research_id, tree_snapshot)
            logger.debug(f"Published tree state for experiment {self.tree.research_id}")

        except Exception as e:
            logger.error(f"Failed to publish tree state: {e}", exc_info=True)

    def _update_tree_stats(self):
        """Update tree-level statistics"""
        if not self.tree:
            return

        # Update stats while preserving existing fields like "created" and "expanded"
        self.tree.stats.update({
            "total_nodes": len(self.tree.nodes),
            "total_edges": len(self.tree.edges),
            "max_depth": self.tree.calculate_max_depth(),
            "total_cost": self.stats["total_cost"],
            "total_tokens": self.stats["total_tokens"],
        })

        # Publish tree state to API endpoint
        self._publish_tree_to_api()

    async def _control_loop(self):
        """
        Process control commands in background.

        Listens for control messages via ControlBus and applies them
        to the running research tree.

        Supported actions:
        - pause: Pause research (stop scheduling new nodes)
        - resume: Resume paused research
        - cancel: Cancel entire experiment
        - cancel_node: Cancel specific branch/node
        - reprioritize: Adjust node priorities
        - steer: Send guidance to running adapters
        """
        if not self.tree:
            return

        logger.info(f"Starting control loop for research {self.tree.research_id}")

        try:
            async for cmd in self.control_bus.subscribe(self.tree.research_id):
                logger.info(
                    f"Received control command: {cmd.action} "
                    f"(target={cmd.target}, sender={cmd.sender})"
                )

                if cmd.action == "pause":
                    self._paused = True
                    logger.info(f"Paused research: {self.tree.research_id}")

                elif cmd.action == "resume":
                    self._paused = False
                    logger.info(f"Resumed research: {self.tree.research_id}")

                elif cmd.action == "cancel":
                    self._cancelled = True
                    # Cancel all running tasks
                    for node_id, task in list(self._running_tasks.items()):
                        if not task.done():
                            task.cancel()
                            logger.info(f"Cancelled task for node {node_id}")
                            self.event_bus.stop_branch_heartbeat(node_id)
                    logger.info(f"Cancelled research: {self.tree.research_id}")

                elif cmd.action == "cancel_node":
                    node_id = cmd.target.get("node_id")
                    if node_id and node_id in self._running_tasks:
                        task = self._running_tasks[node_id]
                        if not task.done():
                            task.cancel()
                            # Mark node as cancelled in tree
                            if node_id in self.tree.nodes:
                                self.tree.nodes[node_id].status = NodeStatus.FAILED
                            logger.info(f"Cancelled node: {node_id}")
                            self.event_bus.stop_branch_heartbeat(node_id)

                elif cmd.action == "reprioritize":
                    # Adjust node priors
                    delta = cmd.payload.get("delta", 0.1)
                    target_adapter = cmd.target.get("adapter")
                    target_node_type = cmd.target.get("node_type")

                    # Update priors for matching nodes
                    for node in self.tree.nodes.values():
                        if node.status == NodeStatus.PENDING:
                            # Check if node matches target criteria
                            if target_adapter and hasattr(node, 'adapter'):
                                if node.adapter == target_adapter:
                                    node.prior = min(1.0, node.prior + delta)
                            elif target_node_type:
                                if node.type.value == target_node_type:
                                    node.prior = min(1.0, node.prior + delta)

                    logger.info(
                        f"Reprioritized nodes: adapter={target_adapter}, "
                        f"type={target_node_type}, delta={delta}"
                    )

                elif cmd.action == "steer":
                    # Store steering directive
                    target_id = (
                        cmd.target.get("node_id") or
                        cmd.target.get("branch_id") or
                        cmd.target.get("adapter")
                    )
                    steer_text = cmd.payload.get("text", "")

                    if target_id:
                        self._steer_map[target_id] = steer_text

                        delivered = False
                        if target_id in self._running_tasks:
                            node = self.tree.nodes.get(target_id)
                            adapter_name = getattr(node, "adapter", None) if node else None
                            if adapter_name:
                                delivered = await self._send_adapter_message(adapter_name, steer_text)
                                if delivered:
                                    self._steer_map.pop(target_id, None)

                        adapter_target = cmd.target.get("adapter")
                        if not delivered and adapter_target:
                            delivered = await self._send_adapter_message(adapter_target, steer_text)
                            if delivered:
                                self._steer_map.pop(target_id, None)

                        if delivered:
                            logger.info(
                                f"Delivered steer directive for {target_id}: {steer_text[:100]}"
                            )
                        else:
                            logger.info(
                                f"Stored steer directive for {target_id}: {steer_text[:100]}"
                            )

                elif cmd.action == "add_node":
                    # Add new research direction
                    parent_id = cmd.payload.get("parent_id", "root")
                    node_data = cmd.payload.get("node", {})

                    if parent_id in self.tree.nodes:
                        new_node = ResearchNode(
                            id=f"manual-{uuid.uuid4().hex[:8]}",
                            type=NodeType[node_data.get("type", "IDEA")],
                            title=node_data.get("title", "Manual node"),
                            content=node_data.get("content", ""),
                            status=NodeStatus.PENDING,
                            prior=node_data.get("prior", 0.5),
                        )
                        self.tree.add_node(new_node, parent_id=parent_id)
                        logger.info(f"Added manual node {new_node.id} under {parent_id}")
                        logger.info(
                            "Manual node will be evaluated in the next PUCT iteration"
                        )

        except asyncio.CancelledError:
            logger.info(f"Control loop cancelled for {self.tree.research_id}")
        except Exception as e:
            logger.error(f"Error in control loop: {e}", exc_info=True)

    async def cancel(self):
        """Cancel ongoing tree search"""
        self._cancelled = True

        # Cancel control loop
        if self._control_task and not self._control_task.done():
            self._control_task.cancel()

        # Cancel all running tasks
        for task_id, task in self._running_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled task {task_id}")

        self._running_tasks.clear()
        self._steer_map.clear()

        if self.tree:
            for node in self.tree.nodes.values():
                self.event_bus.stop_branch_heartbeat(node.id)

        # Clear event log
        if self.event_bus and self.tree:
            try:
                self.event_bus.clear_event_log(self.tree.research_id)
                logger.info(f"Cleared event log for {self.tree.research_id}")
            except Exception as e:
                logger.warning(f"Failed to clear event log: {e}")

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
