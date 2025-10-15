"""
Tree Search Orchestrator

Manages parallel research tree exploration using PUCT (AlphaZero-style) scoring.
Coordinates multiple adapters to explore different branches concurrently.
"""

import asyncio
import logging
import math
from typing import Dict, List, Optional, AsyncIterator, Set, Any
from datetime import datetime
from collections import defaultdict, Counter
import traceback
import uuid

from ..uagent_research.models.research_tree import (
    ResearchTree,
    ResearchNode,
    NodeType,
    NodeStatus,
    Task,
    Context,
    Budget,
    ExperimentContext,
)
from ..uagent_research.models.events import ResearchEvent, EventType, CompleteEvent
from ..adapters.base.agent_adapter import AgentAdapter, adapter_registry
from ..router.skill_router import SkillRouter
from .event_bus import EventBus
from ..control.control_bus import ControlBus, ControlMessage
# Delayed import to avoid circular dependency
# from openhands.events.agent_event import ProgressUpdateEvent, NodeCompleteEvent, CommandEvent
from ..services.idea_generation_service import IdeaGenerationService

# Import SubAgent observation events for experiment tracking
try:
    from openhands.events.observation.sub_agent import SubAgentSpawnedObservation
    SUB_AGENT_EVENTS_AVAILABLE = True
except ImportError:
    SUB_AGENT_EVENTS_AVAILABLE = False
    logger.warning("SubAgent observation events not available")


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
        llm = None,
        idea_service: Optional[IdeaGenerationService] = None,
        message_bus: Optional[Any] = None,  # MessageBus for inter-agent communication
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

        # MessageBus for inter-agent communication
        self.message_bus = message_bus

        # Intelligent node expansion
        self.llm = llm
        self.idea_service = idea_service
        if self.idea_service is None and self.llm is not None:
            try:
                from ..config import (
                    ENABLE_INTELLIGENT_EXPANSION,
                    MAX_RESEARCH_IDEAS,
                    MAX_HYPOTHESES_PER_IDEA,
                    MAX_EXPERIMENTS_PER_HYPOTHESIS,
                    IDEA_GENERATION_RETRY_COUNT,
                )
                if ENABLE_INTELLIGENT_EXPANSION:
                    config = {
                        'max_ideas': MAX_RESEARCH_IDEAS,
                        'max_hypotheses': MAX_HYPOTHESES_PER_IDEA,
                        'max_experiments': MAX_EXPERIMENTS_PER_HYPOTHESIS,
                        'retry_count': IDEA_GENERATION_RETRY_COUNT,
                    }
                    self.idea_service = IdeaGenerationService(llm=self.llm, config=config)
                    logger.info("Created IdeaGenerationService for intelligent node expansion")
            except Exception as e:
                logger.warning(f"Failed to create IdeaGenerationService: {e}")
                self.idea_service = None
        
        self.use_intelligent_expansion = self.idea_service is not None
        if self.use_intelligent_expansion:
            logger.info("Intelligent node expansion ENABLED")
        else:
            logger.warning("Intelligent node expansion DISABLED (using fallback placeholders)")

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
            "running_nodes": 0,
            "pending_nodes": 0,
            "total_cost": 0.0,
            "total_tokens": 0,
            "iterations": 0,
        }

        # Debounce for broadcast flood prevention (500ms threshold)
        self._last_broadcast_ts = 0.0
        self._broadcast_debounce_ms = 500

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
            normalized_iterations = None
            if max_iterations is not None:
                if isinstance(max_iterations, str):
                    max_iterations = max_iterations.strip()
                normalized_iterations = int(max_iterations)
        except (TypeError, ValueError):
            normalized_iterations = None

        if normalized_iterations is None or normalized_iterations <= 0:
            fallback_iterations = getattr(self.budget, "max_iterations", None)
            try:
                if fallback_iterations is not None:
                    normalized_iterations = int(fallback_iterations)
            except (TypeError, ValueError):
                normalized_iterations = None

        if normalized_iterations is None or normalized_iterations <= 0:
            normalized_iterations = 1

        max_iterations = normalized_iterations
        self._max_iterations = max_iterations
        if self.budget:
            self.budget.max_iterations = max_iterations

        logger.info(f"[ORCHESTRATOR] run() called with goal={goal[:100]}, max_iterations={max_iterations}, research_id={research_id}")
        logger.info(f"[ORCHESTRATOR] Budget: max_cost={self.budget.max_cost}, max_iterations={self.budget.max_iterations}")
        logger.info(f"[ORCHESTRATOR] Concurrency: max_parallel={self.max_parallel}")
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
                status=NodeStatus.RUNNING,
                started_at=datetime.utcnow(),
            )

            # Initialize tree
            self.tree = ResearchTree(research_id=research_id)

            # Add root node to tree
            self.tree.add_node(root_node)

            # Seed statistics with the root node so UI reflects active work
            self.stats["total_nodes"] = 1
            self.stats["completed_nodes"] = 0
            self.stats["failed_nodes"] = 0
            self.stats["running_nodes"] = 1
            self.stats["pending_nodes"] = 0

            # Register orchestrator with MessageBus
            if self.message_bus:
                self.message_bus.register_agent(
                    agent_id=research_id,
                    agent_type="research_orchestrator",
                    capabilities=["tree_search", "parallel_execution", "budget_management"]
                )

            # Publish initial tree state so UI reflects activity immediately
            self._update_tree_stats(force=True)

            self._cancelled = False
            self._paused = False


            logger.info(f"Starting tree search for: {goal} (research_id: {research_id})")
            # Ensure adapters are registered
            from ..adapters.ensure_adapters import ensure_research_adapters_registered
            ensure_research_adapters_registered()

            
            logger.info(f"[ORCHESTRATOR] Starting PUCT loop for {research_id}")
            
            # Verify API imports are working
            try:
                from ..uagent_research.api.tree_publisher import update_tree_state, broadcast_tree_update
                logger.info("[ORCHESTRATOR] ✅ API imports verified successfully")
            except ImportError as e:
                logger.error(f"[ORCHESTRATOR] ❌ API imports FAILED: {e}", exc_info=True)
                logger.error("[ORCHESTRATOR] Tree updates will NOT be published to UI!")

            from ..adapters.base.agent_adapter import adapter_registry
            if hasattr(adapter_registry, '_adapters'):
                registered = list(adapter_registry._adapters.keys())
            else:
                registered = 'unknown'
            logger.info(f"[ORCHESTRATOR] Adapter registry status: {registered}")

            # DIAGNOSTIC: Log adapter registry state
            registered_adapters = list(adapter_registry.get_all_adapters())
            logger.info(f"[DIAGNOSTIC] Adapter Registry State:")
            logger.info(f"[DIAGNOSTIC] Total registered adapters: {len(registered_adapters)}")
            if len(registered_adapters) == 0:
                logger.error(f"[DIAGNOSTIC] ERROR: No adapters registered! Parallel research will fail.")
            else:
                for adapter in registered_adapters:
                    logger.info(f"[DIAGNOSTIC]   - {adapter.name}: {adapter.description}")


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
                    
                    logger.info(f"[ORCHESTRATOR] PUCT iteration {iteration + 1}/{max_iterations}")
                    logger.info(f"[ORCHESTRATOR] Tree state: {len(self.tree.nodes)} nodes, {len(self.tree.edges)} edges")
                    logger.info(f"[ORCHESTRATOR] Budget check: cost={self.stats['total_cost']:.3f}/{self.budget.max_cost}, iterations={iteration + 1}/{max_iterations}")

                    # DIAGNOSTIC: Log iteration state
                    logger.info(f"[DIAGNOSTIC] PUCT Iteration {iteration + 1}/{max_iterations}")
                    logger.info(f"[DIAGNOSTIC] Tree state: {len(self.tree.nodes)} nodes, {len(self.tree.edges)} edges")
                    logger.info(f"[DIAGNOSTIC] Budget: cost=${self.stats['total_cost']:.3f}, tokens={self.stats['total_tokens']}, iterations={self.stats['iterations']}")


                    # Select best node using PUCT
                    logger.info(f"[ORCHESTRATOR] Selecting best node for expansion...")
                    node = self._select_best_node()
                    if node:
                        logger.info(f"[ORCHESTRATOR] Selected node: {node.id} (type={node.type}, visits={node.visits})")
                    else:
                        logger.warning(f"[ORCHESTRATOR] No expandable node found! Tree may be exhausted.")

                    if node is None:
                        logger.info("No more nodes to explore")
                        break

                    # Expand node (generate children)
                    logger.info(f"[ORCHESTRATOR] Expanding node {node.id}...")
                    children = await self._expand_node(node, goal, context)
                    logger.info(f"[ORCHESTRATOR] Expansion generated {len(children)} children")
                    for i, child in enumerate(children):
                        logger.info(f"[ORCHESTRATOR]   - Child {i+1}: {child.id} (type={child.type})")

                    if not children:
                        logger.info(f"No children generated for node {node.id}")
                        continue

                    # Execute children in parallel
                    logger.info(f"[ORCHESTRATOR] Starting parallel execution of {len(children)} children (max_parallel={self.max_parallel})")
                    await self._execute_children_parallel(children)
                    logger.info(f"[ORCHESTRATOR] Parallel execution completed")

                    # Update tree statistics
                    self._update_tree_stats(force=True)

                logger.info(f"[ORCHESTRATOR] PUCT loop finished")
                logger.info(f"[ORCHESTRATOR] Final tree: {len(self.tree.nodes)} nodes, {len(self.tree.edges)} edges")
                logger.info(f"[ORCHESTRATOR] Final cost: ${self.stats['total_cost']:.3f}")
                logger.info(f"\nTree search completed: {self.stats}")

                # Mark root node as complete so UI reflects finished state
                if self.tree and "root" in self.tree.nodes:
                    root_node = self.tree.nodes["root"]
                    if root_node.status != NodeStatus.COMPLETE:
                        root_node.status = NodeStatus.COMPLETE
                        root_node.completed_at = datetime.utcnow()
                    if root_node.visits <= 0:
                        root_node.visits = 1

                # Ensure final tree snapshot is published
                self._update_tree_stats(force=True)

            finally:
                # Cleanup: Cancel control loop
                if self._control_task and not self._control_task.done():
                    self._control_task.cancel()
                    try:
                        await self._control_task
                    except asyncio.CancelledError:
                        pass

                # Unregister from MessageBus
                if self.message_bus and self.tree:
                    self.message_bus.unregister_agent(self.tree.research_id)

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
            # Apply max_children check to all nodes including ROOT
            # IDEA nodes should be expandable immediately (don't wait for COMPLETE)
            # because they're conceptual and need to generate HYPOTHESIS + EXPERIMENT children
            if (node.status == NodeStatus.COMPLETE or 
                node.type == NodeType.ROOT or 
                node.type == NodeType.IDEA):
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

        # DIAGNOSTIC: Log node selection details
        if best_node:
            logger.info(f"[DIAGNOSTIC] Node selected: ID={best_node.id}, Type={best_node.type}, Status={best_node.status}")
            logger.info(f"[DIAGNOSTIC]   PUCT score={best_score:.3f}, Children={len(self.tree.get_children(best_node.id))}/{self._get_max_children(best_node.type)}")
        else:
            node_statuses = {}
            for node in self.tree.nodes.values():
                status = node.status.value if hasattr(node.status, 'value') else str(node.status)
                node_statuses[status] = node_statuses.get(status, 0) + 1
            logger.warning(f"[DIAGNOSTIC] No node selected! Total nodes: {len(self.tree.nodes)}, Statuses: {node_statuses}")

        return best_node

    def _get_max_children(self, node_type: NodeType) -> int:
        """
        Get maximum children for node type, reading from config.
        
        Supports new sibling structure:
        - ROOT: generates IDEAS
        - IDEA: generates HYPOTHESES + EXPERIMENTS as siblings (not chain)
        - HYPOTHESIS: leaf node (no children in new structure)
        """
        # Try to import config values
        try:
            from ..config import (
                MAX_RESEARCH_IDEAS,
                MAX_HYPOTHESES_PER_IDEA,
                MAX_EXPERIMENTS_PER_IDEA,
                MAX_EXPERIMENTS_PER_HYPOTHESIS,  # Deprecated, kept for backward compat
            )
            max_ideas = MAX_RESEARCH_IDEAS
            max_hypotheses = MAX_HYPOTHESES_PER_IDEA
            # Use new constant if available, fallback to old for backward compatibility
            max_experiments_per_idea = MAX_EXPERIMENTS_PER_IDEA
        except ImportError:
            # Fallback to defaults if config not available
            max_ideas = 3
            max_hypotheses = 2
            max_experiments_per_idea = 2
        
        max_children_map = {
            NodeType.ROOT: max_ideas,
            # IDEA generates both hypotheses and experiments as siblings
            NodeType.IDEA: max_hypotheses + max_experiments_per_idea,
            # HYPOTHESIS is now a leaf node (no children)
            NodeType.HYPOTHESIS: 0,
            NodeType.WEB_SEARCH: 0,  # Leaf node
            NodeType.CODE_SEARCH: 0,  # Leaf node
            NodeType.EXPERIMENT: 0,  # Leaf node
        }

        return max_children_map.get(node_type, 0)


    def _generate_placeholder_children(self, node: ResearchNode, goal: str) -> List[ResearchNode]:
        """Generate placeholder children when LLM unavailable."""
        children = []
        
        if node.type == NodeType.ROOT:
            # Generate 3 research ideas
            logger.info(f"Generating 3 placeholder research ideas for root node")
            for i in range(1, 4):
                child_id = f"{node.id}_idea_{i}"
                children.append(ResearchNode(
                    id=child_id,
                    type=NodeType.IDEA,
                    title=f"Research Idea {i}: {goal[:50]}",
                    content=f"Explore approach {i} for: {goal}",
                    status=NodeStatus.PENDING,
                    parent_id=node.id
                ))
        elif node.type == NodeType.IDEA:
            # Generate 2 hypotheses
            logger.info(f"Generating 2 placeholder hypotheses for idea node {node.id}")
            for i in range(1, 3):
                child_id = f"{node.id}_hyp_{i}"
                children.append(ResearchNode(
                    id=child_id,
                    type=NodeType.HYPOTHESIS,
                    title=f"Hypothesis {i} for {node.title}",
                    content=f"Test hypothesis {i}",
                    status=NodeStatus.PENDING,
                    parent_id=node.id
                ))
        elif node.type == NodeType.HYPOTHESIS:
            # Generate 1 experiment
            logger.info(f"Generating 1 placeholder experiment for hypothesis node {node.id}")
            child_id = f"{node.id}_exp_1"
            children.append(ResearchNode(
                id=child_id,
                type=NodeType.EXPERIMENT,
                title=f"Experiment for {node.title}",
                content=f"Run experiment to validate hypothesis",
                status=NodeStatus.PENDING,
                parent_id=node.id
            ))
        
        logger.info(f"Generated {len(children)} placeholder children for {node.id}")
        return children

    async def _expand_node(
        self, node: ResearchNode, goal: str, context: Optional[str]
    ) -> List[ResearchNode]:
        """
        Expand node by generating children.

        Uses IdeaGenerationService for intelligent LLM-based expansion when available,
        falls back to hardcoded placeholders otherwise.

        Args:
            node: Node to expand
            goal: Research goal
            context: Additional context

        Returns:
            List of child nodes
        """
        logger.info(f"[EXPAND] Expanding node {node.id} (type={node.type})")
        logger.info(f"[EXPAND] Intelligent expansion enabled: {self.use_intelligent_expansion}")
        
        logger.info(f"Expanding node {node.id} (type={node.type})")

        # DIAGNOSTIC: Log node expansion details
        logger.info(f"[DIAGNOSTIC] Expanding node: ID={node.id}, Type={node.type}, Title='{node.title}'")
        logger.info(f"[DIAGNOSTIC]   Content preview: {(node.content or '')[:100]}...")
        logger.info(f"[DIAGNOSTIC]   Intelligent expansion enabled: {self.use_intelligent_expansion}")

        children = []

        # Try intelligent expansion first
        if self.use_intelligent_expansion:
            logger.info(f"[EXPAND] Using LLM to generate children for {node.id}")
            try:
                if node.type == NodeType.ROOT:
                    logger.info(f"Using intelligent expansion for ROOT node")
                    children = await self.idea_service.generate_ideas(
                        goal=goal,
                        context=context
                    )
                    if children:
                        logger.info(f"Intelligent expansion generated {len(children)} ideas")
                    else:
                        logger.warning("Intelligent expansion returned no ideas, using fallback")
                
                elif node.type == NodeType.IDEA:
                    logger.info(f"Using intelligent expansion for IDEA node: {node.title[:50]}")
                    # Generate hypotheses first
                    hypotheses = await self.idea_service.generate_hypotheses(
                        idea_content=node.content,
                        parent_node=node
                    )
                    if hypotheses:
                        logger.info(f"Intelligent expansion generated {len(hypotheses)} hypotheses")
                    else:
                        logger.warning("Intelligent expansion returned no hypotheses, using fallback")
                        hypotheses = []
                    
                    # Generate experiments as siblings of hypotheses
                    experiments = await self.idea_service.generate_experiments_for_idea(
                        idea_content=node.content,
                        parent_node=node,
                        hypotheses=hypotheses
                    )
                    if experiments:
                        logger.info(f"Intelligent expansion generated {len(experiments)} experiments")
                    else:
                        logger.warning("Intelligent expansion returned no experiments, using fallback")
                        experiments = []
                    
                    # Combine hypotheses and experiments as siblings
                    children = hypotheses + experiments
                    logger.info(f"IDEA node will have {len(hypotheses)} hypotheses + {len(experiments)} experiments = {len(children)} children")
                
                elif node.type == NodeType.HYPOTHESIS:
                    # HYPOTHESIS nodes are now leaf nodes (no children)
                    logger.info(f"HYPOTHESIS node {node.id} is a leaf node, no children generated")
                    children = []
            
            except Exception as e:
                logger.error(f"Intelligent expansion failed: {e}", exc_info=True)
                logger.warning("Falling back to hardcoded node generation")
                children = []

        # Fallback to hardcoded expansion if intelligent expansion unavailable or failed
        if not children:
            logger.info(f"Using fallback placeholder expansion for {node.type}")
            
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
                # Generate hypotheses as siblings
                hypotheses = [
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
                
                # Generate experiments as siblings (not children of hypotheses)
                # Collect hypothesis IDs for metadata
                hypothesis_ids = [h.id for h in hypotheses]
                
                experiments = []
                for i in range(2):  # Generate 2 experiments per IDEA
                    metadata = {
                        'parent_hypotheses': hypothesis_ids,
                        'parent_idea': node.id,
                        'num_hypotheses': len(hypotheses)
                    }
                    experiments.append(ResearchNode(
                        id=f"{node.id}-exp-{i}",
                        type=NodeType.EXPERIMENT,
                        title=f"Experiment {i+1}",
                        content=f"Execute experiment to test all hypotheses of: {node.content}",
                        status=NodeStatus.PENDING,
                        prior=0.5,
                        metadata=metadata
                    ))
                
                # Combine as siblings
                children = hypotheses + experiments
                logger.info(f"Fallback: Generated {len(hypotheses)} hypotheses + {len(experiments)} experiments = {len(children)} children")

            elif node.type == NodeType.HYPOTHESIS:
                # HYPOTHESIS nodes are now leaf nodes (no children)
                logger.info(f"HYPOTHESIS node {node.id} is a leaf node in new structure")
                children = []

        # Add children to tree
        for child in children:
            self.tree.add_node(child, parent_id=node.id)
            self.stats["total_nodes"] += 1

        logger.info(f"Generated {len(children)} children for node {node.id}")

        # Publish snapshot immediately after expansion so UI reflects new nodes
        try:
            self._update_tree_stats(force=True)
        except Exception:
            logger.debug("Failed to publish tree after expansion", exc_info=True)

        return children

    async def _execute_children_parallel(self, children: List[ResearchNode]):
        """
        Execute children nodes in parallel with bounded concurrency.

        Args:
            children: Child nodes to execute
        """
        
        logger.info(f"[EXECUTE] Starting parallel execution of {len(children)} children")
        logger.info(f"[EXECUTE] Concurrency limit: {self.max_parallel}")
        # DIAGNOSTIC: Log parallel execution start
        logger.info(f"[DIAGNOSTIC] Starting parallel execution of {len(children)} children")
        for child in children:
            logger.info(f"[DIAGNOSTIC]   Queuing: ID={child.id}, Type={child.type}, Title='{child.title}'")
        
        tasks = []

        for child in children:
            task = asyncio.create_task(self._execute_node(child))
            self._running_tasks[child.id] = task
            task.add_done_callback(
                lambda t, node_id=child.id: self._running_tasks.pop(node_id, None)
            )
            tasks.append(task)


        # DIAGNOSTIC: Log task creation
        logger.info(f"[DIAGNOSTIC] Created {len(tasks)} asyncio tasks for parallel execution")
        logger.info(f"[DIAGNOSTIC] Active task IDs: {list(self._running_tasks.keys())}")
        
        try:
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info(f"[EXECUTE] All {len(tasks)} tasks completed")

            # DIAGNOSTIC: Log gather results
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            failure_count = len(results) - success_count
            logger.info(f"[DIAGNOSTIC] Parallel execution complete: {success_count} succeeded, {failure_count} failed")
            
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
        # Initialize context to None for finally block access
        context = None
        
        async with self._semaphore:
            try:
                logger.info(f"Executing node {node.id} (type={node.type})")


                # DIAGNOSTIC: Log node execution details
                logger.info(f"[DIAGNOSTIC] Executing node: ID={node.id}, Type={node.type}")
                logger.info(f"[DIAGNOSTIC]   Goal/Title: {node.title}")
                logger.info(f"[DIAGNOSTIC]   Acquiring semaphore (max_parallel={self.max_parallel})")
                
                node.status = NodeStatus.RUNNING
                node.started_at = datetime.utcnow()

                # Initialize per-node event stream (Issue #24 - UAGENT-24-2)
                if self.event_bus:
                    self.event_bus.init_node_stream(node.id)
                    logger.debug(f"[NODE_EVENTS] Initialized event stream for node {node.id}")

                # Generate virtual conversation ID for EXPERIMENT nodes
                if node.type == NodeType.EXPERIMENT:
                    # Generate unique conversation ID for this experiment
                    experiment_id = f"exp_{self.tree.research_id}_{node.id[:8]}"
                    worktree_branch = f"exp_{experiment_id}"
                    
                    # Initialize node.metadata if not exists
                    if not hasattr(node, 'metadata') or node.metadata is None:
                        node.metadata = {}
                    
                    # Store conversation context in node metadata for UI access
                    node.metadata['conversation_id'] = experiment_id
                    node.metadata['worktree_branch'] = worktree_branch
                    node.metadata['parent_session_id'] = self.tree.research_id
                    
                    logger.info(f"[EXPERIMENT] Generated virtual conversation ID: {experiment_id}")
                    logger.info(f"[EXPERIMENT] Worktree branch: {worktree_branch}")
                    
                    # Emit SubAgentSpawnedObservation event if available
                    if SUB_AGENT_EVENTS_AVAILABLE and self.event_bus:
                        try:
                            spawn_event = SubAgentSpawnedObservation(
                                content=f"Experiment {node.title} started",
                                sub_agent_id=experiment_id,
                                sub_agent_type='experiment',
                                goal=node.content or node.title,
                                session_id=self.tree.research_id
                            )
                            # Publish event to event bus
                            await self.event_bus.publish(spawn_event)
                            logger.info(f"[EXPERIMENT] SubAgentSpawnedObservation emitted for {experiment_id}")
                        except Exception as e:
                            logger.warning(f"Failed to emit SubAgentSpawnedObservation: {e}")

                # Create task for node
                task = Task(
                    id=node.id,
                    goal=node.content,
                    context=node.title,
                )

                # Create experiment context for EXPERIMENT nodes
                experiment_context = None
                if node.type == NodeType.EXPERIMENT:
                    # Extract experiment metadata created earlier
                    experiment_id = node.metadata.get('conversation_id')
                    worktree_branch = node.metadata.get('worktree_branch')
                    
                    if experiment_id and worktree_branch:
                        # Determine parent branch (try to get from config or use default)
                        parent_branch = self.config.get('parent_branch', 'main') if hasattr(self, 'config') and isinstance(self.config, dict) else 'main'
                        
                        # Create ExperimentContext
                        experiment_context = ExperimentContext(
                            conversation_id=experiment_id,
                            worktree_branch=worktree_branch,
                            worktree_path=f"../worktrees/{worktree_branch}",
                            parent_branch=parent_branch
                        )
                        
                        logger.info(f"[EXPERIMENT] Created ExperimentContext: worktree_path={experiment_context.worktree_path}")

                # Create context with optional experiment_context
                # Include node_type in metadata for router to make routing decisions
                context = Context(
                    branch_id=node.id,
                    parent_nodes=[],
                    experiment_context=experiment_context,
                    metadata={'node_type': node.type}
                )

                # Route to adapter
                logger.info(f"[EXECUTE] Routing task for node {node.id} (type={node.type})")
                adapter_name = self.router.route(task, context)
                logger.info(f"[EXECUTE] Router selected adapter: {adapter_name} for node {node.id}")
                
                adapter = adapter_registry.get(adapter_name)

                if not adapter:
                    logger.error(f"❌ CRITICAL: Adapter '{adapter_name}' not found in registry!")
                    available = list(adapter_registry._adapters.keys()) if hasattr(adapter_registry, '_adapters') else []
                    logger.error(f"   Available adapters: {available}")
                    logger.error(f"   Registry instance: {id(adapter_registry)}")
                    
                    # Try to register adapters if none found
                    if len(available) == 0:
                        logger.error(f"   No adapters registered! Attempting to register now...")
                        from ..adapters.ensure_adapters import ensure_research_adapters_registered
                        if ensure_research_adapters_registered():
                            logger.info(f"   Adapters registered successfully, retrying...")
                            adapter = adapter_registry.get(adapter_name)
                            if not adapter:
                                logger.error(f"   Still cannot find adapter '{adapter_name}' after registration")
                                raise Exception(f"Adapter '{adapter_name}' not found even after registration attempt")
                        else:
                            logger.error(f"   Failed to register adapters")
                            raise Exception(f"Adapter '{adapter_name}' not found and registration failed")
                    else:
                        raise Exception(f"Adapter '{adapter_name}' not found (available: {available})")

                node.adapter = adapter_name
                logger.info(f"✅ Routed node {node.id} to adapter: {adapter_name}")
                logger.info(f"   Adapter instance: {type(adapter).__name__}")
                
                # Verify adapter is properly initialized
                if not hasattr(adapter, 'run'):
                    logger.error(f"❌ CRITICAL: Adapter '{adapter_name}' missing run() method")
                    logger.error(f"   Adapter type: {type(adapter)}")
                    logger.error(f"   Adapter methods: {dir(adapter)}")
                    raise Exception(f"Adapter '{adapter_name}' missing run() method")

                logger.info(f"✅ Adapter {adapter_name} verified (has run() method)")
                await self._deliver_pending_steer(node.id, adapter_name)

                # Setup experiment worktree if this is an EXPERIMENT node
                if context.experiment_context:
                    try:
                        await self._setup_experiment_worktree(context.experiment_context)
                    except Exception as e:
                        logger.error(f"[EXPERIMENT] Failed to setup worktree: {e}")
                        raise  # Re-raise to mark node as failed

                # Execute via adapter with timeout
                events_received = 0
                execution_timeout = 300  # 5 minutes per node
                
                logger.info(f"[EXECUTE] Starting adapter execution for node {node.id}")
                logger.info(f"   Adapter: {adapter_name}")
                logger.info(f"   Timeout: {execution_timeout}s")
                logger.info(f"   Task: {task.goal[:100] if task.goal else 'N/A'}")
                
                try:
                    logger.info(f"[EXECUTE] Calling adapter.run() for node {node.id}...")
                    
                    # Create task to handle timeout
                    start_time = asyncio.get_event_loop().time()
                    
                    async for event in adapter.run(task, context):
                        # Check timeout on each event
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed > execution_timeout:
                            raise asyncio.TimeoutError(f"Execution exceeded {execution_timeout}s")
                        
                        events_received += 1
                        logger.info(f"[EXECUTE] Node {node.id} received event #{events_received}: {event.type if hasattr(event, 'type') else type(event).__name__}")

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

                        # Publish event to bus (global stream)
                        await self.event_bus.publish(event)
                        
                        # Publish to per-node stream (Issue #24 - UAGENT-24-2)
                        if hasattr(event, 'node_id'):
                            event.node_id = node.id
                        await self.event_bus.publish_node_event(node.id, event)

                        # Update node on completion
                        if event.type == EventType.COMPLETE:
                            logger.info(f"[EXECUTE] Node {node.id} received COMPLETE event")
                            node.status = NodeStatus.COMPLETE
                            node.visits += 1
                            node.avg_value = 0.8  # Success value

                            self.stats["completed_nodes"] += 1

                            # Update costs
                            cost = await adapter.estimate_cost(task, context)
                            node.cost = cost
                            self.stats["total_cost"] += cost
                            logger.info(f"[EXECUTE] Node {node.id} completed successfully (cost: ${cost:.3f})")

                    # If no events received and status hasn't changed, force completion
                    if events_received == 0 and node.status == NodeStatus.RUNNING:
                        logger.warning(f"[EXECUTE] Node {node.id} received no events, forcing completion")
                        node.status = NodeStatus.COMPLETE
                        node.visits += 1
                        node.avg_value = 0.5  # Neutral value for no-op execution
                        self.stats["completed_nodes"] += 1
                    
                    logger.info(f"[EXECUTE] Node {node.id} execution finished (received {events_received} events)")
                
                except asyncio.TimeoutError:
                    logger.error(f"❌ Node {node.id} execution TIMED OUT after {execution_timeout}s")
                    node.status = NodeStatus.FAILED
                    node.visits += 1
                    node.avg_value = 0.0
                    self.stats["failed_nodes"] += 1
                    
                except Exception as adapter_error:
                    logger.error(f"❌ ADAPTER EXECUTION FAILED for node {node.id}")
                    logger.error(f"   Adapter: {adapter_name}")
                    logger.error(f"   Error type: {type(adapter_error).__name__}")
                    logger.error(f"   Error message: {str(adapter_error)}")
                    logger.error(f"   Traceback:", exc_info=True)
                    raise  # Re-raise to be caught by outer exception handler
                
                logger.info(
                    f"Node {node.id} completed ({events_received} events received)"
                )
                
                # Emit NodeCompleteEvent via MessageBus
                if self.message_bus and self.tree:
                    try:
                        # Delayed import to avoid circular dependency
                        from openhands.events.agent_event import NodeCompleteEvent
                        asyncio.create_task(
                            self.message_bus.send_message(
                                from_agent_id=self.tree.research_id,
                                to_agent_id=None,  # Broadcast
                                message=NodeCompleteEvent(
                                    from_agent_id=self.tree.research_id,
                                    node_id=node.id,
                                    branch_id=node.id,
                                    experiment_id=self.tree.research_id,
                                    result=node.content,
                                    cost=node.cost,
                                    artifacts=node.artifacts
                                )
                            )
                        )
                    except Exception as e:
                        logger.debug(f"Failed to emit NodeCompleteEvent: {e}")

            except Exception as e:
                logger.error(f"Node {node.id} execution failed: {e}", exc_info=True)

                node.status = NodeStatus.FAILED
                node.visits += 1
                node.avg_value = 0.0  # Failure value

                self.stats["failed_nodes"] += 1

                error_message = str(e)
                error_trace = traceback.format_exc()

                try:
                    # Persist error metadata on node for UI diagnostics
                    node.metadata = node.metadata or {}
                    existing_errors = list(node.metadata.get("errors", []))
                    error_entry = {
                        "message": error_message,
                        "traceback": error_trace,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    existing_errors.append(error_entry)
                    # Keep only latest 20 errors per node to bound payload size
                    node.metadata["errors"] = existing_errors[-20:]
                    node.metadata["last_error"] = error_entry
                except Exception:
                    logger.debug("Failed to record error metadata", exc_info=True)

                try:
                    error_event = ErrorEvent(
                        branch_id=node.id,
                        node_id=node.id,
                        message=error_message,
                        traceback=error_trace,
                        experiment_id=self.tree.research_id if self.tree else None,
                    )
                    await self.event_bus.publish(error_event)
                except Exception:
                    logger.debug("Failed to publish error event", exc_info=True)

                # Push updated tree snapshot so UI sees failure immediately
                try:
                    self._update_tree_stats(force=True)
                except Exception:
                    logger.debug("Failed to broadcast tree update after error", exc_info=True)
            finally:
                # Cleanup experiment worktree if this was an EXPERIMENT node
                if context and context.experiment_context:
                    try:
                        await self._cleanup_experiment_worktree(context.experiment_context.worktree_path)
                    except Exception as e:
                        logger.warning(f"[EXPERIMENT] Error during worktree cleanup: {e}")
                        # Don't re-raise, just log - cleanup failures shouldn't fail the node
                
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

    async def _setup_experiment_worktree(self, experiment_context: ExperimentContext) -> None:
        """Setup git worktree for experiment isolation.
        
        Creates a git worktree with a new branch for the experiment.
        All experiment code execution happens in this isolated worktree directory.
        Git worktrees share the .git directory (lightweight) but have separate
        working directories, enabling parallel experiments without interference.
        
        Args:
            experiment_context: Experiment configuration with worktree details
            
        Raises:
            RuntimeError: If worktree creation fails
        """
        import subprocess
        import os
        
        worktree_path = experiment_context.worktree_path
        branch_name = experiment_context.worktree_branch
        parent_branch = experiment_context.parent_branch
        
        logger.info(f"[WORKTREE] Setting up experiment worktree: {worktree_path}")
        logger.info(f"[WORKTREE] Branch: {branch_name}, Parent: {parent_branch}")
        
        # Check if worktree already exists (reuse for idempotency)
        if os.path.exists(worktree_path):
            logger.info(f"[WORKTREE] Worktree {worktree_path} already exists, reusing")
            return
        
        # Ensure parent worktrees directory exists
        worktrees_dir = os.path.dirname(worktree_path)
        if worktrees_dir and not os.path.exists(worktrees_dir):
            try:
                os.makedirs(worktrees_dir, exist_ok=True)
                logger.info(f"[WORKTREE] Created worktrees directory: {worktrees_dir}")
            except Exception as e:
                logger.error(f"[WORKTREE] Failed to create worktrees directory: {e}")
                raise RuntimeError(f"Failed to create worktrees directory: {e}")
        
        # Create git worktree with new branch
        try:
            # Use git worktree add with -b to create new branch
            cmd = ['git', 'worktree', 'add', worktree_path, '-b', branch_name]
            
            logger.debug(f"[WORKTREE] Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd()  # Ensure we're in a git repo
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"[WORKTREE] Failed to create worktree: {error_msg}")
                logger.error(f"[WORKTREE] Command: {' '.join(cmd)}")
                logger.error(f"[WORKTREE] Stdout: {result.stdout}")
                raise RuntimeError(f"Failed to create worktree: {error_msg}")
            
            logger.info(f"[WORKTREE] Successfully created worktree at {worktree_path}")
            logger.debug(f"[WORKTREE] Git output: {result.stdout.strip()}")
            
        except FileNotFoundError:
            logger.error(f"[WORKTREE] Git command not found. Is git installed?")
            raise RuntimeError("Git command not found. Please ensure git is installed.")
        except Exception as e:
            logger.error(f"[WORKTREE] Unexpected error creating worktree: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create worktree: {e}")

    async def _cleanup_experiment_worktree(self, worktree_path: str) -> None:
        """Remove experiment worktree after completion.
        
        Cleans up the git worktree directory and associated branch.
        Uses --force to handle cases where worktree might be partially created
        or has uncommitted changes.
        
        Args:
            worktree_path: Path to the worktree directory to remove
        """
        import subprocess
        import os
        
        if not worktree_path:
            logger.warning(f"[WORKTREE] Empty worktree_path provided for cleanup, skipping")
            return
        
        # Check if worktree exists
        if not os.path.exists(worktree_path):
            logger.debug(f"[WORKTREE] Worktree {worktree_path} doesn't exist, no cleanup needed")
            return
        
        logger.info(f"[WORKTREE] Cleaning up experiment worktree: {worktree_path}")
        
        try:
            # Use --force to remove even if there are uncommitted changes
            cmd = ['git', 'worktree', 'remove', worktree_path, '--force']
            
            logger.debug(f"[WORKTREE] Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.warning(f"[WORKTREE] Failed to remove worktree: {error_msg}")
                logger.debug(f"[WORKTREE] Command: {' '.join(cmd)}")
                logger.debug(f"[WORKTREE] Stdout: {result.stdout}")
                # Don't raise exception on cleanup failure, just log it
            else:
                logger.info(f"[WORKTREE] Successfully removed worktree {worktree_path}")
                logger.debug(f"[WORKTREE] Git output: {result.stdout.strip()}")
                
        except FileNotFoundError:
            logger.warning(f"[WORKTREE] Git command not found during cleanup")
        except Exception as e:
            logger.warning(f"[WORKTREE] Unexpected error during cleanup: {e}")
            # Don't raise exception on cleanup failure

    def _convert_tree_to_frontend_format(self, tree_dict: dict) -> dict:
        """
        Convert internal tree representation to frontend-compatible format.
        
        Internal format: {nodes: {node_id: node_data}, edges: [{from, to, relation}], ...}
        Frontend format: {nodes: [{id, type, position, data}], edges: [{id, source, target, type}], ...}
        
        Args:
            tree_dict: Internal tree dictionary from ResearchTree.to_dict()
            
        Returns:
            Frontend-compatible tree structure
        """
        nodes_data = tree_dict.get('nodes', {})
        edges_list = tree_dict.get('edges', [])

        # Handle both dict and list formats for nodes
        # ResearchTree.to_dict() returns nodes as a list, not dict
        if isinstance(nodes_data, list):
            # Convert list to dict for easier processing
            nodes_dict = {node['id']: node for node in nodes_data}
        else:
            nodes_dict = nodes_data

        # Convert nodes dictionary to array with frontend structure
        frontend_nodes = []
        node_positions = {}  # Track positions for layout

        # Calculate positions using a simple tree layout algorithm
        # Root at top, children spread horizontally below
        root_nodes = []
        child_nodes_by_parent = {}

        # First pass: identify root nodes and group children
        for node_id, node_data in nodes_dict.items():
            parent_id = node_data.get('parent_id')
            if not parent_id or parent_id not in nodes_dict:
                root_nodes.append((node_id, node_data))
            else:
                if parent_id not in child_nodes_by_parent:
                    child_nodes_by_parent[parent_id] = []
                child_nodes_by_parent[parent_id].append((node_id, node_data))
        
        # Layout parameters
        x_start = 400
        y_start = 50
        y_spacing = 150
        x_spacing = 300
        
        # Position root nodes
        for i, (node_id, node_data) in enumerate(root_nodes):
            x = x_start + (i - len(root_nodes) / 2) * x_spacing
            node_positions[node_id] = {'x': x, 'y': y_start}
        
        # BFS layout for children
        from collections import deque
        queue = deque([(node_id, 1) for node_id, _ in root_nodes])
        processed = set()
        
        while queue:
            parent_id, depth = queue.popleft()
            if parent_id in processed:
                continue
            processed.add(parent_id)
            
            children = child_nodes_by_parent.get(parent_id, [])
            if not children:
                continue
                
            # Position children horizontally around parent
            parent_x = node_positions[parent_id]['x']
            y = y_start + depth * y_spacing
            
            for i, (child_id, child_data) in enumerate(children):
                x = parent_x + (i - len(children) / 2) * x_spacing
                node_positions[child_id] = {'x': x, 'y': y}
                queue.append((child_id, depth + 1))
        
        # Build frontend node objects
        for node_id, node_data in nodes_dict.items():
            position = node_positions.get(node_id, {'x': x_start, 'y': y_start})
            
            frontend_node = {
                'id': node_id,
                'type': node_data.get('type', 'default'),
                'position': position,
                'data': {
                    'id': node_id,
                    'type': node_data.get('type', 'default'),
                    'title': node_data.get('title', ''),
                    'description': node_data.get('content', ''),
                    'content': node_data.get('content', ''),  # Frontend expects this for display
                    'status': node_data.get('status', 'pending'),
                    'visits': node_data.get('visits', 0),  # Frontend uses 'visits' not 'visit_count'
                    'avg_value': node_data.get('avg_value', 0.0),
                    'prior': node_data.get('prior', 0.5),
                    'puct_score': 0.0,  # Can be calculated if needed
                    'cost': node_data.get('cost', 0.0),  # Frontend expects at top level
                    'tokens_used': node_data.get('tokens_used', 0),  # Frontend expects at top level
                    'created_at': node_data.get('created_at'),
                    'completed_at': node_data.get('completed_at'),
                    'metadata': {
                        'score': node_data.get('score'),
                        'confidence': node_data.get('confidence'),
                        'novelty': node_data.get('novelty'),
                        'iterations': node_data.get('iterations', 0),
                        'started_at': node_data.get('started_at'),
                        'adapter': node_data.get('adapter'),
                    }
                }
            }
            frontend_nodes.append(frontend_node)
        
        # Convert edges to frontend format
        frontend_edges = []
        for i, edge in enumerate(edges_list):
            edge_id = f"edge_{edge.get('from', '')}_{edge.get('to', '')}"
            frontend_edge = {
                'id': edge_id,
                'source': edge.get('from'),
                'target': edge.get('to'),
                'type': 'smoothstep',
                'label': edge.get('relation', '')
            }
            frontend_edges.append(frontend_edge)
        
        return {
            'nodes': frontend_nodes,
            'edges': frontend_edges,
            'stats': tree_dict.get('stats', {}),
            'version': tree_dict.get('version', 0)
        }

    def _publish_tree_to_api(self, force: bool = False):
        """
        Publish tree state to API and broadcast to WebSocket clients.
        
        This method:
        1. Creates a snapshot of the current tree state
        2. Updates the in-memory state in research_routes (_active_trees)
        3. Broadcasts the update to all connected WebSocket clients
        4. Logs all steps for debugging
        
        Debouncing: Skips broadcast if less than 500ms since last broadcast
        to prevent flooding WebSocket clients with too many updates.
        """
        if not self.tree:
            return
        
        # Debounce: skip if we broadcasted too recently
        import time
        now = time.time()
        time_since_last = (now - self._last_broadcast_ts) * 1000  # convert to ms
        
        if not force and time_since_last < self._broadcast_debounce_ms:
            logger.debug(f"Skipping broadcast (debounce): {time_since_last:.0f}ms since last")
            return
        
        self._last_broadcast_ts = now
        
        try:
            # Import from tree publisher module to avoid circular imports
            from ..uagent_research.api.tree_publisher import update_tree_state, broadcast_tree_update
            logger.info(f"[PUBLISH] Successfully imported API functions")
            
            # Create tree snapshot
            tree_dict = self.tree.to_dict() if hasattr(self.tree, 'to_dict') else {}
            
            # Convert tree format from internal dictionary to frontend array format
            tree_data = self._convert_tree_to_frontend_format(tree_dict)
            
            # Ensure stats are inside data, not at top level
            if 'stats' not in tree_data:
                tree_data['stats'] = {}
            tree_data['stats'].update(self.stats)
            
            tree_snapshot = {
                "version": getattr(self.tree, 'version', 0),
                "experiment_id": getattr(self.tree, 'research_id', 'unknown'),
                "data": tree_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            experiment_id = tree_snapshot["experiment_id"]
            
            # Update in-memory state (for REST API endpoint)
            update_tree_state(experiment_id, tree_snapshot)
            logger.info(f"✅ Tree state updated for {experiment_id}")
            
            # Broadcast to WebSocket clients with fallback for no running loop
            try:
                # Try to get the running loop
                try:
                    loop = asyncio.get_running_loop()
                    # We have a running loop, schedule the broadcast
                    asyncio.create_task(broadcast_tree_update(
                        experiment_id,
                        {"type": "tree_snapshot", **tree_snapshot}
                    ))
                    logger.info(f"✅ Tree broadcast scheduled for {experiment_id}")
                except RuntimeError:
                    # No running loop - use anyio fallback from thread
                    logger.info(f"📡 No running loop, using anyio fallback for broadcast")
                    try:
                        import anyio
                        anyio.from_thread.run(
                            broadcast_tree_update,
                            experiment_id,
                            {"type": "tree_snapshot", **tree_snapshot}
                        )
                        logger.info(f"✅ Tree broadcast sent via anyio fallback for {experiment_id}")
                    except Exception as anyio_err:
                        logger.warning(f"⚠️ Anyio fallback also failed: {anyio_err}")
                        logger.debug("Skipping WebSocket broadcast (no event loop available)")
            except Exception as e:
                logger.warning(f"Could not broadcast tree update: {e}")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import API functions: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"❌ Failed to publish tree state: {e}", exc_info=True)

    def _update_tree_stats(self, force: bool = False):
        """Update tree-level statistics"""
        if not self.tree:
            return

        # Recompute status counts so progress metrics stay accurate
        status_counts = Counter(node.status for node in self.tree.nodes.values())
        self.stats["total_nodes"] = len(self.tree.nodes)
        self.stats["completed_nodes"] = status_counts.get(NodeStatus.COMPLETE, 0)
        self.stats["failed_nodes"] = status_counts.get(NodeStatus.FAILED, 0)
        self.stats["running_nodes"] = status_counts.get(NodeStatus.RUNNING, 0)
        self.stats["pending_nodes"] = status_counts.get(NodeStatus.PENDING, 0)

        # Update stats while preserving existing fields like "created" and "expanded"
        self.tree.stats.update({
            "total_nodes": len(self.tree.nodes),
            "total_edges": len(self.tree.edges),
            "max_depth": self.tree.calculate_max_depth(),
            "total_cost": self.stats["total_cost"],
            "total_tokens": self.stats["total_tokens"],
        })

        # Publish tree state to API endpoint
        self._publish_tree_to_api(force=force)
        
        # Emit ProgressUpdateEvent via MessageBus
        if self.message_bus and self.tree:
            try:
                # Delayed import to avoid circular dependency
                from openhands.events.agent_event import ProgressUpdateEvent
                total_nodes = self.tree.stats.get("total_nodes", 0)
                iterations = self.stats.get("iterations", 0)
                max_iterations = getattr(self, '_max_iterations', 50)
                progress = min(iterations / max(max_iterations, 1), 1.0)
                
                asyncio.create_task(
                    self.message_bus.send_message(
                        from_agent_id=self.tree.research_id,
                        to_agent_id=None,  # Broadcast
                        message=ProgressUpdateEvent(
                            from_agent_id=self.tree.research_id,
                            progress=progress,
                            current_task=f"Iteration {iterations}/{max_iterations}",
                            stats=dict(self.stats)
                        )
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to emit ProgressUpdateEvent: {e}")

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
            # Create tasks for both ControlBus and MessageBus subscriptions
            control_bus_task = asyncio.create_task(self._handle_control_bus())
            message_bus_task = asyncio.create_task(self._handle_message_bus())
            
            # Wait for either task to complete (or both to be cancelled)
            await asyncio.gather(control_bus_task, message_bus_task, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("Control loop cancelled")
        except Exception as e:
            logger.error(f"Error in control loop: {e}", exc_info=True)
    
    
    async def _handle_message_bus(self):
        """Handle MessageBus CommandEvents."""
        if not self.message_bus or not self.tree:
            return
        
        try:
            # Delayed import to avoid circular dependency
            from openhands.events.agent_event import CommandEvent
            async for message in self.message_bus.subscribe(self.tree.research_id):
                if isinstance(message, CommandEvent):
                    logger.info(
                        f"Received MessageBus command: {message.command_type} "
                        f"(from={message.from_agent_id})"
                    )
                    
                    # Map CommandEvent to ControlMessage actions
                    if message.command_type == "pause":
                        self._paused = True
                        logger.info("Research paused via MessageBus")
                    
                    elif message.command_type == "resume":
                        self._paused = False
                        logger.info("Research resumed via MessageBus")
                    
                    elif message.command_type == "cancel":
                        self._cancelled = True
                        logger.info("Research cancelled via MessageBus")
                    
                    elif message.command_type == "steer":
                        # Handle steering commands
                        logger.info(f"Steering command received: {message.payload}")
        
        except asyncio.CancelledError:
            logger.info("MessageBus handler cancelled")
        except Exception as e:
            logger.error(f"Error in MessageBus handler: {e}", exc_info=True)

    async def _handle_control_bus(self):
        """Handle ControlBus commands."""
        if not self.tree:
            return
        
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

    async def generate_final_report(self, idea_node_id: str) -> Optional[str]:
        """
        Generate final research report for an IDEA node after all children complete.
        
        Aggregates results from hypotheses and experiments, generates synthesis report.
        
        Args:
            idea_node_id: ID of the IDEA node
            
        Returns:
            Markdown-formatted report, or None if generation fails
        """
        logger.info(f"[REPORT] Generating final report for IDEA node {idea_node_id}")
        
        if not self.tree or idea_node_id not in self.tree.nodes:
            logger.warning(f"[REPORT] Node {idea_node_id} not found")
            return None
        
        idea_node = self.tree.nodes[idea_node_id]
        children = self.tree.get_children(idea_node_id)
        
        # Separate hypotheses and experiments
        hypotheses = [c for c in children if c.type == NodeType.HYPOTHESIS]
        experiments = [c for c in children if c.type == NodeType.EXPERIMENT]
        
        logger.info(f"[REPORT] Aggregating {len(hypotheses)} hypotheses and {len(experiments)} experiments")
        
        # Build simple markdown report
        report_lines = [
            f"# Research Report: {idea_node.title}",
            "",
            f"**Research Goal:** {idea_node.content}",
            "",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Hypotheses Tested",
            ""
        ]
        
        for i, hyp in enumerate(hypotheses, 1):
            report_lines.extend([
                f"### Hypothesis {i}: {hyp.title}",
                f"{hyp.content}",
                f"- **Status:** {hyp.status.value}",
                f"- **Confidence:** {hyp.prior:.2f}",
                ""
            ])
        
        report_lines.extend([
            "## Experiments Conducted",
            ""
        ])
        
        for i, exp in enumerate(experiments, 1):
            report_lines.extend([
                f"### Experiment {i}: {exp.title}",
                f"{exp.content[:200]}...",
                f"- **Status:** {exp.status.value}",
                f"- **Cost:** ${exp.cost:.3f}",
                f"- **Iterations:** {exp.iterations}",
                ""
            ])
        
        report_lines.extend([
            "## Summary",
            "",
            f"Tested {len(hypotheses)} hypotheses through {len(experiments)} experiments.",
            f"Total cost: ${sum(e.cost for e in experiments):.3f}",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        # Store in node metadata
        if not idea_node.metadata:
            idea_node.metadata = {}
        idea_node.metadata['final_report'] = report
        idea_node.metadata['report_generated_at'] = datetime.utcnow().isoformat()
        
        logger.info(f"[REPORT] Report generated ({len(report)} chars)")
        
        # Emit report complete event
        try:
            from ..models.events import ReportCompleteEvent
            report_event = ReportCompleteEvent(
                branch_id=idea_node_id,
                node_id=idea_node_id,
                report_content=report,
                hypotheses_count=len(hypotheses),
                experiments_count=len(experiments),
                summary=f"Tested {len(hypotheses)} hypotheses through {len(experiments)} experiments"
            )
            await self.event_bus.publish(report_event)
        except Exception as e:
            logger.warning(f"[REPORT] Failed to emit ReportCompleteEvent: {e}")
        
        return report

    async def steer_node(self, node_id: str, message: str) -> bool:
        """
        Send steering message to a specific running node.
        
        Allows users to guide specific experiments or child nodes without
        affecting the main agent. Routes message to the node's adapter via
        send_message() interface.
        
        Args:
            node_id: ID of the node to steer
            message: Steering message to send
            
        Returns:
            True if message was delivered, False otherwise
            
        Example:
            success = await orchestrator.steer_node(
                node_id="experiment-abc123",
                message="Focus on edge cases in your testing"
            )
        """
        logger.info(f"[STEER] Steering node {node_id}: {message[:100]}")
        
        # Validate node exists
        if not self.tree or node_id not in self.tree.nodes:
            logger.warning(f"[STEER] Node {node_id} not found in tree")
            return False
        
        node = self.tree.nodes[node_id]
        
        # Check if node has a running task
        if node_id not in self._running_tasks:
            logger.warning(f"[STEER] Node {node_id} is not currently running")
            return False
        
        # Get adapter for this node
        adapter_name = node.adapter
        if not adapter_name:
            logger.warning(f"[STEER] Node {node_id} has no adapter assigned")
            return False
        
        # Get adapter instance from registry
        from ..adapters.base.agent_adapter import adapter_registry
        adapter = adapter_registry.get(adapter_name)
        
        if not adapter:
            logger.warning(f"[STEER] Adapter '{adapter_name}' not found in registry")
            return False
        
        # Check if adapter supports send_message
        if not hasattr(adapter, 'send_message'):
            logger.warning(f"[STEER] Adapter '{adapter_name}' does not support steering")
            return False
        
        # Send message via adapter
        try:
            await adapter.send_message(message)
            logger.info(f"[STEER] Successfully delivered message to node {node_id}")
            return True
        except Exception as e:
            logger.error(f"[STEER] Failed to deliver message to node {node_id}: {e}")
            return False

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
