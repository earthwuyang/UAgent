"""
AIRA-style MCTS Implementation
Based on "AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench"

Implements proper MCTS with:
- UCT selection with c=0.25 (paper default)
- num_children=5 per expansion
- Empirical mean backup (no simulated rollouts)
- Select→Operate→Run→Score→Backup cycle
"""

import math
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio

from .aira_operators import Artifact, Operator, Memory, ProblemSpec, CodeExecutor, OperatorFactory

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """MCTS node with visit count and empirical mean fitness"""
    artifact: Artifact
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    total_reward: float = 0.0
    mean_fitness: float = 0.0
    is_expanded: bool = False
    is_terminal: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = []

    @property
    def ucb_score(self) -> float:
        """UCB1 score for node selection"""
        if self.visits == 0:
            return float('inf')

        if self.parent is None or self.parent.visits == 0:
            return self.mean_fitness

        # UCT formula: mean + c * sqrt(ln(parent_visits) / visits)
        exploration = UCT_C * math.sqrt(math.log(self.parent.visits) / self.visits)
        return self.mean_fitness + exploration

    def is_expandable(self, max_depth: int, max_children: int) -> bool:
        """Check if node can be expanded"""
        if self.is_terminal or self.artifact.depth >= max_depth:
            return False

        if len(self.children) >= max_children:
            return False

        # For AIRA: node is expandable if it has been evaluated (has fitness score)
        return self.artifact.status == "COMPLETED" and self.artifact.val_metric is not None


# AIRA paper defaults
UCT_C = 0.25  # UCT exploration constant
NUM_CHILDREN = 5  # Children per expansion
MAX_DEPTH = 10  # Maximum tree depth
MAX_ITERATIONS = 400  # Maximum artifacts per goal


class AIRAMCTSPolicy:
    """
    MCTS policy implementing AIRA paper specifications:
    - UCT selection with c=0.25
    - 5 children per expansion
    - Empirical mean backup
    - No simulated rollouts
    """

    def __init__(self,
                 problem_spec: ProblemSpec,
                 llm_client=None,
                 uct_c: float = UCT_C,
                 num_children: int = NUM_CHILDREN,
                 max_depth: int = MAX_DEPTH,
                 max_iterations: int = MAX_ITERATIONS):

        self.problem_spec = problem_spec
        self.llm_client = llm_client
        self.uct_c = uct_c
        self.num_children = num_children
        self.max_depth = max_depth
        self.max_iterations = max_iterations

        # Initialize components
        self.executor = CodeExecutor()
        self.memory = Memory()
        self.operators = {
            'draft': OperatorFactory.create_draft(llm_client=llm_client),
            'improve': OperatorFactory.create_improve(llm_client=llm_client),
            'debug': OperatorFactory.create_debug(llm_client=llm_client)
        }

        # Tree state
        self.root: Optional[MCTSNode] = None
        self.nodes: Dict[str, MCTSNode] = {}
        self.iteration_count = 0

    async def search(self, max_wallclock_hours: float = 24.0) -> MCTSNode:
        """
        Main MCTS search loop following AIRA Select→Operate→Run→Score→Backup
        """
        logger.info(f"Starting AIRA MCTS search for {self.problem_spec.task_name}")

        # Initialize root with draft
        if self.root is None:
            await self._initialize_root()

        start_time = asyncio.get_event_loop().time()
        max_time = max_wallclock_hours * 3600

        while (self.iteration_count < self.max_iterations and
               (asyncio.get_event_loop().time() - start_time) < max_time):

            try:
                # AIRA MCTS cycle: Select → Operate → Run → Score → Backup
                await self._mcts_iteration()
                self.iteration_count += 1

                if self.iteration_count % 50 == 0:
                    logger.info(f"MCTS iteration {self.iteration_count}, "
                              f"best score: {self._get_best_score():.4f}")

            except Exception as e:
                logger.error(f"MCTS iteration {self.iteration_count} failed: {e}")
                continue

        logger.info(f"MCTS search completed after {self.iteration_count} iterations")
        return self._get_best_node()

    async def _initialize_root(self):
        """Initialize root node with initial draft"""
        logger.info("Initializing MCTS root with draft operator")

        draft_op = self.operators['draft']
        root_artifact = await draft_op.apply(self.problem_spec, None, self.memory)

        # Execute root artifact
        score = await self.executor.run_and_score(root_artifact)
        logger.info(f"Root artifact score: {score:.4f}")

        # Create root node
        self.root = MCTSNode(artifact=root_artifact)
        self.root.visits = 1
        self.root.total_reward = score
        self.root.mean_fitness = score
        self.nodes[root_artifact.id] = self.root

    async def _mcts_iteration(self):
        """Single MCTS iteration: Select → Operate → Run → Score → Backup"""

        # 1. SELECT: Traverse tree using UCT to find leaf
        path = self._select_path()
        if not path:
            logger.warning("No selectable path found")
            return

        leaf = path[-1]

        # 2. OPERATE: Expand leaf with operators
        if leaf.is_expandable(self.max_depth, self.num_children):
            children = await self._expand_node(leaf)
            if children:
                # Select one child for evaluation
                child = random.choice(children)
                path.append(child)
                leaf = child

        # 3. RUN & SCORE: Execute and evaluate
        if leaf.artifact.status == "PENDING":
            score = await self.executor.run_and_score(leaf.artifact)
        else:
            score = leaf.artifact.val_metric or 0.0

        # 4. BACKUP: Propagate score up the path
        self._backup_path(path, score)

    def _select_path(self) -> List[MCTSNode]:
        """
        UCT-based selection to find path from root to leaf
        Paper spec: traverse by maximizing UCT score
        """
        if not self.root:
            return []

        path = [self.root]
        current = self.root

        while current.children and current.is_expanded:
            # Select child with highest UCB score
            best_child = max(current.children, key=lambda c: c.ucb_score)
            path.append(best_child)
            current = best_child

        return path

    async def _expand_node(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Expand node with operators to create children
        Paper spec: ~5 children per expansion
        """
        if node.is_expanded or not node.is_expandable(self.max_depth, self.num_children):
            return []

        logger.debug(f"Expanding node {node.artifact.id}")
        children = []

        # Select operators based on parent's status and performance
        operators_to_try = self._select_operators(node)

        for i, op_name in enumerate(operators_to_try[:self.num_children]):
            try:
                operator = self.operators[op_name]

                # Apply operator
                if op_name == 'draft':
                    child_artifact = await operator.apply(self.problem_spec, None, self.memory)
                else:
                    child_artifact = await operator.apply(self.problem_spec, node.artifact, self.memory)

                # Create child node
                child_node = MCTSNode(
                    artifact=child_artifact,
                    parent=node
                )

                node.children.append(child_node)
                children.append(child_node)
                self.nodes[child_artifact.id] = child_node

                logger.debug(f"Created child {child_artifact.id} using {op_name}")

            except Exception as e:
                logger.error(f"Failed to expand with {op_name}: {e}")
                continue

        node.is_expanded = True
        return children

    def _select_operators(self, node: MCTSNode) -> List[str]:
        """Select which operators to use for expansion"""
        operators = []

        # If parent succeeded, try improvements
        if node.artifact.status == "COMPLETED" and (node.artifact.val_metric or 0) > 0.1:
            operators.extend(['improve'] * 3)  # Multiple improvement attempts

        # If parent failed, try debugging
        if node.artifact.status == "FAILED":
            operators.extend(['debug'] * 2)

        # Always include some drafts for diversity
        if node.artifact.depth < 2:  # Only for shallow nodes
            operators.append('draft')

        # Fill remaining slots with improve if we have space
        while len(operators) < self.num_children:
            operators.append('improve')

        return operators

    def _backup_path(self, path: List[MCTSNode], leaf_score: float):
        """
        Backup leaf score along path using empirical mean
        Paper spec: store visit count and empirical mean, no simulated rollouts
        """
        for node in path:
            node.visits += 1
            node.total_reward += leaf_score
            node.mean_fitness = node.total_reward / node.visits

            # Update artifact with latest score
            if node.artifact.val_metric is None or leaf_score > (node.artifact.val_metric or 0):
                node.artifact.visits = node.visits
                node.artifact.mean_fitness = node.mean_fitness

    def _get_best_node(self) -> MCTSNode:
        """Get node with highest mean fitness"""
        if not self.nodes:
            return self.root

        return max(self.nodes.values(),
                  key=lambda n: n.mean_fitness if n.visits > 0 else 0.0)

    def _get_best_score(self) -> float:
        """Get best score seen so far"""
        if not self.nodes:
            return 0.0

        best_node = self._get_best_node()
        return best_node.mean_fitness if best_node else 0.0

    def get_top_k_artifacts(self, k: int = 5) -> List[Artifact]:
        """
        Get top-k artifacts by validation score
        AIRA paper: submit top-k by validation for final selection
        """
        # Filter completed artifacts
        completed_nodes = [
            node for node in self.nodes.values()
            if (node.artifact.status == "COMPLETED" and
                node.artifact.val_metric is not None and
                node.visits > 0)
        ]

        # Sort by mean fitness (higher is better)
        completed_nodes.sort(key=lambda n: n.mean_fitness, reverse=True)

        return [node.artifact for node in completed_nodes[:k]]

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics for monitoring"""
        completed_count = sum(1 for n in self.nodes.values()
                            if n.artifact.status == "COMPLETED")
        failed_count = sum(1 for n in self.nodes.values()
                         if n.artifact.status == "FAILED")

        return {
            'total_iterations': self.iteration_count,
            'total_nodes': len(self.nodes),
            'completed_nodes': completed_count,
            'failed_nodes': failed_count,
            'success_rate': completed_count / len(self.nodes) if self.nodes else 0.0,
            'best_score': self._get_best_score(),
            'tree_depth': max((n.artifact.depth for n in self.nodes.values()), default=0)
        }


class GreedyPolicy:
    """
    Simple greedy policy for comparison (AIDE-style)
    Always expands the best validation node
    """

    def __init__(self, problem_spec: ProblemSpec, llm_client=None):
        self.problem_spec = problem_spec
        self.llm_client = llm_client
        self.executor = CodeExecutor()
        self.memory = Memory()
        self.operators = {
            'draft': OperatorFactory.create_draft(llm_client=llm_client),
            'improve': OperatorFactory.create_improve(llm_client=llm_client),
            'debug': OperatorFactory.create_debug(llm_client=llm_client)
        }
        self.artifacts: Dict[str, Artifact] = {}

    async def search(self, max_iterations: int = 100) -> List[Artifact]:
        """Simple greedy search"""
        logger.info(f"Starting greedy search for {self.problem_spec.task_name}")

        # Initialize with draft
        draft_op = self.operators['draft']
        initial_artifact = await draft_op.apply(self.problem_spec, None, self.memory)
        await self.executor.run_and_score(initial_artifact)
        self.artifacts[initial_artifact.id] = initial_artifact

        for iteration in range(max_iterations):
            # Find best artifact
            best_artifact = max(self.artifacts.values(),
                              key=lambda a: a.val_metric or 0.0)

            # Try to improve it
            try:
                if best_artifact.status == "COMPLETED":
                    improve_op = self.operators['improve']
                    new_artifact = await improve_op.apply(self.problem_spec, best_artifact, self.memory)
                    await self.executor.run_and_score(new_artifact)
                    self.artifacts[new_artifact.id] = new_artifact

                logger.info(f"Greedy iteration {iteration}, "
                          f"best score: {max(a.val_metric or 0 for a in self.artifacts.values()):.4f}")

            except Exception as e:
                logger.error(f"Greedy iteration {iteration} failed: {e}")

        return sorted(self.artifacts.values(),
                     key=lambda a: a.val_metric or 0.0, reverse=True)


# Policy factory
class PolicyFactory:
    """Factory for creating search policies"""

    @staticmethod
    def create_mcts(problem_spec: ProblemSpec, llm_client=None, **kwargs) -> AIRAMCTSPolicy:
        return AIRAMCTSPolicy(problem_spec, llm_client, **kwargs)

    @staticmethod
    def create_greedy(problem_spec: ProblemSpec, llm_client=None) -> GreedyPolicy:
        return GreedyPolicy(problem_spec, llm_client)