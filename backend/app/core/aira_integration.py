"""
AIRA Integration Layer for UAgent
Bridges AIRA operators and MCTS with existing UAgent research tree system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from .research_tree import (
    HierarchicalResearchSystem, ResearchGoal, ResearchNode, ExperimentResult,
    ResearchNodeType, NodeStatus, ExperimentType
)
from .enhanced_research_system import DatabaseIntegratedResearchSystem
from .aira_operators import (
    Artifact, ProblemSpec, Memory, OperatorFactory, CodeExecutor
)
from .aira_mcts import AIRAMCTSPolicy, GreedyPolicy, PolicyFactory

logger = logging.getLogger(__name__)


@dataclass
class AIRAConfig:
    """Configuration for AIRA-enhanced research system"""
    policy: str = "mcts"  # mcts, greedy, evolutionary
    max_wallclock_h: float = 24.0
    max_artifacts: int = 400

    # MCTS settings
    uct_c: float = 0.25
    num_children: int = 5

    # Operator settings
    draft_complexity: str = "normal"  # simple, normal, complex
    improve_complexity: str = "normal"
    debug_max_nodes: int = 10

    # Evaluation settings
    cv_folds: int = 5
    final_selection_top_k: int = 5

    # Environment settings
    per_exec_timeout_h: float = 4.0


class AIRAEnhancedResearchSystem(DatabaseIntegratedResearchSystem):
    """
    Enhanced research system using AIRA operators and MCTS
    Maintains compatibility with existing UAgent interfaces
    """

    def __init__(self, config: AIRAConfig = None, db_path: str = None):
        super().__init__(db_path=db_path)
        self.config = config or AIRAConfig()
        self.aira_policies: Dict[str, Any] = {}  # goal_id -> AIRA policy
        self.aira_artifacts: Dict[str, List[Artifact]] = {}  # goal_id -> artifacts
        self.memory_store: Dict[str, Memory] = {}  # goal_id -> memory

    async def start_research_goal(self, title: str, description: str, success_criteria: List[str],
                                constraints: Optional[Dict[str, Any]] = None,
                                max_depth: int = 5, max_experiments: int = 100) -> str:
        """Start research goal with AIRA enhancement"""

        # Create traditional research goal
        goal_id = await super().start_research_goal(
            title, description, success_criteria, constraints, max_depth, max_experiments
        )

        try:
            # Initialize AIRA components for this goal
            await self._initialize_aira_for_goal(goal_id, title, description, success_criteria, constraints)
            logger.info(f"AIRA components initialized for goal {goal_id}")

        except Exception as e:
            logger.error(f"Failed to initialize AIRA for goal {goal_id}: {e}")
            # Continue with traditional system if AIRA fails

        return goal_id

    async def _initialize_aira_for_goal(self, goal_id: str, title: str, description: str,
                                      success_criteria: List[str], constraints: Dict[str, Any]):
        """Initialize AIRA policy and components for a research goal"""

        # Create problem specification
        problem_spec = ProblemSpec(
            task_name=title,
            description=description,
            data_info=constraints.get("data_info", "Standard ML dataset"),
            constraints=constraints or {},
            success_criteria=success_criteria
        )

        # Initialize memory
        memory = Memory()
        self.memory_store[goal_id] = memory

        # Create AIRA policy
        if self.config.policy == "mcts":
            policy = PolicyFactory.create_mcts(
                problem_spec=problem_spec,
                llm_client=getattr(self, 'llm_client', None),
                uct_c=self.config.uct_c,
                num_children=self.config.num_children,
                max_iterations=self.config.max_artifacts
            )
        elif self.config.policy == "greedy":
            policy = PolicyFactory.create_greedy(
                problem_spec=problem_spec,
                llm_client=getattr(self, 'llm_client', None)
            )
        else:
            raise ValueError(f"Unknown policy: {self.config.policy}")

        self.aira_policies[goal_id] = policy
        self.aira_artifacts[goal_id] = []

        logger.info(f"Initialized AIRA {self.config.policy} policy for goal {goal_id}")

    async def run_goal_iteration(self, goal_id: str) -> bool:
        """Run one iteration of research with AIRA enhancement"""

        if goal_id not in self.active_goals:
            return False

        # Try AIRA-enhanced execution first
        if goal_id in self.aira_policies:
            try:
                return await self._run_aira_iteration(goal_id)
            except Exception as e:
                logger.error(f"AIRA iteration failed for goal {goal_id}: {e}")
                # Fall back to traditional system

        # Fall back to traditional execution
        return await super().run_goal_iteration(goal_id)

    async def _run_aira_iteration(self, goal_id: str) -> bool:
        """Run single AIRA iteration and sync with UAgent tree"""

        policy = self.aira_policies[goal_id]

        if self.config.policy == "mcts":
            # Run MCTS search iteration
            await policy._mcts_iteration()

            # Sync MCTS state with UAgent tree
            await self._sync_mcts_to_uagent(goal_id, policy)

        elif self.config.policy == "greedy":
            # Run greedy iteration
            artifacts = await policy.search(max_iterations=1)
            self.aira_artifacts[goal_id].extend(artifacts)

            # Sync artifacts to UAgent tree
            await self._sync_artifacts_to_uagent(goal_id, artifacts)

        return True

    async def _sync_mcts_to_uagent(self, goal_id: str, mcts_policy: AIRAMCTSPolicy):
        """Sync MCTS nodes to UAgent research tree"""

        if goal_id not in self.research_trees:
            self.research_trees[goal_id] = {}

        tree = self.research_trees[goal_id]

        # Convert MCTS nodes to UAgent nodes
        for artifact_id, mcts_node in mcts_policy.nodes.items():
            if artifact_id not in tree:
                # Create UAgent node from AIRA artifact
                uagent_node = self._artifact_to_research_node(mcts_node.artifact, goal_id)

                # Add MCTS-specific metadata
                uagent_node.visits = mcts_node.visits
                uagent_node.total_reward = mcts_node.total_reward
                uagent_node.confidence = mcts_node.mean_fitness

                # Add result if artifact was executed
                if mcts_node.artifact.status == "COMPLETED" and mcts_node.artifact.val_metric is not None:
                    result = ExperimentResult(
                        experiment_id=f"aira_{artifact_id}",
                        success=True,
                        confidence=mcts_node.mean_fitness,
                        execution_time=mcts_node.artifact.runtime_sec or 0.0,
                        insights=[f"AIRA {mcts_node.artifact.operator} operator result"],
                        metrics={"cv_score": mcts_node.artifact.val_metric},
                        data={"aira_artifact_id": artifact_id}
                    )
                    uagent_node.results.append(result)

                tree[artifact_id] = uagent_node

                # Update parent-child relationships
                if mcts_node.parent and mcts_node.parent.artifact.id in tree:
                    parent_node = tree[mcts_node.parent.artifact.id]
                    if artifact_id not in parent_node.children:
                        parent_node.children.append(artifact_id)

    async def _sync_artifacts_to_uagent(self, goal_id: str, artifacts: List[Artifact]):
        """Sync AIRA artifacts to UAgent research tree"""

        if goal_id not in self.research_trees:
            self.research_trees[goal_id] = {}

        tree = self.research_trees[goal_id]

        for artifact in artifacts:
            if artifact.id not in tree:
                uagent_node = self._artifact_to_research_node(artifact, goal_id)
                tree[artifact.id] = uagent_node

    def _artifact_to_research_node(self, artifact: Artifact, goal_id: str) -> ResearchNode:
        """Convert AIRA artifact to UAgent research node"""

        # Map AIRA operators to UAgent node types
        node_type_mapping = {
            'draft': ResearchNodeType.EXPERIMENT,
            'improve': ResearchNodeType.EXPERIMENT,
            'debug': ResearchNodeType.EXPERIMENT,
            'analysis': ResearchNodeType.ANALYSIS
        }

        # Map AIRA status to UAgent status
        status_mapping = {
            'PENDING': NodeStatus.PROMISING,
            'RUNNING': NodeStatus.PROMISING,
            'COMPLETED': NodeStatus.COMPLETED,
            'FAILED': NodeStatus.FAILED
        }

        node = ResearchNode(
            id=artifact.id,
            parent_id=artifact.parent_id,
            node_type=node_type_mapping.get(artifact.operator, ResearchNodeType.EXPERIMENT),
            title=f"AIRA {artifact.operator}: {artifact.plan_text[:50]}...",
            description=artifact.plan_text,
            depth=artifact.depth,
            status=status_mapping.get(artifact.status, NodeStatus.PROMISING),
            confidence=artifact.val_metric or 0.0,
            experiment_type=ExperimentType.COMPUTATIONAL,
            experiment_config={
                "aira_operator": artifact.operator,
                "code_path": str(artifact.code_path),
                "cv_folds": artifact.cv_folds
            }
        )

        # Set timestamps
        node.created_at = datetime.now()
        if artifact.status in ["RUNNING", "COMPLETED", "FAILED"]:
            node.started_at = datetime.now()
        if artifact.status in ["COMPLETED", "FAILED"]:
            node.completed_at = datetime.now()

        return node

    async def get_research_tree_status(self, goal_id: str) -> Dict[str, Any]:
        """Get enhanced status including AIRA metrics"""

        # Get traditional status
        status = await super().get_research_tree_status(goal_id)

        # Add AIRA-specific information
        if goal_id in self.aira_policies:
            try:
                if self.config.policy == "mcts":
                    mcts_policy = self.aira_policies[goal_id]
                    aira_stats = mcts_policy.get_search_statistics()

                    status["aira"] = {
                        "policy": self.config.policy,
                        "iterations": aira_stats["total_iterations"],
                        "artifacts": aira_stats["total_nodes"],
                        "success_rate": aira_stats["success_rate"],
                        "best_score": aira_stats["best_score"],
                        "tree_depth": aira_stats["tree_depth"]
                    }

                    # Add top artifacts
                    top_artifacts = mcts_policy.get_top_k_artifacts(k=3)
                    status["aira"]["top_artifacts"] = [
                        {
                            "id": a.id,
                            "operator": a.operator,
                            "score": a.val_metric,
                            "plan": a.plan_text[:100] + "..." if len(a.plan_text) > 100 else a.plan_text
                        }
                        for a in top_artifacts
                    ]

            except Exception as e:
                logger.error(f"Failed to get AIRA status for goal {goal_id}: {e}")

        return status

    async def get_final_results(self, goal_id: str) -> Dict[str, Any]:
        """
        Get final results with AIRA top-k selection
        Implements the paper's recommendation for robust final selection
        """

        results = await super().get_research_tree_status(goal_id)

        if goal_id in self.aira_policies and self.config.policy == "mcts":
            try:
                mcts_policy = self.aira_policies[goal_id]

                # Get top-k artifacts by validation score
                top_k_artifacts = mcts_policy.get_top_k_artifacts(k=self.config.final_selection_top_k)

                results["aira_final_selection"] = {
                    "top_k_artifacts": [
                        {
                            "rank": i + 1,
                            "artifact_id": artifact.id,
                            "operator": artifact.operator,
                            "validation_score": artifact.val_metric,
                            "plan": artifact.plan_text,
                            "code_path": str(artifact.code_path)
                        }
                        for i, artifact in enumerate(top_k_artifacts)
                    ],
                    "recommendation": f"Submit top {min(len(top_k_artifacts), 3)} artifacts to test set and select best performing one",
                    "expected_improvement": "AIRA paper reports 9-13 point absolute improvement using this strategy"
                }

            except Exception as e:
                logger.error(f"Failed to get AIRA final results for goal {goal_id}: {e}")

        return results

    def get_aira_config(self) -> Dict[str, Any]:
        """Get current AIRA configuration"""
        return {
            "policy": self.config.policy,
            "max_wallclock_h": self.config.max_wallclock_h,
            "max_artifacts": self.config.max_artifacts,
            "mcts": {
                "uct_c": self.config.uct_c,
                "num_children": self.config.num_children
            },
            "operators": {
                "draft_complexity": self.config.draft_complexity,
                "improve_complexity": self.config.improve_complexity,
                "debug_max_nodes": self.config.debug_max_nodes
            },
            "evaluation": {
                "cv_folds": self.config.cv_folds,
                "final_selection_top_k": self.config.final_selection_top_k
            }
        }

    async def update_aira_config(self, new_config: Dict[str, Any]):
        """Update AIRA configuration"""

        if "policy" in new_config:
            self.config.policy = new_config["policy"]

        if "mcts" in new_config:
            mcts_config = new_config["mcts"]
            if "uct_c" in mcts_config:
                self.config.uct_c = mcts_config["uct_c"]
            if "num_children" in mcts_config:
                self.config.num_children = mcts_config["num_children"]

        if "evaluation" in new_config:
            eval_config = new_config["evaluation"]
            if "cv_folds" in eval_config:
                self.config.cv_folds = eval_config["cv_folds"]
            if "final_selection_top_k" in eval_config:
                self.config.final_selection_top_k = eval_config["final_selection_top_k"]

        logger.info("AIRA configuration updated")


# Factory function for easy integration
def create_aira_enhanced_system(config_dict: Dict[str, Any] = None, db_path: str = None) -> AIRAEnhancedResearchSystem:
    """Create AIRA-enhanced research system with configuration"""

    config = AIRAConfig()

    if config_dict:
        # Update config from dictionary
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return AIRAEnhancedResearchSystem(config, db_path=db_path)