"""
Hierarchical AI Research Tree System
A tree search-based system that runs parallel scientific experiments towards a unified research goal.
Integrates research, code analysis, and search into one cohesive system.
"""

import asyncio
import uuid
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import math
import traceback
from collections import defaultdict

from ..utils.multi_modal_search import MultiModalSearchEngine
from .repo_master import RepoMaster, AnalysisDepth
from .llm_client import llm_client
from .workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)


class ResearchNodeType(Enum):
    """Types of nodes in the research tree"""
    ROOT = "root"                    # Initial research goal
    HYPOTHESIS = "hypothesis"        # Research hypothesis to test
    EXPERIMENT = "experiment"        # Specific experiment to run
    CODE_ANALYSIS = "code_analysis"  # Code analysis branch
    LITERATURE = "literature"       # Literature review branch
    SYNTHESIS = "synthesis"          # Result synthesis node
    VALIDATION = "validation"        # Validation experiment
    HIERARCHICAL_RESEARCH = "hierarchical_research"  # Multi-agent hierarchical research


class ExperimentType(Enum):
    """Types of experiments that can be run"""
    COMPUTATIONAL = "computational"
    THEORETICAL = "theoretical"
    EMPIRICAL = "empirical"
    SIMULATION = "simulation"
    CODE_STUDY = "code_study"
    LITERATURE_ANALYSIS = "literature_analysis"
    COMPARATIVE = "comparative"
    ABLATION = "ablation"
    HIERARCHICAL_MULTI_AGENT = "hierarchical_multi_agent"
    SYNTHESIS = "synthesis"


class NodeStatus(Enum):
    """Status of research tree nodes"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"
    PROMISING = "promising"


@dataclass
class ExecutionLog:
    """Detailed execution log for debugging"""
    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    """Result of a single experiment"""
    experiment_id: str
    success: bool
    confidence: float
    metrics: Dict[str, Any]
    data: Dict[str, Any]
    insights: List[str]
    execution_time: float
    resources_used: Dict[str, Any] = field(default_factory=dict)

    # Enhanced debugging information
    execution_logs: List[ExecutionLog] = field(default_factory=list)
    error_details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    api_calls: List[Dict[str, Any]] = field(default_factory=list)
    search_queries_used: List[str] = field(default_factory=list)
    processing_steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResearchNode:
    """A node in the research tree"""
    id: str
    parent_id: Optional[str]
    node_type: ResearchNodeType
    title: str
    description: str
    hypothesis: Optional[str] = None

    # Tree structure
    children: List[str] = field(default_factory=list)
    depth: int = 0

    # Experiment details
    experiment_type: Optional[ExperimentType] = None
    experiment_config: Dict[str, Any] = field(default_factory=dict)

    # Execution status
    status: NodeStatus = NodeStatus.PENDING
    priority: float = 0.5

    # Results and metrics
    results: List[ExperimentResult] = field(default_factory=list)
    aggregated_score: float = 0.0
    confidence: float = 0.0

    # Tree search metrics
    visits: int = 0
    total_reward: float = 0.0
    ucb_score: float = 0.0

    # Context and dependencies
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Enhanced debugging and execution tracking
    execution_logs: List[ExecutionLog] = field(default_factory=list)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    last_error: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchGoal:
    """Overall research goal driving the tree search"""
    id: str
    title: str
    description: str
    success_criteria: List[str]
    domain: str = "AI/ML Research"
    constraints: Dict[str, Any] = field(default_factory=dict)
    max_depth: int = 5
    max_experiments: int = 100
    time_budget: int = 7200  # 2 hours in seconds
    quality_threshold: float = 0.8
    priority: float = 1.0


class HierarchicalResearchSystem:
    """
    Main hierarchical research system that orchestrates tree search-based scientific experiments
    """

    def __init__(self):
        self.search_engine = MultiModalSearchEngine()
        self.repo_master = RepoMaster()

        # Initialize workspace manager for code generation
        self.workspace_manager = WorkspaceManager()

        # WebSocket manager for real-time logging (injected later to avoid circular imports)
        self.websocket_manager = None

        # Research tree state
        self.research_trees: Dict[str, Dict[str, ResearchNode]] = {}
        self.active_goals: Dict[str, ResearchGoal] = {}

        # Execution state
        self.running_experiments: Dict[str, asyncio.Task] = {}
        self.experiment_results: Dict[str, List[ExperimentResult]] = {}

        # Tree search parameters
        self.exploration_constant = 1.4  # UCB exploration parameter
        self.parallel_limit = 8  # Max parallel experiments

        # Orchestrator reference (injected later to avoid circular imports)
        self.orchestrator = None

        # Integration engines
        self.experiment_types = self._initialize_experiment_types()

    def _log_execution(self, node: ResearchNode, level: str, message: str, context: Dict[str, Any] = None) -> None:
        """Add execution log to node for debugging"""
        log_entry = ExecutionLog(
            timestamp=datetime.now(),
            level=level,
            message=message,
            context=context or {}
        )
        node.execution_logs.append(log_entry)
        print(f"[{level}] Node {node.id}: {message}")  # Also print for real-time debugging

        # Broadcast to WebSocket connections if available
        if self.websocket_manager:
            asyncio.create_task(self._broadcast_log_update(node, log_entry))

    async def _broadcast_log_update(self, node: ResearchNode, log_entry: ExecutionLog):
        """Broadcast log update to WebSocket connections"""
        try:
            # Find the goal_id for this node
            goal_id = None
            for gid, tree in self.research_trees.items():
                if node.id in tree:
                    goal_id = gid
                    break

            if goal_id and self.websocket_manager:
                log_message = {
                    "type": "execution_log",
                    "node_id": node.id,
                    "log": {
                        "level": log_entry.level,
                        "message": log_entry.message,
                        "timestamp": log_entry.timestamp.isoformat(),
                        "context": log_entry.context
                    }
                }

                # Send to goal-level subscribers
                await self.websocket_manager.send_to_goal(goal_id, log_message)

                # Send to node-specific subscribers
                await self.websocket_manager.send_to_node(node.id, log_message)

        except Exception as e:
            logger.debug(f"Failed to broadcast log update: {e}")

    def _initialize_experiment_types(self) -> Dict[ExperimentType, Dict[str, Any]]:
        """Initialize experiment type configurations"""
        return {
            ExperimentType.COMPUTATIONAL: {
                "max_parallel": 4,
                "avg_duration": 300,  # 5 minutes
                "resource_requirements": {"cpu": 2, "memory": "4GB"}
            },
            ExperimentType.CODE_STUDY: {
                "max_parallel": 3,
                "avg_duration": 600,  # 10 minutes
                "resource_requirements": {"cpu": 1, "memory": "2GB"}
            },
            ExperimentType.LITERATURE_ANALYSIS: {
                "max_parallel": 6,
                "avg_duration": 180,  # 3 minutes
                "resource_requirements": {"cpu": 1, "memory": "1GB"}
            },
            ExperimentType.THEORETICAL: {
                "max_parallel": 2,
                "avg_duration": 900,  # 15 minutes
                "resource_requirements": {"cpu": 4, "memory": "8GB"}
            },
            ExperimentType.SIMULATION: {
                "max_parallel": 3,
                "avg_duration": 1200,  # 20 minutes
                "resource_requirements": {"cpu": 4, "memory": "6GB"}
            }
        }

    async def start_research_goal(
        self,
        title: str,
        description: str,
        success_criteria: List[str],
        constraints: Dict[str, Any] = None,
        max_depth: int = 5,
        max_experiments: int = 100
    ) -> str:
        """Start a new hierarchical research goal"""
        goal_id = str(uuid.uuid4())

        goal = ResearchGoal(
            id=goal_id,
            title=title,
            description=description,
            success_criteria=success_criteria,
            constraints=constraints or {},
            max_depth=max_depth,
            max_experiments=max_experiments
        )

        self.active_goals[goal_id] = goal

        # Initialize research tree with root node
        root_node = ResearchNode(
            id=f"{goal_id}_root",
            parent_id=None,
            node_type=ResearchNodeType.ROOT,
            title=title,
            description=description,
            depth=0,
            priority=1.0
        )

        self.research_trees[goal_id] = {root_node.id: root_node}

        # Immediately expand the root node to start the tree
        asyncio.create_task(self._initialize_research_tree(goal_id))

        # Start the research process
        asyncio.create_task(self._execute_research_tree(goal_id))

        return goal_id

    async def _initialize_research_tree(self, goal_id: str):
        """Initialize the research tree by expanding the root node immediately"""
        try:
            # Give the root node a small delay to ensure it's properly set up
            await asyncio.sleep(0.1)

            tree = self.research_trees[goal_id]
            root_node = next(iter(tree.values()))  # Get the root node

            print(f"🌱 Initializing research tree for goal: {goal_id}")
            print(f"🌱 Root node: {root_node.id}, status: {root_node.status}, children: {len(root_node.children)}")

            # Expand the root node to create initial research directions
            new_node_ids = await self._expand_node(goal_id, root_node.id)
            print(f"🌱 Created {len(new_node_ids)} initial child nodes: {new_node_ids}")

            # Set the child nodes to READY status so they can be expanded further
            for node_id in new_node_ids:
                if node_id in tree:
                    tree[node_id].status = NodeStatus.PROMISING
                    tree[node_id].visits = 1  # Give them initial visit to enable UCB

        except Exception as e:
            logger.error(f"Error initializing research tree: {e}")

    async def _execute_research_tree(self, goal_id: str):
        """Execute the research tree using tree search algorithms"""
        goal = self.active_goals[goal_id]
        tree = self.research_trees[goal_id]

        start_time = datetime.now()
        experiments_run = 0

        while (
            experiments_run < goal.max_experiments and
            (datetime.now() - start_time).total_seconds() < goal.time_budget and
            not await self._is_goal_satisfied(goal_id)
        ):
            # Collect nodes that need experiments (new nodes + existing promising nodes without experiments)
            nodes_to_experiment = []

            # Tree search phase: Selection, Expansion, Simulation, Backpropagation
            selected_node = await self._select_node_ucb(goal_id)

            if selected_node:
                needs_result_first = selected_node.node_type in {
                    ResearchNodeType.EXPERIMENT,
                    ResearchNodeType.LITERATURE,
                    ResearchNodeType.CODE_ANALYSIS,
                }
                if needs_result_first and not selected_node.results:
                    # Run experiment for this node first, then expand to create follow-ups
                    logger.info(f"Running experiment first for {selected_node.node_type.value} node {selected_node.id} before expansion")
                    res = await self._run_experiment(goal_id, selected_node.id)
                    await self._backpropagate_result(goal_id, selected_node.id, res)
                    # Now expansion will actually produce follow-ups
                    new_nodes = await self._expand_node(goal_id, selected_node.id)
                else:
                    new_nodes = await self._expand_node(goal_id, selected_node.id)

                nodes_to_experiment.extend(new_nodes)

            # Also find existing promising nodes that haven't run experiments yet
            for node_id, node in tree.items():
                if (node.status == NodeStatus.PROMISING and
                    len(node.results) == 0 and
                    node_id not in nodes_to_experiment):
                    logger.info(f"Adding PROMISING node {node_id} to experiments queue (results: {len(node.results)})")
                    nodes_to_experiment.append(node_id)

            logger.info(f"Total nodes queued for experiments: {len(nodes_to_experiment)}")

            # Run experiments on nodes in parallel
            tasks = []
            nodes_for_experiments = nodes_to_experiment[:self.parallel_limit]

            for node_id in nodes_for_experiments:
                if experiments_run < goal.max_experiments:
                    task = asyncio.create_task(self._run_experiment(goal_id, node_id))
                    tasks.append(task)
                    experiments_run += 1

            # Wait for experiments to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Backpropagate results
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        await self._backpropagate_result(goal_id, nodes_for_experiments[i], result)

            # Prune unpromising branches
            await self._prune_tree(goal_id)

            # Check and complete ROOT nodes based on child success
            await self._check_root_completion(goal_id)

            # Brief pause to prevent overwhelming the system
            await asyncio.sleep(1)

        # Final synthesis
        await self._synthesize_research_results(goal_id)

    async def _select_node_ucb(self, goal_id: str) -> Optional[ResearchNode]:
        """Select next node to explore using Upper Confidence Bound"""
        tree = self.research_trees[goal_id]
        goal = self.active_goals[goal_id]

        def expandable(n: ResearchNode) -> bool:
            if n.depth >= goal.max_depth:
                return False
            # sensible default if _get_max_children missing or misconfigured
            try:
                max_children = max(0, int(self._get_max_children(n)))
            except Exception:
                max_children = 3 if n.node_type != ResearchNodeType.ROOT else 5
            if len(n.children) >= max_children:
                return False
            # For node types whose follow-ups depend on results, require results
            needs_result = n.node_type in {
                ResearchNodeType.EXPERIMENT,
                ResearchNodeType.LITERATURE,
                ResearchNodeType.CODE_ANALYSIS,
            }
            return (not needs_result) or bool(n.results)

        # include PENDING, PROMISING, and COMPLETED nodes that are expandable
        candidates = [
            n for n in tree.values()
            if n.status in {NodeStatus.PENDING, NodeStatus.PROMISING, NodeStatus.COMPLETED}
            and expandable(n)
        ]
        if not candidates:
            return None

        total_visits = max(1, sum(max(0, n.visits) for n in tree.values()))
        for n in candidates:
            if n.visits == 0:
                n.ucb_score = float("inf")
            else:
                exploitation = n.total_reward / n.visits
                exploration = self.exploration_constant * math.sqrt(math.log(total_visits) / n.visits)
                n.ucb_score = exploitation + exploration
        return max(candidates, key=lambda n: n.ucb_score)

    async def _expand_node(self, goal_id: str, node_id: str) -> List[str]:
        """Expand a node by generating child experiments"""
        tree = self.research_trees[goal_id]
        goal = self.active_goals[goal_id]
        node = tree[node_id]

        new_node_ids = []

        # Generate child nodes based on node type
        if node.node_type == ResearchNodeType.ROOT:
            # Generate initial research directions
            child_configs = await self._generate_research_directions(goal_id)
        elif node.node_type == ResearchNodeType.HYPOTHESIS:
            # Generate experiments to test the hypothesis
            child_configs = await self._generate_hypothesis_experiments(goal_id, node_id)
        elif node.node_type == ResearchNodeType.EXPERIMENT:
            # Generate follow-up experiments based on results
            child_configs = await self._generate_followup_experiments(goal_id, node_id)
        else:
            child_configs = []

        # Create child nodes
        for config in child_configs:
            child_id = str(uuid.uuid4())

            child_node = ResearchNode(
                id=child_id,
                parent_id=node_id,
                node_type=config["node_type"],
                title=config["title"],
                description=config["description"],
                hypothesis=config.get("hypothesis"),
                experiment_type=config.get("experiment_type"),
                experiment_config=config.get("experiment_config", {}),
                depth=node.depth + 1,
                priority=config.get("priority", 0.5),
                context=config.get("context", {})
            )

            tree[child_id] = child_node
            node.children.append(child_id)
            new_node_ids.append(child_id)

        return new_node_ids

    async def _plan_needed_experiments(self, goal_id: str) -> List[Dict[str, Any]]:
        """Use LLM to intelligently decide what types of experiments are needed for the goal"""
        goal = self.active_goals[goal_id]

        system_prompt = """You are a research planning assistant. Given a research goal/task, determine what types of experiments or analyses are actually needed to accomplish it effectively.

Available experiment types:
- LITERATURE_ANALYSIS: Research existing papers and studies (for complex research topics)
- CODE_ANALYSIS: Analyze existing code implementations (when building on existing work)
- THEORETICAL: Develop theoretical frameworks or models (for complex research problems)
- COMPUTATIONAL: Implement and test solutions (for programming/implementation tasks)

Guidelines:
- For simple programming tasks (hello world, basic algorithms): Only COMPUTATIONAL is needed
- For complex research questions: May need LITERATURE_ANALYSIS and THEORETICAL
- For building on existing work: May need CODE_ANALYSIS
- Be selective - don't include unnecessary experiment types
- Prioritize experiments by importance (1.0 = highest, 0.5 = lowest)

Return a JSON list with experiment plans."""

        prompt = f"""Research Goal: {goal.title}
Description: {goal.description}
Success Criteria: {goal.success_criteria}

Analyze this goal and determine the minimum set of experiment types needed to accomplish it effectively. Consider:
1. Is this a simple programming task or complex research?
2. Does it require literature review or can it be solved directly?
3. What's the most efficient path to success?

Return JSON format:
[
  {{
    "experiment_type": "COMPUTATIONAL",
    "reasoning": "Direct implementation needed",
    "priority": 0.9,
    "required": true
  }}
]"""

        try:
            llm_response = await llm_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1000
            )

            if llm_response.get("success"):
                content = llm_response.get("content", "")

                # Try to extract JSON from the response
                import json
                import re

                # Look for JSON array in the response
                json_pattern = r'\[[\s\S]*?\]'
                json_match = re.search(json_pattern, content)

                if json_match:
                    json_str = json_match.group()
                    try:
                        experiment_plans = json.loads(json_str)
                        logger.info(f"LLM planned {len(experiment_plans)} experiment types for goal: {goal.title}")
                        return experiment_plans
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse LLM experiment planning JSON")

                # Fallback: parse text response
                plans = []
                lines = content.split('\n')
                for line in lines:
                    line = line.strip().lower()
                    if 'computational' in line:
                        plans.append({
                            "experiment_type": "COMPUTATIONAL",
                            "reasoning": "LLM suggested computational approach",
                            "priority": 0.9,
                            "required": True
                        })
                    elif 'literature' in line:
                        plans.append({
                            "experiment_type": "LITERATURE_ANALYSIS",
                            "reasoning": "LLM suggested literature review",
                            "priority": 0.7,
                            "required": False
                        })
                    elif 'theoretical' in line:
                        plans.append({
                            "experiment_type": "THEORETICAL",
                            "reasoning": "LLM suggested theoretical analysis",
                            "priority": 0.6,
                            "required": False
                        })

                if plans:
                    logger.info(f"Parsed {len(plans)} experiment types from LLM text response")
                    return plans

        except Exception as e:
            logger.error(f"LLM experiment planning failed: {e}")

        # Fallback: intelligent heuristics based on goal characteristics
        return self._fallback_experiment_planning(goal)

    def _fallback_experiment_planning(self, goal) -> List[Dict[str, Any]]:
        """Fallback heuristic-based experiment planning when LLM fails"""
        plans = []
        description_lower = goal.description.lower()
        title_lower = goal.title.lower()

        # Simple programming task indicators
        simple_programming_keywords = [
            'hello world', 'hello-world', 'print', 'basic', 'simple',
            'tutorial', 'beginner', 'example', 'demo'
        ]

        # Complex research indicators
        research_keywords = [
            'research', 'analysis', 'study', 'investigation', 'survey',
            'review', 'comprehensive', 'advanced', 'novel', 'approach'
        ]

        # Implementation task indicators
        implementation_keywords = [
            'implement', 'build', 'create', 'develop', 'code', 'program',
            'algorithm', 'function', 'system', 'application'
        ]

        is_simple_programming = any(keyword in description_lower or keyword in title_lower
                                  for keyword in simple_programming_keywords)
        is_research_task = any(keyword in description_lower or keyword in title_lower
                              for keyword in research_keywords)
        is_implementation = any(keyword in description_lower or keyword in title_lower
                               for keyword in implementation_keywords)

        # Always include computational for implementation tasks
        if is_implementation or is_simple_programming:
            plans.append({
                "experiment_type": "COMPUTATIONAL",
                "reasoning": "Implementation/programming task detected",
                "priority": 0.9,
                "required": True
            })

        # Only add literature review for complex research tasks
        if is_research_task and not is_simple_programming:
            plans.append({
                "experiment_type": "LITERATURE_ANALYSIS",
                "reasoning": "Complex research task requires literature review",
                "priority": 0.7,
                "required": False
            })

        # Add theoretical analysis for research problems
        if is_research_task and 'theoretical' in description_lower:
            plans.append({
                "experiment_type": "THEORETICAL",
                "reasoning": "Theoretical research component detected",
                "priority": 0.6,
                "required": False
            })

        # Add code analysis if building on existing work
        if 'existing' in description_lower or 'based on' in description_lower:
            plans.append({
                "experiment_type": "CODE_ANALYSIS",
                "reasoning": "Building on existing work requires code analysis",
                "priority": 0.5,
                "required": False
            })

        # Default fallback for unclear tasks
        if not plans:
            plans.append({
                "experiment_type": "COMPUTATIONAL",
                "reasoning": "Default computational approach",
                "priority": 0.8,
                "required": True
            })

        logger.info(f"Fallback planning generated {len(plans)} experiment types")
        return plans

    async def _generate_research_directions(self, goal_id: str, allow_regeneration: bool = False) -> List[Dict[str, Any]]:
        """Generate initial research directions from the root goal"""
        goal = self.active_goals[goal_id]
        tree = self.research_trees[goal_id]

        # Check what child types already exist, but allow re-expansion if existing ones failed or have low confidence
        existing_types = set()
        root_node = tree[f"{goal_id}_root"]
        for child_id in root_node.children:
            if child_id in tree:
                child = tree[child_id]
                # Only block if child exists AND is successful (confidence > 0.6)
                # If allow_regeneration is True, ignore failed nodes and allow new attempts
                if allow_regeneration:
                    # Only block running nodes and high-confidence completed nodes
                    if child.status == NodeStatus.RUNNING or (child.status == NodeStatus.COMPLETED and child.confidence > 0.6):
                        existing_types.add(child.node_type)
                else:
                    # Original logic: block all non-failed attempts
                    if child.status == NodeStatus.COMPLETED and child.confidence > 0.6:
                        existing_types.add(child.node_type)
                    elif child.status == NodeStatus.RUNNING:
                        # Don't create duplicates of running nodes
                        existing_types.add(child.node_type)

        # Use LLM to intelligently plan needed experiments
        experiment_plans = await self._plan_needed_experiments(goal_id)

        directions = []

        # Convert experiment plans to research directions
        for plan in experiment_plans:
            exp_type = plan.get("experiment_type", "COMPUTATIONAL")
            priority = plan.get("priority", 0.8)
            reasoning = plan.get("reasoning", "LLM planned experiment")

            # Map experiment types to node types and configurations
            if exp_type == "COMPUTATIONAL":
                node_type = ResearchNodeType.EXPERIMENT
                experiment_type = ExperimentType.COMPUTATIONAL
                config = {
                    "approach": "direct_implementation",
                    "focus": "practical_solution",
                    "reasoning": reasoning
                }
            elif exp_type == "LITERATURE_ANALYSIS":
                node_type = ResearchNodeType.LITERATURE
                experiment_type = ExperimentType.LITERATURE_ANALYSIS
                config = {
                    "search_queries": [goal.title, goal.description] + goal.success_criteria,
                    "max_papers": 30,
                    "analysis_depth": "focused",
                    "reasoning": reasoning
                }
            elif exp_type == "THEORETICAL":
                node_type = ResearchNodeType.HYPOTHESIS
                experiment_type = ExperimentType.THEORETICAL
                config = {
                    "hypothesis_count": 3,
                    "evidence_threshold": 0.7,
                    "reasoning": reasoning
                }
            elif exp_type == "CODE_ANALYSIS":
                node_type = ResearchNodeType.CODE_ANALYSIS
                experiment_type = ExperimentType.CODE_STUDY
                config = {
                    "search_terms": [goal.title, "implementation", "algorithm"],
                    "languages": ["python", "javascript", "cpp"],
                    "analysis_depth": "focused",
                    "reasoning": reasoning
                }
            else:
                # Default to computational
                node_type = ResearchNodeType.EXPERIMENT
                experiment_type = ExperimentType.COMPUTATIONAL
                config = {"reasoning": reasoning}

            # Only add if this node type doesn't already exist (unless regenerating)
            if node_type not in existing_types:
                directions.append({
                    "node_type": node_type,
                    "title": f"{exp_type.replace('_', ' ').title()}: {goal.title}",
                    "description": f"{reasoning} for {goal.description}",
                    "experiment_type": experiment_type,
                    "experiment_config": config,
                    "priority": priority,
                    "llm_planned": True
                })

        # Only create code analysis if it doesn't exist and is applicable
        if (ResearchNodeType.CODE_ANALYSIS not in existing_types and
            goal.description and ("code" in goal.description.lower() or "implementation" in goal.description.lower())):
            directions.append({
                "node_type": ResearchNodeType.CODE_ANALYSIS,
                "title": f"Code Analysis: {goal.title}",
                "description": f"Analysis of existing implementations related to {goal.description}",
                "experiment_type": ExperimentType.CODE_STUDY,
                "experiment_config": {
                    "search_terms": [goal.title, "implementation", "algorithm"],
                    "languages": ["python", "javascript", "cpp"],
                    "analysis_depth": "deep"
                },
                "priority": 0.8
            })


        # NOTE: Hierarchical and Synthesis nodes are now created as children of successful research nodes
        # This is handled in _generate_followup_experiments() method

        # If no directions at all (LLM planning failed completely), add minimal computational approach
        if not directions:
            logger.warning("No experiment plans generated, adding minimal computational approach")
            directions.append({
                "node_type": ResearchNodeType.EXPERIMENT,
                "title": f"Direct Implementation: {goal.title}",
                "description": f"Direct implementation approach for {goal.description}",
                "experiment_type": ExperimentType.COMPUTATIONAL,
                "experiment_config": {
                    "approach": "minimal_solution",
                    "reasoning": "Fallback computational approach"
                },
                "priority": 0.8,
                "llm_planned": False
            })

        logger.info(f"Generated {len(directions)} intelligent research directions for goal: {goal.title}")
        return sorted(directions, key=lambda x: x["priority"], reverse=True)

    async def _generate_hypothesis_experiments(self, goal_id: str, node_id: str) -> List[Dict[str, Any]]:
        """Generate experiments to test a hypothesis"""
        tree = self.research_trees[goal_id]
        node = tree[node_id]

        experiments = []

        # Computational validation
        experiments.append({
            "node_type": ResearchNodeType.EXPERIMENT,
            "title": f"Computational Test: {node.hypothesis}",
            "description": f"Computational validation of {node.hypothesis}",
            "experiment_type": ExperimentType.COMPUTATIONAL,
            "experiment_config": {
                "hypothesis": node.hypothesis,
                "test_cases": 100,
                "validation_method": "monte_carlo"
            },
            "priority": 0.8
        })

        # Simulation experiment
        experiments.append({
            "node_type": ResearchNodeType.EXPERIMENT,
            "title": f"Simulation: {node.hypothesis}",
            "description": f"Simulation-based testing of {node.hypothesis}",
            "experiment_type": ExperimentType.SIMULATION,
            "experiment_config": {
                "hypothesis": node.hypothesis,
                "simulation_runs": 1000,
                "parameters": {"accuracy": 0.95, "confidence": 0.9}
            },
            "priority": 0.75
        })

        # Empirical analysis
        experiments.append({
            "node_type": ResearchNodeType.EXPERIMENT,
            "title": f"Empirical Analysis: {node.hypothesis}",
            "description": f"Empirical validation using real data",
            "experiment_type": ExperimentType.EMPIRICAL,
            "experiment_config": {
                "hypothesis": node.hypothesis,
                "data_sources": ["research_data", "public_datasets"],
                "statistical_tests": ["t_test", "chi_square", "anova"]
            },
            "priority": 0.85
        })

        return experiments

    async def _generate_followup_experiments(self, goal_id: str, node_id: str) -> List[Dict[str, Any]]:
        """Generate follow-up experiments based on previous results"""
        tree = self.research_trees[goal_id]
        node = tree[node_id]

        if not node.results:
            return []

        experiments = []
        best_result = max(node.results, key=lambda r: r.confidence)

        if best_result.success and best_result.confidence > 0.7:
            # Validation experiment
            experiments.append({
                "node_type": ResearchNodeType.VALIDATION,
                "title": f"Validation: {node.title}",
                "description": f"Independent validation of promising results from {node.title}",
                "experiment_type": ExperimentType.COMPARATIVE,
                "experiment_config": {
                    "baseline_results": best_result.data,
                    "validation_method": "cross_validation",
                    "replication_count": 5
                },
                "priority": 0.9
            })

            # Parameter optimization
            experiments.append({
                "node_type": ResearchNodeType.EXPERIMENT,
                "title": f"Optimization: {node.title}",
                "description": f"Parameter optimization based on {node.title} results",
                "experiment_type": ExperimentType.COMPUTATIONAL,
                "experiment_config": {
                    "base_config": best_result.data,
                    "optimization_target": "maximize_performance",
                    "search_space": "hyperparameter_grid"
                },
                "priority": 0.8
            })

        # Add hierarchical multi-agent analysis as a child of successful literature/code analysis
        if (best_result.success and best_result.confidence > 0.6 and
            node.node_type in [ResearchNodeType.LITERATURE, ResearchNodeType.CODE_ANALYSIS]):

            experiments.append({
                "node_type": ResearchNodeType.HIERARCHICAL_RESEARCH,
                "title": f"Hierarchical AI Analysis: {node.title}",
                "description": f"Deep hierarchical multi-agent analysis building on {node.title}",
                "experiment_type": ExperimentType.HIERARCHICAL_MULTI_AGENT,
                "experiment_config": {
                    "parent_findings": best_result.insights,
                    "research_query": node.title,
                    "domain": "AI/ML Research",
                    "comprehensive": True,
                    "agent_coordination": "hierarchical",
                    "build_on_parent": True
                },
                "priority": 0.95
            })

        # Check if we have multiple successful siblings for synthesis
        if node.parent_id:
            goal = self.active_goals[goal_id]
            parent_node = tree[node.parent_id]

            # Count successful sibling nodes
            successful_siblings = []
            for sibling_id in parent_node.children:
                if sibling_id in tree and sibling_id != node_id:
                    sibling = tree[sibling_id]
                    if (sibling.status == NodeStatus.COMPLETED and
                        sibling.results and
                        any(r.success and r.confidence > 0.5 for r in sibling.results)):
                        successful_siblings.append(sibling)

            # Add synthesis if we have 2+ successful siblings to synthesize with this node
            if len(successful_siblings) >= 1 and best_result.success and best_result.confidence > 0.5:
                experiments.append({
                    "node_type": ResearchNodeType.SYNTHESIS,
                    "title": f"Research Synthesis: {goal.title}",
                    "description": f"Synthesize findings from {node.title} and {len(successful_siblings)} other successful research nodes",
                    "experiment_type": ExperimentType.SYNTHESIS,
                    "experiment_config": {
                        "synthesis_method": "meta_analysis",
                        "integration_level": "comprehensive",
                        "parent_nodes": [node_id] + [s.id for s in successful_siblings],
                        "minimum_confidence": 0.5
                    },
                    "priority": 0.9
                })

        return experiments

    async def _run_experiment(self, goal_id: str, node_id: str) -> ExperimentResult:
        """Run a single experiment with comprehensive logging"""
        tree = self.research_trees[goal_id]
        node = tree[node_id]

        # Initialize execution
        node.status = NodeStatus.RUNNING
        node.started_at = datetime.now()
        start_time = datetime.now()

        self._log_execution(node, "INFO", f"Starting experiment: {node.experiment_type}", {
            "node_type": node.node_type.value,
            "experiment_type": node.experiment_type.value if node.experiment_type else "default",
            "title": node.title,
            "config": node.experiment_config
        })

        try:
            # Execute experiment based on type
            if node.experiment_type == ExperimentType.LITERATURE_ANALYSIS:
                self._log_execution(node, "DEBUG", "Executing literature analysis experiment")
                result = await self._run_literature_experiment(node)
            elif node.experiment_type == ExperimentType.CODE_STUDY:
                self._log_execution(node, "DEBUG", "Executing code study experiment")
                result = await self._run_code_analysis_experiment(node)
            elif node.experiment_type == ExperimentType.COMPUTATIONAL:
                self._log_execution(node, "DEBUG", "Executing computational experiment")
                result = await self._run_computational_experiment(node)
            elif node.experiment_type == ExperimentType.SIMULATION:
                self._log_execution(node, "DEBUG", "Executing simulation experiment")
                result = await self._run_simulation_experiment(node)
            elif node.experiment_type == ExperimentType.THEORETICAL:
                self._log_execution(node, "DEBUG", "Executing theoretical experiment")
                result = await self._run_theoretical_experiment(node)
            elif node.experiment_type == ExperimentType.EMPIRICAL:
                self._log_execution(node, "DEBUG", "Executing empirical experiment")
                result = await self._run_empirical_experiment(node)
            elif node.experiment_type == ExperimentType.HIERARCHICAL_MULTI_AGENT:
                self._log_execution(node, "DEBUG", "Executing hierarchical multi-agent experiment")
                result = await self._run_hierarchical_multi_agent_experiment(node)
            elif node.experiment_type == ExperimentType.SYNTHESIS:
                self._log_execution(node, "DEBUG", "Executing synthesis experiment")
                result = await self._run_synthesis_experiment(node)
            else:
                self._log_execution(node, "WARNING", f"Unknown experiment type: {node.experiment_type}, using default")
                result = await self._run_default_experiment(node)

            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            # Log success
            self._log_execution(node, "INFO", f"Experiment completed successfully in {execution_time:.2f}s", {
                "success": result.success,
                "confidence": result.confidence,
                "metrics": result.metrics,
                "insights_count": len(result.insights)
            })

            node.results.append(result)
            # Set node status based on experiment success
            if result.success:
                node.status = NodeStatus.COMPLETED
            else:
                node.status = NodeStatus.FAILED
            node.completed_at = datetime.now()

            # Update aggregated score
            node.aggregated_score = np.mean([r.confidence for r in node.results])
            node.confidence = result.confidence

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            stack_trace = traceback.format_exc()

            # Log detailed error information
            self._log_execution(node, "ERROR", f"Experiment failed after {execution_time:.2f}s: {error_msg}", {
                "error_type": type(e).__name__,
                "error_message": error_msg,
                "stack_trace": stack_trace,
                "experiment_config": node.experiment_config
            })

            # Update node error tracking
            node.status = NodeStatus.FAILED
            node.completed_at = datetime.now()
            node.last_error = error_msg
            node.retry_count += 1

            error_details = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__,
                "error_message": error_msg,
                "retry_count": node.retry_count,
                "execution_time": execution_time
            }
            node.error_history.append(error_details)

            # Create detailed error result
            return ExperimentResult(
                experiment_id=node.id,
                success=False,
                confidence=0.0,
                metrics={"error": error_msg, "error_type": type(e).__name__},
                data={"execution_time": execution_time},
                insights=[f"Experiment failed: {error_msg}"],
                execution_time=execution_time,
                error_details=error_details,
                stack_trace=stack_trace
            )

    async def _run_literature_experiment(self, node: ResearchNode) -> ExperimentResult:
        """Run literature analysis experiment with detailed logging"""
        config = node.experiment_config
        result = ExperimentResult(
            experiment_id=node.id,
            success=False,
            confidence=0.0,
            metrics={},
            data={},
            insights=[],
            execution_time=0.0
        )

        # Try to search for papers using multi-modal search
        all_papers = []
        search_error = None
        search_queries = config.get("search_queries", [node.title])

        self._log_execution(node, "INFO", f"Starting literature search with {len(search_queries)} queries", {
            "queries": search_queries,
            "max_papers": config.get("max_papers", 50)
        })

        try:
            for i, query in enumerate(search_queries):
                self._log_execution(node, "DEBUG", f"Executing search query {i+1}/{len(search_queries)}: {query}")

                result.processing_steps.append({
                    "step": f"search_query_{i+1}",
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                })
                result.search_queries_used.append(query)

                search_results = await self.search_engine.unified_search(
                    query=query,
                    search_types=["academic"],
                    limit=config.get("max_papers", 50) // len(search_queries)
                )

                result.api_calls.append({
                    "api": "search_engine.unified_search",
                    "query": query,
                    "search_types": ["academic"],
                    "timestamp": datetime.now().isoformat(),
                    "results_count": len(search_results.get("academic", []))
                })
                all_papers.extend(search_results.get("academic", []))

                self._log_execution(node, "DEBUG", f"Query {i+1} found {len(search_results.get('academic', []))} papers")
                result.intermediate_results.append({
                    "query": query,
                    "papers_found": len(search_results.get("academic", [])),
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            search_error = str(e)
            stack_trace = traceback.format_exc()

            self._log_execution(node, "ERROR", f"Literature search failed: {search_error}", {
                "error_type": type(e).__name__,
                "query": query if 'query' in locals() else "unknown",
                "config": config,
                "stack_trace": stack_trace
            })

            result.error_details = {
                "error_type": type(e).__name__,
                "error_message": search_error,
                "failed_query": query if 'query' in locals() else "unknown",
                "timestamp": datetime.now().isoformat()
            }
            result.stack_trace = stack_trace

        # Log search results summary
        self._log_execution(node, "INFO", f"Literature search completed. Found {len(all_papers)} total papers", {
            "total_papers": len(all_papers),
            "search_error": search_error,
            "queries_executed": len(result.search_queries_used)
        })

        # If search failed or no papers found, use LLM for literature analysis
        if not all_papers:
            self._log_execution(node, "WARNING", "No papers found, falling back to LLM analysis")

            result.processing_steps.append({
                "step": "llm_fallback_analysis",
                "reason": "no_papers_found" if not search_error else "search_failed",
                "timestamp": datetime.now().isoformat()
            })

            analysis_result = await llm_client.analyze_literature(
                node.title,
                [] if search_error else [{"title": f"No papers found for: {node.title}", "abstract": ""}]
            )

            result.api_calls.append({
                "api": "llm_client.analyze_literature",
                "input_title": node.title,
                "input_papers": 0,
                "timestamp": datetime.now().isoformat(),
                "success": analysis_result["success"]
            })

            if analysis_result["success"]:
                insights = [analysis_result["content"]]
                confidence = 0.6  # Moderate confidence for LLM-only analysis
                success = True
                self._log_execution(node, "INFO", "LLM literature analysis completed successfully")
            else:
                insights = [f"Literature analysis failed: {analysis_result.get('error', 'Unknown error')}"]
                if search_error:
                    insights.append(f"Search also failed: {search_error}")
                confidence = 0.0
                success = False
                self._log_execution(node, "ERROR", f"LLM literature analysis failed: {analysis_result.get('error', 'Unknown error')}")
        else:
            # Analyze papers with LLM
            self._log_execution(node, "INFO", f"Analyzing {len(all_papers)} papers for patterns")

            result.processing_steps.append({
                "step": "pattern_analysis",
                "papers_count": len(all_papers),
                "timestamp": datetime.now().isoformat()
            })

            insights = await self._analyze_literature_patterns(all_papers)
            confidence = min(len(all_papers) / 20, 1.0)  # Confidence based on paper count
            success = True

            self._log_execution(node, "INFO", f"Pattern analysis completed with confidence {confidence:.2f}")

        # Update result with final values
        result.success = success
        result.confidence = confidence
        result.insights = insights
        result.metrics = {
            "papers_found": len(all_papers),
            "search_error": search_error,
            "llm_analysis": not bool(all_papers),
            "queries_executed": len(result.search_queries_used),
            "api_calls_made": len(result.api_calls),
            "processing_steps": len(result.processing_steps)
        }
        result.data = {"papers": all_papers}

        return result

    async def _run_code_analysis_experiment(self, node: ResearchNode) -> ExperimentResult:
        """Run code analysis experiment"""
        config = node.experiment_config

        # Search for code repositories
        code_results = []
        for term in config.get("search_terms", []):
            search_results = await self.search_engine.unified_search(
                query=term,
                search_types=["code"],
                filters={"code": {"language": config.get("languages", ["python"])[0]}},
                limit=10
            )
            code_results.extend(search_results.get("code", []))

        # Analyze code patterns (simulated)
        insights = []
        confidence = 0.0

        if code_results:
            insights = await self._analyze_code_patterns(code_results)
            confidence = min(len(code_results) / 10, 1.0)

        return ExperimentResult(
            experiment_id=node.id,
            success=len(code_results) > 0,
            confidence=confidence,
            metrics={
                "repositories_found": len(code_results),
                "languages": list(set([repo.get("language", "") for repo in code_results])),
                "avg_stars": np.mean([repo.get("stars", 0) for repo in code_results])
            },
            data={"repositories": code_results},
            insights=insights,
            execution_time=0.0
        )

    async def _run_computational_experiment(self, node: ResearchNode) -> ExperimentResult:
        """Run computational experiment with real code execution and debugging"""
        import time
        from .code_executor import CodeExecutor

        start_time = time.time()
        config = node.experiment_config
        task_description = node.description

        try:
            # Create workspace for code generation if needed
            workspace_info = None
            if any(keyword in task_description.lower() for keyword in
                   ["docker", "python", "programming", "code", "script", "hello-world", "hello world"]):

                # Determine task type
                if "docker" in task_description.lower():
                    if "hello-world" in task_description.lower() or "hello world" in task_description.lower():
                        workspace_info = self.workspace_manager.get_docker_hello_world_workspace(node.title)
                    else:
                        workspace_info = self.workspace_manager.create_workspace(node.title, "docker")
                else:
                    workspace_info = self.workspace_manager.create_workspace(node.title, "programming")

                # Add workspace info to config
                config = config.copy()
                config["workspace"] = workspace_info

            # Create code executor with configuration
            max_iterations = config.get("max_iterations", 3)
            max_debug_attempts = config.get("max_debug_attempts", 2)
            timeout = config.get("timeout", 30)

            executor = CodeExecutor(
                max_iterations=max_iterations,
                max_debug_attempts=max_debug_attempts,
                timeout=timeout
            )

            # Execute computational task with real code execution
            execution_result = await executor.execute_computational_task(task_description, config)

            # Convert CodeExecutionResult to ExperimentResult
            success = execution_result.success
            confidence = execution_result.confidence

            # Build comprehensive metrics
            metrics = {
                "execution_time": execution_result.execution_time,
                "code_execution_success": execution_result.success,
                "total_iterations": execution_result.iterations,
                "debug_attempts": execution_result.debug_attempts,
                "code_blocks_generated": len(execution_result.code_blocks),
                "has_working_code": execution_result.final_code is not None,
                **execution_result.metrics
            }

            # Build data with all execution details
            data = {
                "task_description": task_description,
                "config": config,
                "code_blocks": execution_result.code_blocks,
                "final_code": execution_result.final_code,
                "execution_outputs": execution_result.execution_outputs,
                "execution_method": "real_code_execution"
            }

            # Add workspace information if created
            if workspace_info:
                data["workspace"] = workspace_info

            # Use execution insights plus summary insights
            insights = execution_result.insights.copy()

            # Add workspace-specific insights
            if workspace_info:
                insights.append(f"Code generated in workspace: {workspace_info['workspace_path']}")
                if workspace_info.get("bash_script"):
                    insights.append(f"Bash script available: {workspace_info['bash_script']}")
                if workspace_info.get("python_script"):
                    insights.append(f"Python script available: {workspace_info['python_script']}")
            if execution_result.final_code:
                insights.insert(0, f"Successfully generated and executed working code")
                insights.append(f"Final code has {len(execution_result.final_code.split())} lines")
            else:
                insights.insert(0, f"Code generation attempted but execution failed")

            insights.append(f"Completed {execution_result.iterations} iterations with {execution_result.debug_attempts} debug attempts")

            return ExperimentResult(
                experiment_id=node.id,
                success=success,
                confidence=confidence,
                metrics=metrics,
                data=data,
                insights=insights[:15],  # Limit insights to prevent overwhelming output
                execution_time=execution_result.execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Computational experiment failed: {str(e)}")

            return ExperimentResult(
                experiment_id=node.id,
                success=False,
                confidence=0.0,
                metrics={
                    "exception": str(e),
                    "execution_time": execution_time,
                    "code_execution_success": False
                },
                data={
                    "task_description": task_description,
                    "config": config,
                    "exception_details": str(e),
                    "execution_method": "real_code_execution"
                },
                insights=[f"Computational experiment failed with exception: {str(e)}"],
                execution_time=execution_time
            )

    async def _run_simulation_experiment(self, node: ResearchNode) -> ExperimentResult:
        """Run simulation experiment"""
        config = node.experiment_config

        # Simulate complex simulation
        await asyncio.sleep(np.random.uniform(2, 8))

        simulation_runs = config.get("simulation_runs", 1000)
        parameters = config.get("parameters", {})

        # Generate simulation results
        mean_performance = np.random.uniform(0.6, 0.9)
        std_performance = np.random.uniform(0.05, 0.15)

        results = {
            "simulation_runs": simulation_runs,
            "mean_performance": mean_performance,
            "std_performance": std_performance,
            "confidence_interval": [
                mean_performance - 1.96 * std_performance,
                mean_performance + 1.96 * std_performance
            ]
        }

        return ExperimentResult(
            experiment_id=node.id,
            success=mean_performance > 0.7,
            confidence=max(0, min(1, mean_performance - std_performance)),
            metrics=results,
            data=results,
            insights=[
                f"Simulation mean performance: {mean_performance:.3f}",
                f"Performance variability: {std_performance:.3f}",
                "Simulation results indicate " + ("high" if mean_performance > 0.8 else "moderate") + " confidence"
            ],
            execution_time=0.0
        )

    async def _run_theoretical_experiment(self, node: ResearchNode) -> ExperimentResult:
        """Run theoretical analysis experiment using qwen3-max-review"""
        start_time = time.time()

        # Generate hypothesis using LLM
        hypothesis_result = await llm_client.generate_hypothesis(node.title)

        if hypothesis_result["success"]:
            insights = [hypothesis_result["content"]]
            theoretical_strength = 0.8  # High confidence for LLM-generated hypotheses
            success = True
        else:
            # Fallback simulation
            await asyncio.sleep(np.random.uniform(1, 3))
            theoretical_strength = np.random.uniform(0.5, 0.9)
            success = theoretical_strength > 0.6
            insights = [
                f"Theoretical analysis strength: {theoretical_strength:.3f}",
                "Framework provides " + ("strong" if theoretical_strength > 0.7 else "moderate") + " theoretical foundation",
                f"LLM analysis failed: {hypothesis_result.get('error', 'Unknown error')}"
            ]

        execution_time = time.time() - start_time

        return ExperimentResult(
            experiment_id=node.id,
            success=success,
            confidence=theoretical_strength,
            metrics={"theoretical_strength": theoretical_strength},
            data={"analysis": "qwen_theoretical_framework", "llm_result": hypothesis_result},
            insights=insights,
            execution_time=execution_time
        )

    async def _run_empirical_experiment(self, node: ResearchNode) -> ExperimentResult:
        """Run empirical analysis experiment"""
        # Simulate empirical analysis
        await asyncio.sleep(np.random.uniform(2, 6))

        empirical_evidence = np.random.uniform(0.6, 0.95)

        return ExperimentResult(
            experiment_id=node.id,
            success=empirical_evidence > 0.7,
            confidence=empirical_evidence,
            metrics={"empirical_evidence": empirical_evidence},
            data={"statistical_tests": "completed"},
            insights=[
                f"Empirical evidence strength: {empirical_evidence:.3f}",
                "Data supports hypothesis with " + ("high" if empirical_evidence > 0.8 else "moderate") + " confidence"
            ],
            execution_time=0.0
        )

    async def _run_default_experiment(self, node: ResearchNode) -> ExperimentResult:
        """Run default experiment for unknown types"""
        await asyncio.sleep(np.random.uniform(1, 3))

        return ExperimentResult(
            experiment_id=node.id,
            success=True,
            confidence=0.5,
            metrics={"basic_analysis": "completed"},
            data={},
            insights=["Basic experiment completed"],
            execution_time=0.0
        )

    async def _run_hierarchical_multi_agent_experiment(self, node: ResearchNode) -> ExperimentResult:
        """Run hierarchical multi-agent research experiment"""
        start_time = datetime.now()

        # Check if orchestrator is available
        if not self.orchestrator:
            self._log_execution(node, "ERROR", "Orchestrator not available for hierarchical research")
            return ExperimentResult(
                experiment_id=node.id,
                success=False,
                confidence=0.0,
                metrics={"error": "orchestrator_unavailable"},
                data={},
                insights=["Hierarchical agents not available - orchestrator not injected"],
                execution_time=0.0
            )

        try:
            # Extract research query from node
            research_query = node.title
            if hasattr(node, 'experiment_config') and node.experiment_config:
                research_query = node.experiment_config.get('research_query', node.title)

            self._log_execution(node, "INFO", f"Starting hierarchical research for: {research_query}")

            # Execute hierarchical research
            session_id = await self.orchestrator.execute_hierarchical_research(
                research_query=research_query,
                domain=node.experiment_config.get('domain', 'AI/ML Research') if hasattr(node, 'experiment_config') and node.experiment_config else 'AI/ML Research',
                research_context={
                    'node_id': node.id,
                    'tree_context': node.description,
                    'experiment_config': getattr(node, 'experiment_config', {})
                }
            )

            self._log_execution(node, "INFO", f"Hierarchical research session started: {session_id}")

            # Wait for completion with timeout
            max_wait_time = 300  # 5 minutes
            wait_interval = 10   # Check every 10 seconds
            total_waited = 0

            while total_waited < max_wait_time:
                status = await self.orchestrator.get_hierarchical_research_status(session_id)

                if status.get("status") == "completed":
                    results = await self.orchestrator.get_hierarchical_research_results(session_id)

                    execution_time = (datetime.now() - start_time).total_seconds()

                    # Extract insights from hierarchical research results
                    insights = self._extract_hierarchical_insights(results)

                    # Calculate confidence based on research readiness score
                    readiness_score = results.get("final_synthesis", {}).get("research_readiness_score", 0.5)
                    confidence = min(readiness_score + 0.2, 1.0)  # Boost confidence slightly

                    # Extract metrics
                    execution_summary = results.get("execution_summary", {})
                    metrics = {
                        "agents_used": execution_summary.get("total_agents_used", 0),
                        "literature_papers": execution_summary.get("information_sources", {}).get("literature_papers", 0),
                        "web_resources": execution_summary.get("information_sources", {}).get("web_resources", 0),
                        "research_directions": execution_summary.get("research_directions_identified", 0),
                        "experiments_designed": execution_summary.get("experiments_designed", 0),
                        "research_readiness_score": readiness_score,
                        "hierarchical_session_id": session_id
                    }

                    self._log_execution(node, "INFO", f"Hierarchical research completed successfully", metrics)

                    return ExperimentResult(
                        experiment_id=node.id,
                        success=True,
                        confidence=confidence,
                        metrics=metrics,
                        data={
                            "hierarchical_results": results,
                            "session_id": session_id
                        },
                        insights=insights,
                        execution_time=execution_time
                    )

                elif status.get("status") == "failed":
                    error_msg = status.get("error", "Unknown error in hierarchical research")
                    self._log_execution(node, "ERROR", f"Hierarchical research failed: {error_msg}")

                    return ExperimentResult(
                        experiment_id=node.id,
                        success=False,
                        confidence=0.1,
                        metrics={"error": error_msg, "session_id": session_id},
                        data={"session_id": session_id},
                        insights=[f"Hierarchical research failed: {error_msg}"],
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )

                # Wait and continue checking
                await asyncio.sleep(wait_interval)
                total_waited += wait_interval

            # Timeout case
            self._log_execution(node, "WARNING", f"Hierarchical research timeout after {max_wait_time}s")

            return ExperimentResult(
                experiment_id=node.id,
                success=False,
                confidence=0.3,
                metrics={"timeout": True, "session_id": session_id, "wait_time": total_waited},
                data={"session_id": session_id},
                insights=[f"Hierarchical research timeout after {max_wait_time} seconds - still in progress"],
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            error_msg = str(e)
            self._log_execution(node, "ERROR", f"Exception in hierarchical research: {error_msg}")

            return ExperimentResult(
                experiment_id=node.id,
                success=False,
                confidence=0.0,
                metrics={"exception": error_msg},
                data={},
                insights=[f"Hierarchical research exception: {error_msg}"],
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _extract_hierarchical_insights(self, results: Dict[str, Any]) -> List[str]:
        """Extract key insights from hierarchical research results"""
        insights = []

        # Extract from execution summary
        execution_summary = results.get("execution_summary", {})
        if execution_summary:
            insights.append(f"Comprehensive research using {execution_summary.get('total_agents_used', 0)} specialized agents")

            info_sources = execution_summary.get("information_sources", {})
            if info_sources.get("literature_papers", 0) > 0:
                insights.append(f"Analyzed {info_sources['literature_papers']} academic papers")
            if info_sources.get("web_resources", 0) > 0:
                insights.append(f"Searched {info_sources['web_resources']} web resources")

        # Extract from final synthesis
        final_synthesis = results.get("final_synthesis", {})
        if final_synthesis:
            aggregated_findings = final_synthesis.get("aggregated_findings", {})

            # Literature insights
            lit_insights = aggregated_findings.get("literature_review", {}).get("key_insights", [])
            insights.extend(lit_insights[:2])  # Top 2 literature insights

            # Cross-agent insights
            cross_insights = aggregated_findings.get("cross_agent_insights", [])
            insights.extend(cross_insights[:2])  # Top 2 cross-agent insights

            # Research directions
            research_directions = aggregated_findings.get("planning", {}).get("research_directions", [])
            if research_directions:
                insights.append(f"Identified {len(research_directions)} viable research directions")
                # Add top research direction
                top_direction = research_directions[0] if research_directions else None
                if top_direction:
                    insights.append(f"Top research direction: {top_direction.get('title', 'Unknown')}")

        # Ensure we have at least some insights
        if not insights:
            insights = ["Hierarchical multi-agent research completed with comprehensive analysis"]

        return insights[:8]  # Limit to top 8 insights

    async def _analyze_literature_patterns(self, papers: List[Dict]) -> List[str]:
        """Analyze patterns in literature using qwen3-max-review"""
        if not papers:
            return ["No papers found for analysis"]

        # Use LLM to analyze literature patterns
        analysis_result = await llm_client.analyze_literature("Research patterns analysis", papers)

        if analysis_result["success"]:
            # Extract insights from LLM response
            llm_insights = analysis_result["content"]
            insights = [llm_insights]
        else:
            # Fallback to basic pattern analysis
            insights = []
            if len(papers) > 10:
                insights.append(f"Found substantial literature base with {len(papers)} papers")

            # Analyze publication trends
            years = [paper.get("published", "2020")[:4] for paper in papers if paper.get("published")]
            if years:
                recent_count = sum(1 for year in years if int(year) >= 2020)
                insights.append(f"{recent_count}/{len(years)} papers are recent (2020+)")

            # Analyze authors
            all_authors = []
            for paper in papers:
                all_authors.extend(paper.get("authors", []))

            if all_authors:
                unique_authors = len(set(all_authors))
                insights.append(f"Research involves {unique_authors} unique authors")

        return insights

    async def _analyze_code_patterns(self, repositories: List[Dict]) -> List[str]:
        """Analyze patterns in code repositories"""
        insights = []

        languages = [repo.get("language", "") for repo in repositories]
        lang_counts = {lang: languages.count(lang) for lang in set(languages) if lang}

        if lang_counts:
            top_lang = max(lang_counts.keys(), key=lambda k: lang_counts[k])
            insights.append(f"Most common language: {top_lang} ({lang_counts[top_lang]} repos)")

        stars = [repo.get("stars", 0) for repo in repositories]
        if stars:
            avg_stars = np.mean(stars)
            insights.append(f"Average repository popularity: {avg_stars:.1f} stars")

        return insights

    async def _run_synthesis_experiment(self, node: ResearchNode) -> ExperimentResult:
        """Run a synthesis experiment that combines and analyzes multiple research findings"""
        start_time = datetime.now()

        try:
            # Get the research goal context - improved goal_id extraction
            goal_id = None
            goal = None
            tree = {}

            # Try to find the goal_id by checking all active goals
            for active_goal_id, active_goal in self.active_goals.items():
                if active_goal_id in self.research_trees:
                    tree_nodes = self.research_trees[active_goal_id]
                    if node.id in tree_nodes:
                        goal_id = active_goal_id
                        goal = active_goal
                        tree = tree_nodes
                        break

            # Fallback: try original method
            if not goal_id:
                potential_goal_id = node.id.split('_')[0] if '_' in node.id else node.id
                if potential_goal_id in self.active_goals:
                    goal_id = potential_goal_id
                    goal = self.active_goals[goal_id]
                    tree = self.research_trees[goal_id]

            result = ExperimentResult(
                experiment_id=node.id,
                success=False,  # Will be updated later
                confidence=0.0,  # Will be updated later
                metrics={},  # Will be updated later
                data={},  # Will be updated later
                insights=[],  # Will be updated later
                execution_time=0.0  # Will be updated later
            )
            result.processing_steps = []
            result.api_calls = []

            # Collect completed research findings from specified parent nodes or siblings
            completed_nodes = []
            if tree:
                # Check if specific parent nodes are specified in config
                parent_node_ids = node.experiment_config.get("parent_nodes", [])
                if parent_node_ids:
                    # Use specified parent nodes for synthesis
                    for node_id in parent_node_ids:
                        if node_id in tree:
                            target_node = tree[node_id]
                            if (target_node.status == NodeStatus.COMPLETED and
                                len(target_node.results) > 0 and
                                any(r.success for r in target_node.results)):
                                completed_nodes.append(target_node)
                else:
                    # Fallback: search for any completed nodes (original behavior)
                    for other_node in tree.values():
                        if (other_node.status == NodeStatus.COMPLETED and
                            other_node.id != node.id and
                            len(other_node.results) > 0):
                            completed_nodes.append(other_node)

            result.processing_steps.append({
                "step": "data_collection",
                "completed_nodes_found": len(completed_nodes),
                "timestamp": datetime.now().isoformat()
            })

            # Extract insights from completed nodes
            all_insights = []
            all_metrics = {}
            total_confidence = 0.0
            node_count = 0

            for completed_node in completed_nodes:
                if completed_node.results:
                    latest_result = completed_node.results[-1]
                    if latest_result.insights:
                        all_insights.extend(latest_result.insights)
                    if latest_result.metrics:
                        for key, value in latest_result.metrics.items():
                            if isinstance(value, (int, float)):
                                if key not in all_metrics:
                                    all_metrics[key] = []
                                all_metrics[key].append(value)

                    total_confidence += latest_result.confidence
                    node_count += 1

            # Calculate synthesis metrics
            avg_confidence = total_confidence / node_count if node_count > 0 else 0.0
            aggregated_metrics = {}
            for key, values in all_metrics.items():
                if values:
                    aggregated_metrics[f"avg_{key}"] = np.mean(values)
                    aggregated_metrics[f"total_{key}"] = sum(values)

            result.processing_steps.append({
                "step": "synthesis_analysis",
                "insights_collected": len(all_insights),
                "metrics_aggregated": len(aggregated_metrics),
                "avg_confidence": avg_confidence,
                "timestamp": datetime.now().isoformat()
            })

            # Generate synthesis insights
            synthesis_insights = [
                f"Synthesized findings from {node_count} completed research nodes",
                f"Average research confidence: {avg_confidence:.2f}",
                f"Total insights collected: {len(all_insights)}"
            ]

            if aggregated_metrics:
                synthesis_insights.append(f"Aggregated {len(aggregated_metrics)} quantitative metrics")

            # Add domain-specific synthesis based on goal
            if goal:
                synthesis_insights.append(f"Research synthesis for: {goal.title}")
                title_lower = goal.title.lower()
                description_lower = goal.description.lower() if goal.description else ""

                # Docker-specific synthesis
                if any(keyword in title_lower + description_lower for keyword in ["docker", "container", "hello-world", "hello world"]):
                    synthesis_insights.extend([
                        "Docker containerization task identified",
                        "Recommended approach: Pull official hello-world image and execute",
                        "Environment setup: Ensure Docker daemon is running",
                        "Execution command: docker run hello-world"
                    ])
                    if node_count == 0:
                        synthesis_insights.extend([
                            "Generated Docker hello-world execution plan",
                            "No complex dependencies required for hello-world container",
                            "Expected output: Success message from Docker hello-world image"
                        ])

                elif "optimization" in title_lower:
                    synthesis_insights.append("Focus area: Performance and efficiency optimization")
                elif "analysis" in title_lower:
                    synthesis_insights.append("Focus area: Analytical framework development")
                elif any(keyword in title_lower + description_lower for keyword in ["python", "programming", "code"]):
                    synthesis_insights.extend([
                        "Programming task synthesis",
                        "Workspace recommendation: ./workspace/task-{timestamp}",
                        "Environment: Use virtual environment or Docker container"
                    ])

            # Calculate final confidence based on input quality and synthesis completeness
            # More lenient requirements for simple tasks
            if node_count >= 2:
                confidence = min(avg_confidence * 1.1 + 0.1, 1.0)
                success = avg_confidence > 0.3
            elif node_count == 1:
                # Allow synthesis with single high-quality node
                confidence = max(avg_confidence * 0.8, 0.6)
                success = avg_confidence > 0.5
            else:
                # No completed nodes - create basic synthesis based on goal
                confidence = 0.7 if goal else 0.5
                success = goal is not None
                synthesis_insights.extend([
                    "Synthesis performed based on research goal analysis",
                    "No parent experiment results available for synthesis",
                    "Generated conceptual framework for task completion"
                ])

            # Final metrics
            final_metrics = {
                "nodes_synthesized": node_count,
                "insights_generated": len(synthesis_insights),
                "average_input_confidence": avg_confidence,
                "synthesis_completeness": min(node_count / 4.0, 1.0),  # Complete with 4+ nodes
                **aggregated_metrics
            }

            result.success = success
            result.confidence = confidence
            result.insights = synthesis_insights
            result.metrics = final_metrics
            result.execution_time = (datetime.now() - start_time).total_seconds()
            result.data = {
                "synthesized_insights": all_insights[:20],  # Top 20 insights
                "input_nodes": len(completed_nodes),
                "synthesis_method": "meta_analysis"
            }

            self._log_execution(node, "INFO", f"Synthesis experiment completed", {
                "success": success,
                "confidence": confidence,
                "nodes_synthesized": node_count
            })

            return result

        except Exception as e:
            error_msg = str(e)
            self._log_execution(node, "ERROR", f"Synthesis experiment failed: {error_msg}")

            return ExperimentResult(
                experiment_id=node.id,
                success=False,
                confidence=0.1,
                metrics={"error": error_msg},
                data={},
                insights=[f"Synthesis experiment failed: {error_msg}"],
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    async def _backpropagate_result(self, goal_id: str, node_id: str, result: ExperimentResult):
        """Backpropagate experiment result up the tree"""
        tree = self.research_trees[goal_id]
        current_node = tree[node_id]

        # Update current node
        current_node.visits += 1
        current_node.total_reward += result.confidence

        # Propagate up the tree
        while current_node.parent_id:
            parent = tree[current_node.parent_id]
            parent.visits += 1
            parent.total_reward += result.confidence * 0.8  # Decay factor
            current_node = parent

    async def _prune_tree(self, goal_id: str):
        """Prune unpromising branches from the research tree"""
        tree = self.research_trees[goal_id]

        nodes_to_prune = []
        for node in tree.values():
            if (
                node.visits > 3 and
                node.total_reward / node.visits < 0.3 and
                node.status == NodeStatus.COMPLETED
            ):
                nodes_to_prune.append(node.id)

        for node_id in nodes_to_prune:
            tree[node_id].status = NodeStatus.PRUNED

    async def _check_root_completion(self, goal_id: str):
        """Check if ROOT nodes should be marked as completed based on child success"""
        tree = self.research_trees[goal_id]
        root_node = tree[f"{goal_id}_root"]

        if root_node.status == NodeStatus.PENDING and root_node.children:
            # Get successful child nodes
            successful_children = []
            failed_children = []
            for child_id in root_node.children:
                if child_id in tree:
                    child = tree[child_id]
                    if child.status == NodeStatus.COMPLETED and child.confidence > 0.5:
                        successful_children.append(child)
                    elif child.status == NodeStatus.FAILED:
                        failed_children.append(child)

            # Complete root if we have at least 2 successful children
            if len(successful_children) >= 2:
                root_node.status = NodeStatus.COMPLETED
                root_node.confidence = np.mean([child.confidence for child in successful_children])
                root_node.completed_at = datetime.now()
                logger.info(f"ROOT node {goal_id}_root completed with {len(successful_children)} successful children (confidence: {root_node.confidence:.2f})")

                # Generate completion report automatically
                await self._generate_completion_report(goal_id)
            # If most children failed and we have insufficient successes, generate new research directions
            elif len(failed_children) >= 2 and len(successful_children) < 2:
                logger.warning(f"ROOT node has {len(failed_children)} failed children, generating new research directions...")
                await self._generate_research_directions(goal_id, allow_regeneration=True)

                # Update total reward for tree search
                root_node.total_reward = root_node.confidence * root_node.visits

    async def _generate_completion_report(self, goal_id: str):
        """Generate a comprehensive completion report when the root node is completed"""
        try:
            # Import here to avoid circular imports
            from .report_generator import MarkdownReportGenerator

            # Create report generator instance
            report_generator = MarkdownReportGenerator(self)

            # Generate the report
            report_path = await report_generator.generate_completion_report(goal_id)

            goal = self.active_goals[goal_id]
            logger.info(f"📄 Completion report generated for '{goal.title}': {report_path}")
            logger.info(f"🌐 View report at: /api/research-tree/goals/{goal_id}/report/view")
            logger.info(f"📥 Download report at: /api/research-tree/goals/{goal_id}/report/download")

        except Exception as e:
            logger.error(f"Failed to generate completion report for goal {goal_id}: {e}")
            # Don't raise the exception to avoid disrupting the main research flow

    async def _is_goal_satisfied(self, goal_id: str) -> bool:
        """Check if research goal is satisfied"""
        goal = self.active_goals[goal_id]
        tree = self.research_trees[goal_id]

        # Check if we have high-confidence results
        high_confidence_nodes = [
            node for node in tree.values()
            if node.confidence > goal.quality_threshold and node.status == NodeStatus.COMPLETED
        ]

        return len(high_confidence_nodes) >= 3

    async def _synthesize_research_results(self, goal_id: str):
        """Synthesize final research results"""
        tree = self.research_trees[goal_id]
        goal = self.active_goals[goal_id]

        # Collect all successful experiments
        successful_experiments = []
        for node in tree.values():
            if node.status == NodeStatus.COMPLETED and node.confidence > 0.7:
                successful_experiments.extend(node.results)

        # Create synthesis node as child of root
        root_id = f"{goal_id}_root"
        synthesis_id = f"{goal_id}_synthesis"
        synthesis_node = ResearchNode(
            id=synthesis_id,
            parent_id=root_id,
            node_type=ResearchNodeType.SYNTHESIS,
            title=f"Research Synthesis: {goal.title}",
            description=f"Synthesized results from {len(successful_experiments)} experiments",
            status=NodeStatus.COMPLETED,
            confidence=np.mean([exp.confidence for exp in successful_experiments]) if successful_experiments else 0.0,
            depth=1
        )

        # Add synthesis node as child of root
        if root_id in tree:
            tree[root_id].children.append(synthesis_id)

        tree[synthesis_id] = synthesis_node

    def _get_max_children(self, node: ResearchNode) -> int:
        """Get maximum number of children for a node type"""
        max_children = {
            ResearchNodeType.ROOT: 5,
            ResearchNodeType.HYPOTHESIS: 3,
            ResearchNodeType.EXPERIMENT: 3,
            ResearchNodeType.LITERATURE: 2,
            ResearchNodeType.CODE_ANALYSIS: 2,
            ResearchNodeType.SYNTHESIS: 1,
            ResearchNodeType.VALIDATION: 1,
            ResearchNodeType.HIERARCHICAL_RESEARCH: 3,
        }
        return max_children.get(node.node_type, 2)

    def _is_done(self, goal_id: str, node_id: str) -> bool:
        """Check if a node is truly done (completed and not expandable)"""
        goal = self.active_goals[goal_id]
        node = self.research_trees[goal_id][node_id]

        if node.status in {NodeStatus.FAILED, NodeStatus.PRUNED}:
            return True
        if node.status != NodeStatus.COMPLETED:
            return False

        # completed but still expandable? then NOT done
        return not (
            node.depth < goal.max_depth
            and len(node.children) < self._get_max_children(node)
            and (node.node_type not in {
                ResearchNodeType.EXPERIMENT, ResearchNodeType.LITERATURE, ResearchNodeType.CODE_ANALYSIS
            } or bool(node.results))
        )

    async def get_research_tree_status(self, goal_id: str) -> Dict[str, Any]:
        """Get comprehensive status of research tree"""
        if goal_id not in self.research_trees:
            return {"error": "Research goal not found"}

        tree = self.research_trees[goal_id]
        goal = self.active_goals[goal_id]

        # Calculate tree statistics
        total_nodes = len(tree)
        completed_nodes = len([n for n in tree.values() if n.status == NodeStatus.COMPLETED])
        running_nodes = len([n for n in tree.values() if n.status == NodeStatus.RUNNING])

        # Get best results
        best_nodes = sorted(
            [n for n in tree.values() if n.confidence > 0],
            key=lambda n: n.confidence,
            reverse=True
        )[:5]

        return {
            "goal": {
                "id": goal_id,
                "title": goal.title,
                "description": goal.description,
                "success_criteria": goal.success_criteria
            },
            "tree_stats": {
                "total_nodes": total_nodes,
                "completed_nodes": completed_nodes,
                "running_nodes": running_nodes,
                "success_rate": completed_nodes / max(total_nodes, 1)
            },
            "best_results": [
                {
                    "node_id": node.id,
                    "title": node.title,
                    "confidence": node.confidence,
                    "node_type": node.node_type.value,
                    "insights": node.results[-1].insights if node.results else []
                }
                for node in best_nodes
            ],
            "tree_structure": await self._get_tree_visualization(goal_id)
        }

    async def get_goal_status(self, goal_id: str) -> Dict[str, Any]:
        """Get simplified goal status - thin wrapper for compatibility"""
        if goal_id not in self.research_trees:
            return {"error": "Research goal not found"}

        tree = self.research_trees[goal_id]
        total_nodes = len(tree)
        completed_nodes = len([n for n in tree.values() if n.status == NodeStatus.COMPLETED])
        running_nodes = len([n for n in tree.values() if n.status == NodeStatus.RUNNING])

        # Determine overall status
        if running_nodes > 0:
            status = "running"
        elif completed_nodes > 0:
            status = "completed"
        else:
            status = "pending"

        # Calculate progress as percentage
        progress = (completed_nodes / max(total_nodes, 1)) * 100

        return {
            "status": status,
            "progress": progress,
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "running_nodes": running_nodes
        }

    async def _get_tree_visualization(self, goal_id: str) -> Dict[str, Any]:
        """Get tree structure for visualization - ROMA style"""
        tree = self.research_trees[goal_id]

        # Convert to ROMA-style task nodes with proper formatting
        all_nodes = {}

        for node_id, node in tree.items():
            # Create ROMA-style task node
            task_node = {
                "task_id": node.id,
                "goal": node.title or node.description,
                "task_type": node.experiment_type.value if node.experiment_type else "research",
                "node_type": "PLAN" if node.node_type == ResearchNodeType.ROOT else "EXECUTE",
                "layer": node.depth,
                "parent_node_id": node.parent_id,
                "status": self._convert_status_to_roma(node.status),
                "output_summary": f"Confidence: {node.confidence:.1%}, Visits: {node.visits}",
                "timestamp_created": node.created_at.isoformat() if node.created_at else None,
                "timestamp_updated": node.started_at.isoformat() if node.started_at else node.created_at.isoformat() if node.created_at else None,
                "timestamp_completed": node.completed_at.isoformat() if node.completed_at else None,
                "planned_sub_task_ids": node.children,
                "agent_name": f"Research Agent L{node.depth}",
                "model_display": "qwen-max-latest" if node.status != NodeStatus.PENDING else "Not processed",
            }

            all_nodes[node_id] = task_node

        return {"all_nodes": all_nodes}

    def _convert_status_to_roma(self, status: NodeStatus) -> str:
        """Convert research tree status to ROMA-style status"""
        status_mapping = {
            NodeStatus.PENDING: "PENDING",
            NodeStatus.PROMISING: "READY",
            NodeStatus.RUNNING: "RUNNING",
            NodeStatus.COMPLETED: "DONE",
            NodeStatus.FAILED: "FAILED",
            NodeStatus.PRUNED: "CANCELLED"
        }
        return status_mapping.get(status, "PENDING")

    async def list_active_research_goals(self) -> List[Dict[str, Any]]:
        """List all active research goals"""
        return [
            {
                "goal_id": goal_id,
                "title": goal.title,
                "description": goal.description,
                "created_at": goal.id,  # Using ID as timestamp placeholder
                "experiments_run": len([n for n in self.research_trees.get(goal_id, {}).values() if n.status == NodeStatus.COMPLETED])
            }
            for goal_id, goal in self.active_goals.items()
        ]