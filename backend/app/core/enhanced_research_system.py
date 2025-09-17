"""
Enhanced Research System with SQLite Database Integration
Combines the hierarchical research system with comprehensive database persistence
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .research_tree import HierarchicalResearchSystem, ResearchGoal, ResearchNode, ExperimentResult, ExecutionLog
from .database import ResearchDatabase

logger = logging.getLogger(__name__)


class DatabaseIntegratedResearchSystem(HierarchicalResearchSystem):
    """Research system with full database integration"""

    def __init__(self, db_path: str = None):
        super().__init__()
        if db_path is None:
            # Use relative path: go up from backend/app/core/ to project root, then to data
            import os
            current_dir = os.path.dirname(__file__)  # app/core/
            backend_dir = os.path.dirname(os.path.dirname(current_dir))  # backend/
            project_root = os.path.dirname(backend_dir)  # project root
            db_path = os.path.join(project_root, "data", "research_history.db")
        self.db = ResearchDatabase(db_path)
        logger.info(f"DatabaseIntegratedResearchSystem initialized with database: {db_path}")
        self._load_active_goals()

    def _load_active_goals(self):
        """Load active goals from database on startup"""
        try:
            logger.info("Loading active goals from database...")
            active_goals_data = self.db.get_research_history(status_filter='active')
            logger.info(f"Found {len(active_goals_data)} active goals in database")

            for goal_data in active_goals_data:
                goal_id = goal_data['id']

                # Reconstruct ResearchGoal object
                goal = ResearchGoal(
                    title=goal_data['title'],
                    description=goal_data['description'],
                    success_criteria=goal_data['success_criteria'],
                    constraints=goal_data['constraints'],
                    max_depth=goal_data['max_depth'],
                    max_experiments=goal_data['max_experiments'],
                    quality_threshold=goal_data['quality_threshold']
                )

                if goal_data['created_at']:
                    goal.created_at = datetime.fromisoformat(goal_data['created_at'])

                self.active_goals[goal_id] = goal

                # Load corresponding tree if it has nodes
                if goal_data['total_nodes'] > 0:
                    self._load_goal_tree(goal_id)

            logger.info(f"Loaded {len(self.active_goals)} active goals from database")

        except Exception as e:
            logger.error(f"Failed to load active goals from database: {e}")

    def _load_goal_tree(self, goal_id: str):
        """Load research tree for a specific goal from database"""
        try:
            goal_details = self.db.get_goal_details(goal_id)
            if not goal_details or not goal_details.get('nodes'):
                return

            # Reconstruct tree structure
            tree = {}
            node_children = {}  # Track children for each node

            for node_data in goal_details['nodes']:
                node = self._reconstruct_node(node_data)
                tree[node.id] = node

                # Track parent-child relationships
                if node.parent_id:
                    if node.parent_id not in node_children:
                        node_children[node.parent_id] = []
                    node_children[node.parent_id].append(node.id)

            # Set children for each node
            for node_id, children in node_children.items():
                if node_id in tree:
                    tree[node_id].children = children

            self.research_trees[goal_id] = tree
            logger.debug(f"Loaded tree for goal {goal_id} with {len(tree)} nodes")

        except Exception as e:
            logger.error(f"Failed to load tree for goal {goal_id}: {e}")

    def _reconstruct_node(self, node_data: Dict) -> ResearchNode:
        """Reconstruct a ResearchNode from database data"""
        from .research_tree import NodeStatus, ExperimentType, ResearchNodeType

        # Convert string enums back to enum objects
        node_type = ResearchNodeType(node_data['node_type']) if node_data['node_type'] else ResearchNodeType.EXPERIMENT
        experiment_type = ExperimentType(node_data['experiment_type']) if node_data['experiment_type'] else None
        status = NodeStatus(node_data['status']) if node_data['status'] else NodeStatus.PENDING

        node = ResearchNode(
            id=node_data['id'],
            parent_id=node_data['parent_id'],
            node_type=node_type,
            title=node_data['title'],
            description=node_data['description'],
            status=status,
            confidence=node_data['confidence'] or 0.0,
            depth=node_data['depth'] or 0,
            experiment_type=experiment_type,
            experiment_config=node_data['experiment_config'] or {},
            hypothesis=node_data['hypothesis'],
            context=node_data['context'] or {},
            priority=node_data['priority'] or 0.5
        )

        # Set timing information
        if node_data['created_at']:
            node.created_at = datetime.fromisoformat(node_data['created_at'])
        if node_data['started_at']:
            node.started_at = datetime.fromisoformat(node_data['started_at'])
        if node_data['completed_at']:
            node.completed_at = datetime.fromisoformat(node_data['completed_at'])

        # Set additional properties
        node.visits = node_data['visits'] or 0
        node.retry_count = node_data['retry_count'] or 0
        node.last_error = node_data['last_error']
        node.total_reward = node_data['total_reward'] or 0.0
        node.aggregated_score = node_data['aggregated_score'] or 0.0

        return node

    async def start_research_goal(self, title: str, description: str, success_criteria: List[str],
                                constraints: Optional[Dict[str, Any]] = None,
                                max_depth: int = 5, max_experiments: int = 100) -> str:
        """Start research goal with database persistence"""

        goal_id = await super().start_research_goal(
            title, description, success_criteria, constraints, max_depth, max_experiments
        )

        # Save to database
        if goal_id in self.active_goals:
            logger.info(f"Saving goal {goal_id} to database")
            success = self.db.save_research_goal(goal_id, self.active_goals[goal_id])
            logger.info(f"Goal {goal_id} save result: {success}")
        else:
            logger.error(f"Goal {goal_id} not found in active_goals during save")

        return goal_id

    async def _initialize_research_tree(self, goal_id: str):
        """Initialize tree with database persistence"""
        await super()._initialize_research_tree(goal_id)

        # Save root node to database
        if goal_id in self.research_trees:
            tree = self.research_trees[goal_id]
            root_id = f"{goal_id}_root"
            if root_id in tree:
                self.db.save_research_node(tree[root_id], goal_id)

    def _log_execution(self, node, level: str, message: str, context=None):
        """Log execution with database persistence"""
        super()._log_execution(node, level, message, context)

        # Save log to database
        if hasattr(node, 'execution_logs') and node.execution_logs:
            latest_log = node.execution_logs[-1]
            self.db.save_execution_log(node.id, latest_log)

        # Periodically save node state (every 3 logs to ensure frequent persistence)
        if hasattr(node, 'execution_logs') and len(node.execution_logs) % 3 == 0:
            # Find goal_id for this node
            for goal_id, tree in self.research_trees.items():
                if node.id in tree:
                    self.db.save_research_node(node, goal_id)
                    break

    async def _run_experiment(self, goal_id: str, node_id: str):
        """Run experiment with database result persistence"""
        result = await super()._run_experiment(goal_id, node_id)

        # Save result to database
        if result:
            self.db.save_experiment_result(node_id, result)

        # Save updated node state
        if goal_id in self.research_trees and node_id in self.research_trees[goal_id]:
            node = self.research_trees[goal_id][node_id]
            self.db.save_research_node(node, goal_id)

        return result

    async def _check_root_completion(self, goal_id: str):
        """Check completion with database status update"""
        await super()._check_root_completion(goal_id)

        # Update goal status in database if completed
        if goal_id in self.research_trees:
            tree = self.research_trees[goal_id]
            root_node = tree.get(f"{goal_id}_root")

            if root_node and hasattr(root_node, 'status'):
                from .research_tree import NodeStatus

                if root_node.status == NodeStatus.COMPLETED:
                    self.db.update_goal_status(goal_id, 'completed', root_node.confidence)
                elif root_node.status == NodeStatus.FAILED:
                    self.db.update_goal_status(goal_id, 'failed', root_node.confidence)

    def archive_goal(self, goal_id: str) -> bool:
        """Archive a research goal"""
        try:
            # Remove from active goals
            if goal_id in self.active_goals:
                del self.active_goals[goal_id]

            # Remove from active trees
            if goal_id in self.research_trees:
                del self.research_trees[goal_id]

            # Update status in database
            return self.db.archive_goal(goal_id)

        except Exception as e:
            logger.error(f"Failed to archive goal {goal_id}: {e}")
            return False

    def restore_goal(self, goal_id: str) -> bool:
        """Restore an archived goal to active state"""
        try:
            # Load goal from database
            goal_details = self.db.get_goal_details(goal_id)
            if not goal_details:
                return False

            # Reconstruct goal object
            goal = ResearchGoal(
                title=goal_details['title'],
                description=goal_details['description'],
                success_criteria=goal_details['success_criteria'],
                constraints=goal_details['constraints'],
                max_depth=goal_details['max_depth'],
                max_experiments=goal_details['max_experiments'],
                quality_threshold=goal_details['quality_threshold']
            )

            if goal_details['created_at']:
                goal.created_at = datetime.fromisoformat(goal_details['created_at'])

            # Add back to active goals
            self.active_goals[goal_id] = goal

            # Load tree if it exists
            if goal_details['nodes']:
                self._load_goal_tree(goal_id)

            # Update status in database
            self.db.update_goal_status(goal_id, 'active')

            logger.info(f"Restored goal {goal_id} to active state")
            return True

        except Exception as e:
            logger.error(f"Failed to restore goal {goal_id}: {e}")
            return False

    def get_research_history(self, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get research history from database"""
        return self.db.get_research_history(limit, status_filter)

    def search_research_history(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search research history"""
        return self.db.search_goals(query, limit)

    def get_goal_details_from_db(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed goal information from database"""
        return self.db.get_goal_details(goal_id)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.db.get_database_stats()

    def save_workspace_to_db(self, goal_id: str, node_id: Optional[str], workspace_info: Dict[str, Any]) -> bool:
        """Save workspace information to database"""
        return self.db.save_workspace(goal_id, node_id, workspace_info)

    async def create_workspace(self, goal_id: str, node_id: str, task_description: str) -> Optional[Dict[str, Any]]:
        """Create workspace and save to database"""
        # Use existing workspace manager
        workspace_info = None

        if any(keyword in task_description.lower() for keyword in ["docker", "hello-world", "hello world"]):
            workspace_info = self.workspace_manager.get_docker_hello_world_workspace(task_description)
        elif any(keyword in task_description.lower() for keyword in ["python", "programming", "code"]):
            workspace_info = self.workspace_manager.create_workspace(task_description, "programming")

        if workspace_info:
            # Save to database
            self.save_workspace_to_db(goal_id, node_id, workspace_info)

        return workspace_info