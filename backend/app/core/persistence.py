"""
Simple Persistence Layer for Research Goals
Saves/loads goals to prevent loss on server restart
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle

from .research_tree import ResearchGoal, ResearchNode, HierarchicalResearchSystem

logger = logging.getLogger(__name__)


class ResearchPersistence:
    """Handles persistence of research goals and trees"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.goals_file = self.data_dir / "active_goals.json"
        self.trees_dir = self.data_dir / "trees"
        self.trees_dir.mkdir(exist_ok=True)

    def save_goal(self, goal_id: str, goal: ResearchGoal) -> bool:
        """Save a single research goal"""
        try:
            # Load existing goals
            goals_data = self._load_goals_data()

            # Add/update this goal
            goals_data[goal_id] = {
                "id": goal_id,
                "title": goal.title,
                "description": goal.description,
                "success_criteria": goal.success_criteria,
                "constraints": goal.constraints,
                "max_depth": goal.max_depth,
                "max_experiments": goal.max_experiments,
                "quality_threshold": goal.quality_threshold,
                "status": getattr(goal, 'status', 'active'),
                "created_at": goal.created_at.isoformat() if goal.created_at else datetime.now().isoformat(),
                "started_at": goal.started_at.isoformat() if getattr(goal, 'started_at', None) else None,
                "completed_at": goal.completed_at.isoformat() if getattr(goal, 'completed_at', None) else None
            }

            # Save back to file
            with open(self.goals_file, 'w') as f:
                json.dump(goals_data, f, indent=2)

            logger.info(f"Saved goal {goal_id} to persistence")
            return True

        except Exception as e:
            logger.error(f"Failed to save goal {goal_id}: {e}")
            return False

    def save_tree(self, goal_id: str, tree: Dict[str, ResearchNode]) -> bool:
        """Save a research tree (using pickle for complex objects)"""
        try:
            tree_file = self.trees_dir / f"{goal_id}.pkl"

            with open(tree_file, 'wb') as f:
                pickle.dump(tree, f)

            logger.info(f"Saved tree {goal_id} to persistence")
            return True

        except Exception as e:
            logger.error(f"Failed to save tree {goal_id}: {e}")
            return False

    def load_goal(self, goal_id: str) -> Optional[ResearchGoal]:
        """Load a single research goal"""
        try:
            goals_data = self._load_goals_data()

            if goal_id not in goals_data:
                return None

            goal_data = goals_data[goal_id]

            # Reconstruct ResearchGoal object
            goal = ResearchGoal(
                id=goal_id,
                title=goal_data["title"],
                description=goal_data["description"],
                success_criteria=goal_data["success_criteria"],
                constraints=goal_data.get("constraints", {}),
                max_depth=goal_data.get("max_depth", 5),
                max_experiments=goal_data.get("max_experiments", 100),
                quality_threshold=goal_data.get("quality_threshold", 0.7),
                status=goal_data.get("status", "active")
            )

            # Set timestamps if available
            if "created_at" in goal_data:
                goal.created_at = datetime.fromisoformat(goal_data["created_at"])
            if goal_data.get("started_at"):
                goal.started_at = datetime.fromisoformat(goal_data["started_at"])
            if goal_data.get("completed_at"):
                goal.completed_at = datetime.fromisoformat(goal_data["completed_at"])

            return goal

        except Exception as e:
            logger.error(f"Failed to load goal {goal_id}: {e}")
            return None

    def load_tree(self, goal_id: str) -> Optional[Dict[str, ResearchNode]]:
        """Load a research tree"""
        try:
            tree_file = self.trees_dir / f"{goal_id}.pkl"

            if not tree_file.exists():
                return None

            with open(tree_file, 'rb') as f:
                tree = pickle.load(f)

            logger.info(f"Loaded tree {goal_id} from persistence")
            return tree

        except Exception as e:
            logger.error(f"Failed to load tree {goal_id}: {e}")
            return None

    def load_all_goals(self) -> Dict[str, ResearchGoal]:
        """Load all saved goals"""
        try:
            goals_data = self._load_goals_data()
            loaded_goals = {}

            for goal_id, goal_data in goals_data.items():
                if goal_data.get("status") == "active":
                    goal = self.load_goal(goal_id)
                    if goal:
                        loaded_goals[goal_id] = goal

            logger.info(f"Loaded {len(loaded_goals)} goals from persistence")
            return loaded_goals

        except Exception as e:
            logger.error(f"Failed to load all goals: {e}")
            return {}

    def load_all_trees(self) -> Dict[str, Dict[str, ResearchNode]]:
        """Load all saved trees"""
        try:
            trees = {}

            for tree_file in self.trees_dir.glob("*.pkl"):
                goal_id = tree_file.stem
                tree = self.load_tree(goal_id)
                if tree:
                    trees[goal_id] = tree

            logger.info(f"Loaded {len(trees)} trees from persistence")
            return trees

        except Exception as e:
            logger.error(f"Failed to load all trees: {e}")
            return {}

    def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal and its tree from persistence"""
        try:
            # Remove from goals file
            goals_data = self._load_goals_data()
            if goal_id in goals_data:
                del goals_data[goal_id]
                with open(self.goals_file, 'w') as f:
                    json.dump(goals_data, f, indent=2)

            # Remove tree file
            tree_file = self.trees_dir / f"{goal_id}.pkl"
            if tree_file.exists():
                tree_file.unlink()

            logger.info(f"Deleted goal {goal_id} from persistence")
            return True

        except Exception as e:
            logger.error(f"Failed to delete goal {goal_id}: {e}")
            return False

    def list_saved_goals(self) -> List[Dict[str, Any]]:
        """List all saved goals with metadata"""
        try:
            goals_data = self._load_goals_data()

            goals_list = []
            for goal_id, goal_data in goals_data.items():
                # Check if tree exists
                tree_file = self.trees_dir / f"{goal_id}.pkl"
                has_tree = tree_file.exists()

                goals_list.append({
                    "goal_id": goal_id,
                    "title": goal_data["title"],
                    "description": goal_data["description"][:100] + "..." if len(goal_data["description"]) > 100 else goal_data["description"],
                    "status": goal_data.get("status", "unknown"),
                    "created_at": goal_data.get("created_at"),
                    "has_tree": has_tree,
                    "tree_size": self._get_tree_size(goal_id) if has_tree else 0
                })

            return sorted(goals_list, key=lambda x: x["created_at"] or "", reverse=True)

        except Exception as e:
            logger.error(f"Failed to list saved goals: {e}")
            return []

    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get statistics about persisted data"""
        try:
            goals_data = self._load_goals_data()
            tree_files = list(self.trees_dir.glob("*.pkl"))

            # Calculate total size
            total_size = 0
            if self.goals_file.exists():
                total_size += self.goals_file.stat().st_size

            for tree_file in tree_files:
                total_size += tree_file.stat().st_size

            return {
                "goals_count": len(goals_data),
                "trees_count": len(tree_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "data_directory": str(self.data_dir.absolute()),
                "goals_file": str(self.goals_file.absolute()),
                "trees_directory": str(self.trees_dir.absolute())
            }

        except Exception as e:
            logger.error(f"Failed to get persistence stats: {e}")
            return {"error": str(e)}

    def _load_goals_data(self) -> Dict[str, Any]:
        """Load goals data from file"""
        if not self.goals_file.exists():
            return {}

        try:
            with open(self.goals_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load goals file, starting fresh: {e}")
            return {}

    def _get_tree_size(self, goal_id: str) -> int:
        """Get the number of nodes in a tree"""
        try:
            tree = self.load_tree(goal_id)
            return len(tree) if tree else 0
        except Exception:
            return 0

    def backup_data(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of all persistence data"""
        try:
            if not backup_name:
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            backup_dir = self.data_dir / "backups" / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copy goals file
            if self.goals_file.exists():
                import shutil
                shutil.copy2(self.goals_file, backup_dir / "active_goals.json")

            # Copy all tree files
            trees_backup_dir = backup_dir / "trees"
            trees_backup_dir.mkdir(exist_ok=True)

            for tree_file in self.trees_dir.glob("*.pkl"):
                shutil.copy2(tree_file, trees_backup_dir / tree_file.name)

            logger.info(f"Created backup: {backup_dir}")
            return str(backup_dir)

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise


# Enhanced Research System with Persistence
class PersistentResearchSystem(HierarchicalResearchSystem):
    """Research system with automatic persistence"""

    def __init__(self, enable_persistence: bool = True):
        super().__init__()
        self.enable_persistence = enable_persistence

        if enable_persistence:
            self.persistence = ResearchPersistence()
            self._load_persisted_data()
        else:
            self.persistence = None

    def _load_persisted_data(self):
        """Load persisted goals and trees on startup"""
        if not self.persistence:
            return

        try:
            logger.info("Loading persisted research data...")

            # Load goals
            persisted_goals = self.persistence.load_all_goals()
            self.active_goals.update(persisted_goals)

            # Load trees
            persisted_trees = self.persistence.load_all_trees()
            self.research_trees.update(persisted_trees)

            logger.info(f"Loaded {len(persisted_goals)} goals and {len(persisted_trees)} trees from persistence")

        except Exception as e:
            logger.error(f"Failed to load persisted data: {e}")

    async def start_research_goal(self, title: str, description: str, success_criteria: List[str],
                                constraints: Optional[Dict[str, Any]] = None,
                                max_depth: int = 5, max_experiments: int = 100) -> str:
        """Start research goal with automatic persistence"""

        goal_id = await super().start_research_goal(
            title, description, success_criteria, constraints, max_depth, max_experiments
        )

        # Persist the goal
        if self.enable_persistence and goal_id in self.active_goals:
            self.persistence.save_goal(goal_id, self.active_goals[goal_id])

        return goal_id

    async def _initialize_research_tree(self, goal_id: str):
        """Initialize tree with automatic persistence"""
        await super()._initialize_research_tree(goal_id)

        # Persist the tree
        if self.enable_persistence and goal_id in self.research_trees:
            self.persistence.save_tree(goal_id, self.research_trees[goal_id])

    def _log_execution(self, node, level: str, message: str, context=None):
        """Log execution with periodic persistence"""
        super()._log_execution(node, level, message, context)

        # Periodically save tree state (every 10 logs to avoid too frequent saves)
        if self.enable_persistence and hasattr(node, 'execution_logs'):
            if len(node.execution_logs) % 10 == 0:  # Save every 10 logs
                # Find goal_id for this node
                for goal_id, tree in self.research_trees.items():
                    if node.id in tree:
                        self.persistence.save_tree(goal_id, tree)
                        break