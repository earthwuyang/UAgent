"""
SQLite Database Layer for Research History
Stores research goals, nodes, results, and execution logs
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from dataclasses import asdict

from .research_tree import ResearchGoal, ResearchNode, ExperimentResult, ExecutionLog, NodeStatus, ExperimentType, ResearchNodeType

logger = logging.getLogger(__name__)


class ResearchDatabase:
    """SQLite database for research history and persistence"""

    def __init__(self, db_path: str = "data/research_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)

        self._init_database()
        logger.info(f"Research database initialized: {self.db_path}")

    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            # Research goals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_goals (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    success_criteria TEXT NOT NULL,  -- JSON array
                    constraints TEXT,  -- JSON object
                    max_depth INTEGER DEFAULT 5,
                    max_experiments INTEGER DEFAULT 100,
                    quality_threshold REAL DEFAULT 0.7,
                    status TEXT DEFAULT 'active',  -- active, completed, failed, archived
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP NULL,
                    final_confidence REAL NULL,
                    total_experiments INTEGER DEFAULT 0,
                    successful_experiments INTEGER DEFAULT 0,
                    metadata TEXT  -- JSON object for additional data
                )
            """)

            # Research nodes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_nodes (
                    id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    parent_id TEXT NULL,
                    node_type TEXT NOT NULL,  -- ROOT, EXPERIMENT, LITERATURE, etc.
                    experiment_type TEXT NULL,  -- COMPUTATIONAL, THEORETICAL, etc.
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,  -- PENDING, RUNNING, COMPLETED, FAILED
                    confidence REAL DEFAULT 0.0,
                    depth INTEGER DEFAULT 0,
                    visits INTEGER DEFAULT 0,
                    priority REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP NULL,
                    completed_at TIMESTAMP NULL,
                    execution_time REAL NULL,
                    retry_count INTEGER DEFAULT 0,
                    last_error TEXT NULL,
                    experiment_config TEXT NULL,  -- JSON object
                    hypothesis TEXT NULL,
                    context TEXT NULL,  -- JSON object
                    total_reward REAL DEFAULT 0.0,
                    aggregated_score REAL DEFAULT 0.0,
                    FOREIGN KEY (goal_id) REFERENCES research_goals (id) ON DELETE CASCADE
                )
            """)

            # Experiment results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    execution_time REAL NOT NULL,
                    insights TEXT,  -- JSON array
                    metrics TEXT,  -- JSON object
                    data TEXT,  -- JSON object
                    error_details TEXT NULL,
                    stack_trace TEXT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resources_used TEXT,  -- JSON object
                    intermediate_results TEXT,  -- JSON array
                    api_calls TEXT,  -- JSON array
                    search_queries_used TEXT,  -- JSON array
                    processing_steps TEXT,  -- JSON array
                    FOREIGN KEY (node_id) REFERENCES research_nodes (id) ON DELETE CASCADE
                )
            """)

            # Execution logs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    level TEXT NOT NULL,  -- INFO, DEBUG, WARNING, ERROR
                    message TEXT NOT NULL,
                    context TEXT,  -- JSON object
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (node_id) REFERENCES research_nodes (id) ON DELETE CASCADE
                )
            """)

            # Node relationships table (for tree structure)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_relationships (
                    parent_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    relationship_type TEXT DEFAULT 'child',  -- child, synthesis, followup
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (parent_id, child_id),
                    FOREIGN KEY (parent_id) REFERENCES research_nodes (id) ON DELETE CASCADE,
                    FOREIGN KEY (child_id) REFERENCES research_nodes (id) ON DELETE CASCADE
                )
            """)

            # Workspaces table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id TEXT NOT NULL,
                    node_id TEXT NULL,
                    workspace_name TEXT NOT NULL,
                    workspace_path TEXT NOT NULL,
                    task_type TEXT NOT NULL,  -- docker, programming, general
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,  -- JSON object with scripts, files, etc.
                    FOREIGN KEY (goal_id) REFERENCES research_goals (id) ON DELETE CASCADE,
                    FOREIGN KEY (node_id) REFERENCES research_nodes (id) ON DELETE SET NULL
                )
            """)

            # Research patterns table - learn from successful approaches
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,  -- infrastructure, algorithm, debugging, etc.
                    service_type TEXT NULL,  -- mongodb, postgresql, redis, etc.
                    problem_description TEXT NOT NULL,
                    solution_approach TEXT NOT NULL,  -- JSON object describing the approach
                    success_rate REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 1,
                    average_confidence REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT,  -- JSON array of searchable tags
                    constraints_pattern TEXT,  -- JSON object of common constraints
                    success_criteria_pattern TEXT  -- JSON array of typical success criteria
                )
            """)

            # Pattern applications table - track when patterns are used
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id INTEGER NOT NULL,
                    goal_id TEXT NOT NULL,
                    node_id TEXT NULL,
                    application_success BOOLEAN NULL,  -- null if still running
                    final_confidence REAL NULL,
                    execution_time REAL NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP NULL,
                    adaptation_notes TEXT,  -- what was modified from the pattern
                    FOREIGN KEY (pattern_id) REFERENCES research_patterns (id) ON DELETE CASCADE,
                    FOREIGN KEY (goal_id) REFERENCES research_goals (id) ON DELETE CASCADE,
                    FOREIGN KEY (node_id) REFERENCES research_nodes (id) ON DELETE SET NULL
                )
            """)

            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON research_goals (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_created ON research_goals (created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_goal ON research_nodes (goal_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_status ON research_nodes (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON research_nodes (node_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_results_node ON experiment_results (node_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_node ON execution_logs (node_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON execution_logs (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON research_patterns (pattern_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_service ON research_patterns (service_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_success ON research_patterns (success_rate)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_applications_pattern ON pattern_applications (pattern_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_applications_goal ON pattern_applications (goal_id)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()

    def save_research_goal(self, goal_id: str, goal: ResearchGoal) -> bool:
        """Save or update a research goal"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO research_goals
                    (id, title, description, success_criteria, constraints, max_depth,
                     max_experiments, quality_threshold, status, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    goal_id,
                    goal.title,
                    goal.description,
                    json.dumps(goal.success_criteria),
                    json.dumps(goal.constraints) if goal.constraints else None,
                    goal.max_depth,
                    goal.max_experiments,
                    goal.quality_threshold,
                    'active',
                    goal.created_at.isoformat() if goal.created_at else datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    json.dumps({})  # metadata
                ))
                conn.commit()
                logger.debug(f"Saved research goal {goal_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to save research goal {goal_id}: {e}")
            return False

    def save_research_node(self, node: ResearchNode, goal_id: str) -> bool:
        """Save or update a research node"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO research_nodes
                    (id, goal_id, parent_id, node_type, experiment_type, title, description,
                     status, confidence, depth, visits, priority, created_at, started_at,
                     completed_at, execution_time, retry_count, last_error, experiment_config,
                     hypothesis, context, total_reward, aggregated_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    node.id,
                    goal_id,
                    node.parent_id,
                    node.node_type.value if node.node_type else None,
                    node.experiment_type.value if node.experiment_type else None,
                    node.title,
                    node.description,
                    node.status.value if node.status else 'pending',
                    node.confidence,
                    node.depth,
                    node.visits,
                    node.priority,
                    node.created_at.isoformat() if node.created_at else datetime.now().isoformat(),
                    node.started_at.isoformat() if node.started_at else None,
                    node.completed_at.isoformat() if node.completed_at else None,
                    (node.completed_at - node.started_at).total_seconds() if node.started_at and node.completed_at else None,
                    node.retry_count,
                    node.last_error,
                    json.dumps(node.experiment_config) if node.experiment_config else None,
                    node.hypothesis,
                    json.dumps(node.context) if node.context else None,
                    node.total_reward,
                    node.aggregated_score
                ))

                # Save node relationships
                if hasattr(node, 'children') and node.children:
                    for child_id in node.children:
                        conn.execute("""
                            INSERT OR IGNORE INTO node_relationships (parent_id, child_id)
                            VALUES (?, ?)
                        """, (node.id, child_id))

                conn.commit()
                logger.debug(f"Saved research node {node.id}")
                return True
        except Exception as e:
            logger.error(f"Failed to save research node {node.id}: {e}")
            return False

    def save_experiment_result(self, node_id: str, result: ExperimentResult) -> bool:
        """Save an experiment result"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO experiment_results
                    (node_id, experiment_id, success, confidence, execution_time, insights,
                     metrics, data, error_details, stack_trace, resources_used,
                     intermediate_results, api_calls, search_queries_used, processing_steps)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    node_id,
                    result.experiment_id,
                    result.success,
                    result.confidence,
                    result.execution_time,
                    json.dumps(result.insights) if result.insights else None,
                    json.dumps(result.metrics) if result.metrics else None,
                    json.dumps(result.data) if result.data else None,
                    result.error_details,
                    result.stack_trace,
                    json.dumps(result.resources_used) if result.resources_used else None,
                    json.dumps(result.intermediate_results) if result.intermediate_results else None,
                    json.dumps(result.api_calls) if result.api_calls else None,
                    json.dumps(result.search_queries_used) if result.search_queries_used else None,
                    json.dumps(result.processing_steps) if result.processing_steps else None
                ))
                conn.commit()
                logger.debug(f"Saved experiment result for node {node_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to save experiment result for node {node_id}: {e}")
            return False

    def save_execution_log(self, node_id: str, log: ExecutionLog) -> bool:
        """Save an execution log"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO execution_logs (node_id, level, message, context, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    node_id,
                    log.level,
                    log.message,
                    json.dumps(log.context) if log.context else None,
                    log.timestamp.isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save execution log for node {node_id}: {e}")
            return False

    def save_workspace(self, goal_id: str, node_id: Optional[str], workspace_info: Dict[str, Any]) -> bool:
        """Save workspace information"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO workspaces (goal_id, node_id, workspace_name, workspace_path, task_type, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    goal_id,
                    node_id,
                    workspace_info.get('workspace_name', ''),
                    workspace_info.get('workspace_path', ''),
                    workspace_info.get('task_type', 'general'),
                    json.dumps(workspace_info)
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save workspace for goal {goal_id}: {e}")
            return False

    def get_research_history(self, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get research history with summary statistics"""
        try:
            with self._get_connection() as conn:
                where_clause = ""
                params = []

                if status_filter:
                    where_clause = "WHERE g.status = ?"
                    params.append(status_filter)

                cursor = conn.execute(f"""
                    SELECT
                        g.*,
                        COUNT(DISTINCT n.id) as total_nodes,
                        COUNT(DISTINCT CASE WHEN n.status = 'completed' THEN n.id END) as completed_nodes,
                        COUNT(DISTINCT r.id) as total_results,
                        COUNT(DISTINCT CASE WHEN r.success = 1 THEN r.id END) as successful_results,
                        MAX(r.confidence) as max_confidence,
                        AVG(r.confidence) as avg_confidence
                    FROM research_goals g
                    LEFT JOIN research_nodes n ON g.id = n.goal_id
                    LEFT JOIN experiment_results r ON n.id = r.node_id
                    {where_clause}
                    GROUP BY g.id
                    ORDER BY g.created_at DESC
                    LIMIT ?
                """, params + [limit])

                goals = []
                for row in cursor:
                    goal_data = dict(row)

                    # Parse JSON fields
                    goal_data['success_criteria'] = json.loads(goal_data['success_criteria']) if goal_data['success_criteria'] else []
                    goal_data['constraints'] = json.loads(goal_data['constraints']) if goal_data['constraints'] else {}
                    goal_data['metadata'] = json.loads(goal_data['metadata']) if goal_data['metadata'] else {}

                    # Calculate derived statistics
                    goal_data['success_rate'] = (
                        goal_data['successful_results'] / max(goal_data['total_results'], 1) * 100
                    ) if goal_data['total_results'] else 0

                    goal_data['completion_rate'] = (
                        goal_data['completed_nodes'] / max(goal_data['total_nodes'], 1) * 100
                    ) if goal_data['total_nodes'] else 0

                    goals.append(goal_data)

                return goals
        except Exception as e:
            logger.error(f"Failed to get research history: {e}")
            return []

    def get_goal_details(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific goal"""
        try:
            with self._get_connection() as conn:
                # Get goal info
                cursor = conn.execute("SELECT * FROM research_goals WHERE id = ?", (goal_id,))
                goal_row = cursor.fetchone()

                if not goal_row:
                    return None

                goal_data = dict(goal_row)
                goal_data['success_criteria'] = json.loads(goal_data['success_criteria']) if goal_data['success_criteria'] else []
                goal_data['constraints'] = json.loads(goal_data['constraints']) if goal_data['constraints'] else {}
                goal_data['metadata'] = json.loads(goal_data['metadata']) if goal_data['metadata'] else {}

                # Get nodes with results
                cursor = conn.execute("""
                    SELECT n.*,
                           COUNT(r.id) as result_count,
                           AVG(r.confidence) as avg_confidence,
                           MAX(r.success) as has_success
                    FROM research_nodes n
                    LEFT JOIN experiment_results r ON n.id = r.node_id
                    WHERE n.goal_id = ?
                    GROUP BY n.id
                    ORDER BY n.created_at
                """, (goal_id,))

                nodes = []
                for row in cursor:
                    node_data = dict(row)
                    node_data['experiment_config'] = json.loads(node_data['experiment_config']) if node_data['experiment_config'] else {}
                    node_data['context'] = json.loads(node_data['context']) if node_data['context'] else {}
                    nodes.append(node_data)

                goal_data['nodes'] = nodes

                # Get workspaces
                cursor = conn.execute("SELECT * FROM workspaces WHERE goal_id = ?", (goal_id,))
                workspaces = []
                for row in cursor:
                    workspace_data = dict(row)
                    workspace_data['metadata'] = json.loads(workspace_data['metadata']) if workspace_data['metadata'] else {}
                    workspaces.append(workspace_data)

                goal_data['workspaces'] = workspaces

                return goal_data
        except Exception as e:
            logger.error(f"Failed to get goal details for {goal_id}: {e}")
            return None

    def update_goal_status(self, goal_id: str, status: str, final_confidence: Optional[float] = None) -> bool:
        """Update goal status and completion info"""
        try:
            with self._get_connection() as conn:
                update_data = [status, datetime.now().isoformat(), goal_id]
                query = "UPDATE research_goals SET status = ?, updated_at = ?"

                if status in ['completed', 'failed'] and final_confidence is not None:
                    query += ", completed_at = ?, final_confidence = ?"
                    update_data.insert(-1, datetime.now().isoformat())
                    update_data.insert(-1, final_confidence)

                query += " WHERE id = ?"

                conn.execute(query, update_data)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update goal status for {goal_id}: {e}")
            return False

    def archive_goal(self, goal_id: str) -> bool:
        """Archive a research goal"""
        return self.update_goal_status(goal_id, 'archived')

    def delete_goal(self, goal_id: str) -> bool:
        """Permanently delete a research goal and all related data"""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM research_goals WHERE id = ?", (goal_id,))
                conn.commit()
                logger.info(f"Deleted goal {goal_id} and all related data")
                return True
        except Exception as e:
            logger.error(f"Failed to delete goal {goal_id}: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self._get_connection() as conn:
                stats = {}

                # Table counts
                for table in ['research_goals', 'research_nodes', 'experiment_results', 'execution_logs', 'workspaces']:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]

                # Database size
                stats['database_size_bytes'] = self.db_path.stat().st_size
                stats['database_size_mb'] = round(stats['database_size_bytes'] / (1024 * 1024), 2)
                stats['database_path'] = str(self.db_path.absolute())

                # Status distribution
                cursor = conn.execute("SELECT status, COUNT(*) FROM research_goals GROUP BY status")
                stats['goal_status_distribution'] = dict(cursor.fetchall())

                return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}

    def search_goals(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search goals by title, description, or success criteria"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT g.*,
                           COUNT(DISTINCT n.id) as total_nodes,
                           COUNT(DISTINCT CASE WHEN n.status = 'completed' THEN n.id END) as completed_nodes
                    FROM research_goals g
                    LEFT JOIN research_nodes n ON g.id = n.goal_id
                    WHERE g.title LIKE ? OR g.description LIKE ? OR g.success_criteria LIKE ?
                    GROUP BY g.id
                    ORDER BY g.created_at DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))

                results = []
                for row in cursor:
                    goal_data = dict(row)
                    goal_data['success_criteria'] = json.loads(goal_data['success_criteria']) if goal_data['success_criteria'] else []
                    goal_data['constraints'] = json.loads(goal_data['constraints']) if goal_data['constraints'] else {}
                    results.append(goal_data)

                return results
        except Exception as e:
            logger.error(f"Failed to search goals with query '{query}': {e}")
            return []

    # Research Pattern Management Methods

    def save_research_pattern(self, pattern_type: str, service_type: str, problem_description: str,
                            solution_approach: Dict[str, Any], tags: List[str] = None,
                            constraints_pattern: Dict[str, Any] = None,
                            success_criteria_pattern: List[str] = None) -> int:
        """Save a successful research pattern for future reuse"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO research_patterns (
                        pattern_type, service_type, problem_description, solution_approach,
                        tags, constraints_pattern, success_criteria_pattern
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_type,
                    service_type,
                    problem_description,
                    json.dumps(solution_approach),
                    json.dumps(tags or []),
                    json.dumps(constraints_pattern or {}),
                    json.dumps(success_criteria_pattern or [])
                ))
                pattern_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Saved research pattern {pattern_id} for {pattern_type}")
                return pattern_id
        except Exception as e:
            logger.error(f"Failed to save research pattern: {e}")
            return -1

    def find_matching_patterns(self, pattern_type: str = None, service_type: str = None,
                             problem_keywords: List[str] = None, min_success_rate: float = 0.0,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """Find research patterns matching given criteria"""
        try:
            with self._get_connection() as conn:
                query_parts = ["SELECT * FROM research_patterns WHERE 1=1"]
                params = []

                if pattern_type:
                    query_parts.append("AND pattern_type = ?")
                    params.append(pattern_type)

                if service_type:
                    query_parts.append("AND service_type = ?")
                    params.append(service_type)

                if problem_keywords:
                    keyword_conditions = []
                    for keyword in problem_keywords:
                        keyword_conditions.append("problem_description LIKE ?")
                        params.append(f"%{keyword}%")
                    query_parts.append(f"AND ({' OR '.join(keyword_conditions)})")

                if min_success_rate > 0:
                    query_parts.append("AND success_rate >= ?")
                    params.append(min_success_rate)

                query_parts.append("ORDER BY success_rate DESC, usage_count DESC LIMIT ?")
                params.append(limit)

                query = " ".join(query_parts)
                cursor = conn.execute(query, params)

                patterns = []
                for row in cursor:
                    pattern_data = dict(row)
                    pattern_data['solution_approach'] = json.loads(pattern_data['solution_approach'])
                    pattern_data['tags'] = json.loads(pattern_data['tags'] or '[]')
                    pattern_data['constraints_pattern'] = json.loads(pattern_data['constraints_pattern'] or '{}')
                    pattern_data['success_criteria_pattern'] = json.loads(pattern_data['success_criteria_pattern'] or '[]')
                    patterns.append(pattern_data)

                return patterns
        except Exception as e:
            logger.error(f"Failed to find matching patterns: {e}")
            return []

    def update_pattern_success(self, pattern_id: int, application_success: bool, confidence: float = None):
        """Update pattern success metrics when applied"""
        try:
            with self._get_connection() as conn:
                # Get current stats
                cursor = conn.execute("""
                    SELECT usage_count, success_rate, average_confidence
                    FROM research_patterns WHERE id = ?
                """, (pattern_id,))
                row = cursor.fetchone()
                if not row:
                    return False

                usage_count, current_success_rate, avg_confidence = row
                new_usage_count = usage_count + 1

                # Calculate new success rate
                current_successes = int(usage_count * current_success_rate)
                new_successes = current_successes + (1 if application_success else 0)
                new_success_rate = new_successes / new_usage_count

                # Calculate new average confidence
                if confidence is not None:
                    current_confidence_sum = avg_confidence * usage_count
                    new_avg_confidence = (current_confidence_sum + confidence) / new_usage_count
                else:
                    new_avg_confidence = avg_confidence

                # Update the pattern
                conn.execute("""
                    UPDATE research_patterns
                    SET usage_count = ?, success_rate = ?, average_confidence = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_usage_count, new_success_rate, new_avg_confidence, pattern_id))

                conn.commit()
                logger.info(f"Updated pattern {pattern_id} success metrics")
                return True

        except Exception as e:
            logger.error(f"Failed to update pattern success for {pattern_id}: {e}")
            return False

    def record_pattern_application(self, pattern_id: int, goal_id: str, node_id: str = None,
                                 adaptation_notes: str = None) -> int:
        """Record when a pattern is applied to a research goal"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO pattern_applications (pattern_id, goal_id, node_id, adaptation_notes)
                    VALUES (?, ?, ?, ?)
                """, (pattern_id, goal_id, node_id, adaptation_notes))
                application_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Recorded pattern application {application_id}")
                return application_id
        except Exception as e:
            logger.error(f"Failed to record pattern application: {e}")
            return -1

    def get_pattern_performance(self, pattern_type: str = None, service_type: str = None) -> Dict[str, Any]:
        """Get performance statistics for research patterns"""
        try:
            with self._get_connection() as conn:
                query_parts = ["""
                    SELECT pattern_type, service_type,
                           COUNT(*) as total_patterns,
                           AVG(success_rate) as avg_success_rate,
                           AVG(usage_count) as avg_usage_count,
                           AVG(average_confidence) as avg_confidence
                    FROM research_patterns WHERE 1=1
                """]
                params = []

                if pattern_type:
                    query_parts.append("AND pattern_type = ?")
                    params.append(pattern_type)

                if service_type:
                    query_parts.append("AND service_type = ?")
                    params.append(service_type)

                query_parts.append("GROUP BY pattern_type, service_type ORDER BY avg_success_rate DESC")

                query = " ".join(query_parts)
                cursor = conn.execute(query, params)

                stats = []
                for row in cursor:
                    stats.append(dict(row))

                return {
                    "pattern_performance": stats,
                    "total_unique_patterns": len(stats)
                }

        except Exception as e:
            logger.error(f"Failed to get pattern performance: {e}")
            return {"pattern_performance": [], "total_unique_patterns": 0}