"""
Research Tree API Router
REST API for the hierarchical tree search-based research system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
import asyncio
import math

from ..core.research_tree import HierarchicalResearchSystem
from ..core.unified_orchestrator import UnifiedOrchestrator
from ..core.report_generator import MarkdownReportGenerator
from ..core.persistence import PersistentResearchSystem
from ..core.enhanced_research_system import DatabaseIntegratedResearchSystem

router = APIRouter(prefix="/research-tree", tags=["research-tree"])

# Initialize the hierarchical research system with database integration
research_system = DatabaseIntegratedResearchSystem()

# Initialize and inject the orchestrator
orchestrator = UnifiedOrchestrator()
research_system.orchestrator = orchestrator

# Initialize the report generator
report_generator = MarkdownReportGenerator(research_system)


def sanitize_float(value: float) -> float:
    """Sanitize float values to ensure JSON serialization compatibility"""
    if value is None:
        return None
    if math.isinf(value):
        return 999999.0 if value > 0 else -999999.0
    if math.isnan(value):
        return 0.0
    return value


def sanitize_data(data):
    """Recursively sanitize data structures to remove inf/nan values"""
    if data is None:
        return None
    elif isinstance(data, float):
        return sanitize_float(data)
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    else:
        return data


class WebSocketManager:
    """Manages WebSocket connections for real-time logging"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.node_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, goal_id: str, node_id: str = None):
        """Connect a WebSocket for a specific goal and optionally a specific node"""
        await websocket.accept()

        # Add to goal connections
        if goal_id not in self.active_connections:
            self.active_connections[goal_id] = []
        self.active_connections[goal_id].append(websocket)

        # Add to node-specific connections if provided
        if node_id:
            if node_id not in self.node_subscribers:
                self.node_subscribers[node_id] = []
            self.node_subscribers[node_id].append(websocket)

    def disconnect(self, websocket: WebSocket, goal_id: str, node_id: str = None):
        """Disconnect a WebSocket"""
        if goal_id in self.active_connections:
            if websocket in self.active_connections[goal_id]:
                self.active_connections[goal_id].remove(websocket)

        if node_id and node_id in self.node_subscribers:
            if websocket in self.node_subscribers[node_id]:
                self.node_subscribers[node_id].remove(websocket)

    async def send_to_goal(self, goal_id: str, message: dict):
        """Send a message to all connections for a specific goal"""
        if goal_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[goal_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception:
                    dead_connections.append(connection)

            # Remove dead connections
            for dead_conn in dead_connections:
                self.active_connections[goal_id].remove(dead_conn)

    async def send_to_node(self, node_id: str, message: dict):
        """Send a message to all connections for a specific node"""
        if node_id in self.node_subscribers:
            dead_connections = []
            for connection in self.node_subscribers[node_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception:
                    dead_connections.append(connection)

            # Remove dead connections
            for dead_conn in dead_connections:
                self.node_subscribers[node_id].remove(dead_conn)


# Initialize WebSocket manager
websocket_manager = WebSocketManager()

# Inject WebSocket manager into research system
research_system.websocket_manager = websocket_manager


class ResearchGoalRequest(BaseModel):
    """Request model for starting a research goal"""
    title: str
    description: str
    success_criteria: List[str]
    constraints: Optional[Dict[str, Any]] = None
    max_depth: Optional[int] = 5
    max_experiments: Optional[int] = 100
    time_budget: Optional[int] = 7200  # 2 hours


class ResearchGoalResponse(BaseModel):
    """Response model for research goal creation"""
    goal_id: str
    title: str
    description: str
    status: str
    message: str


class TreeStatusResponse(BaseModel):
    """Response model for research tree status"""
    goal: Dict[str, Any]
    tree_stats: Dict[str, Any]
    best_results: List[Dict[str, Any]]
    tree_structure: Dict[str, Any]


class ActiveGoalsResponse(BaseModel):
    """Response model for active research goals"""
    active_goals: List[Dict[str, Any]]
    total_goals: int


class ResearchHistoryResponse(BaseModel):
    """Response model for research history"""
    history: List[Dict[str, Any]]
    total_count: int
    filters_applied: Dict[str, Any]


class GoalDetailsResponse(BaseModel):
    """Response model for detailed goal information"""
    goal: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    workspaces: List[Dict[str, Any]]
    statistics: Dict[str, Any]


# Research History Endpoints

@router.get("/history", response_model=ResearchHistoryResponse)
async def get_research_history(
    limit: int = 50,
    status: Optional[str] = None,
    search: Optional[str] = None
):
    """Get research history with optional filtering"""
    try:
        if search:
            # Search functionality
            history = research_system.search_research_history(search, limit)
        else:
            # Regular history with status filter
            history = research_system.get_research_history(limit, status)

        return ResearchHistoryResponse(
            history=history,
            total_count=len(history),
            filters_applied={
                "limit": limit,
                "status": status,
                "search": search
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get research history: {str(e)}")


@router.get("/history/{goal_id}", response_model=GoalDetailsResponse)
async def get_historical_goal_details(goal_id: str):
    """Get detailed information about a historical research goal"""
    try:
        goal_details = research_system.get_goal_details_from_db(goal_id)

        if not goal_details:
            raise HTTPException(status_code=404, detail=f"Historical goal '{goal_id}' not found")

        # Calculate statistics
        nodes = goal_details.get('nodes', [])
        completed_nodes = [n for n in nodes if n['status'] == 'completed']
        failed_nodes = [n for n in nodes if n['status'] == 'failed']

        statistics = {
            "total_nodes": len(nodes),
            "completed_nodes": len(completed_nodes),
            "failed_nodes": len(failed_nodes),
            "success_rate": len(completed_nodes) / max(len(nodes), 1) * 100,
            "avg_confidence": sum(n.get('avg_confidence', 0) or 0 for n in nodes) / max(len(nodes), 1),
            "total_workspaces": len(goal_details.get('workspaces', []))
        }

        return GoalDetailsResponse(
            goal={k: v for k, v in goal_details.items() if k not in ['nodes', 'workspaces']},
            nodes=nodes,
            workspaces=goal_details.get('workspaces', []),
            statistics=statistics
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get historical goal details: {str(e)}")


@router.post("/history/{goal_id}/restore")
async def restore_archived_goal(goal_id: str):
    """Restore an archived goal to active state"""
    try:
        success = research_system.restore_goal(goal_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found or cannot be restored")

        return {
            "message": f"Goal '{goal_id}' restored to active state",
            "goal_id": goal_id,
            "status": "active"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore goal: {str(e)}")


@router.post("/history/{goal_id}/archive")
async def archive_goal(goal_id: str):
    """Archive an active goal"""
    try:
        success = research_system.archive_goal(goal_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found or cannot be archived")

        return {
            "message": f"Goal '{goal_id}' archived successfully",
            "goal_id": goal_id,
            "status": "archived"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to archive goal: {str(e)}")


@router.delete("/history/{goal_id}")
async def delete_historical_goal(goal_id: str):
    """Permanently delete a historical research goal"""
    try:
        success = research_system.db.delete_goal(goal_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found")

        return {
            "message": f"Goal '{goal_id}' permanently deleted",
            "goal_id": goal_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete goal: {str(e)}")


@router.get("/history/stats/database")
async def get_database_statistics():
    """Get database and research history statistics"""
    try:
        stats = research_system.get_database_stats()
        return {
            "database_stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")


@router.post("/goals/start", response_model=ResearchGoalResponse)
async def start_research_goal(request: ResearchGoalRequest):
    """Start a new hierarchical research goal with tree search"""
    try:
        goal_id = await research_system.start_research_goal(
            title=request.title,
            description=request.description,
            success_criteria=request.success_criteria,
            constraints=request.constraints or {},
            max_depth=request.max_depth,
            max_experiments=request.max_experiments
        )

        return ResearchGoalResponse(
            goal_id=goal_id,
            title=request.title,
            description=request.description,
            status="started",
            message=f"Hierarchical research goal '{request.title}' started with tree search optimization"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start research goal: {str(e)}")


@router.get("/goals/{goal_id}/status", response_model=TreeStatusResponse)
async def get_research_tree_status(goal_id: str):
    """Get comprehensive status of a research tree"""
    try:
        # Add detailed debugging for missing goals
        if goal_id not in research_system.active_goals:
            available_goals = list(research_system.active_goals.keys())
            error_detail = f"Goal '{goal_id}' not found. Available goals: {available_goals if available_goals else 'None'}"
            raise HTTPException(status_code=404, detail=error_detail)

        status = await research_system.get_research_tree_status(goal_id)

        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])

        return TreeStatusResponse(
            goal=status["goal"],
            tree_stats=status["tree_stats"],
            best_results=status["best_results"],
            tree_structure=status["tree_structure"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get research tree status: {str(e)}")


@router.get("/goals/{goal_id}/visualization")
async def get_tree_visualization(goal_id: str):
    """Get tree visualization data for frontend rendering"""
    try:
        # Check if goal exists first
        if goal_id not in research_system.active_goals:
            available_goals = list(research_system.active_goals.keys())
            error_detail = f"Goal '{goal_id}' not found. Available goals: {available_goals if available_goals else 'None'}"
            raise HTTPException(status_code=404, detail=error_detail)

        status = await research_system.get_research_tree_status(goal_id)

        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])

        # Debug logging
        print(f"🔍 Tree visualization for goal {goal_id}:")
        print(f"  Total nodes: {status['tree_stats']['total_nodes']}")
        if status.get('tree_structure') and status['tree_structure'].get('all_nodes'):
            nodes = status['tree_structure']['all_nodes']
            print(f"  All nodes: {list(nodes.keys())}")
            for node_id, node in nodes.items():
                print(f"    {node_id}: {node.get('goal', 'No goal')} (Layer {node.get('layer', 0)}, Status: {node.get('status', 'Unknown')})")
        else:
            print("  No tree structure or all_nodes found")

        # ROMA-style API response
        viz_data = {
            "overall_project_goal": status["goal"]["title"],
            "all_nodes": status["tree_structure"]["all_nodes"],
            "graphs": {
                "main_graph": {
                    "edges": []  # Will be populated from parent_node_id relationships
                }
            }
        }

        # Build edges from parent-child relationships
        edges = []
        all_nodes = status["tree_structure"]["all_nodes"]
        for node_id, node in all_nodes.items():
            if node.get("parent_node_id"):
                edges.append({
                    "source": node["parent_node_id"],
                    "target": node_id
                })

        viz_data["graphs"]["main_graph"]["edges"] = edges

        return viz_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tree visualization: {str(e)}")


@router.get("/goals/{goal_id}/insights")
async def get_research_insights(goal_id: str):
    """Get AI-generated insights from research tree results"""
    try:
        # Check if goal exists in both active goals and research trees
        if goal_id not in research_system.active_goals:
            available_goals = list(research_system.active_goals.keys())
            error_detail = f"Goal '{goal_id}' not found in active goals. Available goals: {available_goals if available_goals else 'None'}"
            raise HTTPException(status_code=404, detail=error_detail)

        if goal_id not in research_system.research_trees:
            error_detail = f"Research tree for goal '{goal_id}' not found. Goal exists but no tree created yet."
            raise HTTPException(status_code=404, detail=error_detail)

        tree = research_system.research_trees[goal_id]
        goal = research_system.active_goals[goal_id]

        # Collect insights from all completed experiments
        all_insights = []
        confidence_scores = []
        experiment_types = []

        for node in tree.values():
            if node.results:
                for result in node.results:
                    all_insights.extend(result.insights)
                    confidence_scores.append(result.confidence)
                    if node.experiment_type:
                        experiment_types.append(node.experiment_type.value)

        # Generate meta-insights
        meta_insights = []

        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            meta_insights.append(f"Average experiment confidence: {avg_confidence:.2%}")

            high_conf_count = sum(1 for c in confidence_scores if c > 0.8)
            meta_insights.append(f"High-confidence results: {high_conf_count}/{len(confidence_scores)} experiments")

        if experiment_types:
            type_counts = {t: experiment_types.count(t) for t in set(experiment_types)}
            most_common = max(type_counts.keys(), key=lambda k: type_counts[k])
            meta_insights.append(f"Most utilized experiment type: {most_common}")

        # Research trajectory analysis
        completed_nodes = [n for n in tree.values() if n.status.value == "completed"]
        if completed_nodes:
            avg_depth = sum(n.depth for n in completed_nodes) / len(completed_nodes)
            meta_insights.append(f"Average exploration depth: {avg_depth:.1f}")

        return {
            "goal_id": goal_id,
            "research_goal": goal.title,
            "experiment_insights": all_insights,
            "meta_insights": meta_insights,
            "research_trajectory": {
                "total_experiments": len(confidence_scores),
                "success_rate": sum(1 for c in confidence_scores if c > 0.7) / max(len(confidence_scores), 1),
                "exploration_breadth": len(set(n.node_type.value for n in tree.values())),
                "max_depth_reached": max([n.depth for n in tree.values()], default=0)
            },
            "recommendations": await _generate_research_recommendations(goal_id)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get research insights: {str(e)}")


@router.get("/goals/active", response_model=ActiveGoalsResponse)
async def list_active_research_goals():
    """List all active research goals"""
    try:
        active_goals = await research_system.list_active_research_goals()

        return ActiveGoalsResponse(
            active_goals=active_goals,
            total_goals=len(active_goals)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list research goals: {str(e)}")


@router.delete("/goals/{goal_id}")
async def stop_research_goal(goal_id: str):
    """Stop and archive a research goal"""
    try:
        if goal_id not in research_system.active_goals:
            available_goals = list(research_system.active_goals.keys())
            error_detail = f"Research goal '{goal_id}' not found. Available goals: {available_goals if available_goals else 'None'}"
            raise HTTPException(status_code=404, detail=error_detail)

        # Archive the goal (in real implementation, would save to database)
        goal = research_system.active_goals.pop(goal_id)

        # Also remove from research trees if it exists
        if goal_id in research_system.research_trees:
            research_system.research_trees.pop(goal_id)

        return {
            "message": f"Research goal '{goal.title}' stopped and archived",
            "goal_id": goal_id,
            "final_stats": await research_system.get_research_tree_status(goal_id) if goal_id in research_system.research_trees else {}
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop research goal: {str(e)}")


@router.post("/goals/{goal_id}/nodes/{node_id}/expand")
async def expand_node(goal_id: str, node_id: str):
    """Expand a specific node in the research tree"""
    try:
        if goal_id not in research_system.research_trees:
            raise HTTPException(status_code=404, detail="Research goal not found")

        tree = research_system.research_trees[goal_id]
        if node_id not in tree:
            raise HTTPException(status_code=404, detail="Node not found")

        # Simulate node expansion
        node = tree[node_id]
        children_count = len(node.children) if node.children else 0
        new_children = min(3, 5 - children_count)  # Add up to 3 new children, max 5 total

        if node.children is None:
            node.children = []

        for i in range(new_children):
            child_id = f"{node_id}_child_{children_count + i + 1}"
            node.children.append(child_id)

        return {
            "message": f"Node expanded with {new_children} new children",
            "node_id": node_id,
            "new_children": new_children,
            "total_children": len(node.children)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to expand node: {str(e)}")


@router.post("/goals/{goal_id}/nodes/{node_id}/experiment")
async def run_node_experiment(goal_id: str, node_id: str):
    """Run an experiment on a specific node"""
    try:
        if goal_id not in research_system.research_trees:
            raise HTTPException(status_code=404, detail="Research goal not found")

        tree = research_system.research_trees[goal_id]
        if node_id not in tree:
            raise HTTPException(status_code=404, detail="Node not found")

        node = tree[node_id]

        # Update node status to running
        from ..core.research_tree import NodeStatus
        node.status = NodeStatus.RUNNING

        # Simulate experiment execution (in real system, would queue actual experiment)
        return {
            "message": "Experiment started",
            "node_id": node_id,
            "status": "running",
            "estimated_completion": "3-8 minutes"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run experiment: {str(e)}")


@router.get("/goals/{goal_id}/nodes/{node_id}/report")
async def get_node_report(goal_id: str, node_id: str):
    """Get detailed debugging report for a specific node"""
    try:
        if goal_id not in research_system.research_trees:
            raise HTTPException(status_code=404, detail="Research goal not found")

        tree = research_system.research_trees[goal_id]
        if node_id not in tree:
            raise HTTPException(status_code=404, detail="Node not found")

        node = tree[node_id]

        # Generate comprehensive debugging report
        report = {
            "node_id": node_id,
            "node_type": node.node_type.value,
            "status": node.status.value,
            "confidence": sanitize_float(node.confidence),
            "depth": node.depth,
            "visits": node.visits,
            "title": node.title,
            "description": node.description,

            # Enhanced debugging information
            "execution_logs": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level,
                    "message": log.message,
                    "context": log.context
                } for log in node.execution_logs
            ],

            "error_history": node.error_history,
            "last_error": node.last_error,
            "retry_count": node.retry_count,
            "debug_info": node.debug_info,

            # Timing information
            "timestamps": {
                "created_at": node.created_at.isoformat() if node.created_at else None,
                "started_at": node.started_at.isoformat() if node.started_at else None,
                "completed_at": node.completed_at.isoformat() if node.completed_at else None,
                "total_execution_time": (node.completed_at - node.started_at).total_seconds() if node.started_at and node.completed_at else None
            },

            # Experiment configuration and parameters
            "experiment_config": node.experiment_config,
            "experiment_type": node.experiment_type.value if node.experiment_type else None,
            "hypothesis": node.hypothesis,
            "context": node.context,
            "dependencies": node.dependencies,

            # Detailed results with debugging info
            "results": [],

            # UCB and tree search metrics
            "tree_search_metrics": {
                "ucb_score": sanitize_float(node.confidence + (2 * (2 * 0.693 / max(node.visits, 1)) ** 0.5) if node.visits > 0 else 999999.0),
                "exploration_bonus": sanitize_float((2 * 0.693 / max(node.visits, 1)) ** 0.5 if node.visits > 0 else 1.0),
                "exploitation_score": sanitize_float(node.confidence),
                "total_reward": sanitize_float(node.total_reward),
                "aggregated_score": sanitize_float(node.aggregated_score),
                "priority": sanitize_float(node.priority)
            }
        }

        # Add detailed results with debugging information
        if node.results:
            for i, result in enumerate(node.results):
                detailed_result = {
                    "result_index": i,
                    "success": result.success,
                    "confidence": sanitize_float(result.confidence),
                    "execution_time": sanitize_float(result.execution_time),
                    "insights": result.insights,
                    "metrics": sanitize_data(result.metrics),
                    "data": sanitize_data(result.data),
                    "resources_used": sanitize_data(result.resources_used),

                    # Enhanced debugging information
                    "execution_logs": [
                        {
                            "timestamp": log.timestamp.isoformat(),
                            "level": log.level,
                            "message": log.message,
                            "context": log.context
                        } for log in result.execution_logs
                    ],

                    "error_details": result.error_details,
                    "stack_trace": result.stack_trace,
                    "intermediate_results": result.intermediate_results,
                    "api_calls": result.api_calls,
                    "search_queries_used": result.search_queries_used,
                    "processing_steps": result.processing_steps
                }
                report["results"].append(detailed_result)

        return sanitize_data(report)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get node report: {str(e)}")


@router.delete("/goals/{goal_id}/nodes/{node_id}/prune")
async def prune_node(goal_id: str, node_id: str):
    """Prune a node and its children from the research tree"""
    try:
        if goal_id not in research_system.research_trees:
            raise HTTPException(status_code=404, detail="Research goal not found")

        tree = research_system.research_trees[goal_id]
        if node_id not in tree:
            raise HTTPException(status_code=404, detail="Node not found")

        node = tree[node_id]

        # Count nodes that will be pruned
        def count_descendants(node_id: str) -> int:
            count = 1
            node = tree.get(node_id)
            if node and node.children:
                for child_id in node.children:
                    count += count_descendants(child_id)
            return count

        pruned_count = count_descendants(node_id)

        # Remove node and descendants (simplified - in real system would need proper cleanup)
        def remove_descendants(node_id: str):
            node = tree.get(node_id)
            if node and node.children:
                for child_id in node.children:
                    remove_descendants(child_id)
                    if child_id in tree:
                        del tree[child_id]

        remove_descendants(node_id)
        if node_id in tree:
            del tree[node_id]

        return {
            "message": f"Pruned {pruned_count} nodes from the tree",
            "pruned_node_id": node_id,
            "pruned_count": pruned_count
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prune node: {str(e)}")


@router.post("/goals/{goal_id}/manual-experiment")
async def trigger_manual_experiment(
    goal_id: str,
    experiment_request: Dict[str, Any]
):
    """Manually trigger a specific experiment in the research tree"""
    try:
        if goal_id not in research_system.research_trees:
            raise HTTPException(status_code=404, detail="Research goal not found")

        # This would integrate with the tree expansion logic
        # For now, return success message
        return {
            "message": "Manual experiment queued",
            "goal_id": goal_id,
            "experiment_config": experiment_request,
            "estimated_completion": "5-15 minutes"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger manual experiment: {str(e)}")


@router.get("/goals/{goal_id}/export")
async def export_research_results(goal_id: str, format: str = "json"):
    """Export research results in various formats"""
    try:
        if goal_id not in research_system.research_trees:
            raise HTTPException(status_code=404, detail="Research goal not found")

        status = await research_system.get_research_tree_status(goal_id)
        insights = await get_research_insights(goal_id)

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "research_goal": status["goal"],
            "tree_statistics": status["tree_stats"],
            "best_results": status["best_results"],
            "research_insights": insights,
            "full_tree_structure": status["tree_structure"]
        }

        if format.lower() == "json":
            return export_data
        else:
            # For other formats, would implement conversion logic
            return {"message": f"Export format '{format}' not yet supported", "data": export_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export research results: {str(e)}")


@router.get("/system/metrics")
async def get_system_metrics():
    """Get overall system performance metrics"""
    try:
        total_goals = len(research_system.active_goals)
        total_trees = len(research_system.research_trees)

        all_nodes = []
        for tree in research_system.research_trees.values():
            all_nodes.extend(tree.values())

        total_experiments = len([n for n in all_nodes if n.results])
        successful_experiments = len([n for n in all_nodes if n.results and any(r.success for r in n.results)])

        running_experiments = len(research_system.running_experiments)

        return {
            "system_overview": {
                "active_research_goals": total_goals,
                "research_trees": total_trees,
                "total_experiments_run": total_experiments,
                "successful_experiments": successful_experiments,
                "success_rate": successful_experiments / max(total_experiments, 1),
                "currently_running": running_experiments
            },
            "performance_metrics": {
                "average_experiments_per_goal": total_experiments / max(total_goals, 1),
                "system_utilization": min(running_experiments / research_system.parallel_limit, 1.0),
                "exploration_efficiency": successful_experiments / max(total_experiments, 1)
            },
            "resource_usage": {
                "parallel_limit": research_system.parallel_limit,
                "current_load": running_experiments,
                "available_slots": research_system.parallel_limit - running_experiments
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


@router.get("/debug/system-state")
async def debug_system_state():
    """Debug endpoint to see current system state"""
    try:
        # Check database path and connection
        db_path = research_system.db.db_path if hasattr(research_system, 'db') else "No database"

        # Get database stats
        db_stats = {}
        if hasattr(research_system, 'db'):
            try:
                db_stats = research_system.get_database_stats()
            except Exception as e:
                db_stats = {"error": str(e)}

        return {
            "system_type": type(research_system).__name__,
            "database_path": db_path,
            "database_stats": db_stats,
            "active_goals_count": len(research_system.active_goals),
            "active_goals": [
                {
                    "id": goal_id,
                    "title": goal.title,
                    "description": goal.description
                } for goal_id, goal in research_system.active_goals.items()
            ],
            "research_trees_count": len(research_system.research_trees),
            "research_trees": [
                {
                    "tree_id": tree_id,
                    "nodes_count": len(tree)
                } for tree_id, tree in research_system.research_trees.items()
            ],
            "running_experiments_count": len(research_system.running_experiments)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get debug info: {str(e)}")


@router.post("/debug/create-test-goal")
async def create_test_goal():
    """Create a test goal for debugging"""
    try:
        goal_id = await research_system.start_research_goal(
            title="Test Docker Hello World Task",
            description="Pull a hello-world docker and run it for testing",
            success_criteria=[
                "Successfully pull hello-world Docker image",
                "Successfully run hello-world container"
            ],
            max_depth=2,
            max_experiments=3
        )

        return {
            "message": "Test goal created successfully",
            "goal_id": goal_id,
            "title": "Test Docker Hello World Task"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create test goal: {str(e)}")


@router.get("/debug/frontend-state")
async def get_frontend_state():
    """Get state information for frontend debugging"""
    try:
        active_goals_list = await research_system.list_active_research_goals()

        # If no goals exist, suggest creating one
        if not active_goals_list:
            return {
                "status": "no_active_goals",
                "message": "No active research goals found. The frontend may be trying to access a stale goal ID.",
                "suggestion": "Create a new goal using POST /api/research-tree/goals/start",
                "test_goal_endpoint": "POST /api/research-tree/debug/create-test-goal",
                "active_goals": []
            }

        return {
            "status": "goals_available",
            "active_goals": active_goals_list,
            "total_goals": len(active_goals_list),
            "message": "Active goals found"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get frontend state: {str(e)}"
        }


@router.websocket("/goals/{goal_id}/logs")
async def websocket_goal_logs(websocket: WebSocket, goal_id: str):
    """WebSocket endpoint for real-time goal logging"""
    try:
        await websocket_manager.connect(websocket, goal_id)

        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": f"Connected to real-time logs for goal {goal_id}",
            "timestamp": datetime.now().isoformat(),
            "goal_id": goal_id
        }))

        # Send current goal status
        if goal_id in research_system.active_goals:
            goal = research_system.active_goals[goal_id]
            await websocket.send_text(json.dumps({
                "type": "goal_status",
                "goal": {
                    "id": goal_id,
                    "title": goal.title,
                    "description": goal.description,
                    "status": "active"
                },
                "timestamp": datetime.now().isoformat()
            }))

            # Send current tree status
            if goal_id in research_system.research_trees:
                tree = research_system.research_trees[goal_id]
                nodes_info = []
                for node_id, node in tree.items():
                    nodes_info.append({
                        "id": node_id,
                        "title": node.title,
                        "status": node.status.value,
                        "confidence": sanitize_float(node.confidence),
                        "depth": node.depth
                    })

                await websocket.send_text(json.dumps({
                    "type": "tree_status",
                    "nodes": nodes_info,
                    "timestamp": datetime.now().isoformat()
                }))

        # Keep connection alive and listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                # Echo back any client messages (for testing)
                await websocket.send_text(json.dumps({
                    "type": "echo",
                    "received": data,
                    "timestamp": datetime.now().isoformat()
                }))
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    finally:
        websocket_manager.disconnect(websocket, goal_id)


@router.websocket("/goals/{goal_id}/nodes/{node_id}/logs")
async def websocket_node_logs(websocket: WebSocket, goal_id: str, node_id: str):
    """WebSocket endpoint for real-time node-specific logging"""
    try:
        await websocket_manager.connect(websocket, goal_id, node_id)

        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": f"Connected to real-time logs for node {node_id}",
            "timestamp": datetime.now().isoformat(),
            "goal_id": goal_id,
            "node_id": node_id
        }))

        # Send current node status if it exists
        if goal_id in research_system.research_trees:
            tree = research_system.research_trees[goal_id]
            if node_id in tree:
                node = tree[node_id]
                await websocket.send_text(json.dumps({
                    "type": "node_status",
                    "node": {
                        "id": node_id,
                        "title": node.title,
                        "status": node.status.value,
                        "confidence": sanitize_float(node.confidence),
                        "depth": node.depth,
                        "experiment_type": node.experiment_type.value if node.experiment_type else None
                    },
                    "timestamp": datetime.now().isoformat()
                }))

                # Send recent execution logs
                if hasattr(node, 'execution_logs') and node.execution_logs:
                    for log_entry in node.execution_logs[-10:]:  # Last 10 logs
                        await websocket.send_text(json.dumps({
                            "type": "execution_log",
                            "log": {
                                "level": log_entry.level,
                                "message": log_entry.message,
                                "timestamp": log_entry.timestamp.isoformat(),
                                "context": log_entry.context
                            },
                            "node_id": node_id
                        }))

        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                await websocket.send_text(json.dumps({
                    "type": "echo",
                    "received": data,
                    "timestamp": datetime.now().isoformat()
                }))
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    finally:
        websocket_manager.disconnect(websocket, goal_id, node_id)


@router.post("/goals/{goal_id}/generate-report")
async def generate_completion_report(goal_id: str):
    """Generate a comprehensive markdown report for a completed research goal"""
    try:
        if goal_id not in research_system.active_goals:
            raise HTTPException(status_code=404, detail="Research goal not found")

        # Check if the goal is completed
        goal = research_system.active_goals[goal_id]
        tree = research_system.research_trees.get(goal_id, {})

        # Basic completion check - at least one successful experiment
        has_successful_results = any(
            node.results and any(r.success for r in node.results)
            for node in tree.values()
        )

        if not has_successful_results:
            raise HTTPException(
                status_code=400,
                detail="Cannot generate report: research goal has no successful results yet"
            )

        # Generate the report
        report_path = await report_generator.generate_completion_report(goal_id)

        return {
            "message": "Report generated successfully",
            "goal_id": goal_id,
            "report_path": report_path,
            "filename": os.path.basename(report_path),
            "view_url": f"/api/research-tree/goals/{goal_id}/report/view",
            "download_url": f"/api/research-tree/goals/{goal_id}/report/download"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/goals/{goal_id}/report/view")
async def view_report(goal_id: str):
    """View the markdown report for a research goal in the browser"""
    try:
        if goal_id not in research_system.active_goals:
            raise HTTPException(status_code=404, detail="Research goal not found")

        # Get the latest report path
        report_path = report_generator.get_report_path(goal_id)

        if not report_path or not os.path.exists(report_path):
            raise HTTPException(
                status_code=404,
                detail="Report not found. Generate a report first using /generate-report endpoint"
            )

        # Read the markdown content
        with open(report_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Convert markdown to HTML for better web viewing
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report - {goal_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 2em;
            margin-bottom: 0.5em;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        pre {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
        }}
        code {{
            background: #f1f3f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 1.5em 0;
            padding-left: 20px;
            color: #7f8c8d;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .download-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .download-button:hover {{
            background: #2980b9;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <a href="/api/research-tree/goals/{goal_id}/report/download" class="download-button">📥 Download Report</a>
    <div class="timestamp">Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    <div id="content">{_markdown_to_basic_html(markdown_content)}</div>
</body>
</html>"""

        return Response(content=html_content, media_type="text/html")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to view report: {str(e)}")


@router.get("/goals/{goal_id}/report/download")
async def download_report(goal_id: str):
    """Download the markdown report file for a research goal"""
    try:
        if goal_id not in research_system.active_goals:
            raise HTTPException(status_code=404, detail="Research goal not found")

        # Get the latest report path
        report_path = report_generator.get_report_path(goal_id)

        if not report_path or not os.path.exists(report_path):
            raise HTTPException(
                status_code=404,
                detail="Report not found. Generate a report first using /generate-report endpoint"
            )

        # Get the filename
        filename = os.path.basename(report_path)

        return FileResponse(
            path=report_path,
            filename=filename,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download report: {str(e)}")


@router.get("/goals/{goal_id}/report/raw")
async def get_raw_report(goal_id: str):
    """Get the raw markdown content of the report"""
    try:
        if goal_id not in research_system.active_goals:
            raise HTTPException(status_code=404, detail="Research goal not found")

        # Get the latest report path
        report_path = report_generator.get_report_path(goal_id)

        if not report_path or not os.path.exists(report_path):
            raise HTTPException(
                status_code=404,
                detail="Report not found. Generate a report first using /generate-report endpoint"
            )

        # Read and return the markdown content
        with open(report_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        return {
            "goal_id": goal_id,
            "filename": os.path.basename(report_path),
            "markdown_content": markdown_content,
            "generated_at": datetime.fromtimestamp(os.path.getctime(report_path)).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get raw report: {str(e)}")


# Helper functions


def _markdown_to_basic_html(markdown_text: str) -> str:
    """Convert basic markdown to HTML for web viewing"""
    import re

    # Simple markdown to HTML conversion
    html = markdown_text

    # Headers
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)

    # Bold and italic
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)

    # Code blocks
    html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)

    # Lists
    lines = html.split('\n')
    in_list = False
    result_lines = []

    for line in lines:
        if line.strip().startswith('- '):
            if not in_list:
                result_lines.append('<ul>')
                in_list = True
            result_lines.append(f'<li>{line.strip()[2:]}</li>')
        else:
            if in_list:
                result_lines.append('</ul>')
                in_list = False
            result_lines.append(line)

    if in_list:
        result_lines.append('</ul>')

    html = '\n'.join(result_lines)

    # Paragraphs
    paragraphs = html.split('\n\n')
    formatted_paragraphs = []

    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith('<'):
            para = f'<p>{para}</p>'
        formatted_paragraphs.append(para)

    html = '\n\n'.join(formatted_paragraphs)

    # Line breaks
    html = html.replace('\n', '<br>\n')

    return html


# Helper functions

async def _get_experiment_distribution(goal_id: str) -> Dict[str, int]:
    """Get distribution of experiment types"""
    tree = research_system.research_trees.get(goal_id, {})

    distribution = {}
    for node in tree.values():
        if node.experiment_type:
            exp_type = node.experiment_type.value
            distribution[exp_type] = distribution.get(exp_type, 0) + 1

    return distribution


async def _get_research_timeline(goal_id: str) -> List[Dict[str, Any]]:
    """Get research timeline events"""
    tree = research_system.research_trees.get(goal_id, {})

    timeline = []
    for node in tree.values():
        if node.completed_at:
            timeline.append({
                "timestamp": node.completed_at.isoformat(),
                "event": f"Completed: {node.title}",
                "node_type": node.node_type.value,
                "confidence": node.confidence,
                "depth": node.depth
            })

    # Sort by timestamp
    timeline.sort(key=lambda x: x["timestamp"])

    return timeline


async def _generate_research_recommendations(goal_id: str) -> List[str]:
    """Generate AI recommendations for research direction"""
    tree = research_system.research_trees.get(goal_id, {})
    goal = research_system.active_goals.get(goal_id)

    recommendations = []

    if not tree or not goal:
        return recommendations

    # Analyze current state
    completed_nodes = [n for n in tree.values() if n.status.value == "completed"]
    high_confidence_nodes = [n for n in completed_nodes if n.confidence > 0.8]

    if len(high_confidence_nodes) >= 2:
        recommendations.append("Consider focusing on validation experiments for high-confidence results")

    if len(completed_nodes) > 10 and len(high_confidence_nodes) < 2:
        recommendations.append("Current approach may need refinement - consider pivoting research direction")

    # Check exploration balance
    experiment_types = [n.experiment_type.value for n in tree.values() if n.experiment_type]
    type_counts = {t: experiment_types.count(t) for t in set(experiment_types)}

    if len(type_counts) < 3:
        recommendations.append("Consider diversifying experiment types for broader exploration")

    # Depth analysis
    max_depth = max([n.depth for n in tree.values()], default=0)
    if max_depth < 3:
        recommendations.append("Research could benefit from deeper exploration of promising paths")

    return recommendations