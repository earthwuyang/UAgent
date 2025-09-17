"""
Research Tree API Router
REST API for the hierarchical tree search-based research system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.research_tree import HierarchicalResearchSystem

router = APIRouter(prefix="/research-tree", tags=["research-tree"])

# Initialize the hierarchical research system
research_system = HierarchicalResearchSystem()


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
        if goal_id not in research_system.research_trees:
            raise HTTPException(status_code=404, detail="Research goal not found")

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
            raise HTTPException(status_code=404, detail="Research goal not found")

        # Archive the goal (in real implementation, would save to database)
        goal = research_system.active_goals.pop(goal_id)

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
            "confidence": node.confidence,
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
                "ucb_score": node.confidence + (2 * (2 * 0.693 / max(node.visits, 1)) ** 0.5) if node.visits > 0 else float('inf'),
                "exploration_bonus": (2 * 0.693 / max(node.visits, 1)) ** 0.5 if node.visits > 0 else 1.0,
                "exploitation_score": node.confidence,
                "total_reward": node.total_reward,
                "aggregated_score": node.aggregated_score,
                "priority": node.priority
            }
        }

        # Add detailed results with debugging information
        if node.results:
            for i, result in enumerate(node.results):
                detailed_result = {
                    "result_index": i,
                    "success": result.success,
                    "confidence": result.confidence,
                    "execution_time": result.execution_time,
                    "insights": result.insights,
                    "metrics": result.metrics,
                    "data": result.data,
                    "resources_used": result.resources_used,

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

        return report

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