"""
Unified Workflow API Router
Provides REST API endpoints for the unified orchestrator
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..core.unified_orchestrator import UnifiedOrchestrator, WorkflowType, OrchestrationStrategy, WorkflowConfig, CollaborationPattern

router = APIRouter(prefix="/unified", tags=["unified-workflow"])

# Initialize the unified orchestrator
orchestrator = UnifiedOrchestrator()


class WorkflowRequest(BaseModel):
    """Request model for workflow execution"""
    workflow_type: str
    template_name: Optional[str] = None
    strategy: Optional[str] = "adaptive"
    components: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    inputs: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = 3600
    collaboration_pattern: Optional[str] = None


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    workflow_id: str
    workflow_type: str
    status: str
    components_used: List[str]
    execution_time: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]
    progress: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Request model for intelligent search"""
    query: str
    search_types: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 20


class ResearchRequest(BaseModel):
    """Request model for research project"""
    title: str
    description: Optional[str] = None
    research_questions: Optional[List[str]] = None
    collaboration_enabled: Optional[bool] = True
    auto_iteration: Optional[bool] = True


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis"""
    repository_path: str
    analysis_depth: Optional[str] = "semantic"
    pattern_detection: Optional[bool] = True
    collaboration_enabled: Optional[bool] = True


@router.post("/workflows/execute", response_model=Dict[str, str])
async def execute_workflow(request: WorkflowRequest):
    """Execute a unified workflow"""
    try:
        # Handle template-based workflows
        if request.template_name:
            workflow_id = await orchestrator.execute_workflow(
                workflow_config=request.template_name,
                inputs=request.inputs or {}
            )
        else:
            # Create custom workflow config
            try:
                workflow_type = WorkflowType(request.workflow_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid workflow type: {request.workflow_type}")

            try:
                strategy = OrchestrationStrategy(request.strategy) if request.strategy else OrchestrationStrategy.ADAPTIVE
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")

            collaboration_pattern = None
            if request.collaboration_pattern:
                try:
                    collaboration_pattern = CollaborationPattern(request.collaboration_pattern)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid collaboration pattern: {request.collaboration_pattern}")

            config = WorkflowConfig(
                workflow_type=workflow_type,
                strategy=strategy,
                components=request.components or [],
                parameters=request.parameters or {},
                timeout=request.timeout,
                collaboration_pattern=collaboration_pattern
            )

            workflow_id = await orchestrator.execute_workflow(
                workflow_config=config,
                inputs=request.inputs or {}
            )

        return {"workflow_id": workflow_id, "status": "started"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute workflow: {str(e)}")


@router.get("/workflows/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get status of a specific workflow"""
    workflow_result = await orchestrator.get_workflow_status(workflow_id)

    if not workflow_result:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return WorkflowStatusResponse(
        workflow_id=workflow_result.workflow_id,
        workflow_type=workflow_result.workflow_type.value,
        status=workflow_result.status,
        components_used=workflow_result.components_used,
        execution_time=workflow_result.execution_time,
        created_at=workflow_result.created_at,
        completed_at=workflow_result.completed_at
    )


@router.get("/workflows/{workflow_id}/results")
async def get_workflow_results(workflow_id: str):
    """Get results of a completed workflow"""
    workflow_result = await orchestrator.get_workflow_status(workflow_id)

    if not workflow_result:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if workflow_result.status not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Workflow not yet completed")

    return {
        "workflow_id": workflow_id,
        "status": workflow_result.status,
        "results": workflow_result.results,
        "metadata": workflow_result.metadata
    }


@router.get("/workflows/active")
async def list_active_workflows():
    """List all active workflows"""
    active_workflows = await orchestrator.list_active_workflows()

    return {
        "active_workflows": [
            {
                "workflow_id": w.workflow_id,
                "workflow_type": w.workflow_type.value,
                "status": w.status,
                "created_at": w.created_at,
                "components_used": w.components_used
            }
            for w in active_workflows
        ]
    }


@router.delete("/workflows/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """Cancel an active workflow"""
    success = await orchestrator.cancel_workflow(workflow_id)

    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel workflow (not found or not active)")

    return {"message": "Workflow cancelled successfully"}


@router.post("/research/start")
async def start_research_project(request: ResearchRequest):
    """Start an automated research project"""
    try:
        inputs = {
            "title": request.title,
            "description": request.description,
            "research_questions": request.research_questions or []
        }

        parameters = {
            "collaboration_enabled": request.collaboration_enabled,
            "auto_iteration": request.auto_iteration
        }

        config = WorkflowConfig(
            workflow_type=WorkflowType.AUTOMATED_RESEARCH,
            strategy=OrchestrationStrategy.ROMA_RECURSIVE,
            components=["ai_scientist", "search_engine", "agent_lab", "meta_agent"],
            parameters=parameters
        )

        workflow_id = await orchestrator.execute_workflow(config, inputs)

        return {
            "workflow_id": workflow_id,
            "project_title": request.title,
            "status": "started",
            "message": "Research project initiated"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start research project: {str(e)}")


@router.post("/code/analyze")
async def analyze_code_repository(request: CodeAnalysisRequest):
    """Analyze a code repository"""
    try:
        inputs = {
            "repository_path": request.repository_path
        }

        parameters = {
            "analysis_depth": request.analysis_depth,
            "pattern_detection": request.pattern_detection,
            "collaboration_enabled": request.collaboration_enabled
        }

        config = WorkflowConfig(
            workflow_type=WorkflowType.CODE_ANALYSIS,
            strategy=OrchestrationStrategy.SEQUENTIAL,
            components=["repo_master", "agent_lab", "search_engine"],
            parameters=parameters
        )

        workflow_id = await orchestrator.execute_workflow(config, inputs)

        return {
            "workflow_id": workflow_id,
            "repository_path": request.repository_path,
            "status": "started",
            "message": "Code analysis initiated"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze repository: {str(e)}")


@router.post("/search/intelligent")
async def intelligent_search(request: SearchRequest):
    """Perform intelligent multi-modal search"""
    try:
        search_results = await orchestrator.search_engine.intelligent_search(
            query=request.query,
            context=request.context or {}
        )

        return {
            "query": request.query,
            "results": search_results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/search/unified")
async def unified_search(request: SearchRequest):
    """Perform unified search across multiple sources"""
    try:
        search_results = await orchestrator.search_engine.unified_search(
            query=request.query,
            search_types=request.search_types,
            filters=request.filters,
            limit=request.limit or 20
        )

        return {
            "query": request.query,
            "search_types": request.search_types,
            "results": search_results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/templates")
async def list_workflow_templates():
    """List available workflow templates"""
    return {
        "templates": list(orchestrator.workflow_templates.keys()),
        "template_details": {
            name: {
                "workflow_type": config.workflow_type.value,
                "strategy": config.strategy.value,
                "components": config.components,
                "description": f"Template for {config.workflow_type.value} workflow"
            }
            for name, config in orchestrator.workflow_templates.items()
        }
    }


@router.get("/system/status")
async def get_system_status():
    """Get overall system status"""
    try:
        status = await orchestrator.get_system_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.get("/system/health")
async def health_check():
    """System health check"""
    try:
        # Perform basic health checks
        status = await orchestrator.get_system_status()

        # Check if core components are functional
        health_status = "healthy"
        issues = []

        if status["components"]["meta_agent"]["total_agents"] == 0:
            issues.append("No agents available in meta-agent system")
            health_status = "degraded"

        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "components_status": status["components"],
            "issues": issues
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }