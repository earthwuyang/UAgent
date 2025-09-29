"""Direct OpenHands runner endpoints for targeted testing.

Bypasses UAgent routers/engines and launches the single-container
OpenHands runtime directly with a provided goal. Useful for quick
manual or automated tests of the OpenHands pipeline.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..integrations.openhands_single_container import (
    OpenHandsSingleContainer,
    SingleContainerConfig,
    SingleContainerResult,
)


logger = logging.getLogger(__name__)
router = APIRouter()


class DirectRunRequest(BaseModel):
    goal: str = Field(..., description="Full OpenHands task/goal text")
    session_name: Optional[str] = Field(None, description="Session name for experiment directory")
    max_steps: Optional[int] = Field(None, description="Max OpenHands steps/iterations (env fallback)")
    max_minutes: Optional[int] = Field(None, description="Max wall clock time in minutes; <=0 disables timeout (env fallback)")
    workspace_root: Optional[str] = Field(None, description="Override workspace root directory")


class DirectRunResponse(BaseModel):
    success: bool
    exit_code: int
    duration_seconds: float
    final_json: Optional[Dict[str, Any]] = None
    stdout_logs: Optional[str] = None
    stderr_logs: Optional[str] = None
    error_message: Optional[str] = None
    workspace: str
    session_name: str


@router.post("/direct-run", response_model=DirectRunResponse, summary="Run OpenHands single-container directly with a goal")
async def openhands_direct_run(request: DirectRunRequest) -> DirectRunResponse:
    if not request.goal or not request.goal.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Goal is required")

    # Determine workspace root
    root = request.workspace_root or os.getenv("UAGENT_WORKSPACE_DIR") or "/tmp/uagent-workspace"
    workspace = Path(root).expanduser().resolve() / "direct_runs"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error("Failed to create workspace root %s: %s", workspace, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    # Session name
    import uuid
    session_name = request.session_name or f"exp_{uuid.uuid4().hex[:8]}"
    session_dir = workspace

    # Resolve defaults from environment if not provided
    env_max_steps = int(os.getenv("UAGENT_OPENHANDS_MAX_STEPS", "999999999"))
    env_max_minutes = int(os.getenv("UAGENT_OPENHANDS_MAX_MINUTES", "0"))  # 0 -> no timeout

    cfg = SingleContainerConfig(
        goal=request.goal,
        workspace=session_dir,
        session_name=session_name,
        max_steps=int(request.max_steps) if request.max_steps is not None else env_max_steps,
        max_minutes=int(request.max_minutes) if request.max_minutes is not None else env_max_minutes,
        llm_model=os.getenv("LLM_MODEL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
    )

    runner = OpenHandsSingleContainer()
    try:
        result: SingleContainerResult = await runner.run_async(cfg)
    except Exception as exc:
        logger.error("OpenHands direct run failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return DirectRunResponse(
        success=result.success,
        exit_code=result.exit_code,
        duration_seconds=result.duration_seconds,
        final_json=result.final_json,
        stdout_logs=(result.stdout_logs[-2000:] if result.stdout_logs else None),
        stderr_logs=(result.stderr_logs[-2000:] if result.stderr_logs else None),
        error_message=result.error_message,
        workspace=str(session_dir),
        session_name=session_name,
    )
