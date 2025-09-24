"""Artifacts router for listing and previewing files in a session workspace."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..core.app_state import get_app_state
from ..core.openhands.client import OpenHandsClient


router = APIRouter(prefix="/artifacts", tags=["artifacts"])


def _get_openhands_client() -> OpenHandsClient:
    client = get_app_state().get("openhands_client")
    if not isinstance(client, OpenHandsClient):
        raise HTTPException(status_code=503, detail="OpenHands client unavailable")
    return client


def _resolve_workspace(session_id: str) -> Path:
    client = _get_openhands_client()
    state = client.session_states.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown session")
    ws_path = client.workspace_manager.get_workspace_path(state.workspace_id)
    if not ws_path:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws_path


@router.get("/{session_id}/list")
async def list_artifacts(
    session_id: str,
    path: Optional[str] = Query(None, description="Relative path under workspace to list"),
) -> Dict[str, Any]:
    base = _resolve_workspace(session_id)
    root = base if not path else (base / path)
    try:
        root = root.resolve()
        base_resolved = base.resolve()
        # Prevent path traversal
        root.relative_to(base_resolved)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not root.exists():
        raise HTTPException(status_code=404, detail="Path does not exist")
    if not root.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    entries: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        try:
            stat = child.stat()
            entries.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(base)),
                    "is_dir": child.is_dir(),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
        except Exception:
            continue
    return {"base": str(base), "path": str(root.relative_to(base)), "entries": entries}


@router.get("/{session_id}/read")
async def read_artifact(
    session_id: str,
    path: str = Query(..., description="Relative file path under workspace"),
    max_bytes: int = Query(65536, ge=1, le=1048576),
) -> Dict[str, Any]:
    base = _resolve_workspace(session_id)
    target = (base / path)
    try:
        target = target.resolve()
        base_resolved = base.resolve()
        target.relative_to(base_resolved)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    data: bytes
    try:
        with target.open("rb") as f:
            data = f.read(max_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}")

    # Best-effort text preview
    preview: Optional[str] = None
    try:
        preview = data.decode("utf-8", errors="replace")
    except Exception:
        preview = None

    return {
        "path": str(target.relative_to(base)),
        "size": target.stat().st_size,
        "preview": preview,
        "truncated": target.stat().st_size > len(data),
    }

