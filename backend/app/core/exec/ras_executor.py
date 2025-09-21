from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .ras import ResearchActionSpec
from ...utils.atomic_fs import atomic_write_json


class RASExecutor:
    """Minimal Research Action Spec executor scaffold.

    This scaffolding is generic. It expects external plugins or existing services
    to implement build, run, and artifact collection. For now it records intent
    and writes a summary artifact so callers can verify invocation.
    """

    def __init__(self, ws_mgr: Any | None = None, capability_mgr: Any | None = None, openhands_rt: Any | None = None, plugins: Dict[str, Any] | None = None):
        self.ws_mgr = ws_mgr
        self.caps = capability_mgr
        self.oh = openhands_rt
        self.plugins = plugins or {}

    async def execute(self, ras: ResearchActionSpec, run_dir: Path, goal_id: str, node_id: str) -> List[Dict[str, Any]]:
        run_dir.mkdir(parents=True, exist_ok=True)
        # Capability negotiation if a manager is provided
        granted = {"ok": True, "granted": ras.run.get("capabilities_required", [])}
        if self.caps is not None:
            grant = await self.caps.negotiate(list(ras.run.get("capabilities_required", [])))
            if not getattr(grant, "ok", False):
                raise RuntimeError(f"CAPABILITY_REFUSED: {getattr(grant, 'reason', 'unspecified')}")
            granted = grant.__dict__ if hasattr(grant, "__dict__") else {"ok": True}

        # Optionally connect OpenHands runtime (Socket.IO bridge) if provided
        if self.oh is not None:
            try:
                await self.oh.start_session(goal_id, node_id)
            except Exception:
                # Non-fatal here; higher layers may require OH elsewhere
                pass

        # Placeholder execution loop; integrate with plugins if present
        results: List[Dict[str, Any]] = []
        for step in ras.steps:
            results.append({"id": step.id, "kind": step.kind, "status": "planned"})

        atomic_write_json(run_dir / "artifacts" / "exec_summary.json", {
            "granted": granted,
            "steps": [r for r in results],
        })
        return results

