"""Deterministic, idempotent runner built on OpenHandsClientV2.

This module intentionally removes the legacy CodeAct LLM tool-calling loop.
Only V2 helpers remain for predictable file operations and process execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..integrations.openhands_runtime import OpenHandsClientV2
from ..integrations.openhands.types import ActionResult


__all__ = ["CodeActRunnerV2"]


class CodeActRunnerV2:
    """Deterministic, idempotent runner wrapping OpenHandsClientV2.

    Provides: ensure_bootstrap, create_if_absent, write, read, run.
    """

    def __init__(self, client: OpenHandsClientV2, workspace_dir: Path):
        self.client = client
        self.ws_dir = workspace_dir

    async def ensure_bootstrap(self) -> Dict[str, Any]:
        """Ensure venv exists and requirements (if present) are installed.
        Writes a small stamp at ./.uagent/bootstrap.json to skip repeat installs.
        """
        stamp_fp = self.ws_dir / ".uagent" / "bootstrap.json"
        try:
            if stamp_fp.exists():
                return json.loads(stamp_fp.read_text())
        except Exception:
            pass

        # create venv and upgrade pip/wheel
        jid = await self.client.start_job(
            "python3 -m venv ./.venv && ./.venv/bin/pip install -U pip wheel",
            timeout_sec=900,
        )
        res = await self.client.wait_job(jid, 1200)
        if not res.success:
            raise RuntimeError(f"bootstrap venv failed: {getattr(res.error, 'message', '')}")

        # install requirements if present
        req = self.ws_dir / "requirements.txt"
        req_sha = None
        if req.exists():
            req_sha = hashlib.sha256(req.read_bytes()).hexdigest()
            jid2 = await self.client.start_job("./.venv/bin/pip install -r requirements.txt", timeout_sec=3600)
            res2 = await self.client.wait_job(jid2, 5400)
            if not res2.success:
                raise RuntimeError(f"pip install failed: {getattr(res2.error, 'message', '')}")

        stamp_fp.parent.mkdir(parents=True, exist_ok=True)
        payload = {"venv_path": "./.venv", "requirements_sha256": req_sha, "ready": True}
        stamp_fp.write_text(json.dumps(payload))
        return payload

    async def create_if_absent(self, path: str, content: str) -> ActionResult:
        return await self.client.create_if_absent(path, content)

    async def write(self, path: str, content: str, overwrite: bool = True) -> ActionResult:
        return await self.client.write(path, content, overwrite=overwrite)

    async def read(self, path: str) -> ActionResult:
        return await self.client.read(path)

    async def run(self, cmd: str, timeout_sec: int = 300) -> ActionResult:
        jid = await self.client.start_job(cmd, timeout_sec=timeout_sec)
        return await self.client.wait_job(jid, max_wait_sec=max(timeout_sec + 60, 120))

