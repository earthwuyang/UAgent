from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .ras import ResearchActionSpec
from ..assertions import command_executed, file_exists, git_diff_nonempty
from ...utils.atomic_fs import atomic_write_json


class RASExecutionError(RuntimeError):
    """Raised when a RAS step fails."""


class RASExecutor:
    """Execute Research Action Specs with evidence logging.

    The executor favours real command execution with strict failure handling.
    Each step logs events to `artifacts/ras_events.jsonl` and command invocations
    to `artifacts/exec.log` so upstream components can assert on behaviour.
    """

    def __init__(
        self,
        ws_mgr: Any | None = None,
        capability_mgr: Any | None = None,
        openhands_rt: Any | None = None,
        plugins: Dict[str, Any] | None = None,
    ) -> None:
        self.ws_mgr = ws_mgr
        self.caps = capability_mgr
        self.oh = openhands_rt
        self.plugins = plugins or {}
        self._events_lock = asyncio.Lock()

    async def execute(self, ras: ResearchActionSpec, run_dir: Path, goal_id: str, node_id: str) -> List[Dict[str, Any]]:
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        events_path = artifacts_dir / "ras_events.jsonl"
        exec_log_path = artifacts_dir / "exec.log"

        granted = {"ok": True, "granted": ras.run.get("capabilities_required", [])}
        if self.caps is not None:
            grant = await self.caps.negotiate(list(ras.run.get("capabilities_required", [])))
            if not getattr(grant, "ok", False):
                raise RuntimeError(f"CAPABILITY_REFUSED: {getattr(grant, 'reason', 'unspecified')}")
            granted = grant.__dict__ if hasattr(grant, "__dict__") else {"ok": True}

        if self.oh is not None:
            try:
                await self.oh.start_session(goal_id, node_id)
            except Exception:
                # Higher layers may treat absence as fatal; we simply proceed.
                pass

        results: List[Dict[str, Any]] = []
        for step in ras.steps:
            step_record: Dict[str, Any] = {
                "id": step.id,
                "kind": step.kind,
                "status": "in_progress",
                "started_at": datetime.utcnow().isoformat() + "Z",
            }
            results.append(step_record)
            await self._append_event(events_path, {"type": "step_start", **step_record})

            try:
                outcome = await self._execute_step(step, run_dir, exec_log_path)
                step_record.update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                    "outcome": outcome,
                })
                await self._append_event(events_path, {"type": "step_complete", **step_record})
            except Exception as exc:
                step_record.update({
                    "status": "failed",
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                    "error": str(exc),
                })
                await self._append_event(events_path, {"type": "step_failed", **step_record})
                break

        assertion_records = []
        if ras.assertions:
            assertion_records = self._evaluate_assertions(
                ras.assertions,
                run_dir,
                exec_log_path,
            )
            atomic_write_json(artifacts_dir / "assertions.json", assertion_records)

        atomic_write_json(artifacts_dir / "exec_summary.json", {
            "granted": granted,
            "steps": [r for r in results],
            "assertions": assertion_records,
        })
        return results

    async def _append_event(self, path: Path, payload: Dict[str, Any]) -> None:
        async with self._events_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    async def _execute_step(self, step, run_dir: Path, exec_log_path: Path) -> Dict[str, Any]:
        plugin = self.plugins.get(step.kind) if self.plugins else None
        if plugin is not None:
            return await plugin({"run_dir": run_dir}, step.dict(by_alias=True))

        if step.kind == "run_commands":
            return await self._handle_run_commands(step, run_dir, exec_log_path)
        if step.kind == "fetch_repo":
            return await self._handle_fetch_repo(step, run_dir, exec_log_path)
        if step.kind == "collect_artifacts":
            return self._handle_collect_artifacts(step, run_dir)
        if step.kind == "code_edit":
            raise RASExecutionError("code_edit requires an OpenHands runtime plugin; none configured")
        raise RASExecutionError(f"Unhandled RAS step kind: {step.kind}")

    async def _handle_run_commands(self, step, run_dir: Path, exec_log_path: Path) -> Dict[str, Any]:
        cfg = step.dict(by_alias=True).get("with") or {}
        commands = cfg.get("commands")
        if not isinstance(commands, list) or not commands:
            raise RASExecutionError("run_commands step missing non-empty 'commands' list")
        cwd = cfg.get("cwd") or "."
        timeout = int(cfg.get("timeout_sec", 3600))
        env = cfg.get("env") or {}
        if not isinstance(env, dict):
            raise RASExecutionError("run_commands env must be a dict")

        proc_cwd = (run_dir / cwd).resolve()
        proc_cwd.mkdir(parents=True, exist_ok=True)
        base_env = os.environ.copy()
        base_env.update({str(k): str(v) for k, v in env.items()})

        cmd_results: List[Dict[str, Any]] = []
        for index, command in enumerate(commands):
            if isinstance(command, str):
                args = ["bash", "-lc", command]
            elif isinstance(command, list):
                args = [str(part) for part in command]
            else:
                raise RASExecutionError("Commands must be strings or lists")

            started_at = datetime.utcnow().isoformat() + "Z"
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(proc_cwd),
                env=base_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RASExecutionError(f"Command timed out after {timeout}s: {' '.join(args)}")

            rc = process.returncode
            stdout_text = stdout.decode("utf-8", "replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", "replace") if stderr else ""

            await self._append_event(exec_log_path, {
                "type": "run",
                "step_id": step.id,
                "index": index,
                "cmd": args,
                "cwd": str(proc_cwd),
                "rc": rc,
                "stdout": stdout_text[-2000:],
                "stderr": stderr_text[-2000:],
                "started_at": started_at,
                "completed_at": datetime.utcnow().isoformat() + "Z",
            })

            if rc != 0:
                raise RASExecutionError(
                    f"Command exited with code {rc}: {' '.join(args)}\nSTDERR: {stderr_text.strip()}"
                )

            cmd_results.append({
                "command": args,
                "rc": rc,
                "stdout_tail": stdout_text[-400:].strip(),
                "stderr_tail": stderr_text[-400:].strip(),
            })

        return {"commands": cmd_results, "cwd": cwd}

    async def _handle_fetch_repo(self, step, run_dir: Path, exec_log_path: Path) -> Dict[str, Any]:
        cfg = step.dict(by_alias=True).get("with") or {}
        url = cfg.get("url")
        dest = cfg.get("dest")
        ref = cfg.get("ref", "HEAD")
        if not url or not dest:
            raise RASExecutionError("fetch_repo requires 'url' and 'dest'")
        dest_path = (run_dir / dest).resolve()
        if dest_path.exists():
            raise RASExecutionError(f"Destination already exists: {dest_path}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Reuse run command handler for clone/checkout
        wrapper = step.copy(deep=True)
        wrapper.with_ = {
            "commands": [
                ["git", "clone", "--filter=tree:0", url, str(dest_path)],
                ["git", "-C", str(dest_path), "checkout", ref],
            ],
            "cwd": ".",
        }
        await self._handle_run_commands(wrapper, run_dir, exec_log_path)
        return {"url": url, "dest": str(dest_path), "ref": ref}

    def _handle_collect_artifacts(self, step, run_dir: Path) -> Dict[str, Any]:
        cfg = step.dict(by_alias=True).get("with") or {}
        patterns = cfg.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            raise RASExecutionError("collect_artifacts requires 'patterns' list")
        matched: List[str] = []
        for pattern in patterns:
            for path in run_dir.glob(pattern):
                matched.append(str(path.relative_to(run_dir)))
        return {"patterns": patterns, "matched": matched}

    def _evaluate_assertions(self, items: List[Dict[str, Any]], run_dir: Path, exec_log_path: Path) -> List[Dict[str, Any]]:
        handlers = {
            "file_exists": lambda cfg: file_exists(run_dir, cfg.get("path", "*"), int(cfg.get("min_count", 1))),
            "git_diff_nonempty": lambda cfg: git_diff_nonempty(run_dir / cfg.get("repo", ".")),
            "command_executed": lambda cfg: command_executed(exec_log_path, cfg.get("id"), cfg.get("expect_rc")),
        }

        results: List[Dict[str, Any]] = []
        failures = []
        for idx, assertion in enumerate(items):
            if not isinstance(assertion, dict):
                failures.append({"index": idx, "error": "Assertion entry must be a dict"})
                continue
            a_type = assertion.get("type")
            if a_type not in handlers:
                failures.append({"index": idx, "type": a_type, "error": "Unsupported assertion type"})
                continue
            cfg = {k: v for k, v in assertion.items() if k != "type"}
            try:
                ok = handlers[a_type](cfg)
            except Exception as exc:
                results.append({"type": a_type, "config": cfg, "status": "error", "error": str(exc)})
                failures.append({"index": idx, "type": a_type, "error": str(exc)})
                continue
            results.append({"type": a_type, "config": cfg, "status": "passed" if ok else "failed"})
            if not ok:
                failures.append({"index": idx, "type": a_type, "config": cfg})

        if failures:
            raise RASExecutionError(
                "Assertions failed: " + ", ".join(
                    f"#{item['index']}:{item.get('type', 'unknown')}" for item in failures
                )
            )
        return results
