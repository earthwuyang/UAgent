"""In-process OpenHands runtime runner for UAgent.

This integration embeds the OpenHands action execution runtime directly in the
current Python process. It mirrors the behaviour of the previous
``OpenHandsActionServerRunner`` but avoids spawning the HTTP bridge, allowing
UAgent to call OpenHands via direct imports.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from .openhands_runtime import (
    DEFAULT_ACTION_TIMEOUT,
    OPENHANDS_MAX_ACTION_TIMEOUT,
    OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT,
    OPENHANDS_RUN_ADAPTIVE_MULTIPLIER,
    OPENHANDS_RUN_MAX_ATTEMPTS,
    OpenHandsActionResult,
    _collect_output_text,
    _lookup_username,
)
from .openhands_runtime import _ensure_openhands_on_path as _ensure_oh  # ensure sys.path contains OpenHands
from .stream_monitor import CommandStreamMonitor
from .openhands.types import ActionError, ActionResult


logger = logging.getLogger(__name__)

_IMPORT_ERROR: Optional[Exception]

try:  # pragma: no cover - import availability depends on environment
    # Ensure OpenHands repository is importable (e.g., repo/OpenHands)
    _ensure_oh()
    from openhands.events.serialization import event_from_dict, event_to_dict
    from openhands.runtime.action_execution_server import ActionExecutor
    from openhands.runtime.plugins import ALL_PLUGINS
except Exception as exc:  # pragma: no cover - best-effort fallback
    ActionExecutor = None  # type: ignore[assignment]
    ALL_PLUGINS = {}  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:  # pragma: no cover - only executed when imports succeed
    _IMPORT_ERROR = None


# In-process job service removed; use blocking run_cmd and async wrappers


def _observation_to_execution_result(observation: Dict[str, Any]) -> ExecutionResult:
    from ..core.openhands.code_executor import ExecutionResult

    metadata = observation.get("metadata", {}) or {}
    if not metadata and isinstance(observation.get("extras"), dict):
        extras_meta = observation["extras"].get("metadata")
        if isinstance(extras_meta, dict):
            metadata = extras_meta

    action_name = (
        metadata.get("action")
        or observation.get("action")
        or metadata.get("command")
        or metadata.get("tool_name")
        or observation.get("tool_name")
        or ""
    )

    exit_code = metadata.get("exit_code")
    if exit_code is None:
        exit_code = observation.get("exit_code")
    if exit_code is None:
        exit_code = -1

    success_flag = observation.get("success")
    stdout_text = _collect_output_text(observation, ("content", "stdout"))
    stderr_text = _collect_output_text(observation, ("stderr",)) if isinstance(observation, dict) else ""
    command_text = metadata.get("command") or metadata.get("action", "")
    working_dir = metadata.get("cwd", ".")
    env_snapshot = metadata.get("env", {}) if isinstance(metadata.get("env"), dict) else {}

    # Check for success message in observation
    message = observation.get("message", "")
    observation_type = observation.get("observation", "")

    if exit_code != 0:
        if success_flag is True:
            exit_code = 0
        elif message.strip() and any(phrase in message.lower() for phrase in [
            "wrote to the file", "read the file", "file created successfully", "has been edited",
            "file edited successfully", "created at:", "saved to"
        ]):
            # Success message indicates operation completed
            exit_code = 0
        elif stdout_text.strip():
            cmd_name = (metadata.get("command") or observation.get("command") or "").lower()
            act = (action_name or "").lower()
            if act in {"list", "read", "view"} or cmd_name in {"list", "read", "view"}:
                exit_code = 0
            elif act in {"edit", "write"} or observation_type in {"edit", "write"}:
                low = stdout_text.lower()
                if any(
                    phrase in low
                    for phrase in (
                        "file created successfully",
                        "has been edited",
                        "file edited successfully",
                        "created at:",
                    )
                ):
                    exit_code = 0
        elif observation_type in {"read", "write", "edit"} and not stderr_text.strip():
            # File operations without stderr are usually successful
            exit_code = 0

    return ExecutionResult(
        success=exit_code == 0,
        exit_code=exit_code,
        stdout=stdout_text,
        stderr=stderr_text,
        execution_time=0.0,
        files_created=[],
        files_modified=[],
        command=command_text,
        working_directory=working_dir,
        env=env_snapshot,
    )


class OpenHandsInProcessRunner:
    """Direct OpenHands runtime runner."""

    def __init__(self) -> None:
        self._available = ActionExecutor is not None
        self._import_error = _IMPORT_ERROR

    @property
    def is_available(self) -> bool:
        return self._available

    async def open_session(self, workspace_path: Path, enable_browser: bool = False) -> "OpenHandsInProcessRunner.Session":
        if not self._available:
            raise RuntimeError(
                "OpenHands in-process runtime is unavailable"
                + (f": {_IMPORT_ERROR}" if _IMPORT_ERROR else "")
            )
        session = OpenHandsInProcessRunner.Session(workspace_path, enable_browser)
        await session.start()
        return session

    async def execute_python_file(
        self,
        workspace_path: Path,
        script_relative_path: str,
        timeout: int = 300,
    ) -> OpenHandsActionResult:
        session = await self.open_session(workspace_path, enable_browser=False)
        try:
            command = f"python {script_relative_path}"
            return await session.run_cmd(command, timeout=timeout, blocking=True)
        finally:
            await session.close()

    class Session:
        def __init__(self, workspace_path: Path, enable_browser: bool) -> None:
            self._workspace_path = Path(workspace_path).resolve()
            self._enable_browser = enable_browser
            self._executor: Optional[ActionExecutor] = None
            self._closed = False
            self._stream_monitor = CommandStreamMonitor(self._workspace_path)
            self._current_log_path: Optional[Path] = None
            self._lock = asyncio.Lock()
            self._live_log_dir = self._workspace_path / "logs" / "openhands_inprocess"
            self._live_stdout_path: Optional[Path] = None
            self._live_stderr_path: Optional[Path] = None
            self._live_combined_path: Optional[Path] = None

        async def start(self) -> None:
            if self._executor is not None:
                return
            username = _lookup_username()
            try:
                user_id = os.getuid()
            except AttributeError:  # pragma: no cover - Windows fallback
                user_id = 1000

            plugin_specs = os.getenv("OPENHANDS_RUNTIME_PLUGINS", "").strip()
            plugins = []
            if plugin_specs:
                for name in [p.strip() for p in plugin_specs.split(",") if p.strip()]:
                    plugin_cls = ALL_PLUGINS.get(name)
                    if plugin_cls:
                        plugins.append(plugin_cls())  # type: ignore[misc]

            executor = ActionExecutor(
                plugins,
                work_dir=str(self._workspace_path),
                username=username,
                user_id=user_id,
                enable_browser=self._enable_browser,
                browsergym_eval_env=os.getenv("OPENHANDS_BROWSERGYM_ENV"),
            )
            await executor.ainit()
            self._executor = executor

            # Initialize live streaming logs similar to single-container bridge
            try:
                self._live_log_dir.mkdir(parents=True, exist_ok=True)
                self._live_stdout_path = self._live_log_dir / "live_stdout.log"
                self._live_stderr_path = self._live_log_dir / "live_stderr.log"
                self._live_combined_path = self._live_log_dir / "live_combined.log"
                for path in (self._live_stdout_path, self._live_stderr_path, self._live_combined_path):
                    path.write_text("", encoding="utf-8")
            except Exception as exc:
                self._live_stdout_path = None
                self._live_stderr_path = None
                self._live_combined_path = None
                logger.warning("Failed to initialize in-process live logs: %s", exc)

        @property
        def is_running(self) -> bool:
            return self._executor is not None and not self._closed

        @property
        def workspace_path(self) -> Path:
            return self._workspace_path

        async def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            if self._executor is not None:
                self._executor.close()

        async def send_action(
            self,
            action_dict: Dict[str, Any],
            timeout: int = DEFAULT_ACTION_TIMEOUT,
            retry_with_yes: bool = True,
        ) -> OpenHandsActionResult:
            if not self.is_running:
                raise RuntimeError("OpenHands in-process session is not running")

            action_name = action_dict.get("action")
            args = dict(action_dict.get("args", {}))

            def _remap_path(path_value: Optional[str]) -> Optional[str]:
                if not path_value:
                    return path_value
                candidate = Path(path_value)
                if candidate.is_absolute():
                    if str(candidate).startswith("/workspace"):
                        relative = str(candidate).replace("/workspace", "", 1).lstrip("/")
                        return str((self._workspace_path / relative).resolve())
                    return str(candidate)
                return str((self._workspace_path / candidate).resolve())

            if action_name in {"read", "write", "edit", "str_replace_editor"}:
                path_value = args.get("path")
                remapped = _remap_path(path_value)
                if remapped:
                    args["path"] = remapped
                    if action_name == "read" and os.path.isdir(remapped):
                        command = f"ls -pa {shlex.quote(str(remapped))} | head -n 200"
                        args = {
                            "command": command,
                            "is_input": False,
                            "thought": "",
                            "blocking": True,
                            "is_static": False,
                            "cwd": None,
                            "hidden": False,
                        }
                        action_name = "run"
                Path(args.get("path", self._workspace_path)).parent.mkdir(parents=True, exist_ok=True)
                if action_name == "write" and "start" not in args:
                    args.setdefault("start", 1)
                    args.setdefault("end", -1)

            if action_name == "run" and isinstance(args.get("command"), str):
                command_text = args["command"]
                if "/workspace" in command_text:
                    command_text = command_text.replace(
                        "/workspace/",
                        str(self._workspace_path.resolve()) + "/",
                    )
                stripped = command_text.strip()
                if "\n" in command_text or stripped.startswith("#") or stripped.startswith("{") or stripped.startswith("["):
                    command_text = "bash -lc " + json.dumps(stripped)
                args["command"] = command_text

            rewritten = dict(action_dict)
            if action_name:
                rewritten["action"] = action_name
            rewritten["args"] = args

            command_str = args.get("command") if action_name == "run" else None
            log_path = self._stream_monitor.create_log_file(action_name or "unknown", command_str)
            self._stream_monitor.write_to_log(log_path, json.dumps(args, indent=2, default=str), "CONTEXT")
            self._current_log_path = log_path
            self._append_live_log("combined", f"ACTION_START action={action_name or 'unknown'} args={json.dumps(args, default=str)}")
            if command_str:
                self._append_live_log("stdout", f"$ {command_str}\n")

            actual_timeout = timeout
            if action_name == "run" and isinstance(args.get("command"), str):
                cmd = args["command"]
                docker_cmds = ["docker stop", "docker-compose down", "docker-compose stop", "docker kill", "docker rm"]
                if any(dc in cmd for dc in docker_cmds):
                    actual_timeout = max(20, timeout // 3)
                else:
                    for pkg_cmd in [
                        "apt-get",
                        "apt ",
                        "yum ",
                        "dnf ",
                        "zypper",
                        "pacman",
                        "emerge",
                        "conda install",
                        "conda update",
                        "npm install",
                        "yarn add",
                        "brew install",
                    ]:
                        if pkg_cmd in cmd:
                            min_timeout = OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT
                            actual_timeout = max(timeout, min_timeout)
                            break

            attempt = 0
            command_to_run = rewritten
            while True:
                try:
                    result = await self._execute_action(command_to_run, actual_timeout, log_path)
                    exec_result = result.execution_result
                    self._append_live_log(
                        "combined",
                        f"ACTION_RESULT action={action_name or 'unknown'} exit_code={exec_result.exit_code} success={exec_result.success} duration={exec_result.execution_time:.2f}s",
                    )
                    return result
                except asyncio.TimeoutError as exc:
                    if not (retry_with_yes and action_name == "run" and isinstance(args.get("command"), str)):
                        self._append_live_log(
                            "stderr",
                            f"TIMEOUT after {actual_timeout}s while executing action {action_name or 'unknown'}\n",
                        )
                        raise exc

                    cmd = args["command"]
                    replacements = [
                        ("apt-get", "-y"),
                        ("apt ", "-y"),
                        ("yum", "-y"),
                        ("dnf", "-y"),
                        ("zypper", "--non-interactive"),
                        ("pacman", "--noconfirm"),
                        ("emerge", "--ask n"),
                        ("conda install", "-y"),
                        ("conda update", "-y"),
                        ("conda upgrade", "-y"),
                        ("npm install", "--yes"),
                        ("yarn add", "--non-interactive"),
                        ("brew install", "-q"),
                    ]
                    applied = False
                    for marker, flag in replacements:
                        if marker in cmd and flag not in cmd:
                            args["command"] = f"{cmd} {flag}".strip()
                            command_to_run = dict(rewritten)
                            command_to_run["args"] = args
                            applied = True
                            actual_timeout = min(int(actual_timeout * OPENHANDS_RUN_ADAPTIVE_MULTIPLIER), OPENHANDS_MAX_ACTION_TIMEOUT)
                            attempt += 1
                            if attempt > OPENHANDS_RUN_MAX_ATTEMPTS:
                                raise exc
                            break
                    if not applied:
                        raise exc

        async def _execute_action(
            self,
            action_payload: Dict[str, Any],
            timeout: int,
            log_path: Path,
        ) -> OpenHandsActionResult:
            assert self._executor is not None
            action = event_from_dict(action_payload)
            blocking = bool(action_payload.get("args", {}).get("blocking", True))
            if hasattr(action, "set_hard_timeout"):
                action.set_hard_timeout(timeout, blocking=blocking)

            start = time.monotonic()
            observation = await asyncio.wait_for(self._executor.run_action(action), timeout=timeout + 5)
            elapsed = time.monotonic() - start
            observation_dict = event_to_dict(observation)
            exec_result = _observation_to_execution_result(observation_dict)
            exec_result.execution_time = elapsed

            stdout_text = _collect_output_text(observation_dict, ("content", "stdout"))
            stderr_text = _collect_output_text(observation_dict, ("stderr",))

            if observation_dict:
                if stdout_text:
                    self._stream_monitor.write_to_log(log_path, stdout_text, "STDOUT")
                if stderr_text:
                    self._stream_monitor.write_to_log(log_path, stderr_text, "STDERR")
                self._stream_monitor.write_json_to_log(log_path, observation_dict, "FULL_RESPONSE")

            if stdout_text:
                self._append_live_log("stdout", stdout_text)
            if stderr_text:
                self._append_live_log("stderr", stderr_text)
            combined_payload = []
            if stdout_text:
                combined_payload.append("STDOUT:\n" + stdout_text.rstrip())
            if stderr_text:
                combined_payload.append("STDERR:\n" + stderr_text.rstrip())
            if combined_payload:
                self._append_live_log("combined", "\n".join(combined_payload))

            return OpenHandsActionResult(
                execution_result=exec_result,
                raw_observation=observation_dict,
                stdout=stdout_text,
                stderr=stderr_text,
                server_logs="",
            )

        def _append_live_log(self, stream: str, content: str) -> None:
            if not content:
                return
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{timestamp}] {content.rstrip()}\n"
            try:
                if stream == "stdout" and self._live_stdout_path is not None:
                    with self._live_stdout_path.open("a", encoding="utf-8") as fh:
                        fh.write(entry)
                if stream == "stderr" and self._live_stderr_path is not None:
                    with self._live_stderr_path.open("a", encoding="utf-8") as fh:
                        fh.write(entry)
                if self._live_combined_path is not None:
                    with self._live_combined_path.open("a", encoding="utf-8") as fh:
                        fh.write(entry)
            except Exception as exc:
                logger.warning("Failed to append live %s log: %s", stream, exc)

        async def run_cmd(self, command: str, timeout: int = DEFAULT_ACTION_TIMEOUT, blocking: bool = True, cwd: Optional[str] = None) -> OpenHandsActionResult:
            payload = {
                "action": "run",
                "args": {
                    "command": command,
                    "is_input": False,
                    "thought": "",
                    "blocking": bool(blocking),
                    "is_static": False,
                    "cwd": cwd,
                    "hidden": False,
                },
            }
            return await self.send_action(payload, timeout=timeout)

        async def ipython_run(self, code: str, timeout: int = DEFAULT_ACTION_TIMEOUT) -> OpenHandsActionResult:
            payload = {
                "action": "run_ipython",
                "args": {"code": code, "thought": "", "include_extra": True},
            }
            return await self.send_action(payload, timeout=timeout)

        async def file_read(self, path: str, start: int = 0, end: int = -1, timeout: int = DEFAULT_ACTION_TIMEOUT) -> OpenHandsActionResult:
            payload = {
                "action": "read",
                "args": {
                    "path": path,
                    "start": int(start),
                    "end": int(end),
                    "thought": "",
                    "impl_source": "oh_aci",
                },
            }
            return await self.send_action(payload, timeout=timeout)

        async def file_write(self, path: str, content: str, timeout: int = DEFAULT_ACTION_TIMEOUT) -> OpenHandsActionResult:
            payload = {
                "action": "write",
                "args": {
                    "path": path,
                    "content": content,
                    "start": 1,
                    "end": -1,
                    "thought": "",
                },
            }
            return await self.send_action(payload, timeout=timeout)

        async def file_edit(
            self,
            path: str,
            command: str,
            *,
            file_text: Optional[str] = None,
            old_str: Optional[str] = None,
            new_str: Optional[str] = None,
            insert_line: Optional[int] = None,
            timeout: int = DEFAULT_ACTION_TIMEOUT,
        ) -> OpenHandsActionResult:
            args: Dict[str, Any] = {
                "path": path,
                "command": command,
                "impl_source": "oh_aci",
            }
            if file_text is not None:
                args["file_text"] = file_text
            if old_str is not None:
                args["old_str"] = old_str
            if new_str is not None:
                args["new_str"] = new_str
            if insert_line is not None:
                args["insert_line"] = int(insert_line)

            payload = {"action": "edit", "args": args}
            return await self.send_action(payload, timeout=timeout)

        # jobs_* methods removed; callers should use run_cmd with appropriate timeouts


__all__ = ["OpenHandsInProcessRunner"]
