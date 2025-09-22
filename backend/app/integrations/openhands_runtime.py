"""Integration helpers for launching the OpenHands action execution server headlessly."""

from __future__ import annotations

import asyncio
import os
import secrets
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import httpx


def _ensure_openhands_on_path() -> Optional[Path]:
    integrations_dir = Path(__file__).resolve().parent
    project_root = integrations_dir.parent.parent  # backend/
    openhands_dir = project_root.parent / "OpenHands"
    if openhands_dir.exists() and str(openhands_dir) not in sys.path:
        sys.path.insert(0, str(openhands_dir))
    return openhands_dir if openhands_dir.exists() else None


_openhands_dir_for_imports = _ensure_openhands_on_path()

logger = logging.getLogger(__name__)

# Avoid importing OpenHands modules in the parent process to prevent heavy deps.
CmdRunAction = None  # type: ignore
event_to_dict = None  # type: ignore


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _lookup_username() -> str:
    try:
        return os.getlogin()
    except OSError:  # Fallback when running without a controlling terminal
        import getpass

        return getpass.getuser()


@dataclass
class OpenHandsActionResult:
    execution_result: ExecutionResult
    raw_observation: dict
    stdout: str
    stderr: str
    server_logs: str


class OpenHandsActionServerRunner:
    """Utility that spins up the OpenHands action execution server to run commands."""

    def __init__(self) -> None:
        # Determine the location of the OpenHands repository relative to this file
        integrations_dir = Path(__file__).resolve().parent
        project_root = integrations_dir.parent.parent  # backend/
        openhands_dir = _openhands_dir_for_imports or (project_root.parent / "OpenHands")

        if openhands_dir is None or not openhands_dir.exists():
            self._script_path = Path()
            self._openhands_path = Path()
            self._available = False
            return

        self._script_path = openhands_dir / "openhands" / "runtime" / "action_execution_server.py"
        self._openhands_path = openhands_dir
        self._available = self._script_path.exists()

    @property
    def is_available(self) -> bool:
        return self._available

    class Session:
        def __init__(self, runner: 'OpenHandsActionServerRunner', process, port: int, api_key: str, workspace_path: Path):
            self._runner = runner
            self._process = process
            self._port = port
            self._api_key = api_key
            self._workspace_path = workspace_path

        @property
        def is_running(self) -> bool:
            return self._process.returncode is None

        async def send_action(self, action_dict: dict, timeout: int = 300) -> 'OpenHandsActionResult':
            if not self.is_running:
                raise RuntimeError("OpenHands action server session is not running")
            # Work on a shallow copy of the action payload so we can safely rewrite paths
            action_name = action_dict.get("action")
            args = dict(action_dict.get("args", {}))

            # Remap file paths to the host workspace when the agent uses /workspace prefixes
            def _remap_path(path_value: Optional[str]) -> Optional[str]:
                if not path_value:
                    return path_value
                if os.path.isabs(path_value):
                    if path_value.startswith("/workspace"):
                        relative = path_value[len("/workspace"):].lstrip("/")
                        return str((self._workspace_path / relative).resolve())
                    return path_value
                normalized = path_value.lstrip("./")
                return str((self._workspace_path / normalized).resolve())
                return path_value

            if action_name in {"read", "write", "edit"}:
                path_value = args.get("path")
                remapped = _remap_path(path_value)
                if remapped:
                    args["path"] = remapped
                if action_name == "write" and "start" not in args:
                    args.setdefault("start", 1)
                    args.setdefault("end", -1)
                if action_name == "edit":
                    file_text = args.get("file_text")
                    if isinstance(file_text, str):
                        args["file_text"] = file_text

            if action_name == "run" and "command" in args and isinstance(args["command"], str):
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

            # Persist any rewrites made above back onto the outgoing payload
            rewritten_action = dict(action_dict)
            rewritten_action["args"] = args
            payload = {"action": rewritten_action}

            preview = {}
            for key in ("command", "path", "code", "file_text", "content"):
                if key in args:
                    value = args[key]
                    if isinstance(value, str) and len(value) > 120:
                        value = value[:117] + "..."
                    preview[key] = value
            logger.info(
                "[CodeAct] sending action=%s workspace=%s args_preview=%s",
                action_name,
                self._workspace_path,
                preview,
            )
            async with httpx.AsyncClient(timeout=timeout + 10, trust_env=False) as client:
                response = await client.post(
                    f"http://127.0.0.1:{self._port}/execute_action",
                    json=payload,
                    headers={"X-Session-API-Key": self._api_key},
                )
            response.raise_for_status()
            observation = response.json()
            execution_result = self._runner._observation_to_execution_result(observation)
            preview = observation.get("content") or observation.get("stdout") or ""
            if isinstance(preview, str) and len(preview) > 200:
                preview = preview[:197] + "..."
            logger.info(
                "[CodeAct] action=%s exit_code=%s success=%s output=%s",
                action_name,
                execution_result.exit_code,
                execution_result.success,
                preview,
            )
            return OpenHandsActionResult(
                execution_result=execution_result,
                raw_observation=observation,
                stdout=observation.get("content", ""),
                stderr="",
                server_logs="",
            )

        async def run_cmd(self, command: str, timeout: int = 300, blocking: bool = True, cwd: str | None = None) -> 'OpenHandsActionResult':
            action = {
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
                "timeout": timeout,
            }
            return await self.send_action(action, timeout=timeout)

        async def ipython_run(self, code: str, timeout: int = 300) -> 'OpenHandsActionResult':
            action = {
                "action": "run_ipython",
                "args": {
                    "code": code,
                    "thought": "",
                    "include_extra": True,
                },
                "timeout": timeout,
            }
            return await self.send_action(action, timeout=timeout)

        async def file_read(self, path: str, start: int = 0, end: int = -1, timeout: int = 60) -> 'OpenHandsActionResult':
            action = {
                "action": "read",
                "args": {
                    "path": path,
                    "start": int(start),
                    "end": int(end),
                    "thought": "",
                    "impl_source": "oh_aci",
                },
                "timeout": timeout,
            }
            return await self.send_action(action, timeout=timeout)

        async def file_write(self, path: str, content: str, timeout: int = 120) -> 'OpenHandsActionResult':
            action = {
                "action": "write",
                "args": {
                    "path": path,
                    "content": content,
                    "start": 1,
                    "end": -1,
                    "thought": "",
                },
                "timeout": timeout,
            }
            return await self.send_action(action, timeout=timeout)

        async def file_edit(self, path: str, command: str, *, file_text: str | None = None, old_str: str | None = None, new_str: str | None = None, insert_line: int | None = None, timeout: int = 120) -> 'OpenHandsActionResult':
            args: dict = {
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
            action = {"action": "edit", "args": args, "timeout": timeout}
            return await self.send_action(action, timeout=timeout)

        async def close(self) -> None:
            if self._process and self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()

    async def open_session(
        self,
        workspace_path: Path,
        enable_browser: bool = False,
    ) -> 'OpenHandsActionServerRunner.Session':
        if not self._available:
            raise RuntimeError("OpenHands runtime is not available in the current environment")

        # Ensure core third-party dependencies for the server are available in this Python
        port = _find_free_port()
        session_api_key = secrets.token_hex(16)

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self._openhands_path) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        env["SESSION_API_KEY"] = session_api_key
        env.setdefault("PYTHONUNBUFFERED", "1")

        username = _lookup_username()
        try:
            user_id = os.getuid()
        except AttributeError:
            user_id = 1000

        cmd = [
            sys.executable,
            str(self._script_path),
            str(port),
            "--working-dir",
            str(workspace_path),
            "--username",
            username,
            "--user-id",
            str(user_id),
        ]
        cmd.append("--enable-browser" if enable_browser else "--no-enable-browser")

        logger.info(
            "[CodeAct] launching action server on port %s (workspace=%s enable_browser=%s)",
            port,
            workspace_path,
            enable_browser,
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        await self._wait_until_ready(process, port, session_api_key)
        return OpenHandsActionServerRunner.Session(self, process, port, session_api_key, workspace_path)

    async def execute_python_file(
        self,
        workspace_path: Path,
        script_relative_path: str,
        timeout: int = 300,
    ) -> OpenHandsActionResult:
        if not self._available:
            raise RuntimeError("OpenHands runtime is not available in the current environment")

        session = await self.open_session(workspace_path, enable_browser=False)
        try:
            result = await session.run_cmd(f"python {script_relative_path}", timeout=timeout, blocking=True)
            return result
        finally:
            await session.close()

    async def _wait_until_ready(self, process, port: int, api_key: str, retries: int = 600, delay: float = 0.5) -> None:
        url = f"http://127.0.0.1:{port}/alive"
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            for _ in range(retries):
                if process.returncode is not None:
                    stdout, stderr = await process.communicate()
                    raise RuntimeError(
                        "OpenHands action execution server exited early with code "
                        f"{process.returncode}.\nSTDOUT:\n{stdout.decode() if stdout else ''}\nSTDERR:\n{stderr.decode() if stderr else ''}"
                    )
                try:
                    response = await client.get(url, headers={"X-Session-API-Key": api_key})
                    if response.status_code == 200:
                        try:
                            payload = response.json()
                        except ValueError:
                            payload = {}
                        status = payload.get("status") if isinstance(payload, dict) else None
                        if status == "ok":
                            logger.info("[CodeAct] action server ready on port %s", port)
                            return
                        # The server is up but still initializing – keep waiting.
                except (httpx.HTTPError, ConnectionError):
                    pass
                await asyncio.sleep(delay)
        raise RuntimeError("OpenHands action execution server failed to start in time")

    def _observation_to_execution_result(self, observation: dict):
        from ..core.openhands.code_executor import ExecutionResult
        metadata = observation.get("metadata", {}) or {}
        if not metadata and isinstance(observation.get("extras"), dict):
            extras_meta = observation["extras"].get("metadata")
            if isinstance(extras_meta, dict):
                metadata = extras_meta
        exit_code = metadata.get("exit_code", -1)
        output = observation.get("content", "")
        command_text = metadata.get("command") or metadata.get("action", "")
        working_dir = metadata.get("cwd", ".")
        env_snapshot = metadata.get("env", {}) if isinstance(metadata.get("env"), dict) else {}

        return ExecutionResult(
            success=exit_code == 0,
            exit_code=exit_code,
            stdout=output,
            stderr="",
            execution_time=0.0,
            files_created=[],
            files_modified=[],
            command=command_text,
            working_directory=working_dir,
            env=env_snapshot,
        )

    def _ensure_packages(self, packages: list[str]) -> None:
        """Ensure server's Python can import required third-party packages.

        This installs missing packages into the current interpreter (sys.executable)
        so the action_execution_server.py can import them. Idempotent and best-effort.
        """
        # No automatic installation; assume prerequisites are installed in the environment
        return

    async def execute_ipython_code(
        self,
        workspace_path: Path,
        code: str,
        timeout: int = 300,
    ) -> OpenHandsActionResult:
        if not self._available:
            raise RuntimeError("OpenHands runtime is not available in the current environment")
        session = await self.open_session(workspace_path, enable_browser=False)
        try:
            return await session.ipython_run(code, timeout=timeout)
        finally:
            await session.close()
