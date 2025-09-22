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

import httpx


def _ensure_openhands_on_path() -> Optional[Path]:
    integrations_dir = Path(__file__).resolve().parent
    project_root = integrations_dir.parent.parent  # backend/
    openhands_dir = project_root.parent / "OpenHands"
    if openhands_dir.exists() and str(openhands_dir) not in sys.path:
        sys.path.insert(0, str(openhands_dir))
    return openhands_dir if openhands_dir.exists() else None


_openhands_dir_for_imports = _ensure_openhands_on_path()

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
            payload = {"action": action_dict}
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                response = await client.post(
                    f"http://127.0.0.1:{self._port}/execute_action",
                    json=payload,
                    headers={"X-Session-API-Key": self._api_key},
                )
            response.raise_for_status()
            observation = response.json()
            execution_result = self._runner._observation_to_execution_result(observation)
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
        self._ensure_packages([
            "puremagic",
            "binaryornot",
            "fastapi",
            "uvicorn",
            "starlette",
            "pydantic",
            "openhands-aci",
            "litellm",
        ])

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
            "--enable-browser",
            str(bool(enable_browser)),
        ]

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

    async def _wait_until_ready(self, process, port: int, api_key: str, retries: int = 50, delay: float = 0.2) -> None:
        url = f"http://127.0.0.1:{port}/alive"
        async with httpx.AsyncClient(timeout=2.0) as client:
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
                        return
                except (httpx.HTTPError, ConnectionError):
                    pass
                await asyncio.sleep(delay)
        raise RuntimeError("OpenHands action execution server failed to start in time")

    def _observation_to_execution_result(self, observation: dict):
        from ..core.openhands.code_executor import ExecutionResult
        metadata = observation.get("metadata", {}) or {}
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
        for pkg in packages:
            mod = pkg.replace('-', '_')
            try:
                __import__(mod)
                continue
            except Exception:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
                except Exception:
                    # Non-fatal; the server may still run if this import isn't needed in current path
                    pass

        # Ensure the OpenHands repo itself is installed (captures remaining deps)
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(self._openhands_path)], check=True)
        except Exception:
            pass

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
