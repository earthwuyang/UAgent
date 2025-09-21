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

# Lazy imports of OpenHands modules – attempt after ensuring PYTHONPATH contains repository.
try:
    from openhands.events.action.commands import CmdRunAction
    from openhands.events.serialization import event_to_dict
except ModuleNotFoundError:  # pragma: no cover - OpenHands may not be installed
    CmdRunAction = None  # type: ignore
    event_to_dict = None  # type: ignore

from ..core.openhands.code_executor import ExecutionResult


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
        self._available = self._script_path.exists() and CmdRunAction is not None and event_to_dict is not None

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

        async def send_action(self, action, timeout: int = 300) -> 'OpenHandsActionResult':
            if not self.is_running:
                raise RuntimeError("OpenHands action server session is not running")
            payload = {"action": event_to_dict(action)}
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

        await self._wait_until_ready(port, session_api_key)
        return OpenHandsActionServerRunner.Session(self, process, port, session_api_key, workspace_path)

    async def execute_python_file(
        self,
        workspace_path: Path,
        script_relative_path: str,
        timeout: int = 300,
    ) -> OpenHandsActionResult:
        if not self._available:
            raise RuntimeError("OpenHands runtime is not available in the current environment")

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
        except AttributeError:  # Windows
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
            "False",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            await self._wait_until_ready(port, session_api_key)

            action = CmdRunAction(command=f"python {script_relative_path}")
            action.set_hard_timeout(timeout, blocking=True)
            payload = {"action": event_to_dict(action)}

            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}/execute_action",
                    json=payload,
                    headers={"X-Session-API-Key": session_api_key},
                )
            response.raise_for_status()
            observation = response.json()
            execution_result = self._observation_to_execution_result(observation)

            stdout_data, stderr_data = await process.communicate()
            server_logs = "".join(filter(None, [stdout_data.decode(), stderr_data.decode()]))

            return OpenHandsActionResult(
                execution_result=execution_result,
                raw_observation=observation,
                stdout=observation.get("content", ""),
                stderr="",
                server_logs=server_logs.strip(),
            )
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

    async def _wait_until_ready(self, port: int, api_key: str, retries: int = 50, delay: float = 0.2) -> None:
        url = f"http://127.0.0.1:{port}/alive"
        async with httpx.AsyncClient(timeout=2.0) as client:
            for _ in range(retries):
                try:
                    response = await client.get(url, headers={"X-Session-API-Key": api_key})
                    if response.status_code == 200:
                        return
                except (httpx.HTTPError, ConnectionError):
                    pass
                await asyncio.sleep(delay)
        raise RuntimeError("OpenHands action execution server failed to start in time")

    def _observation_to_execution_result(self, observation: dict) -> ExecutionResult:
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

    async def execute_ipython_code(
        self,
        workspace_path: Path,
        code: str,
        timeout: int = 300,
    ) -> OpenHandsActionResult:
        if not self._available:
            raise RuntimeError("OpenHands runtime is not available in the current environment")

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
            "False",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            await self._wait_until_ready(port, session_api_key)

            from openhands.events.action.commands import IPythonRunCellAction  # lazy import

            action = IPythonRunCellAction(code=code)
            action.set_hard_timeout(timeout, blocking=True)
            payload = {"action": event_to_dict(action)}

            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}/execute_action",
                    json=payload,
                    headers={"X-Session-API-Key": session_api_key},
                )
            response.raise_for_status()
            observation = response.json()
            execution_result = self._observation_to_execution_result(observation)

            stdout_data, stderr_data = await process.communicate()
            server_logs = "".join(filter(None, [stdout_data.decode(), stderr_data.decode()]))

            return OpenHandsActionResult(
                execution_result=execution_result,
                raw_observation=observation,
                stdout=observation.get("content", ""),
                stderr="",
                server_logs=server_logs.strip(),
            )
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
