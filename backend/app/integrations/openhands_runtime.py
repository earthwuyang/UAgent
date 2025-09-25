"""Integration helpers for launching the OpenHands action execution server headlessly."""

from __future__ import annotations

import logging
import asyncio
import json
import shlex
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

logger = logging.getLogger(__name__)

# Avoid importing OpenHands modules in the parent process to prevent heavy deps.
CmdRunAction = None  # type: ignore
event_to_dict = None  # type: ignore


def _collect_output_text(observation: object, keys: tuple[str, ...]) -> str:
    if not isinstance(observation, dict):
        return str(observation or "").strip()
    parts: list[str] = []
    for key in keys:
        value = observation.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            if text not in parts:
                parts.append(text)
    return "\n".join(parts).strip()


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


def _detect_local_proxy(host: str = "127.0.0.1", port: int = 7890, timeout: float = 0.25) -> Optional[str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return f"http://{host}:{port}"
    except OSError:
        return None


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

        async def send_action(self, action_dict: dict, timeout: int = 300, retry_with_yes: bool = True) -> 'OpenHandsActionResult':
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
                    if action_name == "read" and os.path.isdir(remapped):
                        command = (
                            f"ls -pa {shlex.quote(str(remapped))} | head -n 200"
                        )
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
            if action_name and rewritten_action.get("action") != action_name:
                rewritten_action["action"] = action_name
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

            # Use appropriate timeout based on command type
            actual_timeout = timeout
            if action_name == "run" and "command" in args:
                cmd = args["command"]

                # Docker commands need longer timeout (especially docker stop which waits 10s by default)
                docker_cmds = ["docker stop", "docker-compose down", "docker-compose stop", "docker kill", "docker rm"]
                for docker_cmd in docker_cmds:
                    if docker_cmd in cmd:
                        # Docker stop waits 10 seconds by default, so we need at least 15 seconds
                        actual_timeout = max(20, timeout // 3)  # At least 20 seconds for docker commands
                        logger.info(f"[CodeAct] Using docker timeout ({actual_timeout}s) for docker command")
                        break
                else:
                    # Check if it's a package manager command
                    package_cmds = ["apt-get", "apt ", "yum ", "dnf ", "zypper", "pacman", "emerge",
                                   "conda install", "conda update", "npm install", "yarn add", "brew install"]
                    for pkg_cmd in package_cmds:
                        if pkg_cmd in cmd:
                            # Use shorter timeout (30 seconds) to detect hanging quickly
                            # unless it already has non-interactive flags
                            if not any(flag in cmd for flag in ["-y", "--yes", "--no-input", "--noconfirm", "--non-interactive"]):
                                actual_timeout = min(30, timeout)
                                logger.info(f"[CodeAct] Using short timeout ({actual_timeout}s) for potentially interactive command")
                            break

            # Try the action with timeout handling
            try:
                response = await self._post_action(payload, actual_timeout)
            except (httpx.TimeoutException, asyncio.TimeoutError) as timeout_exc:
                # Check if this is a command that might be waiting for user input
                if retry_with_yes and action_name == "run" and "command" in args:
                    cmd = args["command"]
                    # Check if it's a package manager command that might need -y
                    package_managers = [
                        ("apt-get", "-y"),
                        ("apt", "-y"),
                        ("yum", "-y"),
                        ("dnf", "-y"),
                        ("zypper", "--non-interactive"),
                        ("pacman", "--noconfirm"),
                        ("emerge", "--ask n"),
                        ("conda install", "-y"),
                        ("conda update", "-y"),
                        ("conda upgrade", "-y"),
                        ("pip install", "--no-input"),
                        ("pip3 install", "--no-input"),
                        ("npm install", "--yes"),
                        ("yarn add", "--yes"),
                        ("brew install", "--yes"),
                    ]

                    for pm, flag in package_managers:
                        if pm in cmd and flag not in cmd:
                            # Add the non-interactive flag
                            logger.warning(
                                f"[CodeAct] Command timeout detected for {pm}. Retrying with {flag} flag"
                            )

                            # Special handling for different package managers
                            if "pip" in pm or "conda" in pm or "npm" in pm or "yarn" in pm or "brew" in pm:
                                # These already have the full command pattern in pm
                                cmd = cmd.replace(pm, f"{pm} {flag}")
                            else:
                                # Traditional package managers (apt, yum, etc.)
                                # Modify the command to add the flag
                                if f"{pm} install" in cmd:
                                    cmd = cmd.replace(f"{pm} install", f"{pm} install {flag}")
                                elif f"{pm} upgrade" in cmd:
                                    cmd = cmd.replace(f"{pm} upgrade", f"{pm} upgrade {flag}")
                                elif f"{pm} update" in cmd:
                                    cmd = cmd.replace(f"{pm} update", f"{pm} update {flag}")
                                elif f"{pm} remove" in cmd:
                                    cmd = cmd.replace(f"{pm} remove", f"{pm} remove {flag}")
                                else:
                                    # Generic insertion after the package manager name
                                    cmd = cmd.replace(pm, f"{pm} {flag}", 1)

                            # Update the args and payload
                            args["command"] = cmd
                            rewritten_action["args"] = args
                            payload = {"action": rewritten_action}

                            logger.info(
                                "[CodeAct] Retrying with non-interactive command: %s",
                                cmd[:200]
                            )

                            # Retry with the modified command (use original timeout for retry)
                            try:
                                response = await self._post_action(payload, timeout)
                            except (httpx.TimeoutException, asyncio.TimeoutError):
                                logger.error(
                                    "[CodeAct] Command still timed out after adding %s flag",
                                    flag
                                )
                                # Return timeout error instead of raising
                                timeout_retry_msg = (
                                    f"ERROR: Command still timed out after {timeout} seconds even with {flag} flag.\n"
                                    f"The command still appears to be hanging.\n"
                                    f"This might be a long-running process that needs more time.\n"
                                    f"Consider:\n"
                                    f"1. Running in background: nohup {cmd[:100]} &\n"
                                    f"2. Breaking into smaller steps\n"
                                    f"3. Checking if the command is correct"
                                )
                                from ..core.openhands.code_executor import ExecutionResult
                                return OpenHandsActionResult(
                                    execution_result=ExecutionResult(
                                        success=False,
                                        exit_code=124,
                                        stdout=timeout_retry_msg,
                                        stderr=f"Command timed out even with {flag} flag after {timeout} seconds",
                                        execution_time=float(timeout),
                                        files_created=[],
                                        files_modified=[],
                                        command=cmd[:200],
                                        working_directory=str(self._workspace_path),
                                        env={},
                                    ),
                                    raw_observation={
                                        "error": "timeout",
                                        "timeout_seconds": timeout,
                                        "retry_attempted": True,
                                        "content": timeout_retry_msg
                                    },
                                    stdout=timeout_retry_msg,
                                    stderr=f"Command timed out even with {flag} flag after {timeout} seconds",
                                    server_logs=f"Timeout error: Command did not complete within {timeout} seconds even after adding {flag}",
                                )
                            break
                    else:
                        # Not a package manager command or already has flags
                        logger.error(
                            "[CodeAct] Command timeout: %s",
                            cmd[:200] if cmd else "unknown"
                        )
                        # Return timeout error result with clear error message
                        cmd_display = cmd[:100] if len(cmd) > 100 else cmd

                        # Special message for docker commands
                        if "docker stop" in cmd:
                            timeout_msg = (
                                f"ERROR: Docker stop command timed out after {actual_timeout} seconds.\n"
                                f"Docker stop waits 10 seconds by default for graceful shutdown.\n"
                                f"To stop immediately, use: docker stop -t 0 <container_name>\n"
                                f"Or use: docker kill <container_name> for immediate termination.\n"
                                f"The original command was: {cmd_display}"
                            )
                        elif "docker" in cmd:
                            timeout_msg = (
                                f"ERROR: Docker command timed out after {actual_timeout} seconds.\n"
                                f"The command '{cmd_display}' is taking too long.\n"
                                f"Possible issues:\n"
                                f"1. Container is not responding\n"
                                f"2. Docker daemon is busy\n"
                                f"3. Network issues pulling images\n"
                                f"Try checking docker status: docker ps -a"
                            )
                        else:
                            timeout_msg = (
                                f"ERROR: Command timed out after {actual_timeout} seconds.\n"
                                f"The command '{cmd_display}' appears to be hanging.\n"
                                f"Possible issues:\n"
                                f"1. The script is waiting for input\n"
                                f"2. The script has an infinite loop\n"
                                f"3. The script is taking too long to process data\n"
                                f"Try running with timeout command: timeout 20 {cmd_display}\n"
                                f"Or check if the script exists and has proper permissions."
                            )
                        from ..core.openhands.code_executor import ExecutionResult
                        return OpenHandsActionResult(
                            execution_result=ExecutionResult(
                                success=False,
                                exit_code=124,  # Standard timeout exit code
                                stdout=timeout_msg,  # Put message in stdout so it's visible
                                stderr=f"Command timed out after {actual_timeout} seconds",
                                execution_time=float(actual_timeout),
                                files_created=[],
                                files_modified=[],
                                command=cmd[:200] if cmd else "unknown",
                                working_directory=str(self._workspace_path),
                                env={},
                            ),
                            raw_observation={
                                "error": "timeout",
                                "timeout_seconds": actual_timeout,
                                "content": timeout_msg  # Also in content for visibility
                            },
                            stdout=timeout_msg,
                            stderr=f"Command timed out after {actual_timeout} seconds",
                            server_logs=f"Timeout error: Command did not complete within {actual_timeout} seconds",
                        )
                else:
                    # Return timeout error for non-command actions
                    from ..core.openhands.code_executor import ExecutionResult
                    return OpenHandsActionResult(
                        execution_result=ExecutionResult(
                            success=False,
                            exit_code=-1,
                            stdout="",
                            stderr=f"Action {action_name} timed out after {actual_timeout} seconds",
                            execution_time=float(actual_timeout),
                            files_created=[],
                            files_modified=[],
                            command="",
                            working_directory=str(self._workspace_path),
                            env={},
                        ),
                        raw_observation={"error": "timeout", "timeout_seconds": actual_timeout},
                        stdout="",
                        stderr=f"Action {action_name} timed out after {actual_timeout} seconds",
                        server_logs=f"Timeout error: Action did not complete within {actual_timeout} seconds",
                    )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                observation: dict[str, object]
                error_detail: str = ""
                try:
                    parsed = exc.response.json()
                    if isinstance(parsed, dict):
                        observation = dict(parsed)
                    else:
                        observation = {"content": str(parsed)}
                except ValueError:
                    error_detail = exc.response.text
                    observation = {"content": error_detail or exc.message}

                metadata = observation.get("metadata") if isinstance(observation, dict) else None
                if not isinstance(metadata, dict):
                    metadata = {}
                    observation["metadata"] = metadata
                metadata.setdefault("action", action_name)
                metadata.setdefault("exit_code", observation.get("exit_code", -1))
                stdout_text = _collect_output_text(observation, ("content", "stdout", "detail"))
                stderr_text = _collect_output_text(observation, ("stderr",)) if isinstance(observation, dict) else ""
                exec_result = self._runner._observation_to_execution_result(observation)
                logger.error(
                    "[CodeAct] action=%s server error status=%s detail=%s",
                    action_name,
                    exc.response.status_code if exc.response is not None else "unknown",
                    stdout_text or stderr_text or error_detail or str(exc),
                )
                return OpenHandsActionResult(
                    execution_result=exec_result,
                    raw_observation=observation,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    server_logs=stdout_text or stderr_text or error_detail or str(exc),
                )

            observation = response.json()
            stdout_text = _collect_output_text(observation, ("content", "stdout"))
            stderr_text = _collect_output_text(observation, ("stderr",)) if isinstance(observation, dict) else ""
            execution_result = self._runner._observation_to_execution_result(observation)
            preview = stdout_text or stderr_text
            if preview and len(preview) > 200:
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
                stdout=stdout_text,
                stderr=stderr_text,
                server_logs="",
            )

        async def _post_action(self, payload: dict, timeout: int) -> httpx.Response:
            async with httpx.AsyncClient(timeout=timeout + 10, trust_env=False) as client:
                return await client.post(
                    f"http://127.0.0.1:{self._port}/execute_action",
                    json=payload,
                    headers={"X-Session-API-Key": self._api_key},
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

        proxy_url = _detect_local_proxy()
        if proxy_url:
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
                env.setdefault(key, proxy_url)
            env.setdefault("GIT_HTTP_PROXY", proxy_url)
            env.setdefault("GIT_HTTPS_PROXY", proxy_url)

        workspace_mount = f"{workspace_path}:/workspace:rw"
        sandbox_volumes = env.get("SANDBOX_VOLUMES")
        if sandbox_volumes:
            if workspace_mount not in sandbox_volumes.split(","):
                env["SANDBOX_VOLUMES"] = f"{workspace_mount},{sandbox_volumes}"
        else:
            env["SANDBOX_VOLUMES"] = workspace_mount
        try:
            env["SANDBOX_USER_ID"] = str(os.getuid())
        except AttributeError:
            env.setdefault("SANDBOX_USER_ID", "1000")

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
        output = stdout_text
        command_text = metadata.get("command") or metadata.get("action", "")
        working_dir = metadata.get("cwd", ".")
        env_snapshot = metadata.get("env", {}) if isinstance(metadata.get("env"), dict) else {}

        # Directory listings (and some file reads) return exit_code=-1 even when successful.
        if exit_code != 0:
            if success_flag is True:
                exit_code = 0
            elif isinstance(output, str) and output.strip() and action_name in {"list", "read"}:
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
