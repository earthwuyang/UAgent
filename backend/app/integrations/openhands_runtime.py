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
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from .stream_monitor import CommandStreamMonitor

# Read default timeout from environment variable (default: 120 seconds = 2 minutes)
DEFAULT_ACTION_TIMEOUT = int(os.getenv("OPENHANDS_ACTION_TIMEOUT", "120"))
OPENHANDS_MAX_ACTION_TIMEOUT = int(os.getenv("OPENHANDS_ACTION_MAX_TIMEOUT", "900"))
OPENHANDS_RUN_ADAPTIVE_MULTIPLIER = float(os.getenv("OPENHANDS_RUN_TIMEOUT_MULTIPLIER", "1.75"))
OPENHANDS_RUN_MAX_ATTEMPTS = int(os.getenv("OPENHANDS_RUN_MAX_ATTEMPTS", "3"))
OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT = int(os.getenv("OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT", "600"))


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
            self._stream_monitor = CommandStreamMonitor(workspace_path)

        @property
        def is_running(self) -> bool:
            return self._process.returncode is None

        async def send_action(self, action_dict: dict, timeout: int = DEFAULT_ACTION_TIMEOUT, retry_with_yes: bool = True) -> 'OpenHandsActionResult':
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

                # Set long-running commands to non-blocking to prevent timeouts on other operations
                long_running_patterns = [
                    "pip install", "pip3 install", "npm install", "yarn install",
                    "apt-get", "apt install", "yum install", "brew install",
                    "docker build", "docker pull", "make", "cmake",
                    "wget", "curl -O", "git clone", "sleep"
                ]

                # Check if this is a long-running command
                is_long_running = any(pattern in command_text for pattern in long_running_patterns)

                # Override blocking setting for long-running commands
                if is_long_running and args.get("blocking", True):
                    logger.info(f"[CodeAct] Setting long-running command to non-blocking: {command_text[:50]}...")
                    args["blocking"] = False
                    # Add a follow-up check command
                    args["thought"] = "Running in background to prevent blocking other operations"

                    # Wrap command for background execution with output streaming
                    # Create a realtime log path based on the workspace
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_cmd = command_text[:30].replace("/", "_").replace(" ", "_")
                    log_realtime = f"/workspace/logs/commands/{timestamp}_{safe_cmd}.realtime"

                    # Ensure the logs directory exists
                    wrapped_command = f"mkdir -p /workspace/logs/commands && nohup {command_text} > {log_realtime} 2>&1 & echo $! > {log_realtime}.pid && echo 'Started background process PID: '$(cat {log_realtime}.pid)"
                    args["command"] = wrapped_command
                    logger.info(f"[CodeAct] Wrapped for background execution. Monitor with: tail -f {log_realtime}")
                # For blocking long-running commands, wrap with script for real-time output capture
                elif is_long_running and args.get("blocking", True) == True:
                    # Create a realtime log path
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_cmd = command_text[:30].replace("/", "_").replace(" ", "_")
                    log_realtime = f"/workspace/logs/commands/{timestamp}_{safe_cmd}.realtime"

                    # Use script command to capture all output including progress bars
                    wrapped_command = f"mkdir -p /workspace/logs/commands && script -q -c '{command_text}' {log_realtime}"
                    args["command"] = wrapped_command
                    logger.info(f"[CodeAct] Wrapped for real-time streaming. Monitor with: tail -f {log_realtime}")

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

            # Create log file using stream monitor for ALL commands
            command_str = args.get("command") if action_name == "run" else None
            log_path = self._stream_monitor.create_log_file(action_name, command_str)

            # Write additional context
            self._stream_monitor.write_to_log(log_path, f"Arguments: {json.dumps(args, indent=2, default=str)}", "CONTEXT")

            # Print monitoring instructions for important commands
            if command_str and any(cmd in command_str for cmd in ["pip install", "npm install", "apt", "docker"]):
                self._stream_monitor.print_monitoring_instructions(log_path)

            logger.info(f"[CodeAct] Streaming output to: {log_path}")

            # Store log path for later use
            self._current_log_path = log_path

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
                            # Allow ample time for package managers; interactive prompts are handled via retries.
                            min_timeout = int(os.getenv('OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT', '600'))
                            if not any(flag in cmd for flag in ['-y', '--yes', '--no-input', '--noconfirm', '--non-interactive']):
                                logger.info('[CodeAct] Command may prompt for input; will retry with a non-interactive flag if needed')
                            actual_timeout = max(timeout, min_timeout)
                            logger.info(f'[CodeAct] Using extended timeout ({actual_timeout}s) for package manager command')
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
                                # Log timeout to file
                                if hasattr(self, '_current_log_path') and self._current_log_path:
                                    try:
                                        with self._current_log_path.open("a", encoding="utf-8") as f:
                                            f.write(f"\n=== TIMEOUT (Retry with {flag}) ===\n")
                                            f.write(f"Time: {datetime.now().isoformat()}\n")
                                            f.write(f"Timeout after: {timeout} seconds\n")
                                            f.write(f"Command: {cmd}\n")
                                            f.write(f"{'=' * 50}\n")
                                            f.flush()
                                    except Exception:
                                        pass

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

            # Stream output to log file using stream monitor
            if hasattr(self, '_current_log_path') and self._current_log_path:
                try:
                    # Extract key information
                    if isinstance(observation, dict):
                        content = observation.get("content", "")
                        stdout = observation.get("stdout", "")
                        stderr = observation.get("stderr", "")
                        metadata = observation.get("metadata", {})
                        exit_code = metadata.get("exit_code", observation.get("exit_code"))

                        # Write output sections
                        if content:
                            self._stream_monitor.write_to_log(self._current_log_path, content, "OUTPUT")
                        if stdout and stdout != content:
                            self._stream_monitor.write_to_log(self._current_log_path, stdout, "STDOUT")
                        if stderr:
                            self._stream_monitor.write_to_log(self._current_log_path, stderr, "STDERR")
                        if exit_code is not None:
                            self._stream_monitor.write_to_log(self._current_log_path, f"Exit Code: {exit_code}", "STATUS")

                        # Write full response for debugging
                        self._stream_monitor.write_json_to_log(self._current_log_path, observation, "FULL_RESPONSE")

                        # Special handling for pip/npm install progress
                        if action_name == "run" and "command" in args:
                            cmd = args["command"]
                            if "pip install" in cmd or "npm install" in cmd:
                                if exit_code == 0:
                                    self._stream_monitor.write_to_log(self._current_log_path, "✓ Installation completed successfully", "SUCCESS")
                                elif exit_code is not None and exit_code != 0:
                                    self._stream_monitor.write_to_log(self._current_log_path, f"✗ Installation failed with exit code {exit_code}", "ERROR")
                    else:
                        self._stream_monitor.write_to_log(self._current_log_path, str(observation), "RESPONSE")

                    # Print monitoring reminder for long commands
                    if action_name == "run" and "command" in args:
                        cmd = args["command"]
                        if any(keyword in cmd for keyword in ["pip", "npm", "apt", "docker", "make", "build"]):
                            logger.info(f"[CodeAct] Monitor progress: tail -f {self._current_log_path}")

                except Exception as e:
                    logger.warning(f"[CodeAct] Failed to write to log: {e}")

            # Check if the response indicates the command wasn't actually executed
            # This can happen when the action server accepts the request but fails to execute it
            if isinstance(observation, dict):
                # Check for signs that the command wasn't executed
                metadata = observation.get("metadata", {})
                extras = observation.get("extras", {}) if isinstance(observation, dict) else {}
                if isinstance(extras, dict) and not metadata:
                    metadata = extras.get("metadata", {}) or {}
                exit_code = metadata.get("exit_code")
                if exit_code is None:
                    exit_code = observation.get("exit_code")
                content = observation.get("content", "")

                # If we have a run command with no exit code and no content, it likely didn't execute
                if action_name == "run" and exit_code is None and not content:
                    logger.warning(
                        "[CodeAct] Command appears to have not executed - no exit code or output received"
                    )
                    # Create an error response indicating the command didn't execute
                    from ..core.openhands.code_executor import ExecutionResult
                    error_msg = (
                        "ERROR: Command was accepted by the action server but appears not to have executed.\n"
                        "This can happen when:\n"
                        "1. The action server is overloaded or not responding properly\n"
                        "2. The workspace or container has issues\n"
                        "3. The command path or environment is not set up correctly\n"
                        "Please try the command again or check the action server status."
                    )
                    return OpenHandsActionResult(
                        execution_result=ExecutionResult(
                            success=False,
                            exit_code=-1,
                            stdout=error_msg,
                            stderr="Command did not execute",
                            execution_time=0.0,
                            files_created=[],
                            files_modified=[],
                            command=payload.get("action", {}).get("args", {}).get("command", "unknown"),
                            working_directory=str(self._workspace_path),
                            env={},
                        ),
                        raw_observation=observation,
                        stdout=error_msg,
                        stderr="Command did not execute",
                        server_logs="Action server accepted request but command did not execute",
                    )

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

        async def _execute_with_backend_fallback(self, command: str, last_timeout: int, attempts: int) -> 'OpenHandsActionResult':
            """Execute command directly via backend subprocess when action server times out."""
            from ..core.openhands.code_executor import ExecutionResult

            log_dir = self._workspace_path / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"backend_fallback_{timestamp}.log"
            logger.warning(
                "[CodeAct] Executing command via backend fallback. Log file: %s",
                log_path,
            )

            # Wrap command for better execution
            shell_command = f"cd {shlex.quote(str(self._workspace_path))} && {command}"
            env = os.environ.copy()
            start_time = time.perf_counter()

            process = await asyncio.create_subprocess_shell(
                shell_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout_chunks = []
            stderr_chunks = []

            async def _stream(pipe, prefix, collector):
                if pipe is None:
                    return
                while True:
                    chunk = await pipe.readline()
                    if not chunk:
                        break
                    text = chunk.decode(errors="replace")
                    collector.append(text)
                    log_handle.write(f"[{prefix}] {text}")
                    log_handle.flush()

            log_handle = log_path.open("a", encoding="utf-8", errors="ignore")
            reporter = asyncio.create_task(self._periodic_log_report(log_path, process))
            try:
                await asyncio.gather(
                    _stream(process.stdout, "STDOUT", stdout_chunks),
                    _stream(process.stderr, "STDERR", stderr_chunks),
                )
                await process.wait()
            finally:
                reporter.cancel()
                try:
                    await reporter
                except asyncio.CancelledError:
                    pass
                log_handle.close()

            duration = time.perf_counter() - start_time
            exit_code = process.returncode if process.returncode is not None else -1
            stdout_text = ''.join(stdout_chunks).strip()
            stderr_text = ''.join(stderr_chunks).strip()

            summary = (
                "Command executed via backend fallback. "
                f"Review detailed logs at {log_path}."
            )
            if stdout_text:
                stdout_payload = f"{summary}\n{stdout_text}"
            else:
                stdout_payload = summary

            exec_result = ExecutionResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout_payload,
                stderr=stderr_text,
                execution_time=duration,
                files_created=[],
                files_modified=[],
                command=command[:200],
                working_directory=str(self._workspace_path),
                env={},
            )

            raw_observation = {
                "content": summary,
                "metadata": {
                    "action": "backend_fallback",
                    "exit_code": exit_code,
                    "log_path": str(log_path),
                    "attempts": attempts,
                    "last_timeout_seconds": last_timeout,
                },
            }

            return OpenHandsActionResult(
                execution_result=exec_result,
                raw_observation=raw_observation,
                stdout=stdout_payload,
                stderr=stderr_text,
                server_logs=f"Backend fallback executed. Logs stored at {log_path}",
            )

        async def _periodic_log_report(self, log_path: Path, process: asyncio.subprocess.Process) -> None:
            """Periodically report progress from log file."""
            last_size = 0
            try:
                while True:
                    await asyncio.sleep(5)
                    if process.returncode is not None:
                        break
                    try:
                        stat_info = log_path.stat()
                    except FileNotFoundError:
                        continue
                    if stat_info.st_size == last_size:
                        continue
                    last_size = stat_info.st_size
                    try:
                        with log_path.open("r", encoding="utf-8", errors="ignore") as log_file:
                            if stat_info.st_size > 4096:
                                log_file.seek(stat_info.st_size - 4096)
                            tail = log_file.read().strip().splitlines()
                            tail_snippet = " | ".join(tail[-5:]) if tail else ""
                    except FileNotFoundError:
                        continue
                    if tail_snippet:
                        logger.info(
                            "[CodeAct] Backend fallback progress (%s): %s",
                            log_path.name,
                            tail_snippet,
                        )
            except asyncio.CancelledError:
                pass

        async def _post_action(self, payload: dict, timeout: int) -> httpx.Response:
            # Determine HTTP timeout based on action type
            action = payload.get("action", {})
            action_name = action.get("action", "unknown")
            args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}
            command = args.get("command", "")
            blocking = bool(args.get("blocking", True))

            # Auto-wrap commands to stream output to log file in real-time
            if action_name == "run" and command and hasattr(self, '_current_log_path'):
                log_realtime = str(self._current_log_path) + ".realtime"

                # Check if this is a non-blocking long-running command
                is_non_blocking = not args.get("blocking", True)

                # For important long-running commands, add logging wrapper
                if any(keyword in command for keyword in ["pip install", "npm install", "apt", "docker", "make", "wget", "curl", "git clone"]):
                    if is_non_blocking:
                        # For non-blocking, run in background with output to file
                        wrapped_command = f"nohup {command} > {log_realtime} 2>&1 & echo $! > {log_realtime}.pid && echo 'Started background process PID: '$(cat {log_realtime}.pid)"
                        args["command"] = wrapped_command
                        action["args"] = args
                        payload = {"action": action}
                        logger.info(f"[CodeAct] Non-blocking execution with output to: {log_realtime}")
                        logger.info(f"[CodeAct] Monitor with: tail -f {log_realtime}")
                    else:
                        # Use script command to capture all output in real-time
                        wrapped_command = f"script -q -c '{command}' {log_realtime}"
                        args["command"] = wrapped_command
                        action["args"] = args
                        payload = {"action": action}
                        logger.info(f"[CodeAct] Real-time output capture to: {log_realtime}")
                        logger.info(f"[CodeAct] Monitor with: tail -f {log_realtime}")
                elif ">" not in command and "|" not in command and not is_non_blocking:
                    # For simple blocking commands without redirection, use tee
                    wrapped_command = f"({command}) 2>&1 | tee -a {log_realtime}"
                    args["command"] = wrapped_command
                    action["args"] = args
                    payload = {"action": action}
                    logger.info(f"[CodeAct] Output streaming to: {log_realtime}")

            # Adjust HTTP timeout based on action type
            # IMPORTANT: If server is busy with another command, we need quick timeouts for file operations
            if action_name == "read":
                # Read operations should be instant, use very short timeout
                http_timeout_seconds = 10
                logger.debug(f"[CodeAct] Read operation, using quick timeout: {http_timeout_seconds}s")
            elif action_name in ["edit", "write", "str_replace_editor"]:
                # File operations should be fast, but may need some processing
                http_timeout_seconds = min(30, timeout)
                logger.debug(f"[CodeAct] File operation {action_name}, using timeout: {http_timeout_seconds}s")
            elif action_name == "run":
                if not blocking:
                    # Non-blocking commands return immediately after starting
                    http_timeout_seconds = 15
                    logger.debug(f"[CodeAct] Non-blocking run, using quick timeout: {http_timeout_seconds}s")
                else:
                    # Blocking commands need full timeout
                    http_timeout_seconds = min(timeout + 10, 300)
                    logger.debug(f"[CodeAct] Blocking run, using timeout: {http_timeout_seconds}s")
            else:
                http_timeout_seconds = min(timeout + 10, 60)
                logger.debug(f"[CodeAct] Action {action_name}, using default timeout: {http_timeout_seconds}s")

            # Construct a per-phase timeout so connect/write don't inherit the full read budget
            http_timeout_seconds = max(5.0, http_timeout_seconds)
            http_timeout = httpx.Timeout(
                connect=min(10.0, http_timeout_seconds),
                read=http_timeout_seconds,
                write=http_timeout_seconds,
                pool=None,
            )

            # Special logging for problematic commands and edit actions
            if action_name in ["edit", "write", "str_replace_editor"]:
                logger.info(
                    f"[CodeAct] Executing {action_name} action with action_timeout={timeout}s, http_timeout={http_timeout_seconds}s"
                )
            elif "pip install" in command or "npm install" in command or "apt" in command:
                logger.info(
                    f"[CodeAct] Executing potentially slow command: {command[:100]}... with action_timeout={timeout}s, http_timeout={http_timeout_seconds}s"
                )
            else:
                logger.debug(
                    f"[CodeAct] Sending HTTP request for action={action_name} with http_timeout={http_timeout_seconds}s"
                )

            async with httpx.AsyncClient(timeout=http_timeout, trust_env=False) as client:
                try:
                    response = await client.post(
                        f"http://127.0.0.1:{self._port}/execute_action",
                        json=payload,
                        headers={"X-Session-API-Key": self._api_key},
                    )
                    # Log successful receipt of response for slow operations
                    if action_name in ["edit", "write", "str_replace_editor"]:
                        logger.info(f"[CodeAct] Received response for {action_name} action")
                    elif "pip install" in command or "npm install" in command or "apt" in command:
                        logger.info(f"[CodeAct] Received response for command: {command[:50]}...")
                    return response
                except httpx.TimeoutException as e:
                    logger.error(
                        f"[CodeAct] HTTP timeout after {http_timeout_seconds}s for action={action_name}, "
                        f"command={command[:100] if command else 'N/A'}"
                    )
                    raise

        async def run_cmd(self, command: str, timeout: int = DEFAULT_ACTION_TIMEOUT, blocking: bool = True, cwd: str | None = None) -> 'OpenHandsActionResult':
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

        async def ipython_run(self, code: str, timeout: int = DEFAULT_ACTION_TIMEOUT) -> 'OpenHandsActionResult':
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

        async def file_read(self, path: str, start: int = 0, end: int = -1, timeout: int = DEFAULT_ACTION_TIMEOUT) -> 'OpenHandsActionResult':
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

        async def file_write(self, path: str, content: str, timeout: int = DEFAULT_ACTION_TIMEOUT) -> 'OpenHandsActionResult':
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

        async def file_edit(self, path: str, command: str, *, file_text: str | None = None, old_str: str | None = None, new_str: str | None = None, insert_line: int | None = None, timeout: int = DEFAULT_ACTION_TIMEOUT) -> 'OpenHandsActionResult':
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

        # Resolve workspace path to an absolute path for the action server.
        if not isinstance(workspace_path, Path):
            workspace_path = Path(workspace_path)
        workspace_path = workspace_path.resolve()

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
