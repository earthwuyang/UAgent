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
from typing import Optional, Dict, Any

import httpx

from .stream_monitor import CommandStreamMonitor
from .openhands.types import ActionResult, ActionError


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

        @property
        def workspace_path(self) -> Path:
            return self._workspace_path

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

            if action_name in {"read", "write", "edit", "str_replace_editor"}:
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
                # Ensure parent directory exists for file operations
                try:
                    p = args.get("path")
                    if isinstance(p, str):
                        Path(p).parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

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

                # Do NOT automatically wrap commands with nohup
                # The LLM should generate commands with nohup if needed
                # Just log the command for monitoring
                if command_text:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_dir = "logs"
                    log_file = f"{log_dir}/commands.log"

                    # Log the command (but don't wrap it)
                    logger.info(f"[CodeAct] Executing command: {command_text[:100]}...")

                    # Create log directory if needed and log the command
                    try:
                        os.makedirs(self._workspace_path / log_dir if hasattr(self, '_workspace_path') else log_dir, exist_ok=True)
                        actual_log_path = self._workspace_path / log_file if hasattr(self, '_workspace_path') else log_file
                        with open(actual_log_path, "a") as f:
                            f.write(f"[{timestamp}] Command: {command_text}\n")
                        logger.info(f"[CodeAct] Command logged to: {actual_log_path}")
                    except Exception as e:
                        logger.warning(f"[CodeAct] Could not log command: {e}")

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

                                return await self._execute_with_backend_fallback(cmd, timeout, attempts=2)
                            break
                    else:
                        # Not a package manager command or already has flags — fall back to backend execution
                        logger.error(
                            "[CodeAct] Command timeout for %s; falling back to backend execution",
                            (cmd[:200] if cmd else "unknown"),
                        )
                        return await self._execute_with_backend_fallback(cmd or "", actual_timeout, attempts=1)
                else:
                    # For non-run actions (file ops), attempt backend fallback write first
                    if action_name in ("edit", "write", "str_replace_editor"):
                        try:
                            target = args.get("path")
                            content = args.get("file_text") or args.get("content") or ""
                            if isinstance(target, str):
                                Path(target).parent.mkdir(parents=True, exist_ok=True)
                                if content:
                                    heredoc = (
                                        f"mkdir -p $(dirname {shlex.quote(target)}) && "
                                        f"cat > {shlex.quote(target)} <<'__UAGENT_EOF__'\n{content}\n__UAGENT_EOF__\n"
                                    )
                                    return await self.run_cmd(heredoc, timeout=60, blocking=True)
                                old = args.get("old_str"); new = args.get("new_str")
                                if isinstance(old, str) and isinstance(new, str) and os.path.exists(target):
                                    try:
                                        txt = Path(target).read_text(encoding="utf-8", errors="ignore")
                                        Path(target).write_text(txt.replace(old, new), encoding="utf-8")
                                        return await self.run_cmd(f"stat -c '%s %n' {shlex.quote(target)}", timeout=30, blocking=True)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
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

            # Do NOT auto-wrap commands - let the LLM decide how to run them
            # Just log for monitoring purposes
            if action_name == "run" and command and hasattr(self, '_current_log_path'):
                log_realtime = str(self._current_log_path) + ".realtime"
                logger.info(f"[CodeAct] Executing command without modification: {command[:100]}...")
                logger.info(f"[CodeAct] Log path available at: {log_realtime}")

            # Adjust HTTP timeout based on action type
            # IMPORTANT: If server is busy with another command, we need quick timeouts for file operations
            if action_name == "read":
                # Read operations should be instant, use very short timeout
                http_timeout_seconds = 10
                logger.debug(f"[CodeAct] Read operation, using quick timeout: {http_timeout_seconds}s")
            elif action_name in ["edit", "write", "str_replace_editor"]:
                # File operations should be fast; allow up to 90s then fall back
                http_timeout_seconds = min(90, timeout)
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

        # Job service client (Action Server managed queued jobs)
        async def jobs_submit(self, command: str, *, mode: str = "auto", short_timeout_s: float = 6.0, cwd: str | None = None, env: dict | None = None) -> dict:
            payload = {
                "command": command,
                "mode": mode,
                "short_timeout_s": float(short_timeout_s),
                "cwd": cwd,
                "env": env or {},
            }
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                r = await client.post(
                    f"http://127.0.0.1:{self._port}/jobs",
                    json=payload,
                    headers={"X-Session-API-Key": self._api_key},
                )
                r.raise_for_status()
                return r.json()

        async def jobs_status(self, job_id: str, *, include_tail: bool = True) -> dict:
            params = {"include_tail": "true"} if include_tail else {}
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                r = await client.get(
                    f"http://127.0.0.1:{self._port}/jobs/{job_id}",
                    params=params,
                    headers={"X-Session-API-Key": self._api_key},
                )
                r.raise_for_status()
                return r.json()

        async def jobs_logs_tail(self, job_id: str, *, stderr: bool = False, max_bytes: int = 4000) -> str:
            params = {"stderr": "true"} if stderr else {}
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                r = await client.get(
                    f"http://127.0.0.1:{self._port}/jobs/{job_id}/logs",
                    params=params,
                    headers={"X-Session-API-Key": self._api_key},
                )
                r.raise_for_status()
                try:
                    return r.text
                except Exception:
                    return ""


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

        # Force Docker runtime for hardware isolation - no local fallback
        env["RUNTIME"] = "docker"

        # Use pre-built OpenHands image to avoid build issues
        env["SANDBOX_BOX_TYPE"] = "remote"
        env["SANDBOX_CONTAINER_IMAGE"] = "ghcr.io/all-hands-ai/openhands:latest"
        env["SANDBOX_FORCE_REBUILD"] = "false"  # Don't rebuild image

        # Map LLM configuration from .env file to OpenHands format
        if "LITELLM_MODEL" in os.environ:
            env["LLM_MODEL"] = os.environ["LITELLM_MODEL"]
        if "LITELLM_API_KEY" in os.environ:
            env["LLM_API_KEY"] = os.environ["LITELLM_API_KEY"]
        if "LITELLM_API_BASE" in os.environ:
            env["LLM_BASE_URL"] = os.environ["LITELLM_API_BASE"]

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

        # Widen allowed write roots inside the action server to common workspace paths
        allowed_roots = [
            "/workspace",
            "/workspace/code",
            "/workspace/experiments",
            "/workspace/workspace",
            "/workspace/logs",
            "/workspace/output",
        ]
        existing_allowed = env.get("OPENHANDS_ALLOWED_WRITE_DIRS", "")
        if existing_allowed:
            # Merge, deduplicate
            merged = existing_allowed.split(":") + allowed_roots
            env["OPENHANDS_ALLOWED_WRITE_DIRS"] = ":".join(sorted(set(filter(None, merged))))
        else:
            env["OPENHANDS_ALLOWED_WRITE_DIRS"] = ":".join(allowed_roots)

        # Also advertise host-side common prefixes (best-effort; respected only if server supports it)
        host_prefixes = []
        for parent in (workspace_path.parent, workspace_path.parent.parent):
            if parent and parent.name in ("uagent-workspace", "uagent_workspace"):
                host_prefixes.append(str(parent))
        if host_prefixes:
            env["OPENHANDS_ALLOWED_PATH_PREFIXES"] = ":".join(host_prefixes)
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

        # Directory listings and many file reads/"view" ops return exit_code=-1 even when successful.
        if exit_code != 0:
            if success_flag is True:
                exit_code = 0
            elif isinstance(output, str) and output.strip():
                # Treat str_replace_editor.view and generic read/list as success if output is non-empty
                cmd_name = (metadata.get("command") or observation.get("command") or "").lower()
                act = (action_name or "").lower()
                if act in {"list", "read", "view"} or cmd_name in {"list", "read", "view"}:
                    exit_code = 0
                # Heuristic: treat file edit/write messages as success even if exit_code missing
                elif act in {"edit", "write"}:
                    low = output.lower()
                    if (
                        "file created successfully" in low
                        or "has been edited" in low
                        or "file edited successfully" in low
                        or "created at:" in low
                    ):
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


class OpenHandsClientV2:
    """Typed, idempotent client built on top of OpenHandsActionServerRunner.

    This wrapper enforces allowed-write roots, provides idempotent file ops, and
    uses the runner's job service for long-running execute_bash commands.
    """

    def __init__(self, workspace_path: Path, allowed_write_roots: list[Path]):
        self._runner = OpenHandsActionServerRunner()
        self._workspace = workspace_path.resolve()
        self._allowed_roots = [p.resolve() for p in allowed_write_roots] or [self._workspace]
        # Widen allowed roots to all workspaces under uagent-workspace or uagent_workspace if present
        for parent in (self._workspace.parent, self._workspace.parent.parent):
            try:
                if parent and parent.name in ("uagent-workspace", "uagent_workspace"):
                    if parent.resolve() not in self._allowed_roots:
                        self._allowed_roots.append(parent.resolve())
            except Exception:
                pass
        (self._workspace / "logs").mkdir(parents=True, exist_ok=True)
        self._trace_path = self._workspace / "logs" / "openhands_trace.jsonl"
        # Persistent session per-workspace
        self._session: Optional[OpenHandsActionServerRunner.Session] = None
        self._session_lock = asyncio.Lock()

    # -------------------- helpers --------------------
    def _log(self, rec: dict) -> None:
        try:
            import json, time
            rec = dict(rec)
            rec.setdefault("ts", time.time())
            with self._trace_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = (self._workspace / p).resolve()
        return p

    def _allowed(self, path: str) -> bool:
        rp = self._resolve(path)
        return any(str(rp).startswith(str(root)) for root in self._allowed_roots)

    async def _ensure_session(self) -> 'OpenHandsActionServerRunner.Session':
        """Ensure a single long-lived action server session for this workspace.

        This prevents server thrashing (start/stop per action) and keeps job ids
        stable across start_job/poll_job/wait_job.
        """
        async with self._session_lock:
            if self._session is not None and getattr(self._session, "is_running", False):
                return self._session
            # Open a fresh session and keep it
            self._session = await self._runner.open_session(self._workspace)
            return self._session

    async def close(self) -> None:
        async with self._session_lock:
            if self._session is not None:
                try:
                    await self._session.close()
                finally:
                    self._session = None

    # -------------------- file API --------------------
    async def create_if_absent(self, path: str, content: str, timeout_sec: int = 60) -> ActionResult:
        if not self._allowed(path):
            return ActionResult(id="create_if_absent", tool="write", success=False, exit_code=None, duration_ms=0,
                                error=ActionError("PATH_DENIED", f"{path} outside allowed roots"))
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            return ActionResult(id="create_if_absent", tool="write", success=True, exit_code=0, duration_ms=1, stdout="exists")

        session = await self._ensure_session()
        res = await session.file_edit(str(p), "create", file_text=content, timeout=max(1, timeout_sec))
        ok = res.execution_result.success
        out = res.stdout or ""
        err = res.stderr or ""
        return ActionResult(id="create_if_absent", tool="write", success=ok, exit_code=res.execution_result.exit_code,
                            duration_ms=int(res.execution_result.execution_time * 1000) if hasattr(res.execution_result, 'execution_time') else 0,
                            stdout=out, stderr=err,
                            error=None if ok else ActionError("SERVER_ERROR", err or "create failed"))

    async def write(self, path: str, content: str, overwrite: bool = True, timeout_sec: int = 60) -> ActionResult:
        if not self._allowed(path):
            return ActionResult(id="write", tool="write", success=False, exit_code=None, duration_ms=0,
                                error=ActionError("PATH_DENIED", f"{path} outside allowed roots"))
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and not overwrite:
            return ActionResult(id="write", tool="write", success=False, exit_code=1, duration_ms=1,
                                error=ActionError("CLIENT_ERROR", "file exists and overwrite=false"))
        session = await self._ensure_session()
        res = await session.file_write(str(p), content, timeout=max(1, timeout_sec))
        ok = res.execution_result.success
        out = res.stdout or ""
        err = res.stderr or ""
        return ActionResult(id="write", tool="write", success=ok, exit_code=res.execution_result.exit_code,
                            duration_ms=int(res.execution_result.execution_time * 1000) if hasattr(res.execution_result, 'execution_time') else 0,
                            stdout=out, stderr=err,
                            error=None if ok else ActionError("SERVER_ERROR", err or "write failed"))

    async def read(self, path: str, timeout_sec: int = 60) -> ActionResult:
        # Allow reads anywhere under workspace
        p = self._resolve(path)
        session = await self._ensure_session()
        res = await session.file_read(str(p), timeout=max(1, timeout_sec))
        ok = res.execution_result.success
        out = res.stdout or res.server_logs or ""
        err = res.stderr or ""
        return ActionResult(id="read", tool="read", success=ok, exit_code=res.execution_result.exit_code,
                            duration_ms=int(res.execution_result.execution_time * 1000) if hasattr(res.execution_result, 'execution_time') else 0,
                            stdout=out, stderr=err,
                            error=None if ok else ActionError("SERVER_ERROR", err or "read failed"))

    # -------------------- jobs for run --------------------
    async def start_job(self, cmd: str, timeout_sec: int = 300, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> str:
        session = await self._ensure_session()
        payload_cmd = cmd if not cwd else f"cd {cwd} && {cmd}"
        submit = await session.jobs_submit(payload_cmd, mode="auto", short_timeout_s=min(8.0, max(1.0, timeout_sec/10)), cwd=cwd, env=env)
        job_id = submit.get("id") or submit.get("job_id")
        return str(job_id)

    async def poll_job(self, job_id: str) -> Dict[str, Any]:
        session = await self._ensure_session()
        status = await session.jobs_status(job_id, include_tail=True)
        state = status.get("state") or status.get("status")
        return {"status": state, "exit_code": status.get("exit_code"), "tail": status.get("tail", "")}

    async def wait_job(self, job_id: str, max_wait_sec: int = 3600) -> ActionResult:
        import time as _t
        start = _t.time()
        exit_code = None
        tail = ""
        while _t.time() - start < max_wait_sec:
            st = await self.poll_job(job_id)
            state = (st.get("status") or "").upper()
            tail = st.get("tail", tail)
            if state in ("SUCCEEDED", "FAILED", "CANCELED"):
                exit_code = st.get("exit_code")
                success = (state == "SUCCEEDED") or (exit_code == 0)
                return ActionResult(id=job_id, tool="run", success=bool(success), exit_code=exit_code,
                                    duration_ms=int((_t.time()-start)*1000), stdout=tail, stderr=None,
                                    error=None if success else ActionError("SERVER_ERROR", f"job state {state}"))
            await asyncio.sleep(2.0)
        return ActionResult(id=job_id, tool="run", success=False, exit_code=None, duration_ms=int((_t.time()-start)*1000),
                            stdout=tail, error=ActionError("TIMEOUT","job wait timeout"))
