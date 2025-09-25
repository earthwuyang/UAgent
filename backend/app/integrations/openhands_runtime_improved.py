"""Improved OpenHands runtime with adaptive retry and background execution."""

import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger(__name__)

# Configuration from environment
DEFAULT_ACTION_TIMEOUT = int(os.getenv("OPENHANDS_ACTION_TIMEOUT", "120"))
MAX_ACTION_TIMEOUT = int(os.getenv("OPENHANDS_MAX_ACTION_TIMEOUT", "900"))
ADAPTIVE_MULTIPLIER = float(os.getenv("OPENHANDS_RUN_ADAPTIVE_MULTIPLIER", "1.75"))
MAX_ATTEMPTS = int(os.getenv("OPENHANDS_RUN_MAX_ATTEMPTS", "3"))
PACKAGE_CMD_MIN_TIMEOUT = int(os.getenv("OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT", "600"))


class ImprovedActionExecutor:
    """Enhanced action executor with retry logic and background execution."""

    def __init__(self, session, workspace_path: Path):
        self.session = session
        self.workspace_path = workspace_path
        self.port = session._port
        self.api_key = session._api_key

    async def execute_with_retry(
        self,
        action_dict: dict,
        base_timeout: int = DEFAULT_ACTION_TIMEOUT,
    ) -> dict:
        """Execute action with adaptive retry and timeout scaling."""

        action_name = action_dict.get("action", "unknown")
        args = action_dict.get("args", {})
        command = args.get("command", "")

        # Determine if this needs special handling
        is_package_cmd = self._is_package_command(command)
        is_long_running = self._is_long_running_command(command)

        # Setup retry configuration
        max_attempts = MAX_ATTEMPTS if action_name == "run" else 1
        current_timeout = self._calculate_initial_timeout(
            base_timeout, action_name, command, is_package_cmd
        )

        attempt = 1
        last_error = None

        while attempt <= max_attempts:
            logger.info(
                f"[CodeAct] Attempt {attempt}/{max_attempts} for {action_name}, "
                f"timeout={current_timeout}s, command={command[:100] if command else 'N/A'}"
            )

            try:
                # Try normal execution first
                response = await self._execute_action(
                    action_dict, current_timeout
                )

                # Check if response indicates success
                if self._is_successful_response(response):
                    return response

                # If command didn't execute, try background execution
                if not self._command_was_executed(response) and is_long_running:
                    logger.warning(
                        f"[CodeAct] Command appears not executed, trying background execution"
                    )
                    return await self._execute_in_background(command, current_timeout)

                return response

            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                last_error = e
                logger.warning(
                    f"[CodeAct] Timeout on attempt {attempt}/{max_attempts} after {current_timeout}s"
                )

                # Try adding non-interactive flags for package managers
                if attempt == 1 and is_package_cmd:
                    modified_cmd = self._add_noninteractive_flags(command)
                    if modified_cmd != command:
                        logger.info(f"[CodeAct] Retrying with non-interactive flags")
                        args["command"] = modified_cmd
                        action_dict["args"] = args
                        command = modified_cmd
                        continue

                # On second timeout, try backend fallback
                if attempt == 2 and action_name == "run":
                    logger.info(f"[CodeAct] Attempting backend fallback execution")
                    return await self._execute_with_backend_fallback(
                        command, current_timeout, attempt
                    )

                # Increase timeout for next attempt
                if attempt < max_attempts:
                    attempt += 1
                    current_timeout = min(
                        int(current_timeout * ADAPTIVE_MULTIPLIER),
                        MAX_ACTION_TIMEOUT
                    )
                else:
                    # Final attempt failed, return error
                    return self._create_timeout_error(
                        action_name, command, current_timeout, attempt
                    )

        return self._create_timeout_error(
            action_name, command, current_timeout, attempt
        )

    def _is_package_command(self, command: str) -> bool:
        """Check if command is a package manager command."""
        package_patterns = [
            "pip install", "pip3 install", "conda install",
            "npm install", "yarn add", "apt-get install",
            "apt install", "yum install", "brew install"
        ]
        return any(pattern in command for pattern in package_patterns)

    def _is_long_running_command(self, command: str) -> bool:
        """Check if command is likely to be long-running."""
        patterns = [
            "pip install", "npm install", "yarn", "make",
            "docker build", "docker pull", "wget", "curl",
            "git clone", "compile", "build"
        ]
        return any(pattern in command for pattern in patterns)

    def _calculate_initial_timeout(
        self,
        base_timeout: int,
        action_name: str,
        command: str,
        is_package_cmd: bool
    ) -> int:
        """Calculate appropriate initial timeout."""

        if action_name in ["edit", "write"]:
            return min(base_timeout + 60, 180)

        if is_package_cmd:
            return max(base_timeout, PACKAGE_CMD_MIN_TIMEOUT)

        if "docker stop" in command:
            return max(20, base_timeout // 3)

        return base_timeout

    def _add_noninteractive_flags(self, command: str) -> str:
        """Add non-interactive flags to package manager commands."""

        replacements = [
            ("pip install", "pip install --no-input"),
            ("pip3 install", "pip3 install --no-input"),
            ("apt-get install", "apt-get install -y"),
            ("apt install", "apt install -y"),
            ("yum install", "yum install -y"),
            ("conda install", "conda install -y"),
            ("npm install", "npm install --yes"),
        ]

        for pattern, replacement in replacements:
            if pattern in command and "--no-input" not in command and "-y" not in command:
                return command.replace(pattern, replacement)

        return command

    async def _execute_action(self, action_dict: dict, timeout: int) -> dict:
        """Execute a single action with given timeout."""

        payload = {"action": action_dict}

        # Determine HTTP timeout based on action type
        action_name = action_dict.get("action", "unknown")
        if action_name in ["edit", "write"]:
            http_timeout = min(timeout + 10, 180)
        else:
            http_timeout = min(timeout + 10, 60)

        async with httpx.AsyncClient(timeout=http_timeout, trust_env=False) as client:
            response = await client.post(
                f"http://127.0.0.1:{self.port}/execute_action",
                json=payload,
                headers={"X-Session-API-Key": self.api_key},
            )
            response.raise_for_status()
            return response.json()

    def _is_successful_response(self, response: dict) -> bool:
        """Check if response indicates success."""

        if not isinstance(response, dict):
            return False

        # Check for explicit success flag
        if "success" in response:
            return response["success"]

        # Check exit code
        metadata = response.get("metadata", {})
        exit_code = metadata.get("exit_code", response.get("exit_code"))
        if exit_code is not None:
            return exit_code == 0

        # Check for content
        return bool(response.get("content"))

    def _command_was_executed(self, response: dict) -> bool:
        """Check if command was actually executed."""

        if not isinstance(response, dict):
            return False

        metadata = response.get("metadata", {})
        exit_code = metadata.get("exit_code", response.get("exit_code"))
        content = response.get("content", "")

        # If no exit code and no content, likely not executed
        return exit_code is not None or bool(content)

    async def _execute_in_background(self, command: str, timeout: int) -> dict:
        """Execute command in background with log monitoring."""

        log_dir = self.workspace_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"background_{timestamp}.log"

        # Wrap command to run in background with logging
        bg_command = (
            f"nohup bash -c '{command}' > {log_file} 2>&1 & "
            f"echo $! > {log_file}.pid && "
            f"echo 'Started background process with PID '$(cat {log_file}.pid)"
        )

        logger.info(f"[CodeAct] Starting background execution: {log_file}")

        # Execute the background command
        result = await self._execute_action(
            {"action": "run", "args": {"command": bg_command}},
            30  # Short timeout for starting background process
        )

        # Monitor the log file for a short time
        monitor_cmd = f"""
for i in {{1..10}}; do
    if [ -f {log_file} ]; then
        echo "=== Log output (attempt $i) ==="
        tail -20 {log_file}
        if grep -q "ERROR\\|FAIL\\|Exception" {log_file}; then
            echo "=== Errors detected in log ==="
            exit 1
        fi
    fi
    sleep 2
done
echo "=== Background process started successfully ==="
echo "Monitor with: tail -f {log_file}"
"""

        monitor_result = await self._execute_action(
            {"action": "run", "args": {"command": monitor_cmd}},
            30
        )

        # Combine results
        return {
            "success": True,
            "content": (
                f"Started background process. Log file: {log_file}\n"
                f"{monitor_result.get('content', '')}"
            ),
            "metadata": {
                "background": True,
                "log_file": str(log_file),
                "pid_file": f"{log_file}.pid",
            }
        }

    async def _execute_with_backend_fallback(
        self,
        command: str,
        timeout: int,
        attempts: int
    ) -> dict:
        """Execute command directly via backend subprocess."""

        log_dir = self.workspace_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"fallback_{timestamp}.log"

        logger.warning(f"[CodeAct] Backend fallback execution to {log_file}")

        # Run command with subprocess
        shell_cmd = f"cd {shlex.quote(str(self.workspace_path))} && {command}"

        try:
            process = await asyncio.create_subprocess_shell(
                shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return self._create_timeout_error(
                    "run", command, timeout, attempts
                )

            # Save output to log
            with open(log_file, "w") as f:
                f.write(f"Command: {command}\n")
                f.write(f"Exit code: {process.returncode}\n")
                f.write(f"=== STDOUT ===\n{stdout.decode(errors='replace')}\n")
                f.write(f"=== STDERR ===\n{stderr.decode(errors='replace')}\n")

            return {
                "success": process.returncode == 0,
                "exit_code": process.returncode,
                "content": stdout.decode(errors='replace'),
                "stderr": stderr.decode(errors='replace'),
                "metadata": {
                    "backend_fallback": True,
                    "log_file": str(log_file),
                    "attempts": attempts,
                }
            }

        except Exception as e:
            logger.error(f"[CodeAct] Backend fallback failed: {e}")
            return {
                "success": False,
                "content": f"Backend fallback execution failed: {str(e)}",
                "metadata": {"error": str(e)}
            }

    def _create_timeout_error(
        self,
        action_name: str,
        command: str,
        timeout: int,
        attempts: int
    ) -> dict:
        """Create a timeout error response."""

        error_msg = (
            f"ERROR: {action_name} timed out after {timeout} seconds "
            f"(attempt {attempts}/{MAX_ATTEMPTS}).\n"
        )

        if command:
            cmd_display = command[:200] if len(command) > 200 else command
            error_msg += f"Command: {cmd_display}\n\n"

            # Add suggestions
            if "pip install" in command:
                error_msg += (
                    "Suggestions:\n"
                    "1. Try running with: nohup pip install --no-cache-dir <packages> &\n"
                    "2. Install packages one by one\n"
                    "3. Use: pip install --timeout 1000 <packages>\n"
                    "4. Check network connectivity\n"
                )
            else:
                error_msg += (
                    "Suggestions:\n"
                    f"1. Run in background: nohup {cmd_display[:50]} > output.log 2>&1 &\n"
                    "2. Use timeout command: timeout 300 <command>\n"
                    "3. Break into smaller steps\n"
                )

        return {
            "success": False,
            "exit_code": 124,  # Standard timeout exit code
            "content": error_msg,
            "metadata": {
                "error": "timeout",
                "timeout_seconds": timeout,
                "attempts": attempts,
            }
        }