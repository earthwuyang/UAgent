"""Patch for send_action method with adaptive retry and background execution."""

# This is the improved send_action method to replace the existing one

async def send_action(self, action_dict: dict, timeout: int = DEFAULT_ACTION_TIMEOUT, retry_with_yes: bool = True) -> 'OpenHandsActionResult':
    """Execute action with adaptive retry, increasing timeouts, and backend fallback."""

    if not self.is_running:
        raise RuntimeError("OpenHands action server session is not running")

    # Work on a shallow copy of the action payload
    action_name = action_dict.get("action")
    args = dict(action_dict.get("args", {}))

    # [Previous path remapping logic stays the same - lines 136-186]
    # ... (keeping existing path remapping code)

    # Prepare payload
    rewritten_action = dict(action_dict)
    rewritten_action["args"] = args
    if action_name and rewritten_action.get("action") != action_name:
        rewritten_action["action"] = action_name
    payload = {"action": rewritten_action}

    # Log the action
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

    # Attempt execution with adaptive timeout handling for long-running commands
    response = None
    max_attempts = OPENHANDS_MAX_RUN_ATTEMPTS if action_name == "run" else 1
    attempt = 1
    current_timeout = max(1, timeout)

    def _resolve_timeout(base_timeout: int) -> int:
        """Calculate appropriate timeout based on command type."""
        actual = max(1, base_timeout)
        if action_name == "run" and "command" in args:
            cmd_local = args["command"]

            # Docker commands
            docker_cmds = ["docker stop", "docker-compose down", "docker-compose stop", "docker kill", "docker rm"]
            for docker_cmd in docker_cmds:
                if docker_cmd in cmd_local:
                    actual = max(20, base_timeout // 3)
                    logger.info("[CodeAct] Using docker timeout (%ss) for docker command", actual)
                    break
            else:
                # Package manager commands
                package_cmds = [
                    "apt-get", "apt ", "yum ", "dnf ", "zypper", "pacman", "emerge",
                    "conda install", "conda update", "npm install", "yarn add", "brew install",
                    "pip install", "pip3 install"
                ]
                for pkg_cmd in package_cmds:
                    if pkg_cmd in cmd_local:
                        actual = max(base_timeout, OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT)
                        logger.info(
                            "[CodeAct] Using extended timeout (%ss) for package manager command",
                            actual,
                        )
                        break
        return int(actual)

    # Package manager retry flags
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

    while attempt <= max_attempts:
        attempt_timeout = _resolve_timeout(current_timeout)

        try:
            response = await self._post_action(payload, attempt_timeout)
            break

        except (httpx.TimeoutException, asyncio.TimeoutError) as timeout_exc:
            # Try adding non-interactive flags first
            if action_name == "run" and retry_with_yes and "command" in args:
                cmd = args["command"]
                updated_cmd = cmd

                for pm, flag in package_managers:
                    if pm in cmd and flag not in cmd:
                        logger.warning(
                            "[CodeAct] Timeout detected for %s. Retrying with %s flag",
                            pm, flag
                        )

                        # Add non-interactive flag
                        if any(token in pm for token in ("pip", "conda", "npm", "yarn", "brew")):
                            updated_cmd = cmd.replace(pm, f"{pm} {flag}")
                        else:
                            if f"{pm} install" in cmd:
                                updated_cmd = cmd.replace(f"{pm} install", f"{pm} install {flag}")
                            elif f"{pm} upgrade" in cmd:
                                updated_cmd = cmd.replace(f"{pm} upgrade", f"{pm} upgrade {flag}")
                            elif f"{pm} update" in cmd:
                                updated_cmd = cmd.replace(f"{pm} update", f"{pm} update {flag}")
                            elif f"{pm} remove" in cmd:
                                updated_cmd = cmd.replace(f"{pm} remove", f"{pm} remove {flag}")
                            else:
                                updated_cmd = cmd.replace(pm, f"{pm} {flag}", 1)
                        break

                if updated_cmd != cmd:
                    args["command"] = updated_cmd
                    rewritten_action["args"] = args
                    payload = {"action": rewritten_action}
                    logger.info("[CodeAct] Retrying with non-interactive command: %s", updated_cmd[:200])
                    continue

            # Adaptive timeout retry
            if attempt < max_attempts and action_name == "run":
                attempt += 1
                new_base = max(current_timeout, attempt_timeout)
                current_timeout = min(
                    int(new_base * OPENHANDS_RUN_ADAPTIVE_MULTIPLIER),
                    OPENHANDS_MAX_ACTION_TIMEOUT,
                )
                logger.warning(
                    "[CodeAct] Command timed out after %ss (attempt %s/%s). Increasing timeout to %ss",
                    attempt_timeout,
                    attempt - 1,
                    max_attempts,
                    current_timeout,
                )
                continue

            # On final timeout for run commands, try backend fallback
            if action_name == "run" and "command" in args:
                logger.error(
                    "[CodeAct] Command timed out after %ss on attempt %s/%s; falling back to backend execution",
                    attempt_timeout,
                    attempt,
                    max_attempts,
                )
                return await self._execute_with_backend_fallback(
                    args["command"],
                    attempt_timeout,
                    attempt,
                )

            # Return timeout error for non-run actions
            from ..core.openhands.code_executor import ExecutionResult
            logger.error(
                "[CodeAct] Action %s timed out after %ss",
                action_name,
                attempt_timeout,
            )
            return OpenHandsActionResult(
                execution_result=ExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Action {action_name} timed out after {attempt_timeout} seconds",
                    execution_time=float(attempt_timeout),
                    files_created=[],
                    files_modified=[],
                    command="",
                    working_directory=str(self._workspace_path),
                    env={},
                ),
                raw_observation={
                    "error": "timeout",
                    "timeout_seconds": attempt_timeout,
                    "attempt": attempt,
                },
                stdout="",
                stderr=f"Action {action_name} timed out after {attempt_timeout} seconds",
                server_logs=f"Timeout error: Action did not complete within {attempt_timeout} seconds",
            )

    if response is None:
        raise RuntimeError("Failed to obtain response from action server")

    # [Rest of the response handling logic stays the same]
    # ... (keeping existing response handling code)