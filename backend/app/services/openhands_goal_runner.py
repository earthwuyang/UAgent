"""OpenHands Goal Delegation Runner - Delegates entire goals to OpenHands CLI.

This runner passes complete research goals to OpenHands CLI, letting it handle
all planning and execution autonomously, avoiding repetitive step-by-step interactions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.llm_client import LLMClient
from ..integrations.openhands_runtime import OpenHandsActionServerRunner

logger = logging.getLogger(__name__)


@dataclass
class OpenHandsGoalResult:
    """Result from delegating a complete goal to OpenHands."""
    success: bool
    message: str
    artifacts: Dict[str, Any]
    logs: str
    workspace_path: Path


class OpenHandsGoalRunner:
    """Delegates entire research goals to OpenHands CLI for autonomous execution."""

    def __init__(
        self,
        openhands_binary: str = "openhands",  # or uvx path
        max_iterations: int = 50,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        self.openhands_binary = openhands_binary
        self.max_iterations = max_iterations
        self.model = model
        self.logger = logging.getLogger(__name__)

    async def run_goal(
        self,
        workspace_path: Path,
        goal: str,
        progress_cb: Optional[Any] = None,
    ) -> OpenHandsGoalResult:
        """Delegate an entire goal to OpenHands CLI.

        Args:
            workspace_path: Directory where OpenHands should work
            goal: Complete research goal description
            progress_cb: Optional callback for progress updates

        Returns:
            OpenHandsGoalResult with success status, outputs, and artifacts
        """
        workspace_path = Path(workspace_path).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Prepare OpenHands CLI command
        # Using programmatic invocation instead of subprocess for better control
        cmd = [
            self.openhands_binary,
            "--task", goal,
            "--workspace", str(workspace_path),
            "--model", self.model,
            "--max-iterations", str(self.max_iterations),
            "--non-interactive",  # Run without user interaction
            "--json-output",  # Get structured output
        ]

        # Set environment variables
        env = os.environ.copy()
        env["OPENHANDS_WORKSPACE"] = str(workspace_path)

        if progress_cb:
            await progress_cb("delegating_to_openhands", {"goal": goal})

        try:
            # Run OpenHands CLI with complete goal
            self.logger.info(f"Delegating goal to OpenHands CLI: {goal[:100]}...")

            # For async execution with streaming output
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workspace_path),
            )

            # Collect output while streaming progress
            stdout_lines = []
            stderr_lines = []

            async def read_stream(stream, lines_list, stream_type):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8', errors='replace').rstrip()
                    lines_list.append(line_str)

                    # Send progress updates
                    if progress_cb and line_str:
                        try:
                            # Parse JSON progress if available
                            if line_str.startswith('{'):
                                progress_data = json.loads(line_str)
                                await progress_cb("openhands_progress", progress_data)
                            else:
                                await progress_cb("openhands_output", {
                                    "type": stream_type,
                                    "line": line_str
                                })
                        except:
                            pass

            # Read both streams concurrently
            await asyncio.gather(
                read_stream(process.stdout, stdout_lines, "stdout"),
                read_stream(process.stderr, stderr_lines, "stderr"),
            )

            # Wait for process completion
            returncode = await process.wait()

            stdout_text = "\n".join(stdout_lines)
            stderr_text = "\n".join(stderr_lines)

            # Parse final result from JSON output
            artifacts = {}
            success_message = ""

            try:
                # Look for JSON result in output
                for line in reversed(stdout_lines):
                    if line.startswith('{"result":'):
                        result_data = json.loads(line)
                        artifacts = result_data.get("artifacts", {})
                        success_message = result_data.get("message", "")
                        break
            except:
                pass

            # Collect generated files as artifacts
            code_dir = workspace_path / "code"
            if code_dir.exists():
                artifacts["generated_files"] = [
                    str(f.relative_to(workspace_path))
                    for f in code_dir.rglob("*")
                    if f.is_file()
                ]

            success = returncode == 0

            if progress_cb:
                await progress_cb("completed", {
                    "success": success,
                    "artifacts": artifacts
                })

            return OpenHandsGoalResult(
                success=success,
                message=success_message or ("Goal completed successfully" if success else "Goal execution failed"),
                artifacts=artifacts,
                logs=f"STDOUT:\n{stdout_text}\n\nSTDERR:\n{stderr_text}",
                workspace_path=workspace_path,
            )

        except Exception as e:
            self.logger.error(f"Failed to delegate goal to OpenHands: {e}")
            return OpenHandsGoalResult(
                success=False,
                message=f"OpenHands delegation failed: {str(e)}",
                artifacts={},
                logs=str(e),
                workspace_path=workspace_path,
            )

    async def run_with_openhands_cli_api(
        self,
        workspace_path: Path,
        goal: str,
        progress_cb: Optional[Any] = None,
    ) -> OpenHandsGoalResult:
        """Alternative: Use OpenHands Python API directly instead of CLI subprocess.

        This approach integrates more tightly with OpenHands internals.
        """
        from openhands.cli.main import run_agent_until_done
        from openhands.core.setup import (
            create_agent,
            create_controller,
            create_runtime,
            generate_sid,
        )
        from openhands.core.config import OpenHandsConfig

        # Configure OpenHands
        config = OpenHandsConfig(
            workspace_path=str(workspace_path),
            model=self.model,
            max_iterations=self.max_iterations,
        )

        # Create runtime and agent
        sid = generate_sid()
        runtime = await create_runtime(config, sid)
        agent = create_agent(config)
        controller = create_controller(agent, runtime)

        try:
            # Initialize workspace
            await runtime.initialize(str(workspace_path))

            # Run the agent with the complete goal
            final_state = await run_agent_until_done(
                controller,
                initial_prompt=goal,
                max_iterations=self.max_iterations,
            )

            # Extract results
            success = final_state.exit_reason == "success"
            message = final_state.summary or "Task completed"

            # Collect artifacts
            artifacts = {
                "final_state": final_state.to_dict(),
                "events": len(final_state.history),
            }

            return OpenHandsGoalResult(
                success=success,
                message=message,
                artifacts=artifacts,
                logs=str(final_state),
                workspace_path=workspace_path,
            )

        finally:
            await runtime.close()


# Integration with existing UAgent system
async def delegate_scientific_research_to_openhands(
    goal: str,
    workspace_path: Path,
    progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    """Delegate a complete scientific research goal to OpenHands.

    This replaces the step-by-step CodeActRunner approach with full delegation.
    """
    runner = OpenHandsGoalRunner()
    result = await runner.run_goal(workspace_path, goal, progress_cb)

    return {
        "success": result.success,
        "message": result.message,
        "artifacts": result.artifacts,
        "workspace_path": str(result.workspace_path),
        "approach": "full_delegation",
    }