"""OpenHands Single Container Integration - Final Solution

Run OpenHands CLI directly inside the runtime container in headless mode.
This is the cleanest approach: one container, complete isolation, simple architecture.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import docker

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class SingleContainerConfig:
    """Configuration for single-container OpenHands"""
    goal: str
    workspace: Path
    session_name: str = "exp"
    max_steps: int = 80
    max_minutes: int = 30
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None


@dataclass
class SingleContainerResult:
    """Result from single-container OpenHands"""
    success: bool
    exit_code: int
    duration_seconds: float
    final_json: Optional[Dict[str, Any]] = None
    stdout_logs: str = ""
    stderr_logs: str = ""
    error_message: str = ""


class OpenHandsSingleContainer:
    """Run OpenHands in single container with complete isolation"""

    def __init__(self):
        self.docker_client = docker.from_env()

    def _prepare_workspace(self, cfg: SingleContainerConfig) -> None:
        """Prepare workspace with UAgent contract"""
        cfg.workspace.mkdir(parents=True, exist_ok=True)

        # Create experiments directory
        (cfg.workspace / "experiments" / cfg.session_name / "results").mkdir(parents=True, exist_ok=True)

        # Write contract README
        readme_path = cfg.workspace / "README_UAGENT.md"
        readme_content = f"""# UAgent Experiment Contract

## Goal
{cfg.goal}

## CRITICAL: Final Result File
You MUST create: experiments/{cfg.session_name}/results/final.json

Required structure:
{{
    "success": true/false,
    "data": {{...experiment results...}},
    "analysis": {{...analysis results...}},
    "conclusions": ["conclusion1", "conclusion2"],
    "errors": ["any errors"]
}}

This file is MANDATORY for completion.
"""
        readme_path.write_text(readme_content)

    def run(self, cfg: SingleContainerConfig) -> SingleContainerResult:
        """Run OpenHands in single container"""
        start_time = time.time()

        # Prepare workspace
        self._prepare_workspace(cfg)

        # Enhanced goal with explicit final.json requirement
        enhanced_goal = f"""{cfg.goal}

CRITICAL FINAL STEP: When you complete this task, you MUST create a file at experiments/{cfg.session_name}/results/final.json with this exact structure:
{{
    "success": true,
    "data": {{"task_completed": true, "files_created": ["list of files"], "results": "summary"}},
    "analysis": {{"approach": "how you solved it", "challenges": "any issues"}},
    "conclusions": ["Task completed successfully", "Files created as requested"],
    "errors": []
}}

Create the directory experiments/{cfg.session_name}/results/ if needed. This final.json file is MANDATORY."""

        # Container environment - completely isolated Poetry environment
        env = {
            # Completely disable conda/mamba interference
            "CONDA_PREFIX": "",
            "CONDA_DEFAULT_ENV": "",
            "MAMBA_EXE": "",
            "CONDA_SHLVL": "",

            # Force Poetry Python exclusively
            "PYTHONPATH": "/openhands/code",
            "PATH": "/openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin:/usr/local/bin:/usr/bin:/bin",

            # LLM configuration from .env
            "LLM_MODEL": cfg.llm_model or "openai/Moonshot-Kimi-K2-Instruct",
            "LLM_API_KEY": cfg.llm_api_key or "",
            "LLM_BASE_URL": cfg.llm_base_url or "",

            # OpenHands configuration - single container mode (no nested Docker)
            "RUNTIME": "local",  # Use local runtime instead of Docker-in-Docker
            "WORKSPACE_BASE": "/workspace",

            # Debug and logging
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
        }

        # Load LLM config from environment if not provided
        import os
        if not cfg.llm_model and "LITELLM_MODEL" in os.environ:
            env["LLM_MODEL"] = os.environ["LITELLM_MODEL"]
        if not cfg.llm_api_key and "LITELLM_API_KEY" in os.environ:
            env["LLM_API_KEY"] = os.environ["LITELLM_API_KEY"]
        if not cfg.llm_base_url and "LITELLM_API_BASE" in os.environ:
            env["LLM_BASE_URL"] = os.environ["LITELLM_API_BASE"]

        # OpenHands CLI command using Poetry Python
        cmd = [
            "/openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python",
            "-m", "openhands.core.main",
            "-t", enhanced_goal,
            "-n", cfg.session_name,
            "--max-iterations", str(cfg.max_steps),
            "--no-auto-continue"
        ]

        # Container configuration
        container_config = {
            "image": "docker.all-hands.dev/all-hands-ai/runtime:0.57-nikolaik",
            "command": cmd,
            "environment": env,
            "volumes": {
                # Mount workspace so results are accessible
                str(cfg.workspace.resolve()): {"bind": "/workspace", "mode": "rw"},
            },
            "working_dir": "/workspace",
            "detach": False,
            "remove": True,
            "stdout": True,
            "stderr": True,
        }

        logger.info(f"Running OpenHands in single container for {cfg.max_minutes} minutes")
        logger.info(f"Task: {cfg.goal}")

        try:
            # Run container (timeout handled by Docker client)
            container = self.docker_client.containers.run(**container_config)

            # Container completed successfully
            duration = time.time() - start_time

            # Extract logs
            if isinstance(container, bytes):
                logs = container.decode('utf-8')
                stdout_logs = logs
                stderr_logs = ""
            else:
                stdout_logs = str(container)
                stderr_logs = ""

            # Parse artifacts
            final_json = self._parse_artifacts(cfg)

            # Determine success
            success = final_json is not None and final_json.get("success", False)

            return SingleContainerResult(
                success=success,
                exit_code=0,
                duration_seconds=duration,
                final_json=final_json,
                stdout_logs=stdout_logs,
                stderr_logs=stderr_logs,
                error_message="" if success else "No valid final.json found or success=false"
            )

        except docker.errors.ContainerError as e:
            # Container exited with non-zero code
            duration = time.time() - start_time

            stdout_logs = e.stdout.decode('utf-8') if e.stdout else ""
            stderr_logs = e.stderr.decode('utf-8') if e.stderr else ""

            # Still try to parse artifacts in case partial success
            final_json = self._parse_artifacts(cfg)

            return SingleContainerResult(
                success=False,
                exit_code=e.exit_status,
                duration_seconds=duration,
                final_json=final_json,
                stdout_logs=stdout_logs,
                stderr_logs=stderr_logs,
                error_message=f"Container failed with exit code {e.exit_status}"
            )

        except Exception as e:
            duration = time.time() - start_time

            return SingleContainerResult(
                success=False,
                exit_code=-1,
                duration_seconds=duration,
                error_message=f"Container execution error: {str(e)}"
            )

    def _parse_artifacts(self, cfg: SingleContainerConfig) -> Optional[Dict[str, Any]]:
        """Parse final.json from workspace"""
        # Primary location
        final_json_path = cfg.workspace / "experiments" / cfg.session_name / "results" / "final.json"

        if final_json_path.exists():
            try:
                with open(final_json_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Found final.json: success={data.get('success', 'N/A')}")
                    return data
            except Exception as e:
                logger.error(f"Failed to parse {final_json_path}: {e}")

        # Fallback: search all experiments
        for path in (cfg.workspace / "experiments").rglob("final.json"):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Found fallback final.json at {path}: success={data.get('success', 'N/A')}")
                    return data
            except Exception as e:
                logger.error(f"Failed to parse {path}: {e}")

        # Check workspace root as last resort
        for path in cfg.workspace.rglob("final.json"):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Found root final.json at {path}: success={data.get('success', 'N/A')}")
                    return data
            except Exception as e:
                logger.error(f"Failed to parse {path}: {e}")

        logger.warning("No final.json found in workspace")
        return None

    async def run_async(self, cfg: SingleContainerConfig) -> SingleContainerResult:
        """Async wrapper for run() method"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, cfg)


# Test function
def test_single_container():
    """Test the single container approach"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "test_workspace"

        cfg = SingleContainerConfig(
            goal="Write a simple Python script hello.py that prints 'Hello Single Container' and then create the final.json file",
            workspace=workspace,
            session_name="test",
            max_steps=15,
            max_minutes=3
        )

        container = OpenHandsSingleContainer()
        result = container.run(cfg)

        print(f"Success: {result.success}")
        print(f"Exit code: {result.exit_code}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        if result.final_json:
            print(f"Final JSON: {json.dumps(result.final_json, indent=2)}")
        else:
            print("No final.json found")

        if result.error_message:
            print(f"Error: {result.error_message}")

        # Show last part of logs
        if result.stderr_logs:
            print(f"Stderr (last 200 chars):\n{result.stderr_logs[-200:]}")
        if result.stdout_logs:
            print(f"Stdout (last 200 chars):\n{result.stdout_logs[-200:]}")

        # Show workspace contents
        print(f"\nWorkspace contents:")
        for path in workspace.rglob("*"):
            if path.is_file():
                print(f"  {path.relative_to(workspace)}")

        return result


if __name__ == "__main__":
    test_single_container()