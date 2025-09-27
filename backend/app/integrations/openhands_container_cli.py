"""OpenHands Container CLI Integration - Hybrid Approach

Run OpenHands CLI directly in a container with proper environment isolation.
This bypasses subprocess environment issues while keeping it simple.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import docker

from dotenv import load_dotenv
from .docker_transition_manager import docker_transition_manager

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ContainerCLIConfig:
    """Configuration for containerized OpenHands CLI"""
    goal: str
    workspace: Path
    session_name: str = "exp"
    max_steps: int = 80
    max_minutes: int = 30
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None


@dataclass
class ContainerCLIResult:
    """Result from containerized OpenHands CLI"""
    success: bool
    exit_code: int
    duration_seconds: float
    final_json: Optional[Dict[str, Any]] = None
    container_logs: str = ""
    error_message: str = ""


class OpenHandsContainerCLI:
    """Run OpenHands CLI in isolated container"""

    def __init__(self):
        self.docker_client = docker.from_env()

    def _prepare_workspace(self, cfg: ContainerCLIConfig) -> None:
        """Prepare workspace with UAgent contract"""
        cfg.workspace.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (cfg.workspace / "experiments").mkdir(exist_ok=True)

        # Write contract README
        readme_path = cfg.workspace / "README_UAGENT.md"
        readme_content = f"""# UAgent Experiment Contract

## Goal
{cfg.goal}

## CRITICAL: Final Result File
When you complete this task, you MUST create: experiments/{cfg.session_name}/results/final.json

Required structure:
{{
    "success": true/false,
    "data": {{...experiment results...}},
    "analysis": {{...analysis results...}},
    "conclusions": ["conclusion1", "conclusion2", ...],
    "errors": ["error1", "error2", ...]
}}

This file is MANDATORY for completion.
"""
        readme_path.write_text(readme_content)

    def run(self, cfg: ContainerCLIConfig) -> ContainerCLIResult:
        """Run OpenHands in isolated container"""
        start_time = time.time()

        # Prepare workspace
        self._prepare_workspace(cfg)

        # Ensure Docker is ready
        if not docker_transition_manager.ensure_docker_ready():
            status = docker_transition_manager.get_transition_status()
            return ContainerCLIResult(
                success=False,
                exit_code=-1,
                duration_seconds=0,
                error_message=f"Docker not ready: {status['recommended_action']}"
            )

        # Get Docker config
        docker_config = docker_transition_manager.get_docker_config()

        # Enhanced goal with explicit final.json requirement
        enhanced_goal = f"""{cfg.goal}

CRITICAL: When you complete this task, you MUST create a file at experiments/{cfg.session_name}/results/final.json with the following structure:
{{
    "success": true/false,
    "data": {{...experiment results...}},
    "analysis": {{...analysis results...}},
    "conclusions": ["conclusion1", "conclusion2", ...],
    "errors": ["error1", "error2", ...]
}}

Create the experiments/{cfg.session_name}/results/ directory if it doesn't exist. This final.json file is mandatory for the experiment to be considered complete."""

        # Container environment - completely isolated
        env = {
            # Completely disable conda/mamba
            "CONDA_PREFIX": "",
            "CONDA_DEFAULT_ENV": "",
            "MAMBA_EXE": "",
            "CONDA_SHLVL": "",

            # Use Poetry python exclusively
            "PYTHONPATH": "/openhands/code",
            "PATH": "/openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin:/usr/local/bin:/usr/bin:/bin",

            # LLM configuration
            "LLM_MODEL": cfg.llm_model or docker_config.get("LLM_MODEL", ""),
            "LLM_API_KEY": cfg.llm_api_key or docker_config.get("LLM_API_KEY", ""),
            "LLM_BASE_URL": cfg.llm_base_url or docker_config.get("LLM_BASE_URL", ""),

            # Runtime configuration
            "RUNTIME": "docker",
            "SANDBOX_BASE_CONTAINER_IMAGE": docker_config.get("SANDBOX_BASE_CONTAINER_IMAGE"),
            "SANDBOX_USE_HOST_NETWORK": "true",
            "SANDBOX_TIMEOUT": "600",

            # Debug
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
        }

        # OpenHands CLI command
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
                "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
                str(cfg.workspace.resolve()): {"bind": "/openhands/code/workspace", "mode": "rw"},
            },
            "working_dir": "/openhands/code/workspace",
            "detach": False,
            "remove": True,
            "stdout": True,
            "stderr": True,
        }

        logger.info(f"Running OpenHands CLI in container for {cfg.max_minutes} minutes")

        try:
            # Run container with timeout
            container = self.docker_client.containers.run(**container_config)

            # Container ran to completion, get results
            duration = time.time() - start_time

            # Get container logs
            logs = container.decode('utf-8') if isinstance(container, bytes) else str(container)

            # Parse artifacts
            final_json = self._parse_artifacts(cfg)

            # Determine success
            success = final_json is not None and final_json.get("success", False)

            return ContainerCLIResult(
                success=success,
                exit_code=0,
                duration_seconds=duration,
                final_json=final_json,
                container_logs=logs,
                error_message="" if success else "No valid final.json found"
            )

        except docker.errors.ContainerError as e:
            duration = time.time() - start_time

            return ContainerCLIResult(
                success=False,
                exit_code=e.exit_status,
                duration_seconds=duration,
                container_logs=e.stderr.decode('utf-8') if e.stderr else "",
                error_message=f"Container failed with exit code {e.exit_status}"
            )

        except Exception as e:
            duration = time.time() - start_time

            return ContainerCLIResult(
                success=False,
                exit_code=-1,
                duration_seconds=duration,
                error_message=f"Container execution error: {str(e)}"
            )

    def _parse_artifacts(self, cfg: ContainerCLIConfig) -> Optional[Dict[str, Any]]:
        """Parse final.json from workspace"""
        # Look for final.json in experiments directory
        exp_dir = cfg.workspace / "experiments" / cfg.session_name
        results_dir = exp_dir / "results"
        final_json_path = results_dir / "final.json"

        if final_json_path.exists():
            try:
                with open(final_json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to parse {final_json_path}: {e}")

        # Fallback: look for any final.json in experiments
        for path in (cfg.workspace / "experiments").rglob("final.json"):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to parse {path}: {e}")

        return None


# Test function
def test_container_cli():
    """Test the container CLI approach"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "test_workspace"

        cfg = ContainerCLIConfig(
            goal="Write a Python script that prints 'Hello Container CLI' and save result to experiments/test/results/final.json with success=true",
            workspace=workspace,
            session_name="test",
            max_steps=10,
            max_minutes=2
        )

        cli = OpenHandsContainerCLI()
        result = cli.run(cfg)

        print(f"Success: {result.success}")
        print(f"Exit code: {result.exit_code}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        if result.final_json:
            print(f"Final JSON: {json.dumps(result.final_json, indent=2)}")
        if result.error_message:
            print(f"Error: {result.error_message}")
        if result.container_logs:
            print(f"Container logs (last 300 chars):\n{result.container_logs[-300:]}")

        return result


if __name__ == "__main__":
    test_container_cli()