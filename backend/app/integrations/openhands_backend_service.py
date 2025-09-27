"""OpenHands Backend-as-Service Integration

This module provides integration with OpenHands by running the OpenHands backend
in a separate Docker container and communicating via API calls.

Architecture:
- UAgent spawns OpenHands backend container
- OpenHands backend manages its own runtime containers
- UAgent communicates with backend via REST API
- Backend handles Docker environment isolation properly
"""

import asyncio
import json
import logging
import subprocess
import time
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import docker

from dotenv import load_dotenv
from .docker_transition_manager import docker_transition_manager

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class OpenHandsBackendConfig:
    """Configuration for OpenHands backend service"""
    goal: str
    workspace: Path
    session_name: str = "exp"
    max_steps: int = 80
    max_minutes: int = 30
    backend_port: int = 3000
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None


@dataclass
class OpenHandsServiceResult:
    """Result from OpenHands backend service"""
    success: bool
    session_id: str
    duration_seconds: float
    final_json: Optional[Dict[str, Any]] = None
    backend_logs: str = ""
    error_message: str = ""


class OpenHandsBackendService:
    """Manages OpenHands backend as a Docker service"""

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize the service manager

        Args:
            repo_root: Root directory of the UAgent repository. If None, auto-detect.
        """
        if repo_root is None:
            # Auto-detect repo root
            integrations_dir = Path(__file__).resolve().parent
            self.repo_root = integrations_dir.parent.parent.parent
        else:
            self.repo_root = Path(repo_root).resolve()

        self.openhands_dir = self.repo_root / "OpenHands"
        if not self.openhands_dir.exists():
            raise RuntimeError(f"OpenHands directory not found at {self.openhands_dir}")

        self.docker_client = docker.from_env()
        self.backend_container = None

    def _prepare_workspace(self, cfg: OpenHandsBackendConfig) -> None:
        """Prepare the workspace with UAgent contract"""
        cfg.workspace.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (cfg.workspace / "logs").mkdir(exist_ok=True)
        (cfg.workspace / "code").mkdir(exist_ok=True)
        (cfg.workspace / "data").mkdir(exist_ok=True)
        (cfg.workspace / "experiments").mkdir(exist_ok=True)
        (cfg.workspace / "workspace").mkdir(exist_ok=True)

        # Write contract README
        readme_path = cfg.workspace / "README_UAGENT.md"
        readme_content = f"""# UAgent Experiment Contract

## Goal
{cfg.goal}

## Requirements
1. Write all results to `experiments/{cfg.session_name}/results/final.json`
2. The final.json must contain:
   - "success": boolean
   - "data": dict with experiment results
   - "analysis": dict with analysis results
   - "conclusions": list of conclusion strings
   - "errors": list of any errors encountered

3. Only read/write within workspace directories
4. Do not fabricate data - use only real computation results
5. Exit cleanly when complete

## Critical Final Step
When you complete this task, you MUST create a file at experiments/{cfg.session_name}/results/final.json
This file is mandatory for the experiment to be considered complete.
"""
        readme_path.write_text(readme_content)

    def _build_backend_environment(self, cfg: OpenHandsBackendConfig) -> Dict[str, str]:
        """Build environment variables for OpenHands backend container"""

        # Ensure Docker is ready
        if not docker_transition_manager.ensure_docker_ready():
            status = docker_transition_manager.get_transition_status()
            raise RuntimeError(f"Docker environment not ready: {status['recommended_action']}")

        # Get Docker configuration
        docker_config = docker_transition_manager.get_docker_config()

        env = {
            # OpenHands backend configuration
            "FRONTEND_PORT": "3001",  # Not used in backend-only mode
            "BACKEND_PORT": str(cfg.backend_port),
            "SANDBOX_RUNTIME_CONTAINER_IMAGE": docker_config.get(
                "SANDBOX_BASE_CONTAINER_IMAGE",
                "docker.all-hands.dev/all-hands-ai/runtime:0.57-nikolaik"
            ),
            "RUNTIME": "docker",
            "WORKSPACE_BASE": "/workspace",

            # LLM configuration
            "LLM_MODEL": cfg.llm_model or docker_config.get("LLM_MODEL", ""),
            "LLM_API_KEY": cfg.llm_api_key or docker_config.get("LLM_API_KEY", ""),
            "LLM_BASE_URL": cfg.llm_base_url or docker_config.get("LLM_BASE_URL", ""),

            # Debug and logging
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",

            # Docker socket access for runtime management
            "DOCKER_HOST": "unix:///var/run/docker.sock",
        }

        # Add other Docker config
        for key, value in docker_config.items():
            if key.startswith("SANDBOX_"):
                env[key] = value

        return env

    def start_backend(self, cfg: OpenHandsBackendConfig) -> str:
        """Start OpenHands backend container

        Returns:
            Container ID
        """
        logger.info("Starting OpenHands backend container...")

        # Prepare workspace
        self._prepare_workspace(cfg)

        # Build environment
        env = self._build_backend_environment(cfg)

        # Container configuration
        container_config = {
            "image": "ghcr.io/all-hands-ai/openhands:main",
            "environment": env,
            "ports": {f"{cfg.backend_port}/tcp": cfg.backend_port},
            "volumes": {
                "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
                str(cfg.workspace.resolve()): {"bind": "/workspace", "mode": "rw"},
            },
            "detach": True,
            "remove": True,  # Auto-remove when stopped
            "name": f"openhands-backend-{cfg.session_name}-{int(time.time())}",
        }

        try:
            self.backend_container = self.docker_client.containers.run(**container_config)
            container_id = self.backend_container.id

            logger.info(f"OpenHands backend started: {container_id}")

            # Wait for backend to be ready
            self._wait_for_backend_ready(cfg.backend_port)

            return container_id

        except Exception as e:
            logger.error(f"Failed to start OpenHands backend: {e}")
            raise

    def _wait_for_backend_ready(self, port: int, timeout: int = 60) -> None:
        """Wait for OpenHands backend to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/api/options/models", timeout=5)
                if response.status_code == 200:
                    logger.info("OpenHands backend is ready")
                    return
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)

        raise RuntimeError(f"OpenHands backend failed to start within {timeout} seconds")

    def submit_task(self, cfg: OpenHandsBackendConfig) -> str:
        """Submit task to OpenHands backend

        Returns:
            Session ID
        """
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

        # Submit task via API
        task_data = {
            "task": enhanced_goal,
            "max_iterations": cfg.max_steps,
            "confirmation_mode": False,
            "auto_continue": True,
        }

        try:
            response = requests.post(
                f"http://localhost:{cfg.backend_port}/api/conversations",
                json=task_data,
                timeout=30
            )
            response.raise_for_status()

            session_data = response.json()
            session_id = session_data.get("conversation_id")

            if not session_id:
                raise RuntimeError("No session ID returned from backend")

            logger.info(f"Task submitted to OpenHands backend: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to submit task to backend: {e}")
            raise

    def monitor_session(self, cfg: OpenHandsBackendConfig, session_id: str) -> OpenHandsServiceResult:
        """Monitor session until completion

        Returns:
            Service result with final status
        """
        start_time = time.time()
        timeout_seconds = cfg.max_minutes * 60

        logger.info(f"Monitoring session {session_id} for up to {cfg.max_minutes} minutes")

        while time.time() - start_time < timeout_seconds:
            try:
                # Check session status
                response = requests.get(
                    f"http://localhost:{cfg.backend_port}/api/conversations/{session_id}",
                    timeout=10
                )
                response.raise_for_status()

                session_data = response.json()
                status = session_data.get("status", "unknown")

                logger.debug(f"Session {session_id} status: {status}")

                # Check if completed
                if status in ["finished", "stopped", "error"]:
                    duration = time.time() - start_time

                    # Parse artifacts
                    final_json = self._parse_artifacts(cfg)

                    # Get backend logs
                    backend_logs = self._get_backend_logs()

                    success = status == "finished" and final_json is not None

                    return OpenHandsServiceResult(
                        success=success,
                        session_id=session_id,
                        duration_seconds=duration,
                        final_json=final_json,
                        backend_logs=backend_logs,
                        error_message="" if success else f"Session ended with status: {status}"
                    )

                time.sleep(5)  # Poll every 5 seconds

            except Exception as e:
                logger.error(f"Error monitoring session: {e}")
                time.sleep(5)

        # Timeout
        duration = time.time() - start_time
        backend_logs = self._get_backend_logs()

        return OpenHandsServiceResult(
            success=False,
            session_id=session_id,
            duration_seconds=duration,
            backend_logs=backend_logs,
            error_message=f"Session timed out after {cfg.max_minutes} minutes"
        )

    def _parse_artifacts(self, cfg: OpenHandsBackendConfig) -> Optional[Dict[str, Any]]:
        """Parse final.json artifacts from workspace"""
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

    def _get_backend_logs(self) -> str:
        """Get logs from backend container"""
        if not self.backend_container:
            return ""

        try:
            logs = self.backend_container.logs(tail=100).decode('utf-8')
            return logs
        except Exception as e:
            logger.error(f"Failed to get backend logs: {e}")
            return ""

    def stop_backend(self) -> None:
        """Stop OpenHands backend container"""
        if self.backend_container:
            try:
                logger.info("Stopping OpenHands backend container")
                self.backend_container.stop(timeout=10)
                self.backend_container = None
            except Exception as e:
                logger.error(f"Error stopping backend container: {e}")

    def run(self, cfg: OpenHandsBackendConfig) -> OpenHandsServiceResult:
        """Run complete OpenHands session with backend service

        Args:
            cfg: Configuration for the session

        Returns:
            Service result with final status
        """
        try:
            # Start backend
            container_id = self.start_backend(cfg)

            # Submit task
            session_id = self.submit_task(cfg)

            # Monitor until completion
            result = self.monitor_session(cfg, session_id)

            return result

        except Exception as e:
            error_msg = f"OpenHands backend service error: {str(e)}"
            logger.error(error_msg)

            return OpenHandsServiceResult(
                success=False,
                session_id="",
                duration_seconds=0,
                error_message=error_msg
            )

        finally:
            # Always clean up
            self.stop_backend()

    async def run_async(self, cfg: OpenHandsBackendConfig) -> OpenHandsServiceResult:
        """Async wrapper for run() method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, cfg)


# Convenience function for testing
def test_backend_service():
    """Test the backend service with a simple goal"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "test_workspace"

        cfg = OpenHandsBackendConfig(
            goal="Write a Python script that prints 'Hello from OpenHands Backend' and saves it to experiments/test/results/final.json with success=true",
            workspace=workspace,
            session_name="test",
            max_steps=15,
            max_minutes=5,
            backend_port=3000
        )

        service = OpenHandsBackendService()
        result = service.run(cfg)

        print(f"Success: {result.success}")
        print(f"Session ID: {result.session_id}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        if result.final_json:
            print(f"Final JSON: {json.dumps(result.final_json, indent=2)}")
        if result.error_message:
            print(f"Error: {result.error_message}")
        if result.backend_logs:
            print(f"Backend logs:\n{result.backend_logs[-500:]}")

        return result


if __name__ == "__main__":
    # Run test if executed directly
    test_backend_service()