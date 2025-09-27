"""OpenHands V3 CodeAct Bridge - Headless subprocess execution for deterministic experiments"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from dotenv import load_dotenv
from .docker_transition_manager import docker_transition_manager

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class CodeActRunConfig:
    """Configuration for a headless OpenHands CodeAct run"""
    goal: str
    workspace: Path
    session_name: str = "exp"
    max_steps: int = 80
    max_minutes: int = 30
    disable_browser: bool = True
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None


@dataclass
class CodeActRunSummary:
    """Summary of a CodeAct run result"""
    success: bool
    exit_code: int
    duration_seconds: float
    artifact_path: Optional[Path] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    reason: str = ""
    final_json: Optional[Dict[str, Any]] = None


class OpenHandsCodeActBridgeV3:
    """Bridge to execute OpenHands CodeAct loop in headless subprocess mode"""

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize the bridge

        Args:
            repo_root: Root directory of the UAgent repository. If None, auto-detect.
        """
        if repo_root is None:
            # Auto-detect repo root
            integrations_dir = Path(__file__).resolve().parent
            self.repo_root = integrations_dir.parent.parent.parent  # backend/app/integrations -> root
        else:
            self.repo_root = Path(repo_root).resolve()

        self.openhands_dir = self.repo_root / "OpenHands"
        if not self.openhands_dir.exists():
            raise RuntimeError(f"OpenHands directory not found at {self.openhands_dir}")

        # Verify main.py exists
        main_path = self.openhands_dir / "openhands" / "core" / "main.py"
        if not main_path.exists():
            raise RuntimeError(f"OpenHands main.py not found at {main_path}")

    def _prepare_workspace(self, cfg: CodeActRunConfig) -> None:
        """Prepare the workspace with README contract"""
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

3. Only read/write within:
   - {cfg.workspace}/code/
   - {cfg.workspace}/data/
   - {cfg.workspace}/experiments/
   - {cfg.workspace}/workspace/

4. Do not fabricate data - use only real computation results
5. Exit cleanly when complete
"""
        readme_path.write_text(readme_content)

    def _build_environment(self, cfg: CodeActRunConfig) -> Dict[str, str]:
        """Build environment variables for subprocess"""
        env = os.environ.copy()

        # Add OpenHands to PYTHONPATH
        pythonpath = str(self.openhands_dir)
        if "PYTHONPATH" in env:
            pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath

        # Ensure Docker is ready for operations
        if not docker_transition_manager.ensure_docker_ready():
            status = docker_transition_manager.get_transition_status()
            logger.error(f"Docker not ready: {status}")
            raise RuntimeError(f"Docker environment not ready: {status['recommended_action']}")

        # Get optimized Docker configuration
        docker_config = docker_transition_manager.get_docker_config()
        env.update(docker_config)

        # Override conda/mamba environment to fully isolate Poetry
        env["CONDA_PREFIX"] = ""
        env["CONDA_DEFAULT_ENV"] = ""
        env["MAMBA_EXE"] = ""

        env["WORKSPACE_BASE"] = str(cfg.workspace.resolve())

        # Disable browser if requested
        if cfg.disable_browser:
            env["ENABLE_BROWSER"] = "false"

        # Enable detailed LLM logging
        env["DEBUG_LLM"] = "true"
        env["DEBUG_LLM_AUTO_CONFIRM"] = "true"  # Auto-confirm for headless operation
        env["LOG_TO_FILE"] = "true"
        env["LOG_LEVEL"] = "DEBUG"
        env["LLM_LOG_FILE"] = str(cfg.workspace / "logs" / "openhands_llm_interactions.log")

        # Map LLM configuration from .env file
        # Priority: explicit config > .env LITELLM_* > LLM_* env
        if cfg.llm_model:
            env["LLM_MODEL"] = cfg.llm_model
        elif "LITELLM_MODEL" in os.environ:
            env["LLM_MODEL"] = os.environ["LITELLM_MODEL"]
        elif "LLM_MODEL" in os.environ:
            env["LLM_MODEL"] = os.environ["LLM_MODEL"]

        if cfg.llm_api_key:
            env["LLM_API_KEY"] = cfg.llm_api_key
        elif "LITELLM_API_KEY" in os.environ:
            env["LLM_API_KEY"] = os.environ["LITELLM_API_KEY"]
        elif "LLM_API_KEY" in os.environ:
            env["LLM_API_KEY"] = os.environ["LLM_API_KEY"]

        if cfg.llm_base_url:
            env["LLM_BASE_URL"] = cfg.llm_base_url
        elif "LITELLM_API_BASE" in os.environ:
            env["LLM_BASE_URL"] = os.environ["LITELLM_API_BASE"]
        elif "LLM_BASE_URL" in os.environ:
            env["LLM_BASE_URL"] = os.environ["LLM_BASE_URL"]

        # Additional OpenHands settings from environment
        if "UAGENT_OPENHANDS_MAX_STEPS" in os.environ:
            cfg.max_steps = int(os.environ["UAGENT_OPENHANDS_MAX_STEPS"])
        if "UAGENT_OPENHANDS_MAX_MINUTES" in os.environ:
            cfg.max_minutes = int(os.environ["UAGENT_OPENHANDS_MAX_MINUTES"])

        return env

    def _build_command(self, cfg: CodeActRunConfig) -> List[str]:
        """Build the command to execute OpenHands"""
        python = sys.executable or "python"

        # Enhance goal with explicit final.json requirement
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

        cmd = [
            python,
            "-m", "openhands.core.main",
            "-t", enhanced_goal,
            "-n", cfg.session_name,
            "--max-iterations", str(cfg.max_steps),
            "--no-auto-continue"
        ]

        return cmd

    def _parse_artifacts(self, cfg: CodeActRunConfig) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
        """Parse artifacts from workspace after run"""
        # Look for final.json in experiments directory
        exp_dir = cfg.workspace / "experiments" / cfg.session_name
        results_dir = exp_dir / "results"
        final_json_path = results_dir / "final.json"

        if final_json_path.exists():
            try:
                with open(final_json_path, 'r') as f:
                    data = json.load(f)
                return final_json_path, data
            except Exception as e:
                logger.error(f"Failed to parse {final_json_path}: {e}")

        # Fallback: look for any final.json in experiments
        for path in (cfg.workspace / "experiments").rglob("final.json"):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return path, data
            except Exception as e:
                logger.error(f"Failed to parse {path}: {e}")

        return None, None

    def run(self, cfg: CodeActRunConfig) -> CodeActRunSummary:
        """Execute OpenHands CodeAct loop in subprocess

        Args:
            cfg: Configuration for the run

        Returns:
            CodeActRunSummary with results
        """
        start_time = time.time()

        # Prepare workspace
        self._prepare_workspace(cfg)

        # Build environment and command
        env = self._build_environment(cfg)
        cmd = self._build_command(cfg)

        # Log files
        stdout_log = cfg.workspace / "logs" / "openhands_stdout.log"
        stderr_log = cfg.workspace / "logs" / "openhands_stderr.log"
        llm_log = cfg.workspace / "logs" / "openhands_llm_interactions.log"

        logger.info(f"Executing OpenHands headless: {' '.join(cmd)}")
        logger.info(f"Workspace: {cfg.workspace}")
        logger.info(f"Max steps: {cfg.max_steps}, Max minutes: {cfg.max_minutes}")

        # Add connection retry settings to help with LocalRuntime
        env["SANDBOX_CONNECTION_TIMEOUT"] = "120"  # Give more time for connection
        env["SANDBOX_CONNECTION_RETRIES"] = "10"  # More retries
        env["SANDBOX_SERVER_STARTUP_TIMEOUT"] = "60"  # More time for server startup

        try:
            with open(stdout_log, 'w') as stdout_f, open(stderr_log, 'w') as stderr_f:
                # Start subprocess
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cfg.workspace),
                    env=env,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True
                )

                # Wait with timeout
                timeout_seconds = cfg.max_minutes * 60
                try:
                    exit_code = proc.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    logger.warning(f"OpenHands timed out after {cfg.max_minutes} minutes, terminating...")
                    proc.terminate()
                    time.sleep(5)  # Give it time to clean up
                    if proc.poll() is None:
                        proc.kill()  # Force kill if still running

                    duration = time.time() - start_time

                    # Read tail of logs
                    stdout_tail = self._read_tail(stdout_log, 1000)
                    stderr_tail = self._read_tail(stderr_log, 500)

                    return CodeActRunSummary(
                        success=False,
                        exit_code=-1,
                        duration_seconds=duration,
                        stdout_tail=stdout_tail,
                        stderr_tail=stderr_tail,
                        reason=f"Timeout after {cfg.max_minutes} minutes"
                    )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to execute OpenHands: {e}")

            # Create standardized error report
            error_report = docker_transition_manager.create_error_report(e, cfg.workspace)

            return CodeActRunSummary(
                success=False,
                exit_code=-1,
                duration_seconds=duration,
                reason=f"Docker execution error: {str(e)}",
                final_json=error_report
            )

        # Process completed
        duration = time.time() - start_time

        # Read tail of logs
        stdout_tail = self._read_tail(stdout_log, 1000)
        stderr_tail = self._read_tail(stderr_log, 500)

        # Parse artifacts
        artifact_path, final_json = self._parse_artifacts(cfg)

        # Determine success
        if exit_code == 0 and artifact_path:
            logger.info(f"OpenHands completed successfully in {duration:.1f}s")
            return CodeActRunSummary(
                success=True,
                exit_code=exit_code,
                duration_seconds=duration,
                artifact_path=artifact_path,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                final_json=final_json,
                reason="Completed successfully"
            )
        else:
            reason = f"Exit code {exit_code}"
            if not artifact_path:
                reason += ", no final.json found"
            logger.warning(f"OpenHands failed: {reason}")
            return CodeActRunSummary(
                success=False,
                exit_code=exit_code,
                duration_seconds=duration,
                artifact_path=artifact_path,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                final_json=final_json,
                reason=reason
            )

    def _read_tail(self, log_path: Path, lines: int = 100) -> str:
        """Read tail of log file"""
        if not log_path.exists():
            return ""

        try:
            with open(log_path, 'r') as f:
                content = f.read()
                lines_list = content.splitlines()
                if len(lines_list) > lines:
                    return "\n".join(lines_list[-lines:])
                return content
        except Exception as e:
            logger.error(f"Failed to read {log_path}: {e}")
            return ""

    async def run_async(self, cfg: CodeActRunConfig) -> CodeActRunSummary:
        """Async wrapper for run() method

        Args:
            cfg: Configuration for the run

        Returns:
            CodeActRunSummary with results
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, cfg)


# Convenience function for quick testing
def test_bridge():
    """Test the bridge with a simple goal"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "test_workspace"

        cfg = CodeActRunConfig(
            goal="Write a Python script that prints 'Hello from OpenHands' and saves it to experiments/test/results/final.json with success=true",
            workspace=workspace,
            session_name="test",
            max_steps=10,
            max_minutes=2,
            disable_browser=True
        )

        bridge = OpenHandsCodeActBridgeV3()
        result = bridge.run(cfg)

        print(f"Success: {result.success}")
        print(f"Exit code: {result.exit_code}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Artifact: {result.artifact_path}")
        if result.final_json:
            print(f"Final JSON: {json.dumps(result.final_json, indent=2)}")
        if result.stdout_tail:
            print(f"Stdout tail:\n{result.stdout_tail[-500:]}")
        if result.stderr_tail:
            print(f"Stderr tail:\n{result.stderr_tail[-500:]}")

        return result


if __name__ == "__main__":
    # Run test if executed directly
    test_bridge()