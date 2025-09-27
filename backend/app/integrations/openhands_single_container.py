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

    def _prepare_workspace_with_permissions(self, cfg: SingleContainerConfig) -> None:
        """Prepare workspace with proper permissions for container access"""
        import os
        import stat

        try:
            # First, ensure the workspace exists
            cfg.workspace.mkdir(parents=True, exist_ok=True)

            # Create subdirectories with proper permissions
            subdirs = ["experiments", "logs", "code", "data", "workspace"]
            for subdir in subdirs:
                dir_path = cfg.workspace / subdir
                dir_path.mkdir(exist_ok=True)
                # Set full permissions for current user and group
                os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            # Create experiment-specific directory
            exp_dir = cfg.workspace / "experiments" / cfg.session_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(exp_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            # Create results directory
            results_dir = exp_dir / "results"
            results_dir.mkdir(exist_ok=True)
            os.chmod(results_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            # Create contract README with permissions
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
            os.chmod(readme_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH)

            # Set workspace root permissions
            os.chmod(cfg.workspace, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            logger.info(f"Workspace prepared with full permissions: {cfg.workspace}")

        except Exception as e:
            logger.error(f"Failed to prepare workspace with permissions: {e}")
            raise

    def _prepare_container_directories(self, cfg: SingleContainerConfig) -> Dict[str, Any]:
        """Prepare container volume mappings with proper directories"""
        import os
        import tempfile

        # Get current user info for permission mapping
        current_uid = os.getuid()
        current_gid = os.getgid()

        # Create temporary directory for OpenHands internal directories
        openhands_temp = tempfile.mkdtemp(prefix="openhands_internal_")

        # Create necessary OpenHands directories that it expects to write to
        os.makedirs(f"{openhands_temp}/logs", mode=0o777, exist_ok=True)
        os.makedirs(f"{openhands_temp}/cache", mode=0o777, exist_ok=True)
        os.makedirs(f"{openhands_temp}/tmp", mode=0o777, exist_ok=True)
        os.makedirs(f"{openhands_temp}/home", mode=0o777, exist_ok=True)

        # Set proper ownership and permissions
        for root, dirs, files in os.walk(openhands_temp):
            for d in dirs:
                dir_path = os.path.join(root, d)
                os.chmod(dir_path, 0o777)
                try:
                    os.chown(dir_path, current_uid, current_gid)
                except PermissionError:
                    pass  # Skip if we can't change ownership

        return {
            "volumes": {
                "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
                str(cfg.workspace.resolve()): {"bind": "/workspace", "mode": "rw"},
                f"{openhands_temp}/logs": {"bind": "/openhands/code/logs", "mode": "rw"},
                f"{openhands_temp}/cache": {"bind": "/openhands/code/cache", "mode": "rw"},
                f"{openhands_temp}/tmp": {"bind": "/tmp/openhands", "mode": "rw"},
                f"{openhands_temp}/home": {"bind": "/tmp/openhands_home", "mode": "rw"},
            },
            "temp_dir": openhands_temp,
            "user": f"{current_uid}:{current_gid}"
        }

    def run(self, cfg: SingleContainerConfig) -> SingleContainerResult:
        """Run OpenHands in single container"""
        start_time = time.time()

        # Prepare workspace with proper permissions
        self._prepare_workspace_with_permissions(cfg)

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

            # User configuration for proper permissions
            "HOME": "/tmp/openhands_home",  # Temporary home directory
            "USER": "containeruser",

            # Disable Docker-in-Docker to avoid permission conflicts
            "SANDBOX_USE_HOST_NETWORK": "false",
            "SANDBOX_TIMEOUT": "300",

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

        # Create live monitoring directory
        monitoring_dir = cfg.workspace / "logs" / "openhands_live"
        monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Prepare container directories with proper permissions
        container_dirs = self._prepare_container_directories(cfg)

        # Container configuration with live monitoring and correct user permissions
        container_config = {
            "image": "docker.all-hands.dev/all-hands-ai/runtime:0.57-nikolaik",
            "command": cmd,
            "environment": env,
            "volumes": {
                # Use the prepared volumes that include OpenHands internal directories
                **container_dirs["volumes"],
                # Add monitoring directory for live logs
                str(monitoring_dir.resolve()): {"bind": "/openhands/live_logs", "mode": "rw"},
            },
            "working_dir": "/workspace",
            "user": container_dirs["user"],  # Use prepared user mapping
            "detach": True,  # Changed to detach so we can monitor
            "remove": False,  # Keep container for debugging
            "stdout": True,
            "stderr": True,
        }

        logger.info(f"Running OpenHands in single container for {cfg.max_minutes} minutes")
        logger.info(f"Task: {cfg.goal}")

        try:
            # Start container in detached mode
            container = self.docker_client.containers.run(**container_config)
            container_id = container.id

            logger.info(f"OpenHands container started: {container_id}")
            logger.info(f"Live monitoring at: {monitoring_dir}")

            # Create monitoring summary file for user
            self._create_monitoring_summary(monitoring_dir, container_id, cfg)

            # Start real-time monitoring
            result = self._monitor_container_with_streaming(container, cfg, start_time, monitoring_dir)

            # Update final monitoring summary
            self._update_monitoring_summary(monitoring_dir, result)

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to start container: {e}")

            return SingleContainerResult(
                success=False,
                exit_code=-1,
                duration_seconds=duration,
                error_message=f"Container startup error: {str(e)}"
            )

        finally:
            # Clean up temporary directories
            try:
                import shutil
                if 'container_dirs' in locals() and 'temp_dir' in container_dirs:
                    shutil.rmtree(container_dirs['temp_dir'], ignore_errors=True)
                    logger.debug(f"Cleaned up temporary directory: {container_dirs['temp_dir']}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")

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

    def _monitor_container_with_streaming(self, container, cfg: SingleContainerConfig, start_time: float, monitoring_dir: Path) -> SingleContainerResult:
        """Monitor container with real-time log streaming"""
        import threading
        import time

        # Create live log files
        live_stdout = monitoring_dir / "live_stdout.log"
        live_stderr = monitoring_dir / "live_stderr.log"
        live_combined = monitoring_dir / "live_combined.log"
        container_status = monitoring_dir / "container_status.json"

        # Initialize log files
        live_stdout.write_text("")
        live_stderr.write_text("")
        live_combined.write_text("")

        timeout_seconds = cfg.max_minutes * 60
        full_stdout = ""
        full_stderr = ""

        def stream_logs():
            """Stream container logs to files in real-time"""
            nonlocal full_stdout, full_stderr

            try:
                # Stream logs with timestamps
                for log_line in container.logs(stream=True, follow=True, stdout=True, stderr=True, timestamps=True):
                    log_str = log_line.decode('utf-8', errors='replace')
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                    # Append to combined log with timestamp
                    with open(live_combined, 'a', encoding='utf-8') as f:
                        f.write(f"[{timestamp}] {log_str}")
                        f.flush()

                    # Separate stdout/stderr (Docker doesn't distinguish in combined stream)
                    if any(keyword in log_str.lower() for keyword in ['error', 'exception', 'traceback', 'stderr']):
                        full_stderr += log_str
                        with open(live_stderr, 'a', encoding='utf-8') as f:
                            f.write(f"[{timestamp}] {log_str}")
                            f.flush()
                    else:
                        full_stdout += log_str
                        with open(live_stdout, 'a', encoding='utf-8') as f:
                            f.write(f"[{timestamp}] {log_str}")
                            f.flush()

            except Exception as e:
                logger.error(f"Log streaming error: {e}")

        # Start log streaming in background thread
        log_thread = threading.Thread(target=stream_logs, daemon=True)
        log_thread.start()

        # Monitor container status
        try:
            # Wait for container to complete or timeout
            exit_code = container.wait(timeout=timeout_seconds)['StatusCode']
            duration = time.time() - start_time

            # Update status file
            status_info = {
                "container_id": container.id,
                "exit_code": exit_code,
                "duration_seconds": duration,
                "status": "completed",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(container_status, 'w') as f:
                json.dump(status_info, f, indent=2)

            # Give log streaming a moment to finish
            time.sleep(2)

            # Parse artifacts
            final_json = self._parse_artifacts(cfg)

            # Clean up container
            try:
                container.remove()
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")

            # Determine success
            success = exit_code == 0 and final_json is not None and final_json.get("success", False)

            return SingleContainerResult(
                success=success,
                exit_code=exit_code,
                duration_seconds=duration,
                final_json=final_json,
                stdout_logs=full_stdout,
                stderr_logs=full_stderr,
                error_message="" if success else f"Exit code {exit_code}" + ("" if final_json else ", no final.json found")
            )

        except Exception as e:
            duration = time.time() - start_time

            # Update status file with error
            status_info = {
                "container_id": container.id,
                "error": str(e),
                "duration_seconds": duration,
                "status": "error",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(container_status, 'w') as f:
                json.dump(status_info, f, indent=2)

            # Try to clean up
            try:
                container.kill()
                container.remove()
            except Exception as cleanup_e:
                logger.warning(f"Failed to cleanup container: {cleanup_e}")

            return SingleContainerResult(
                success=False,
                exit_code=-1,
                duration_seconds=duration,
                stdout_logs=full_stdout,
                stderr_logs=full_stderr,
                error_message=f"Container monitoring error: {str(e)}"
            )

    def _create_monitoring_summary(self, monitoring_dir: Path, container_id: str, cfg: SingleContainerConfig) -> None:
        """Create initial monitoring summary for user"""
        summary_file = monitoring_dir / "README_MONITORING.md"

        summary_content = f"""# OpenHands Live Monitoring

## Container Information
- **Container ID**: `{container_id}`
- **Session**: `{cfg.session_name}`
- **Max Duration**: {cfg.max_minutes} minutes
- **Max Steps**: {cfg.max_steps}
- **Started**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Live Log Files
- 📄 **live_combined.log** - All OpenHands output with timestamps
- 📋 **live_stdout.log** - Standard output (commands, results)
- ⚠️ **live_stderr.log** - Error output and warnings
- 📊 **container_status.json** - Current container status

## Monitoring Commands
```bash
# Watch live combined output
tail -f {monitoring_dir / "live_combined.log"}

# Watch just errors
tail -f {monitoring_dir / "live_stderr.log"}

# Check container status
cat {monitoring_dir / "container_status.json"}

# Check container directly
docker logs {container_id} -f
```

## Workspace Location
- **Host Path**: `{cfg.workspace.resolve()}`
- **Container Path**: `/workspace`
- **Experiments**: `{cfg.workspace.resolve() / "experiments"}`
- **Results**: `{cfg.workspace.resolve() / "experiments" / cfg.session_name / "results"}`

## Goal
```
{cfg.goal[:500]}{"..." if len(cfg.goal) > 500 else ""}
```

---
*This file is updated in real-time during execution*
"""

        summary_file.write_text(summary_content)

    def _update_monitoring_summary(self, monitoring_dir: Path, result: SingleContainerResult) -> None:
        """Update monitoring summary with final results"""
        summary_file = monitoring_dir / "README_MONITORING.md"

        final_summary = f"""

## Final Results
- **Success**: {'✅ YES' if result.success else '❌ NO'}
- **Exit Code**: {result.exit_code}
- **Duration**: {result.duration_seconds:.1f} seconds
- **Final JSON**: {'✅ Found' if result.final_json else '❌ Missing'}
- **Completed**: {time.strftime('%Y-%m-%d %H:%M:%S')}

### Error Message
```
{result.error_message or 'None'}
```

### Final JSON Preview
```json
{json.dumps(result.final_json, indent=2)[:1000] if result.final_json else 'Not found'}{"..." if result.final_json and len(str(result.final_json)) > 1000 else ""}
```
"""

        # Append to existing summary
        with open(summary_file, 'a') as f:
            f.write(final_summary)

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