"""Code Executor for safe code execution in isolated workspaces"""

import asyncio
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
import logging

from .workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    files_created: List[str]
    files_modified: List[str]
    command: str = ""
    working_directory: str = "."
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionCommand:
    """Command to execute in workspace"""
    command: str
    working_directory: str = "."
    timeout: int = 300  # 5 minutes
    capture_output: bool = True
    shell: bool = True
    env_vars: Dict[str, str] = None


class CodeExecutor:
    """Executes code safely in isolated workspaces"""

    def __init__(self, workspace_manager: WorkspaceManager):
        """Initialize code executor

        Args:
            workspace_manager: WorkspaceManager instance for workspace operations
        """
        self.workspace_manager = workspace_manager
        self.allowed_commands = {
            # Python and package managers
            "python", "python3", "pip", "pip3", "ipython", "jupyter",
            # Shell and file ops
            "bash", "sh", "ls", "cat", "head", "tail", "grep", "find", "sed", "awk",
            "mkdir", "touch", "cp", "mv", "rm", "tar", "unzip",
            # VCS and network fetch
            "git", "wget", "curl",
            # Build tools
            "make", "cmake", "ninja", "gcc", "g++", "clang",
            # Node ecosystem (if needed)
            "npm", "node", "yarn",
        }
        self.forbidden_patterns = [
            # keep destructive or privilege-escalation patterns blocked
            "rm -rf /", "sudo", "su",
            # absolute sensitive paths
            "/etc/",
            # path traversal
            "../../", "../etc",
            # disk formatting
            "format", "fdisk", "chmod +x",
        ]

    def _is_command_safe(self, command: str) -> bool:
        """Check if a command is safe to execute

        Args:
            command: Command string to check

        Returns:
            bool: True if command is considered safe
        """
        command_lower = command.lower()

        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern in command_lower:
                logger.warning(f"Forbidden pattern '{pattern}' found in command: {command}")
                return False

        # Extract base command (first word)
        base_command = command.strip().split()[0] if command.strip() else ""
        base_command = base_command.split("/")[-1]  # Remove path

        # Check if base command is allowed
        if base_command not in self.allowed_commands:
            logger.warning(f"Command '{base_command}' not in allowed commands")
            return False

        return True

    async def execute_python_code(
        self,
        workspace_id: str,
        code: str,
        file_name: str = None,
        timeout: int = 300
    ) -> ExecutionResult:
        """Execute Python code in workspace

        Args:
            workspace_id: ID of the workspace
            code: Python code to execute
            file_name: Optional filename to save code as
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult: Result of execution
        """
        if not file_name:
            file_name = f"generated_code_{asyncio.get_event_loop().time():.0f}.py"

        # Write code to workspace
        code_path = f"code/{file_name}"
        if not await self.workspace_manager.write_file(workspace_id, code_path, code):
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Failed to write code file",
                execution_time=0.0,
                files_created=[],
                files_modified=[],
                command=f"write:{code_path}",
                working_directory=".",
                env={}
            )

        # Execute the code
        command = ExecutionCommand(
            command=f"python3 code/{file_name}",
            timeout=timeout
        )

        return await self.execute_command(workspace_id, command)

    async def execute_bash_command(
        self,
        workspace_id: str,
        command: str,
        timeout: int = 300
    ) -> ExecutionResult:
        """Execute bash command in workspace

        Args:
            workspace_id: ID of the workspace
            command: Bash command to execute
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult: Result of execution
        """
        if not self._is_command_safe(command):
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Command not allowed: {command}",
                execution_time=0.0,
                files_created=[],
                files_modified=[],
                command=command,
                working_directory=".",
                env={}
            )

        exec_command = ExecutionCommand(
            command=command,
            timeout=timeout
        )

        return await self.execute_command(workspace_id, exec_command)

    async def execute_command(
        self,
        workspace_id: str,
        command: ExecutionCommand
    ) -> ExecutionResult:
        """Execute a command in the workspace

        Args:
            workspace_id: ID of the workspace
            command: ExecutionCommand object

        Returns:
            ExecutionResult: Result of execution
        """
        workspace_path = self.workspace_manager.get_workspace_path(workspace_id)
        if not workspace_path:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Workspace {workspace_id} not found",
                execution_time=0.0,
                files_created=[],
                files_modified=[],
                command=command.command,
                working_directory=command.working_directory,
                env=command.env_vars or {}
            )

        # Get file states before execution
        files_before = await self._get_workspace_files(workspace_id)

        # Prepare working directory
        work_dir = workspace_path / command.working_directory
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)

        # Prepare environment
        env = os.environ.copy()
        if command.env_vars:
            env.update(command.env_vars)

        # Add workspace to Python path
        env["PYTHONPATH"] = str(workspace_path) + ":" + env.get("PYTHONPATH", "")

        start_time = asyncio.get_event_loop().time()
        stdout_data = ""
        stderr_data = ""
        exit_code = 0

        try:
            # Execute command
            if command.shell:
                process = await asyncio.create_subprocess_shell(
                    command.command,
                    stdout=asyncio.subprocess.PIPE if command.capture_output else None,
                    stderr=asyncio.subprocess.PIPE if command.capture_output else None,
                    cwd=str(work_dir),
                    env=env
                )
            else:
                cmd_parts = command.command.split()
                process = await asyncio.create_subprocess_exec(
                    *cmd_parts,
                    stdout=asyncio.subprocess.PIPE if command.capture_output else None,
                    stderr=asyncio.subprocess.PIPE if command.capture_output else None,
                    cwd=str(work_dir),
                    env=env
                )

            # Track the process
            if workspace_id in self.workspace_manager.active_processes:
                self.workspace_manager.active_processes[workspace_id].append(process)

            # Wait for completion with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=command.timeout
                )

                if command.capture_output:
                    stdout_data = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ""
                    stderr_data = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ""

                exit_code = process.returncode

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                stderr_data = f"Command timed out after {command.timeout} seconds"
                exit_code = -9

        except Exception as e:
            stderr_data = f"Execution error: {str(e)}"
            exit_code = -1

        finally:
            # Remove from active processes
            if workspace_id in self.workspace_manager.active_processes:
                try:
                    self.workspace_manager.active_processes[workspace_id].remove(process)
                except ValueError:
                    pass

        execution_time = asyncio.get_event_loop().time() - start_time

        # Get file changes
        files_after = await self._get_workspace_files(workspace_id)
        files_created = [f for f in files_after if f not in files_before]
        files_modified = [f for f in files_after if f in files_before]  # Simplified check

        success = exit_code == 0

        logger.info(f"Executed command in {workspace_id}: {command.command} "
                   f"(exit_code={exit_code}, time={execution_time:.2f}s)")

        try:
            rel_work_dir = str(work_dir.relative_to(workspace_path)) if work_dir != workspace_path else "."
        except Exception:
            rel_work_dir = command.working_directory

        return ExecutionResult(
            success=success,
            exit_code=exit_code,
            stdout=stdout_data,
            stderr=stderr_data,
            execution_time=execution_time,
            files_created=files_created,
            files_modified=files_modified,
            command=command.command,
            working_directory=rel_work_dir,
            env=command.env_vars or {},
        )

    async def execute_jupyter_notebook(
        self,
        workspace_id: str,
        notebook_content: str,
        timeout: int = 600
    ) -> List[ExecutionResult]:
        """Execute Jupyter notebook cells

        Args:
            workspace_id: ID of the workspace
            notebook_content: Jupyter notebook JSON content
            timeout: Total execution timeout in seconds

        Returns:
            List[ExecutionResult]: Results for each cell
        """
        try:
            notebook_data = json.loads(notebook_content)
        except json.JSONDecodeError as e:
            return [ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Invalid notebook JSON: {e}",
                execution_time=0.0,
                files_created=[],
                files_modified=[],
                command="notebook:load",
                working_directory=".",
                env={}
            )]

        results = []
        cells = notebook_data.get("cells", [])

        for i, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue

            cell_source = "".join(cell.get("source", []))
            if not cell_source.strip():
                continue

            # Execute cell
            result = await self.execute_python_code(
                workspace_id=workspace_id,
                code=cell_source,
                file_name=f"notebook_cell_{i}.py",
                timeout=min(timeout, 120)  # Max 2 minutes per cell
            )
            results.append(result)

            # Stop on first error if specified
            if not result.success:
                logger.warning(f"Notebook cell {i} failed, stopping execution")
                break

        return results

    async def _get_workspace_files(self, workspace_id: str) -> List[str]:
        """Get list of all files in workspace

        Args:
            workspace_id: ID of the workspace

        Returns:
            List[str]: List of file paths relative to workspace root
        """
        files = []
        workspace_path = self.workspace_manager.get_workspace_path(workspace_id)
        if not workspace_path or not workspace_path.exists():
            return files

        try:
            for file_path in workspace_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(workspace_path)
                    files.append(str(rel_path))
        except Exception as e:
            logger.error(f"Error listing workspace files: {e}")

        return files

    async def stream_execution(
        self,
        workspace_id: str,
        command: ExecutionCommand
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream execution output in real-time

        Args:
            workspace_id: ID of the workspace
            command: ExecutionCommand object

        Yields:
            Dict[str, Any]: Stream of execution events
        """
        workspace_path = self.workspace_manager.get_workspace_path(workspace_id)
        if not workspace_path:
            yield {
                "type": "error",
                "message": f"Workspace {workspace_id} not found"
            }
            return

        # Safety check
        if not self._is_command_safe(command.command):
            yield {
                "type": "error",
                "message": f"Command not allowed: {command.command}"
            }
            return

        work_dir = workspace_path / command.working_directory
        work_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if command.env_vars:
            env.update(command.env_vars)
        env["PYTHONPATH"] = str(workspace_path) + ":" + env.get("PYTHONPATH", "")

        yield {
            "type": "start",
            "command": command.command,
            "workspace_id": workspace_id
        }

        try:
            if command.shell:
                process = await asyncio.create_subprocess_shell(
                    command.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir),
                    env=env
                )
            else:
                cmd_parts = command.command.split()
                process = await asyncio.create_subprocess_exec(
                    *cmd_parts,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir),
                    env=env
                )

            # Track process
            if workspace_id in self.workspace_manager.active_processes:
                self.workspace_manager.active_processes[workspace_id].append(process)

            # Stream output
            async def read_stream(stream, stream_type):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    yield {
                        "type": stream_type,
                        "data": line.decode('utf-8', errors='replace').rstrip()
                    }

            # Read stdout and stderr concurrently
            async def stream_outputs():
                async for event in read_stream(process.stdout, "stdout"):
                    yield event
                async for event in read_stream(process.stderr, "stderr"):
                    yield event

            async for event in stream_outputs():
                yield event

            # Wait for process completion
            exit_code = await process.wait()

            yield {
                "type": "complete",
                "exit_code": exit_code,
                "success": exit_code == 0
            }

        except Exception as e:
            yield {
                "type": "error",
                "message": f"Execution error: {str(e)}"
            }

        finally:
            # Clean up process tracking
            if workspace_id in self.workspace_manager.active_processes:
                try:
                    self.workspace_manager.active_processes[workspace_id].remove(process)
                except (ValueError, NameError):
                    pass
