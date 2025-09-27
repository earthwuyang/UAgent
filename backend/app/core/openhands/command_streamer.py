"""OpenHands Command Streaming for Real-time Frontend Updates"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Awaitable

from ..websocket_manager import progress_tracker, EventType

logger = logging.getLogger(__name__)


class OpenHandsCommandStreamer:
    """Stream OpenHands command execution in real-time to WebSocket clients"""

    def __init__(self, session_id: str, workspace_path: Path):
        """Initialize the command streamer

        Args:
            session_id: Research session ID for WebSocket routing
            workspace_path: Path to the OpenHands workspace
        """
        self.session_id = session_id
        self.workspace_path = workspace_path
        self.active_streams = {}

    async def stream_command_execution(
        self,
        command: str,
        command_type: str = "bash",
        working_directory: str = ".",
        stream_callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> str:
        """Stream command execution to WebSocket clients

        Args:
            command: Command being executed
            command_type: Type of command (bash, python, etc.)
            working_directory: Working directory for the command
            stream_callback: Optional callback for custom streaming logic

        Returns:
            Stream ID for tracking this command execution
        """
        stream_id = f"cmd_{int(datetime.now().timestamp() * 1000)}"

        # Send command start event
        await progress_tracker.log_openhands_output(
            session_id=self.session_id,
            output_type="command_start",
            content={
                "stream_id": stream_id,
                "command": command,
                "command_type": command_type,
                "working_directory": working_directory,
                "timestamp": datetime.now().isoformat(),
                "status": "started"
            }
        )

        return stream_id

    async def stream_command_output(
        self,
        stream_id: str,
        output: str,
        output_type: str = "stdout",
        exit_code: Optional[int] = None
    ):
        """Stream command output in real-time

        Args:
            stream_id: Stream ID from stream_command_execution
            output: Command output text
            output_type: Type of output (stdout, stderr, info, error)
            exit_code: Exit code if command completed
        """
        await progress_tracker.log_openhands_output(
            session_id=self.session_id,
            output_type="command_output",
            content={
                "stream_id": stream_id,
                "output": output,
                "output_type": output_type,
                "exit_code": exit_code,
                "timestamp": datetime.now().isoformat(),
                "status": "completed" if exit_code is not None else "running"
            }
        )

    async def stream_file_operation(
        self,
        operation: str,  # "create", "edit", "delete", "read"
        file_path: str,
        content_preview: Optional[str] = None,
        status: str = "success"
    ):
        """Stream file operations to WebSocket clients

        Args:
            operation: Type of file operation
            file_path: Path to the file being operated on
            content_preview: Preview of file content (first few lines)
            status: Operation status (success, error, in_progress)
        """
        await progress_tracker.log_openhands_output(
            session_id=self.session_id,
            output_type="file_operation",
            content={
                "operation": operation,
                "file_path": file_path,
                "content_preview": content_preview,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        )

    async def stream_environment_setup(
        self,
        setup_type: str,  # "dependencies", "environment", "workspace"
        details: Dict[str, Any],
        status: str = "in_progress"
    ):
        """Stream environment setup operations

        Args:
            setup_type: Type of setup operation
            details: Setup details and progress
            status: Setup status
        """
        await progress_tracker.log_openhands_output(
            session_id=self.session_id,
            output_type="environment_setup",
            content={
                "setup_type": setup_type,
                "details": details,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        )

    async def stream_agent_action(
        self,
        action_type: str,  # "think", "plan", "execute", "observe"
        action_content: str,
        step_number: Optional[int] = None,
        total_steps: Optional[int] = None
    ):
        """Stream agent actions and thoughts

        Args:
            action_type: Type of agent action
            action_content: Content of the action
            step_number: Current step number
            total_steps: Total number of steps
        """
        await progress_tracker.log_openhands_output(
            session_id=self.session_id,
            output_type="agent_action",
            content={
                "action_type": action_type,
                "action_content": action_content,
                "step_number": step_number,
                "total_steps": total_steps,
                "timestamp": datetime.now().isoformat()
            }
        )


class OpenHandsStreamingWrapper:
    """Wrapper to add streaming capabilities to existing OpenHands operations"""

    def __init__(self, original_client, session_id: str):
        """Initialize wrapper

        Args:
            original_client: Original OpenHands client instance
            session_id: Research session ID
        """
        self.original_client = original_client
        self.session_id = session_id
        self.streamer = None

    def enable_streaming(self, workspace_path: Path):
        """Enable streaming for this session"""
        self.streamer = OpenHandsCommandStreamer(self.session_id, workspace_path)
        logger.info(f"Enabled OpenHands streaming for session {self.session_id}")

    async def execute_with_streaming(
        self,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute original method with streaming

        Args:
            method_name: Name of the original method to call
            *args: Arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Result from the original method
        """
        if not self.streamer:
            # Fallback to original method if streaming not enabled
            method = getattr(self.original_client, method_name)
            return await method(*args, **kwargs)

        # Extract command information
        command = None
        command_type = "unknown"

        if method_name == "execute_bash_command":
            command = args[1] if len(args) > 1 else kwargs.get("command", "unknown")
            command_type = "bash"
        elif method_name == "execute_python_code":
            command = f"python (code execution)"
            command_type = "python"
        elif method_name == "execute_command":
            cmd_obj = args[1] if len(args) > 1 else kwargs.get("command")
            if cmd_obj and hasattr(cmd_obj, "command"):
                command = cmd_obj.command
                command_type = "command"

        # Start streaming
        if command:
            stream_id = await self.streamer.stream_command_execution(
                command=command,
                command_type=command_type
            )

        try:
            # Execute original method
            method = getattr(self.original_client, method_name)
            result = await method(*args, **kwargs)

            # Stream result
            if command and hasattr(result, "stdout"):
                await self.streamer.stream_command_output(
                    stream_id=stream_id,
                    output=result.stdout,
                    output_type="stdout",
                    exit_code=result.exit_code if hasattr(result, "exit_code") else 0
                )

                if hasattr(result, "stderr") and result.stderr:
                    await self.streamer.stream_command_output(
                        stream_id=stream_id,
                        output=result.stderr,
                        output_type="stderr",
                        exit_code=result.exit_code if hasattr(result, "exit_code") else 0
                    )

            return result

        except Exception as e:
            # Stream error
            if command:
                await self.streamer.stream_command_output(
                    stream_id=stream_id,
                    output=f"Error: {str(e)}",
                    output_type="error",
                    exit_code=1
                )
            raise

    def __getattr__(self, name):
        """Delegate to original client for non-streaming operations"""
        return getattr(self.original_client, name)