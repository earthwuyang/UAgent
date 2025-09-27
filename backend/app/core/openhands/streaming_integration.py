"""Integration module for adding OpenHands command streaming to research engines"""

import logging
from typing import Optional, Any, Dict
from pathlib import Path

from .command_streamer import OpenHandsStreamingWrapper
from ..websocket_manager import progress_tracker

logger = logging.getLogger(__name__)


def enable_openhands_streaming(openhands_client, session_id: str) -> Optional[Any]:
    """Enable streaming for an OpenHands client

    Args:
        openhands_client: The OpenHands client instance
        session_id: Research session ID for WebSocket routing

    Returns:
        Streaming-enabled client wrapper or original client
    """
    if not openhands_client or not session_id:
        return openhands_client

    try:
        # Create streaming wrapper
        streaming_client = OpenHandsStreamingWrapper(openhands_client, session_id)

        # Try to get workspace path for streaming setup
        if hasattr(openhands_client, 'workspace_manager') and openhands_client.workspace_manager:
            try:
                # Enable streaming if we can find any workspace
                streaming_client.enable_streaming(Path("/tmp/openhands_streaming"))
                logger.info(f"Enabled OpenHands streaming for session {session_id}")
            except Exception as e:
                logger.debug(f"Could not fully enable streaming: {e}")

        return streaming_client

    except Exception as e:
        logger.warning(f"Failed to enable OpenHands streaming: {e}")
        return openhands_client


async def stream_openhands_execution_start(session_id: str, command: str, context: Dict[str, Any] = None):
    """Stream the start of an OpenHands execution

    Args:
        session_id: Research session ID
        command: Command being executed
        context: Additional context information
    """
    await progress_tracker.log_openhands_output(
        session_id=session_id,
        output_type="execution_start",
        content={
            "command": command,
            "context": context or {},
            "status": "started"
        }
    )


async def stream_openhands_execution_progress(session_id: str, step: str, progress: float, details: Dict[str, Any] = None):
    """Stream progress during OpenHands execution

    Args:
        session_id: Research session ID
        step: Current execution step
        progress: Progress percentage (0-100)
        details: Additional details about the progress
    """
    await progress_tracker.log_openhands_output(
        session_id=session_id,
        output_type="execution_progress",
        content={
            "step": step,
            "progress": progress,
            "details": details or {},
            "status": "in_progress"
        }
    )


async def stream_openhands_execution_result(session_id: str, result: Any, success: bool = True):
    """Stream the result of an OpenHands execution

    Args:
        session_id: Research session ID
        result: Execution result
        success: Whether the execution was successful
    """
    # Convert result to streamable format
    result_content = {}
    if hasattr(result, 'stdout'):
        result_content['stdout'] = result.stdout
    if hasattr(result, 'stderr'):
        result_content['stderr'] = result.stderr
    if hasattr(result, 'exit_code'):
        result_content['exit_code'] = result.exit_code
    if hasattr(result, 'files_created'):
        result_content['files_created'] = result.files_created
    if hasattr(result, 'files_modified'):
        result_content['files_modified'] = result.files_modified

    await progress_tracker.log_openhands_output(
        session_id=session_id,
        output_type="execution_result",
        content={
            "result": result_content,
            "success": success,
            "status": "completed" if success else "failed"
        }
    )


async def stream_code_generation(session_id: str, code: str, language: str = "python", filename: str = None):
    """Stream code generation events

    Args:
        session_id: Research session ID
        code: Generated code
        language: Programming language
        filename: Target filename if known
    """
    await progress_tracker.log_openhands_output(
        session_id=session_id,
        output_type="code_generation",
        content={
            "code": code[:500] + "..." if len(code) > 500 else code,  # Truncate for streaming
            "language": language,
            "filename": filename,
            "full_length": len(code),
            "status": "generated"
        }
    )


async def stream_experiment_setup(session_id: str, experiment_name: str, setup_details: Dict[str, Any]):
    """Stream experiment setup events

    Args:
        session_id: Research session ID
        experiment_name: Name of the experiment
        setup_details: Details about the setup
    """
    await progress_tracker.log_openhands_output(
        session_id=session_id,
        output_type="experiment_setup",
        content={
            "experiment_name": experiment_name,
            "setup_details": setup_details,
            "status": "setting_up"
        }
    )


# Convenience function to make any OpenHands client streaming-enabled
def make_streaming_client(original_client, session_id: str):
    """Make an OpenHands client streaming-enabled

    This is a simple convenience function that wraps enable_openhands_streaming
    """
    return enable_openhands_streaming(original_client, session_id)