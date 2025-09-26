"""Production-ready OpenHands complete goal delegation runner.

This runner delegates entire research goals to OpenHands using its native
agent controller, avoiding the repetitive step-by-step CodeAct approach.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Add OpenHands to path
openhands_dir = Path(__file__).parent.parent.parent.parent / "OpenHands"
if openhands_dir.exists():
    sys.path.insert(0, str(openhands_dir))

try:
    from openhands.controller import AgentController
    from openhands.core.config import OpenHandsConfig
    from openhands.core.schema import AgentState
    from openhands.core.loop import run_agent_until_done
    from openhands.events.action import MessageAction
    from openhands.memory.memory import Memory
    from openhands.runtime import get_runtime_cls
except ImportError as e:
    import traceback
    print(f"Failed to import OpenHands modules: {e}")
    traceback.print_exc()
    raise RuntimeError("OpenHands complete runner not available") from e

logger = logging.getLogger(__name__)


@dataclass
class OpenHandsCompleteResult:
    """Result from delegating a complete goal to OpenHands."""
    success: bool
    message: str
    final_state: Optional[Any]
    events_count: int
    workspace_path: Path
    generated_files: list[str]


class OpenHandsCompleteRunner:
    """Production-ready runner that delegates complete goals to OpenHands.

    This avoids the repetitive file creation issue by letting OpenHands'
    internal agent handle the entire goal autonomously.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_iterations: int = 50,
        runtime_type: str = "local",  # Use local runtime to avoid Docker
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.runtime_type = runtime_type
        self.logger = logging.getLogger(__name__)

    async def run_complete_goal(
        self,
        workspace_path: Path,
        goal: str,
        progress_cb: Optional[Any] = None,
    ) -> OpenHandsCompleteResult:
        """Delegate a complete goal to OpenHands agent.

        This method creates an OpenHands agent controller and runs it
        with the complete goal, letting it handle all planning and execution
        internally without step-by-step external control.

        Args:
            workspace_path: Directory for the agent to work in
            goal: Complete research/coding goal
            progress_cb: Optional callback for progress updates

        Returns:
            OpenHandsCompleteResult with execution details
        """
        workspace_path = Path(workspace_path).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Configure OpenHands
        from openhands.core.config.llm_config import LLMConfig

        # Create LLM configuration with the model
        llm_config = LLMConfig(model=self.model)

        config = OpenHandsConfig(
            workspace_base=str(workspace_path),
            workspace_mount_path_in_sandbox=str(workspace_path),
            max_iterations=self.max_iterations,
            runtime="local",  # Force local runtime to avoid Docker
            llms={"llm": llm_config},  # Set default LLM configuration
        )

        # Set environment for OpenHands
        os.environ["WORKSPACE_BASE"] = str(workspace_path)
        os.environ["WORKSPACE_MOUNT_PATH"] = str(workspace_path)

        controller = None
        runtime = None

        try:
            # Create event stream and LLM registry first
            from openhands.events.stream import EventStream
            from openhands.llm.llm_registry import LLMRegistry
            from openhands.core.config.agent_config import AgentConfig
            from openhands.agenthub.codeact_agent import CodeActAgent
            from openhands.storage.memory import InMemoryFileStore

            # Create file store and event stream with required parameters
            session_id = f"session_{workspace_path.name}"
            file_store = InMemoryFileStore()
            event_stream = EventStream(sid=session_id, file_store=file_store)
            llm_registry = LLMRegistry(config=config)

            # Create runtime with event stream and LLM registry
            runtime_cls = get_runtime_cls(config.runtime)
            # DockerRuntime and other runtimes use regular __init__, not ainit
            runtime = runtime_cls(
                config=config,
                event_stream=event_stream,
                llm_registry=llm_registry,
                sid=session_id,
            )
            # DockerRuntime doesn't have initialize method - initialization happens in __init__

            self.logger.info(f"Runtime initialized: {runtime.__class__.__name__}")

            # Create memory with event stream
            memory = Memory(
                event_stream=event_stream,
                sid=session_id,
            )

            # Create agent configuration and agent
            agent_config = AgentConfig()
            agent = CodeActAgent(config=agent_config, llm_registry=llm_registry)

            controller = AgentController(
                agent=agent,
                event_stream=event_stream,
                conversation_stats=None,  # ConversationStats not available in this version
                iteration_delta=config.max_iterations,
                initial_state=None,
            )

            self.logger.info("Agent controller created")

            # Set up event subscriber for progress updates
            if progress_cb:
                from openhands.events.stream import EventStreamSubscriber

                async def handle_event(event):
                    """Handle events and send progress updates."""
                    try:
                        if hasattr(event, 'action') and hasattr(event.action, 'thought'):
                            await progress_cb("thought", {
                                "thought": event.action.thought,
                                "action": str(event.action),
                            })
                        elif hasattr(event, 'message'):
                            await progress_cb("message", {
                                "message": event.message,
                            })
                        elif hasattr(event, 'content'):
                            await progress_cb("content", {
                                "content": event.content,
                            })
                    except Exception as e:
                        self.logger.debug(f"Progress callback error: {e}")

                event_stream.subscribe(EventStreamSubscriber.MAIN, handle_event)

            # Send the complete goal as initial message by adding it to event stream
            initial_message = MessageAction(content=goal)
            event_stream.add_event(initial_message)

            self.logger.info("Starting agent with complete goal...")

            # Run agent until done
            # This is the key difference - we let OpenHands handle everything internally
            await run_agent_until_done(
                controller=controller,
                runtime=runtime,
                memory=memory,
                end_states=[
                    AgentState.FINISHED,
                    AgentState.ERROR,
                    AgentState.REJECTED,
                    AgentState.STOPPED,
                ],
            )

            self.logger.info(f"Agent completed with state: {controller.state.agent_state}")

            # Collect results
            final_state = controller.state
            success = final_state.agent_state == AgentState.FINISHED

            # Get generated files
            generated_files = []
            code_dir = workspace_path / "code"
            if code_dir.exists():
                for file_path in code_dir.rglob("*"):
                    if file_path.is_file():
                        generated_files.append(str(file_path.relative_to(workspace_path)))

            # Extract final message
            final_message = ""
            if hasattr(final_state, 'history') and final_state.history:
                for event in reversed(final_state.history):
                    if hasattr(event, 'message') and event.message:
                        final_message = event.message
                        break

            return OpenHandsCompleteResult(
                success=success,
                message=final_message or f"Task completed with state: {final_state.agent_state}",
                final_state=final_state,
                events_count=len(final_state.history) if hasattr(final_state, 'history') else 0,
                workspace_path=workspace_path,
                generated_files=generated_files,
            )

        except Exception as e:
            self.logger.error(f"Error running OpenHands agent: {e}", exc_info=True)
            return OpenHandsCompleteResult(
                success=False,
                message=f"Error: {str(e)}",
                final_state=None,
                events_count=0,
                workspace_path=workspace_path,
                generated_files=[],
            )

        finally:
            # Clean up
            if runtime:
                try:
                    await runtime.close()
                except:
                    pass


async def delegate_to_openhands_complete(
    goal: str,
    workspace_path: Path,
    model: str = "gpt-4o-mini",
    max_iterations: int = 50,
    progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    """High-level function to delegate a complete goal to OpenHands.

    This is the main entry point for UAgent to use OpenHands with
    complete goal delegation instead of step-by-step CodeAct.
    Uses local runtime to avoid Docker dependency.

    Args:
        goal: Complete research/coding goal
        workspace_path: Working directory
        model: LLM model to use
        max_iterations: Maximum iterations for the agent
        progress_cb: Optional progress callback

    Returns:
        Dictionary with results
    """
    runner = OpenHandsCompleteRunner(
        model=model,
        max_iterations=max_iterations,
        runtime_type="local",  # Explicitly use local runtime
    )

    result = await runner.run_complete_goal(
        workspace_path=workspace_path,
        goal=goal,
        progress_cb=progress_cb,
    )

    return {
        "success": result.success,
        "message": result.message,
        "workspace": str(result.workspace_path),
        "generated_files": result.generated_files,
        "events_count": result.events_count,
        "approach": "complete_goal_delegation",
    }