"""
HeadlessAgentSession - Wrapper for embedded OpenHands agent execution

Provides a simplified interface for running OpenHands agents (esp. CodeActAgent)
in headless mode for research tasks.
"""

import asyncio
import logging
from typing import Optional, Type, Any
from datetime import datetime

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.core.schema import AgentState
from openhands.core.loop import run_agent_until_done
from openhands.core.setup import create_memory
from openhands.core.config import AgentConfig, LLMConfig, OpenHandsConfig
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime import get_runtime_cls
from openhands.events import EventStream, EventSource
from openhands.events.action import MessageAction
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.memory import InMemoryFileStore

logger = logging.getLogger(__name__)


class HeadlessAgentSession:
    """
    Headless OpenHands agent session for embedded execution.

    Wraps OpenHands AgentController to provide a simpler interface
    for running agents in research tasks without UI dependencies.

    Example:
        # Create session
        session = HeadlessAgentSession(
            agent_class=CodeActAgent,
            llm_config=llm_config,
            experiment_id="branch-123"
        )

        # Start execution
        await session.start("Implement quicksort in Python")

        # Send additional messages
        await session.send_user_message("Add unit tests")

        # Access events
        async for event in session.event_stream.subscribe():
            print(event)

        # Cleanup
        await session.close()
    """

    def __init__(
        self,
        agent_class: Type[Agent],
        llm_config: LLMConfig,
        agent_config: Optional[AgentConfig] = None,
        experiment_id: Optional[str] = None,
        max_iterations: int = 100,
    ):
        """
        Initialize headless agent session.

        Args:
            agent_class: Agent class to instantiate (e.g., CodeActAgent)
            llm_config: LLM configuration
            agent_config: Optional agent configuration (will create from llm_config if not provided)
            experiment_id: Unique identifier for this session
            max_iterations: Maximum number of agent iterations
        """
        self.agent_class = agent_class
        self.llm_config = llm_config
        self.max_iterations = max_iterations

        # Generate experiment ID
        self.experiment_id = experiment_id or f"headless_{datetime.now().timestamp()}"

        # Create in-memory file store for this session
        self.file_store = InMemoryFileStore()

        # Create isolated event stream for this session
        self.event_stream = EventStream(
            sid=self.experiment_id,
            file_store=self.file_store
        )

        # Store agent config
        self.agent_config = agent_config or AgentConfig()

        # Agent instance will be created on start
        self.agent = None

        # Create conversation stats tracker (headless: no user id)
        self.conversation_stats = ConversationStats(
            file_store=self.file_store,
            conversation_id=self.experiment_id,
            user_id=None,
        )

        # Controller will be created on start
        self.controller: Optional[AgentController] = None

        # Config/registry (set on start)
        self.oh_config: Optional[OpenHandsConfig] = None
        self.llm_registry: Optional[LLMRegistry] = None

        # Runtime will be created on start
        self.runtime = None

        # Control flags
        self._running = False
        self._controller_task: Optional[asyncio.Task] = None

        logger.info(
            f"HeadlessAgentSession initialized: {self.experiment_id}, "
            f"agent={agent_class.__name__}"
        )

    async def start(self, initial_message: str):
        """
        Start agent execution with initial message.

        Args:
            initial_message: The task description to send to the agent

        Example:
            await session.start("Implement quicksort in Python")
        """
        if self._running:
            logger.warning(f"Session {self.experiment_id} already running")
            return

        logger.info(f"Starting session {self.experiment_id}")

        # Build minimal OpenHandsConfig and LLM registry
        oh_config = OpenHandsConfig()
        # Prefer Docker runtime for headless reliability (isolation + fewer host deps)
        # If you need LocalRuntime explicitly, set via settings before start()
        oh_config.runtime = 'docker'
        # Avoid browser dependency in headless runs unless explicitly needed
        oh_config.enable_browser = False
        if self.llm_config is not None:
            oh_config.set_llm_config(self.llm_config, name='llm')
        agent_cfg = self.agent_config or AgentConfig()
        oh_config.set_agent_config(agent_cfg, name='agent')
        llm_registry = LLMRegistry(oh_config)
        self.oh_config = oh_config
        self.llm_registry = llm_registry

        # Create agent instance with proper signature
        self.agent = self.agent_class(config=agent_cfg, llm_registry=llm_registry)

        # Create runtime
        # Note: Runtime creation depends on OpenHands configuration
        # For research tasks, we might use LocalRuntime or RemoteRuntime
        # This is a simplified version - actual implementation needs proper runtime setup
        try:
            self.runtime = await self._create_runtime()
            if hasattr(self.runtime, 'connect'):
                await self.runtime.connect()
        except Exception as e:
            logger.error(f"Failed to create runtime: {e}", exc_info=True)
            raise

        # Create controller
        self.controller = AgentController(
            agent=self.agent,
            event_stream=self.event_stream,
            conversation_stats=self.conversation_stats,
            iteration_delta=self.max_iterations,
            headless_mode=True,
            confirmation_mode=False,  # No confirmation in headless mode
            file_store=self.file_store,
        )

        # Add initial message to event stream
        self.event_stream.add_event(
            MessageAction(content=initial_message),
            EventSource.USER
        )
        # Ensure agent moves out of LOADING/STOPPED to RUNNING promptly
        try:
            await self.controller.set_agent_state_to(AgentState.RUNNING)
        except Exception:
            logger.debug("Initial RUNNING state nudge failed", exc_info=True)

        # Start controller loop (non-blocking)
        self._running = True
        self._controller_task = asyncio.create_task(self._run_controller())

        logger.info(f"Session {self.experiment_id} started with task: {initial_message[:100]}")

    async def send_user_message(self, message: str):
        """
        Send user message to running agent.

        Args:
            message: Message to send to the agent

        Example:
            await session.send_user_message("Add error handling")
        """
        if not self._running:
            # Allow late messages if controller exists; try to revive RUNNING state
            if not self.controller:
                logger.warning("Cannot send message - session not initialized")
                return
            logger.warning("Session not running, delivering message to event stream anyway")

        self.event_stream.add_event(
            MessageAction(content=message),
            EventSource.USER
        )

        # If controller is present and agent not running, set to RUNNING to resume
        try:
            if self.controller and self.controller.get_agent_state() != AgentState.RUNNING:
                await self.controller.set_agent_state_to(AgentState.RUNNING)
        except Exception:
            logger.debug("Failed to nudge controller to RUNNING", exc_info=True)

        logger.info(f"Sent message to session {self.experiment_id}: {message[:100]}")

    async def cancel(self):
        """
        Cancel execution and cleanup.

        Stops the agent controller and closes the runtime.
        """
        logger.info(f"Cancelling session {self.experiment_id}")

        self._running = False

        # Cancel controller task
        if self._controller_task:
            self._controller_task.cancel()
            try:
                await self._controller_task
            except asyncio.CancelledError:
                pass

        # Close controller
        if self.controller:
            try:
                await self.controller.close()
            except Exception as e:
                logger.error(f"Error closing controller: {e}")

        # Close runtime
        if self.runtime:
            try:
                await self.runtime.close()
            except Exception as e:
                logger.error(f"Error closing runtime: {e}")

        # Close event stream to stop background thread
        try:
            if self.event_stream:
                self.event_stream.close()
        except Exception as e:
            logger.warning(f"Error closing event stream: {e}")

        logger.info(f"Session {self.experiment_id} cancelled")

    async def close(self):
        """
        Cleanup resources.

        Alias for cancel() for consistency with other components.
        """
        await self.cancel()

    async def _run_controller(self):
        """
        Run controller loop until completion or cancellation.

        This runs the main agent loop in the background.
        """
        try:
            logger.info(f"Starting controller loop for {self.experiment_id}")

            # Run the actual controller loop
            if not self.controller:
                logger.error("Agent controller not initialized")
                return
            # Create memory and drive the agent until a terminal state.
            # The controller reacts to events and steps itself; we only keep the loop alive.
            memory = create_memory(
                runtime=self.runtime,
                event_stream=self.event_stream,
                sid=self.experiment_id,
            )
            await run_agent_until_done(
                controller=self.controller,
                runtime=self.runtime,
                memory=memory,
                end_states=[
                    AgentState.FINISHED,
                    AgentState.ERROR,
                    AgentState.STOPPED,
                ],
            )

        except asyncio.CancelledError:
            logger.info(f"Controller loop cancelled for {self.experiment_id}")
        except Exception as e:
            logger.error(f"Error in controller loop: {e}", exc_info=True)
        finally:
            self._running = False

    async def _create_runtime(self):
        """
        Create sandbox runtime for agent execution.

        Returns:
            Runtime instance (LocalRuntime or RemoteRuntime)
        """
        # TODO: Implement proper runtime creation
        # This depends on OpenHands runtime configuration
        #
        # Example:
        # from openhands.runtime import create_runtime
        # runtime_config = RuntimeConfig(...)
        # return await create_runtime(runtime_config)

        logger.info(f"Creating runtime for {self.experiment_id}")

        # Build actual runtime from OpenHands config
        if not self.oh_config or not self.llm_registry:
            # Fallback: minimal defaults
            self.oh_config = OpenHandsConfig()
            self.llm_registry = LLMRegistry(self.oh_config)
        runtime_cls = get_runtime_cls(self.oh_config.runtime)
        # Create runtime in headless mode; attach_to_existing False to start fresh
        runtime = runtime_cls(
            config=self.oh_config,
            event_stream=self.event_stream,
            llm_registry=self.llm_registry,
            sid=self.experiment_id,
            plugins=getattr(self.agent_class, 'sandbox_plugins', None),
            attach_to_existing=False,
            headless_mode=True,
        )
        return runtime

    def is_running(self) -> bool:
        """Check if session is currently running"""
        return self._running

    def get_stats(self) -> dict:
        """Get session statistics"""
        return {
            "experiment_id": self.experiment_id,
            "agent": self.agent_class.__name__,
            "running": self._running,
            "max_iterations": self.max_iterations,
        }


# Example usage
async def test_headless_session():
    """Test HeadlessAgentSession"""
    # This requires actual agent class and LLM config
    # For demonstration purposes

    # Mock agent class
    class MockAgent:
        def __init__(self, llm):
            self.llm = llm

    # Mock LLM config
    class MockLLMConfig:
        model = "gpt-4"

    llm_config = MockLLMConfig()

    # Create session
    session = HeadlessAgentSession(
        agent_class=MockAgent,
        llm_config=llm_config,
        experiment_id="test-session"
    )

    logger.info(" HeadlessAgentSession created successfully")
    logger.info(f"Stats: {session.get_stats()}")

    # Start session
    await session.start("Test task")

    # Wait a bit
    await asyncio.sleep(2)

    # Send message
    await session.send_user_message("Additional instruction")

    # Cleanup
    await session.close()

    logger.info(" HeadlessAgentSession test completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_headless_session())
