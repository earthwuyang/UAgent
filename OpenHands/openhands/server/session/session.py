import asyncio
import time
from logging import LoggerAdapter

import socketio

from openhands.controller.agent import Agent
from openhands.core.config import OpenHandsConfig
from openhands.core.config.condenser_config import (
    BrowserOutputCondenserConfig,
    CondenserPipelineConfig,
    ConversationWindowCondenserConfig,
    LLMSummarizingCondenserConfig,
)
from openhands.core.config.mcp_config import OpenHandsMCPConfigImpl
from openhands.core.exceptions import MicroagentValidationError
from openhands.core.logger import OpenHandsLoggerAdapter
from openhands.core.schema import AgentState
from openhands.events.action import MessageAction, NullAction
from openhands.events.event import Event, EventSource
from openhands.events.observation import (
    AgentStateChangedObservation,
    CmdOutputObservation,
    NullObservation,
)
from openhands.events.observation.agent import RecallObservation
from openhands.events.observation.error import ErrorObservation
from openhands.events.serialization import event_from_dict, event_to_dict
from openhands.events.stream import EventStreamSubscriber
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.constants import ROOM_KEY
from openhands.server.services.conversation_stats import ConversationStats
from openhands.server.session.agent_session import AgentSession
from openhands.server.session.conversation_init_data import ConversationInitData
from openhands.storage.data_models.settings import Settings
from openhands.storage.files import FileStore

# Import research middleware for auto-triggering on all messages
try:
    from extensions.uagent_research.middleware.research_middleware import research_middleware
    RESEARCH_MIDDLEWARE_AVAILABLE = True
except ImportError:
    RESEARCH_MIDDLEWARE_AVAILABLE = False
except Exception:
    RESEARCH_MIDDLEWARE_AVAILABLE = False


from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openhands.server.session.multi_agent_coordinator import MultiAgentCoordinator
from openhands.events.agent_event import AgentSpawnedEvent, ProgressUpdateEvent, NodeCompleteEvent
from openhands.events.action import MessageAction


class WebSession:
    """Web server-bound session wrapper.

    This was previously named `Session`. We keep `Session` as a compatibility alias
    (see openhands.server.session.__init__) so downstream imports/tests continue to
    work. The class manages a single web client connection and orchestrates the
    AgentSession lifecycle for that conversation.

    Attributes:
        sid: Stable conversation id across transports.
        sio: Socket.IO server used to emit events to the web client.
        last_active_ts: Unix timestamp of last successful send.
        is_alive: Whether the web connection is still alive.
        agent_session: Core agent session coordinating runtime/LLM.
        loop: The asyncio loop associated with the session.
        config: Effective OpenHands configuration for this conversation.
        llm_registry: Registry responsible for LLM access and retry hooks.
        file_store: File storage interface for this conversation.
        user_id: Optional multi-tenant user identifier.
        logger: Logger with session context.
    """

    sid: str
    sio: socketio.AsyncServer | None
    last_active_ts: int = 0
    is_alive: bool = True
    agent_session: AgentSession
    loop: asyncio.AbstractEventLoop
    config: OpenHandsConfig
    llm_registry: LLMRegistry
    file_store: FileStore
    user_id: str | None
    logger: LoggerAdapter
    active_research_experiment_id: str | None = None  # Track active research experiment
    active_research_experiments: set[str] = None  # Track all active research experiments
    research_coordinator: Optional['MultiAgentCoordinator'] = None  # Coordinator for research sub-agents
    _progress_reporter_task: asyncio.Task | None = None  # Background progress reporting task

    def __init__(
        self,
        sid: str,
        config: OpenHandsConfig,
        llm_registry: LLMRegistry,
        conversation_stats: ConversationStats,
        file_store: FileStore,
        sio: socketio.AsyncServer | None,
        user_id: str | None = None,
    ):
        self.sid = sid
        self.sio = sio
        self.last_active_ts = int(time.time())
        self.file_store = file_store
        self.logger = OpenHandsLoggerAdapter(extra={'session_id': sid})
        self.llm_registry = llm_registry
        self.conversation_stats = conversation_stats
        self.agent_session = AgentSession(
            sid,
            file_store,
            llm_registry=self.llm_registry,
            conversation_stats=conversation_stats,
            status_callback=self.queue_status_message,
            user_id=user_id,
        )
        self.agent_session.event_stream.subscribe(
            EventStreamSubscriber.SERVER, self.on_event, self.sid
        )
        self.config = config

        # Lazy import to avoid circular dependency
        from openhands.experiments.experiment_manager import ExperimentManagerImpl

        self.config = ExperimentManagerImpl.run_config_variant_test(
            user_id, sid, self.config
        )
        self.loop = asyncio.get_event_loop()
        self.user_id = user_id

        self._publish_queue: asyncio.Queue = asyncio.Queue()
        self._monitor_publish_queue_task: asyncio.Task = self.loop.create_task(
            self._monitor_publish_queue()
        )
        self._wait_websocket_initial_complete: bool = True
        self._last_progress_broadcast: float = 0.0

    def get_or_create_coordinator(self) -> 'MultiAgentCoordinator':
        """Get or create the research coordinator for this session."""
        if self.research_coordinator is None:
            from openhands.server.session.multi_agent_coordinator import MultiAgentCoordinator
            self.research_coordinator = MultiAgentCoordinator(
                session_id=self.sid,
                logger=self.logger,
                session=self
            )
            self.logger.info(f"Created MultiAgentCoordinator for session {self.sid}")
            
            # Register session as agent in MessageBus
            self.research_coordinator.message_bus.register_agent(
                self.sid,
                "session",
                ["ui", "status"]
            )
            
            # Subscribe to MessageBus events
            asyncio.create_task(self._monitor_agent_events())
        return self.research_coordinator

    

    async def _monitor_agent_events(self):
        """Monitor MessageBus for agent events and forward to event stream."""
        if not self.research_coordinator or not self.research_coordinator.message_bus:
            return
        
        try:
            async for message in self.research_coordinator.message_bus.subscribe(self.sid):
                # Forward relevant events to main event stream
                if isinstance(message, AgentSpawnedEvent):
                    self.agent_session.event_stream.add_event(
                        MessageAction(
                            content=f"🤖 Research sub-agent spawned: {message.sub_agent_id}\n"
                                    f"Goal: {message.goal}\n"
                                    f"Type: {message.agent_type}"
                        ),
                        EventSource.AGENT
                    )
                elif isinstance(message, ProgressUpdateEvent):
                    # Progress updates already handled by _report_research_progress
                    pass
                elif isinstance(message, NodeCompleteEvent):
                    # Optionally notify user of node completion
                    pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error monitoring agent events: {e}", exc_info=True)

    def get_sub_agents_status(self) -> list[dict]:
        """
        Get status of all sub-agents in this session.
        
        Returns:
            List of sub-agent status dicts
        """
        coord = self.get_or_create_coordinator()
        return coord.get_all_sub_agents()
    
    def get_sub_agent_status(self, sub_agent_id: str) -> dict | None:
        """
        Get status of a specific sub-agent.
        
        Args:
            sub_agent_id: Sub-agent ID
            
        Returns:
            Sub-agent status dict, or None if not found
        """
        coord = self.get_or_create_coordinator()
        return coord.get_sub_agent_status(sub_agent_id)

    def register_experiment(self, experiment_id: str) -> None:
        """
        Register an experiment as active in this session.
        
        Adds the experiment ID to active_research_experiments set and sets
        active_research_experiment_id if not already set (for backward compatibility).
        
        Args:
            experiment_id: The experiment/sub-agent ID to register
        """
        self.logger.info(f"🔬 Registering experiment {experiment_id} for session {self.sid}")
        
        if self.active_research_experiments is None:
            self.active_research_experiments = set()
            self.logger.debug("📝 Initialized active_research_experiments set")
        
        # Check if already registered
        if experiment_id in self.active_research_experiments:
            self.logger.warning(f"⚠️ Experiment {experiment_id} already registered")
            return
        
        self.active_research_experiments.add(experiment_id)
        
        # Set active_research_experiment_id if not set (backward compatibility)
        if not self.active_research_experiment_id:
            self.active_research_experiment_id = experiment_id
            self.logger.debug(f"📌 Set primary experiment ID to {experiment_id}")
        
        self.logger.info(
            f"✅ Registered experiment {experiment_id}. "
            f"Active experiments: {len(self.active_research_experiments)}, "
            f"IDs: {list(self.active_research_experiments)}"
        )
        
    def get_research_diagnostics(self) -> dict:
        """Get diagnostic information about research experiments in this session."""
        return {
            "session_id": self.sid,
            "active_research_experiment_id": self.active_research_experiment_id,
            "active_research_experiments": list(self.active_research_experiments) if self.active_research_experiments else [],
            "total_active": len(self.active_research_experiments) if self.active_research_experiments else 0,
            "has_coordinator": self.research_coordinator is not None,
            "progress_reporter_active": self._progress_reporter_task is not None and not self._progress_reporter_task.done(),
        }

    async def close(self) -> None:
        """Close the session."""
        # Close MessageBus
        if self.research_coordinator and self.research_coordinator.message_bus:
            try:
                await self.research_coordinator.message_bus.close()
                self.logger.info("MessageBus closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing MessageBus: {e}", exc_info=True)
        
        # Cleanup research coordinator and all sub-agents
        if self.research_coordinator:
            try:
                await self.research_coordinator.cleanup_all_sub_agents()
                self.logger.info("Research coordinator cleaned up successfully")
            except Exception as e:
                self.logger.error(f"Error cleaning up research coordinator: {e}", exc_info=True)
        
        self.is_alive = False
        if self.sio:
            await self.sio.emit(
                'oh_event',
                event_to_dict(
                    AgentStateChangedObservation('', AgentState.STOPPED.value)
                ),
                to=ROOM_KEY.format(sid=self.sid),
            )
        self.is_alive = False
        await self.agent_session.close()
        self._monitor_publish_queue_task.cancel()

    async def initialize_agent(
        self,
        settings: Settings,
        initial_message: MessageAction | None,
        replay_json: str | None,
    ) -> None:
        self.agent_session.event_stream.add_event(
            AgentStateChangedObservation('', AgentState.LOADING),
            EventSource.ENVIRONMENT,
        )
        agent_cls = settings.agent or self.config.default_agent
        self.config.security.confirmation_mode = (
            self.config.security.confirmation_mode
            if settings.confirmation_mode is None
            else settings.confirmation_mode
        )
        self.config.security.security_analyzer = (
            self.config.security.security_analyzer
            if settings.security_analyzer is None
            else settings.security_analyzer
        )
        self.config.sandbox.base_container_image = (
            settings.sandbox_base_container_image
            or self.config.sandbox.base_container_image
        )
        self.config.sandbox.runtime_container_image = (
            settings.sandbox_runtime_container_image
            if settings.sandbox_base_container_image
            or settings.sandbox_runtime_container_image
            else self.config.sandbox.runtime_container_image
        )

        # Set Git user configuration if provided in settings
        git_user_name = getattr(settings, 'git_user_name', None)
        if git_user_name is not None:
            self.config.git_user_name = git_user_name
        git_user_email = getattr(settings, 'git_user_email', None)
        if git_user_email is not None:
            self.config.git_user_email = git_user_email
        max_iterations = settings.max_iterations or self.config.max_iterations

        # Prioritize settings over config for max_budget_per_task
        max_budget_per_task = (
            settings.max_budget_per_task
            if settings.max_budget_per_task is not None
            else self.config.max_budget_per_task
        )

        self.config.search_api_key = settings.search_api_key
        if settings.sandbox_api_key:
            self.config.sandbox.api_key = settings.sandbox_api_key.get_secret_value()

        # NOTE: this need to happen AFTER the config is updated with the search_api_key
        self.logger.debug(
            f'MCP configuration before setup - self.config.mcp_config: {self.config.mcp}'
        )

        # Check if settings has custom mcp_config
        mcp_config = getattr(settings, 'mcp_config', None)
        if mcp_config is not None:
            # Use the provided MCP SHTTP servers instead of default setup
            self.config.mcp = self.config.mcp.merge(mcp_config)
            self.logger.debug(f'Merged custom MCP Config: {mcp_config}')

        # Add OpenHands' MCP server by default
        openhands_mcp_server, openhands_mcp_stdio_servers = (
            OpenHandsMCPConfigImpl.create_default_mcp_server_config(
                self.config.mcp_host, self.config, self.user_id
            )
        )

        if openhands_mcp_server:
            self.config.mcp.shttp_servers.append(openhands_mcp_server)
            self.logger.debug('Added default MCP HTTP server to config')

            self.config.mcp.stdio_servers.extend(openhands_mcp_stdio_servers)

        self.logger.debug(
            f'MCP configuration after setup - self.config.mcp: {self.config.mcp}'
        )

        # TODO: override other LLM config & agent config groups (#2075)
        agent_config = self.config.get_agent_config(agent_cls)
        # Pass runtime information to agent config for runtime-specific tool behavior
        agent_config.runtime = self.config.runtime
        agent_name = agent_cls if agent_cls is not None else 'agent'
        llm_config = self.config.get_llm_config_from_agent(agent_name)
        if settings.enable_default_condenser:
            # Default condenser chains three condensers together:
            # 1. a conversation window condenser that handles explicit
            # condensation requests,
            # 2. a condenser that limits the total size of browser observations,
            # and
            # 3. a condenser that limits the size of the view given to the LLM.
            # The order matters: with the browser output first, the summarizer
            # will only see the most recent browser output, which should keep
            # the summarization cost down.
            max_events_for_condenser = settings.condenser_max_size or 120
            default_condenser_config = CondenserPipelineConfig(
                condensers=[
                    ConversationWindowCondenserConfig(),
                    BrowserOutputCondenserConfig(attention_window=2),
                    LLMSummarizingCondenserConfig(
                        llm_config=llm_config,
                        keep_first=4,
                        max_size=max_events_for_condenser,
                    ),
                ]
            )

            self.logger.info(
                f'Enabling pipeline condenser with:'
                f' browser_output_masking(attention_window=2), '
                f' llm(model="{llm_config.model}", '
                f' base_url="{llm_config.base_url}", '
                f' keep_first=4, max_size={max_events_for_condenser})'
            )
            agent_config.condenser = default_condenser_config
        agent = Agent.get_cls(agent_cls)(agent_config, self.llm_registry)

        self.llm_registry.retry_listner = self._notify_on_llm_retry

        git_provider_tokens = None
        selected_repository = None
        selected_branch = None
        custom_secrets = None
        conversation_instructions = None
        if isinstance(settings, ConversationInitData):
            git_provider_tokens = settings.git_provider_tokens
            selected_repository = settings.selected_repository
            selected_branch = settings.selected_branch
            custom_secrets = settings.custom_secrets
            conversation_instructions = settings.conversation_instructions

        # Check if initial message should trigger research mode
        processed_initial_message = initial_message
        if (RESEARCH_MIDDLEWARE_AVAILABLE
            and initial_message
            and isinstance(initial_message, MessageAction)
            and initial_message.content):
            try:
                self.logger.info(f"🔬 Processing initial message through research middleware")
                result = await research_middleware.process_message(
                    user_message=initial_message.content,
                    session_id=self.sid,
                    conversation_metadata={'source': 'initial_message'},
                )

                if result.get('should_trigger_research'):
                    experiment_id = result.get('experiment_id')
                    self.logger.info(
                        f"🔬 Research mode triggered by initial message",
                        extra={
                            'session_id': self.sid,
                            'experiment_id': experiment_id,
                        },
                    )

                    self.active_research_experiment_id = experiment_id
                    self._last_progress_broadcast = 0.0

                    # Register with coordinator
                    coordinator = self.get_or_create_coordinator()
                    try:
                        if coordinator.track_existing_experiment(experiment_id):
                            self.logger.info(f"✅ Research experiment {experiment_id} tracked by coordinator")
                        else:
                            self.logger.warning(f"⚠️ Failed to track experiment {experiment_id} in coordinator")

                        self.register_experiment(experiment_id)
                    except Exception as e:
                        self.logger.error(f"❌ Failed to register research with coordinator: {e}", exc_info=True)

                    # Start progress reporter
                    if self._progress_reporter_task is None or self._progress_reporter_task.done():
                        self.logger.info(f"📊 Starting progress reporter for {experiment_id}")
                        self._progress_reporter_task = asyncio.create_task(
                            self._report_research_progress()
                        )

                    # Append research info to initial message and add monitoring instruction
                    research_info = (
                        "\n\n[System: Research mode activated - "
                        f"Experiment ID: {experiment_id}, "
                        f"Confidence: {result.get('confidence', 0):.2f}. "
                        "Multiple research agents are now working in parallel. "
                        "Check the Research Tree tab for live progress.]\n\n"
                        "Your task now is to monitor the research progress and provide periodic updates. "
                        "Check the research experiment status every 2 minutes and report any significant progress or completed branches. "
                        "Continue monitoring until all research branches are complete, then synthesize the results."
                    )
                    processed_initial_message = MessageAction(
                        content=initial_message.content + research_info,
                        wait_for_response=False,  # Do NOT wait - agent should monitor research progress
                        images_urls=getattr(initial_message, 'images_urls', None),
                        thought="I've activated research mode with multiple parallel research agents. I will now enter monitoring mode.",
                    )
                    self.logger.info(f"✅ Research mode activated for initial message, experiment_id={experiment_id}")
                    
                    # Store experiment ID for monitoring
                    self.active_research_experiment_id = experiment_id
            except Exception as e:
                self.logger.error(f"Failed to process initial message through research middleware: {e}", exc_info=True)

        try:
            await self.agent_session.start(
                runtime_name=self.config.runtime,
                config=self.config,
                agent=agent,
                max_iterations=max_iterations,
                max_budget_per_task=max_budget_per_task,
                agent_to_llm_config=self.config.get_agent_to_llm_config_map(),
                agent_configs=self.config.get_agent_configs(),
                git_provider_tokens=git_provider_tokens,
                custom_secrets=custom_secrets,
                selected_repository=selected_repository,
                selected_branch=selected_branch,
                initial_message=processed_initial_message,
                conversation_instructions=conversation_instructions,
                replay_json=replay_json,
            )
            
            # After initialization, agent should be in AWAITING_USER_INPUT if no initial message
            # The agent controller already sets the correct state internally
            if self.agent_session.controller:
                try:
                    # Only set to RUNNING if there was an initial message that's being processed
                    if initial_message:
                        await self.agent_session.controller.set_agent_state_to(AgentState.RUNNING)
                        self.logger.info(f"✅ Set agent state to RUNNING (processing initial message) for {self.sid}")
                        
                        # Emit state change to frontend
                        state_change_event = AgentStateChangedObservation('', AgentState.RUNNING.value)
                        await self.send(event_to_dict(state_change_event))
                        self.logger.info(f"✅ Emitted agent state change to RUNNING for {self.sid}")
                    else:
                        # No initial message - ensure state is AWAITING_USER_INPUT
                        await self.agent_session.controller.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
                        self.logger.info(f"✅ Set agent state to AWAITING_USER_INPUT (no initial message) for {self.sid}")
                        
                        # Emit state change to frontend
                        state_change_event = AgentStateChangedObservation('', AgentState.AWAITING_USER_INPUT.value)
                        await self.send(event_to_dict(state_change_event))
                        self.logger.info(f"✅ Emitted agent state change to AWAITING_USER_INPUT for {self.sid}")
                except Exception as state_error:
                    self.logger.error(f"❌ Failed to set agent state: {state_error}", exc_info=True)
            else:
                self.logger.warning(f"⚠️ Agent controller not available after start for {self.sid}")
            
        except MicroagentValidationError as e:
            self.logger.exception(f'Error creating agent_session: {e}')
            # For microagent validation errors, provide more helpful information
            await self.send_error(f'Failed to create agent session: {str(e)}')
            return
        except ValueError as e:
            self.logger.exception(f'Error creating agent_session: {e}')
            error_message = str(e)
            # For ValueError related to microagents, provide more helpful information
            if 'microagent' in error_message.lower():
                await self.send_error(
                    f'Failed to create agent session: {error_message}'
                )
            else:
                # For other ValueErrors, just show the error class
                await self.send_error('Failed to create agent session: ValueError')
            return
        except Exception as e:
            self.logger.exception(f'Error creating agent_session: {e}')
            # For other errors, just show the error class to avoid exposing sensitive information
            await self.send_error(
                f'Failed to create agent session: {e.__class__.__name__}'
            )
            return

        # After agent_session.start() completes, if no initial message was provided
        # and research middleware is available, prompt user for research goal
        # Note: RESEARCH_MIDDLEWARE_AVAILABLE is already imported at module level
        # DISABLED: This automatic prompting is causing import errors and is optional
        # Users can manually send research goals starting with "research goal:" to trigger research
        # if (
        #     RESEARCH_MIDDLEWARE_AVAILABLE
        #     and not initial_message
        #     and self.agent_session.controller
        # ):
        #     ... (disabled code)
        pass

    def _notify_on_llm_retry(self, retries: int, max: int) -> None:
        self.queue_status_message(
            'info', RuntimeStatus.LLM_RETRY, f'Retrying LLM request, {retries} / {max}'
        )

    def on_event(self, event: Event) -> None:
        asyncio.get_event_loop().run_until_complete(self._on_event(event))

    async def _on_event(self, event: Event) -> None:
        """Callback function for events that mainly come from the agent.

        Event is the base class for any agent action and observation.

        Args:
            event: The agent event (Observation or Action).
        """
        if isinstance(event, NullAction):
            return
        if isinstance(event, NullObservation):
            return
        if event.source == EventSource.AGENT:
            await self.send(event_to_dict(event))
        elif event.source == EventSource.USER:
            await self.send(event_to_dict(event))
        # NOTE: ipython observations are not sent here currently
        elif event.source == EventSource.ENVIRONMENT and isinstance(
            event,
            (CmdOutputObservation, AgentStateChangedObservation, RecallObservation),
        ):
            # feedback from the environment to agent actions is understood as agent events by the UI
            event_dict = event_to_dict(event)
            event_dict['source'] = EventSource.AGENT.value
            await self.send(event_dict)
            if (
                isinstance(event, AgentStateChangedObservation)
                and event.agent_state == AgentState.ERROR
            ):
                self.logger.error(
                    f'Agent status error: {event.reason}',
                    extra={'signal': 'agent_status_error'},
                )
        elif isinstance(event, ErrorObservation):
            # send error events as agent events to the UI
            event_dict = event_to_dict(event)
            event_dict['source'] = EventSource.AGENT.value
            await self.send(event_dict)

    async def dispatch(self, data: dict) -> None:
        event = event_from_dict(data.copy())
        self.logger.info(f"🎯 [UAG-35 DEBUG] dispatch() called, event type: {type(event).__name__}, research_middleware available: {RESEARCH_MIDDLEWARE_AVAILABLE}")

        result: dict | None = None
        if RESEARCH_MIDDLEWARE_AVAILABLE and isinstance(event, MessageAction) and event.content:
            self.logger.info(f"✅ [UAG-35 DEBUG] Calling research_middleware.process_message() for message: {event.content[:100]}...")
            try:
                result = await research_middleware.process_message(
                    user_message=event.content,
                    session_id=self.sid,
                    conversation_metadata={'source': 'subsequent_message'},
                )

                if result.get('should_trigger_research'):
                    self.logger.info(
                        "🔬 Research mode triggered",
                        extra={
                            'session_id': self.sid,
                            'task_type': result.get('task_type'),
                            'confidence': result.get('confidence'),
                            'experiment_id': result.get('experiment_id'),
                        },
                    )

                    experiment_id = result.get('experiment_id', 'N/A')
                    self.logger.debug(f"📋 Experiment ID: {experiment_id}")
                    
                    self.active_research_experiment_id = experiment_id
                    self._last_progress_broadcast = 0.0

                    # Register with coordinator for proper tracking
                    coordinator = self.get_or_create_coordinator()
                    try:
                        self.logger.debug(f"🔍 Attempting to track experiment {experiment_id} in coordinator")
                        # Track the existing experiment in coordinator
                        if coordinator.track_existing_experiment(experiment_id):
                            self.logger.info(f"✅ Research experiment {experiment_id} tracked by coordinator")
                        else:
                            self.logger.warning(f"⚠️ Failed to track experiment {experiment_id} in coordinator")
                        
                        # Register in session state
                        self.logger.debug(f"📝 Registering experiment {experiment_id} in session")
                        self.register_experiment(experiment_id)
                        self.logger.debug(f"✅ Experiment {experiment_id} registered in session")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to register research with coordinator: {e}", exc_info=True)

                    if self._progress_reporter_task is None or self._progress_reporter_task.done():
                        self.logger.info(f"📊 Starting progress reporter for {experiment_id}")
                        self._progress_reporter_task = asyncio.create_task(
                            self._report_research_progress()
                        )
                    else:
                        self.logger.debug("📊 Progress reporter already active")

                    research_info = (
                        "\n\n[System: Research mode activated - "
                        f"Experiment ID: {experiment_id}, "
                        f"Confidence: {result.get('confidence', 0):.2f}. "
                        "Check the Research Tree tab for live progress.]"
                    )
                    # Send the research activation message to the user
                    self.agent_session.event_stream.add_event(
                        MessageAction(content=event.content + research_info),
                        EventSource.AGENT,
                    )
                    # Set agent state to AWAITING_USER_INPUT since orchestrator is handling the research
                    controller = self.agent_session.controller
                    if controller is not None:
                        await controller.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
                    # Don't pass the event to the agent - orchestrator will handle it
                    return

            except Exception:
                self.logger.error(
                    "Failed to process research middleware", exc_info=True
                )
                result = None

        mode = result.get('mode') if isinstance(result, dict) else None

        if mode == 'progress_query':
            summary = (
                result.get('progress_data', {}).get('summary')
                if isinstance(result, dict)
                else None
            ) or 'No active research found for this conversation.'
            self.agent_session.event_stream.add_event(
                MessageAction(content=summary),
                EventSource.AGENT,
            )
            controller = self.agent_session.controller
            if controller is not None:
                await controller.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
            return

        if mode == 'control_intent':
            control_result = result.get('control_result', {}) if isinstance(result, dict) else {}
            message = control_result.get('message') or 'Control command processed.'
            self.agent_session.event_stream.add_event(
                MessageAction(content=message),
                EventSource.AGENT,
            )
            controller = self.agent_session.controller
            if controller is not None:
                await controller.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
            return

        # This checks if the model supports images
        if isinstance(event, MessageAction) and event.image_urls:
            controller = self.agent_session.controller
            if controller:
                if controller.agent.llm.config.disable_vision:
                    await self.send_error(
                        'Support for images is disabled for this model, try without an image.'
                    )
                    return
                if not controller.agent.llm.vision_is_active():
                    await self.send_error(
                        'Model does not support image upload, change to a different model or try without an image.'
                    )
                    return

        self.agent_session.event_stream.add_event(event, EventSource.USER)

    async def send(self, data: dict[str, object]) -> None:
        self._publish_queue.put_nowait(data)

    async def _monitor_publish_queue(self):
        try:
            while True:
                data: dict = await self._publish_queue.get()
                await self._send(data)
        except asyncio.CancelledError:
            return

    async def _send(self, data: dict[str, object]) -> bool:
        try:
            if not self.is_alive:
                return False

            _start_time = time.time()
            _waiting_times = 1

            if self.sio:
                # Wait once during initialization to avoid event push failures during websocket connection intervals
                while self._wait_websocket_initial_complete and (
                    time.time() - _start_time < 2
                ):
                    if bool(
                        self.sio.manager.rooms.get('/', {}).get(
                            ROOM_KEY.format(sid=self.sid)
                        )
                    ):
                        break
                    self.logger.warning(
                        f'There is no listening client in the current room,'
                        f' waiting for the {_waiting_times}th attempt: {self.sid}'
                    )
                    _waiting_times += 1
                    await asyncio.sleep(0.1)
                self._wait_websocket_initial_complete = False
                await self.sio.emit('oh_event', data, to=ROOM_KEY.format(sid=self.sid))

            await asyncio.sleep(0.001)  # This flushes the data to the client
            self.last_active_ts = int(time.time())
            return True
        except RuntimeError as e:
            self.logger.error(f'Error sending data to websocket: {str(e)}')
            self.is_alive = False
            return False

    async def _report_research_progress(self):
        """Report research progress every 60 seconds while experiment is running."""
        try:
            while self.active_research_experiment_id and self.is_alive:
                await asyncio.sleep(60)  # Backup heartbeat every 60 seconds

                if not self.active_research_experiment_id:
                    break

                # Get progress from middleware
                if RESEARCH_MIDDLEWARE_AVAILABLE:
                    controller = self.agent_session.controller
                    if controller and getattr(controller.agent, '_coordination_mode', False):
                        continue

                    orchestrator = research_middleware.get_orchestrator(self.active_research_experiment_id)

                    if orchestrator and hasattr(orchestrator, 'tree') and orchestrator.tree:
                        tree = orchestrator.tree

                        # Calculate statistics
                        total_nodes = len(tree.nodes) if hasattr(tree, 'nodes') else 0
                        stats = tree.stats if hasattr(tree, 'stats') else {}
                        total_cost = stats.get('total_cost', 0.0)

                        # Count nodes by status
                        completed = sum(1 for node in tree.nodes.values()
                                      if hasattr(node, 'status') and node.status.value == 'complete')
                        running = sum(1 for node in tree.nodes.values()
                                    if hasattr(node, 'status') and node.status.value == 'running')
                        pending = sum(1 for node in tree.nodes.values()
                                    if hasattr(node, 'status') and node.status.value == 'pending')
                        failed = sum(1 for node in tree.nodes.values()
                                   if hasattr(node, 'status') and node.status.value == 'failed')

                        # Create progress message
                        progress_msg = (
                            f"\n\n[Research Progress Update]\n"
                            f"Experiment ID: {self.active_research_experiment_id}\n"
                            f"Total Nodes: {total_nodes}\n"
                            f"Completed: {completed} | Running: {running} | Pending: {pending} | Failed: {failed}\n"
                            f"Total Cost: ${total_cost:.3f}\n"
                        )

                        now = time.time()
                        if now - self._last_progress_broadcast < 60:
                            continue

                        # Send progress message to frontend
                        observation = MessageAction(content=progress_msg)
                        event_dict = event_to_dict(observation)
                        event_dict['source'] = EventSource.AGENT.value
                        await self.send(event_dict)

                        self.logger.info(f"Research progress reported for experiment {self.active_research_experiment_id}: {total_nodes} nodes")
                        
                        # Update global session manager with progress
                        try:
                            from extensions.uagent_research.services.research_session_manager import get_global_session_manager
                            
                            session_mgr = get_global_session_manager()
                            if session_mgr and self.active_research_experiment_id:
                                # Note: ResearchSessionManager doesn't have update_progress method
                                # The progress is already tracked via the orchestrator in the session manager
                                # Just log that we're using the singleton
                                self.logger.debug(f"Progress sync: {total_nodes} nodes for {self.active_research_experiment_id}")
                        except ImportError:
                            self.logger.debug("Research session manager not available")
                        except Exception as e:
                            self.logger.warning(f"Failed to sync with research session manager: {e}")
                        
                        self._last_progress_broadcast = now
                    else:
                        # Orchestrator finished or not found - stop reporting
                        self.logger.info(f"Research orchestrator not found or completed for {self.active_research_experiment_id}, stopping progress reports")
                        self.active_research_experiment_id = None
                        self._last_progress_broadcast = 0.0
                        break

        except asyncio.CancelledError:
            self.logger.info("Research progress reporting cancelled")
        except Exception as e:
            self.logger.error(f"Error in research progress reporting: {str(e)}", exc_info=True)

    async def send_error(self, message: str) -> None:
        """Sends an error message to the client."""
        await self.send({'error': True, 'message': message})

    async def _send_status_message(
        self, msg_type: str, runtime_status: RuntimeStatus, message: str
    ) -> None:
        """Sends a status message to the client."""
        if msg_type == 'error':
            agent_session = self.agent_session
            controller = self.agent_session.controller
            if controller is not None and not agent_session.is_closed():
                await controller.set_agent_state_to(AgentState.ERROR)
            self.logger.error(
                f'Agent status error: {message}',
                extra={'signal': 'agent_status_error'},
            )
        await self.send(
            {
                'status_update': True,
                'type': msg_type,
                'id': runtime_status.value,
                'message': message,
            }
        )

    def queue_status_message(
        self, msg_type: str, runtime_status: RuntimeStatus, message: str
    ) -> None:
        """Queues a status message to be sent asynchronously."""
        asyncio.run_coroutine_threadsafe(
            self._send_status_message(msg_type, runtime_status, message), self.loop
        )


# Backward-compatible alias for external imports that still reference
# openhands.server.session.session import Session
Session = WebSession
