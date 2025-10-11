"""
Research Middleware

Automatically triggers research mode for complex tasks.
Intercepts user messages and routes to appropriate execution mode.

Phase 3 Enhancement:
- Progress query detection ("how's progress?", "what's happening?")
- Control intent detection ("pause research", "cancel idea-2")
- Integration with ResearchSessionManager for real-time status
"""

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass
from threading import RLock
from typing import Optional, Dict, Any, Tuple, List

from extensions.uagent_research.classifier.task_classifier import task_classifier, TaskType
from extensions.uagent_research.orchestrator.tree_orchestrator import TreeSearchOrchestrator
from extensions.uagent_research.uagent_research.models.research_tree import Budget
from ..utils.security import sanitize_identifier

logger = logging.getLogger(__name__)

try:
    from extensions.uagent_research.config import (
        ENABLE_AUTO_RESEARCH_TRIGGER,
        ENABLE_AGENT_COORDINATION,
        RESEARCH_CONFIDENCE_THRESHOLD,
        RESEARCH_POLL_INTERVAL,
        PROGRESS_CACHE_TTL,
    )
except ImportError as config_error:
    logger.warning(
        "Failed to import research config; using safe defaults and disabling auto trigger: %s",
        config_error,
    )
    ENABLE_AUTO_RESEARCH_TRIGGER = False
    ENABLE_AGENT_COORDINATION = True
    RESEARCH_CONFIDENCE_THRESHOLD = 0.7
    RESEARCH_POLL_INTERVAL = 10
    PROGRESS_CACHE_TTL = 2.0

# Enforce single-goal per conversation: do not auto-trigger new research from chat
SINGLE_GOAL_MODE = True

# Import control components
try:
    from extensions.uagent_research.control.control_bus import ControlMessage
    from extensions.uagent_research.services.research_session_manager import (
        ResearchSessionManager,
        ExperimentStatus,
    )
    from extensions.uagent_research.orchestrator.event_bus import get_event_bus
    CONTROL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Control components not available: {e}")
    CONTROL_AVAILABLE = False
    ControlMessage = None
    ResearchSessionManager = None
    ExperimentStatus = None


@dataclass
class ExperimentRecord:
    orchestrator: TreeSearchOrchestrator
    goal: str
    session_id: str
    max_iterations: int
    task: Optional[asyncio.Task] = None


class ResearchMiddleware:
    """
    Middleware that intercepts user messages and triggers research mode when needed.

    Phase 3 Features:
    - Progress query detection and response
    - Control intent detection and routing
    - Integration with ResearchSessionManager
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        enable_auto_trigger: bool = True,
        session_manager: Optional['ResearchSessionManager'] = None,
        poll_interval: float = 10.0,
        progress_cache_ttl: float = 2.0,
        coordination_enabled: bool = True,
    ):
        """
        Initialize research middleware.

        Args:
            confidence_threshold: Minimum confidence to auto-trigger research (0-1)
            enable_auto_trigger: Enable automatic research triggering
            session_manager: Optional ResearchSessionManager instance
        """
        self.confidence_threshold = confidence_threshold
        self.enable_auto_trigger = enable_auto_trigger
        self.active_orchestrators: Dict[str, 'ExperimentRecord'] = {}
        self._active_orchestrators_lock = RLock()
        # Track goal by session for single-goal mode
        self._session_goal: Dict[str, str] = {}
        self._session_goal_lock = RLock()

        # Session manager for progress queries
        self._session_manager = session_manager

        # Coordination / polling configuration
        self.poll_interval = poll_interval
        self.coordination_enabled = coordination_enabled
        self._progress_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._progress_cache_ttl = progress_cache_ttl
        self._progress_cache_lock = RLock()

        # Progress query patterns
        self.progress_patterns = [
            r"how'?s?\s+(the\s+)?progress",
            r"what'?s?\s+(the\s+)?status",
            r"what'?s?\s+happening",
            r"how\s+is\s+(it|research)\s+(going|doing)",
            r"show\s+(me\s+)?(the\s+)?progress",
            r"check\s+status",
            r"research\s+status",
            r"update\s+me",
        ]

        # Control intent patterns (action -> pattern list)
        self.control_patterns = {
            'pause': [
                r"pause\s+(the\s+)?research",
                r"stop\s+(the\s+)?research",
                r"halt\s+(the\s+)?research",
            ],
            'resume': [
                r"resume\s+(the\s+)?research",
                r"continue\s+(the\s+)?research",
                r"restart\s+(the\s+)?research",
            ],
            'cancel': [
                r"cancel\s+(the\s+)?research",
                r"abort\s+(the\s+)?research",
                r"kill\s+(the\s+)?research",
            ],
            'cancel_node': [
                r"cancel\s+(node\s+)?(?P<node_id>[\w\-]+)",
                r"stop\s+(node\s+)?(?P<node_id>[\w\-]+)",
            ],
        }

        # Register adapters for orchestrator
        self._register_adapters()

    def _register_adapters(self):
        """Register all agent adapters with the adapter registry"""
        from extensions.uagent_research.adapters.ensure_adapters import ensure_research_adapters_registered
        ensure_research_adapters_registered()

    # --- Concurrency-safe helpers -------------------------------------------------

    def _list_active_orchestrators(self) -> List[Tuple[str, 'ExperimentRecord']]:
        with self._active_orchestrators_lock:
            return list(self.active_orchestrators.items())

    def _active_experiment_ids(self) -> List[str]:
        with self._active_orchestrators_lock:
            return list(self.active_orchestrators.keys())

    def _get_active_metadata(self, experiment_id: str) -> Optional['ExperimentRecord']:
        with self._active_orchestrators_lock:
            return self.active_orchestrators.get(experiment_id)

    def _set_active_metadata(self, experiment_id: str, metadata: 'ExperimentRecord') -> None:
        with self._active_orchestrators_lock:
            self.active_orchestrators[experiment_id] = metadata

    def _remove_active_metadata(self, experiment_id: str) -> Optional['ExperimentRecord']:
        with self._active_orchestrators_lock:
            return self.active_orchestrators.pop(experiment_id, None)

    def _active_experiment_count(self) -> int:
        with self._active_orchestrators_lock:
            return len(self.active_orchestrators)

    def _set_task_reference(self, experiment_id: str, task: asyncio.Task) -> None:
        with self._active_orchestrators_lock:
            metadata = self.active_orchestrators.get(experiment_id)
            if metadata is not None:
                metadata.task = task

    def _get_session_goal_value(self, session_id: str) -> Optional[str]:
        with self._session_goal_lock:
            return self._session_goal.get(session_id)

    def _set_session_goal_value(self, session_id: str, goal: str) -> None:
        with self._session_goal_lock:
            self._session_goal[session_id] = goal

    def _clear_session_goal_value(self, session_id: str) -> None:
        with self._session_goal_lock:
            self._session_goal.pop(session_id, None)

    def _get_progress_cache_entry(self, experiment_id: str) -> Optional[Tuple[Dict[str, Any], float]]:
        with self._progress_cache_lock:
            return self._progress_cache.get(experiment_id)

    def _set_progress_cache_entry(self, experiment_id: str, status: Dict[str, Any], timestamp: float) -> None:
        with self._progress_cache_lock:
            self._progress_cache[experiment_id] = (status, timestamp)

    def _pop_progress_cache_entry(self, experiment_id: str) -> None:
        with self._progress_cache_lock:
            self._progress_cache.pop(experiment_id, None)

    def _validated_identifier(self, name: str, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            return sanitize_identifier(name, value)
        except ValueError as exc:
            logger.warning("Rejected %s due to invalid value: %s", name, exc)
            return None

    def get_session_manager(self) -> Optional['ResearchSessionManager']:
        """Get or create session manager"""
        if self._session_manager is None and CONTROL_AVAILABLE:
            try:
                from extensions.uagent_research.control.control_bus import ControlBus
                event_bus = get_event_bus()
                control_bus = ControlBus()
                self._session_manager = ResearchSessionManager(
                    event_bus=event_bus,
                    control_bus=control_bus
                )
                logger.info("ResearchSessionManager initialized in middleware")
            except Exception as e:
                logger.error(f"Failed to create session manager: {e}")
        return self._session_manager

    def get_active_experiment_for_session(self, session_id: str) -> Optional[str]:
        """Return the active experiment id for the provided session, if any."""
        for experiment_id, metadata in self._list_active_orchestrators():
            if metadata.session_id == session_id:
                return experiment_id

        # Fallback heuristic: experiment identifiers prefixed with exp_{session_id}_
        for experiment_id in self._active_experiment_ids():
            if experiment_id.startswith(f"exp_{session_id}_"):
                return experiment_id

        return None

    def is_research_active(self, session_id: str) -> bool:
        """Check whether the given session currently has active research."""
        return self.get_active_experiment_for_session(session_id) is not None

    def detect_progress_query(self, message: str) -> bool:
        """
        Detect if message is asking about research progress.

        Args:
            message: User message

        Returns:
            True if progress query detected
        """
        message_lower = message.lower()
        for pattern in self.progress_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True
        return False

    def detect_control_intent(self, message: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Detect control intent in user message.

        Args:
            message: User message

        Returns:
            Tuple of (action, target_dict) or None

        Example:
            "pause research" -> ("pause", {})
            "cancel node idea-2" -> ("cancel_node", {"node_id": "idea-2"})
        """
        message_lower = message.lower()

        for action, patterns in self.control_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower, re.IGNORECASE)
                if match:
                    # Extract any named groups (like node_id)
                    target = match.groupdict()
                    return (action, target)

        return None

    async def handle_progress_query(self, session_id: str) -> Dict[str, Any]:
        """
        Handle progress query by fetching status from session manager.

        Args:
            session_id: Session ID

        Returns:
            Dict with progress information
        """
        session_mgr = self.get_session_manager()

        if not session_mgr:
            return {
                'type': 'progress_query',
                'status': 'unavailable',
                'message': 'Progress tracking not available (session manager not initialized)'
            }

        try:
            experiment_id = self.get_active_experiment_for_session(session_id)

            if not experiment_id:
                return {
                    'type': 'progress_query',
                    'status': 'no_active_research',
                    'message': 'No active research found for this session'
                }

            # Use cached progress when available and fresh
            now = time.time()
            cached = self._get_progress_cache_entry(experiment_id)
            if cached and (now - cached[1]) <= self._progress_cache_ttl:
                status = cached[0]
            else:
                status = session_mgr.get_status(experiment_id)
                self._set_progress_cache_entry(experiment_id, status, now)

            # Format user-friendly summary
            summary = self._format_progress_summary(status)

            return {
                'type': 'progress_query',
                'status': 'success',
                'experiment_id': experiment_id,
                'data': status,
                'summary': summary
            }

        except KeyError:
            return {
                'type': 'progress_query',
                'status': 'not_tracked',
                'message': 'Research is running but not tracked by session manager'
            }
        except Exception as e:
            logger.error(f"Error handling progress query: {e}", exc_info=True)
            return {
                'type': 'progress_query',
                'status': 'error',
                'message': f'Error fetching progress: {str(e)}'
            }

    def _format_progress_summary(self, status: Dict[str, Any]) -> str:
        """
        Format progress status into user-friendly summary.

        Args:
            status: Status dict from ResearchSessionManager

        Returns:
            Human-readable summary string
        """
        lines = []

        # Overall status
        exp_status = status.get('status', 'unknown')
        lines.append(f"**Research Status:** {exp_status}")

        # Node statistics
        stats = status.get('stats', {})
        total = stats.get('total_nodes', 0)
        completed = stats.get('completed', 0)
        failed = stats.get('failed', 0)
        running = stats.get('running', 0)

        if total > 0:
            completion_pct = (completed / total) * 100 if total > 0 else 0
            lines.append(f"**Progress:** {completed}/{total} nodes completed ({completion_pct:.1f}%)")
            if running > 0:
                lines.append(f"**Active:** {running} nodes currently running")
            if failed > 0:
                lines.append(f"**Failed:** {failed} nodes failed")

        # Cost tracking
        total_cost = stats.get('total_cost', 0.0)
        if total_cost > 0:
            lines.append(f"**Cost:** ${total_cost:.3f}")

        # Active branches
        active_branches = status.get('active_branches', [])
        if active_branches:
            lines.append(f"**Active Branches:** {', '.join(active_branches[:3])}" +
                        (f" (+{len(active_branches)-3} more)" if len(active_branches) > 3 else ""))

        # Adapter status
        adapters = status.get('adapters', {})
        if adapters:
            adapter_summary = []
            for adapter_name, adapter_info in adapters.items():
                count = adapter_info.get('running_count', 0)
                if count > 0:
                    adapter_summary.append(f"{adapter_name}({count})")
            if adapter_summary:
                lines.append(f"**Adapters:** {', '.join(adapter_summary)}")

        return "\n".join(lines)

    async def handle_control_intent(
        self,
        action: str,
        target: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Handle control intent by sending command via ControlBus.

        Args:
            action: Control action (pause, resume, cancel, etc.)
            target: Target parameters
            session_id: Session ID

        Returns:
            Dict with control result
        """
        session_mgr = self.get_session_manager()

        if not session_mgr or not CONTROL_AVAILABLE:
            return {
                'type': 'control_intent',
                'status': 'unavailable',
                'message': 'Control system not available'
            }

        # Find active experiment for session
        experiment_id = self.get_active_experiment_for_session(session_id)

        if not experiment_id:
            return {
                'type': 'control_intent',
                'status': 'no_active_research',
                'message': 'No active research to control'
            }

        try:
            # Validate experiment is registered in session manager before sending commands
            try:
                session_mgr.get_status(experiment_id)
            except KeyError:
                return {
                    'type': 'control_intent',
                    'status': 'not_tracked',
                    'message': 'Experiment is not currently tracked; control unavailable'
                }

            # Create control message
            control_msg = ControlMessage(
                action=action,
                target=target,
                payload={},
                sender='middleware'
            )

            # Send control command
            await session_mgr.send_control(experiment_id, control_msg)
            logger.info(
                "Control intent routed",
                extra={
                    'experiment_id': experiment_id,
                    'action': action,
                    'target': target,
                }
            )

            # Handle terminal actions
            if action == 'cancel':
                removed = self._remove_active_metadata(experiment_id)
                if removed:
                    logger.debug("Removed experiment %s from active orchestrators after cancel", experiment_id)
                    session_id_for_goal = removed.session_id
                    if session_id_for_goal:
                        self._clear_session_goal_value(session_id_for_goal)
                    background_task = removed.task
                    if background_task and not background_task.done():
                        background_task.cancel()
                self._pop_progress_cache_entry(experiment_id)

            return {
                'type': 'control_intent',
                'status': 'success',
                'action': action,
                'experiment_id': experiment_id,
                'message': f"Control command '{action}' sent successfully"
            }

        except Exception as e:
            logger.error(f"Error handling control intent: {e}", exc_info=True)
            return {
                'type': 'control_intent',
                'status': 'error',
                'message': f'Error sending control command: {str(e)}'
            }

    async def process_message(
        self,
        user_message: str,
        session_id: str,
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process user message and determine execution mode.

        Phase 3 Enhancement: Detects progress queries and control intents.

        Args:
            user_message: User's input message
            session_id: Conversation/session ID
            conversation_metadata: Optional conversation metadata

        Returns:
            Dict with:
            - mode: "research" or "normal" or "progress_query" or "control_intent"
            - should_trigger_research: bool
            - task_type: TaskType
            - confidence: float
            - reasoning: Dict
            - experiment_id: Optional[str] (if research triggered)
            - progress_data: Optional[Dict] (if progress query)
            - control_result: Optional[Dict] (if control intent)
        """
        # Check for progress query first
        if self.detect_progress_query(user_message):
            logger.info(f"Detected progress query from session {session_id}")
            progress_data = await self.handle_progress_query(session_id)
            return {
                'mode': 'progress_query',
                'should_trigger_research': False,
                'task_type': TaskType.SIMPLE,
                'confidence': 1.0,
                'reasoning': {'decision': 'Progress query detected'},
                'progress_data': progress_data
            }

        # Check for control intent
        control_intent = self.detect_control_intent(user_message)
        if control_intent:
            action, target = control_intent
            logger.info(f"Detected control intent from session {session_id}: action={action}, target={target}")
            sanitized_target = target.copy()
            for key in ("node_id", "experiment_id", "branch_id"):
                if sanitized_target.get(key):
                    safe_value = self._validated_identifier(key, sanitized_target[key])
                    if safe_value is None:
                        return {
                            'mode': 'control_intent',
                            'should_trigger_research': False,
                            'task_type': TaskType.SIMPLE,
                            'confidence': 0.0,
                            'reasoning': {'decision': f'invalid {key}'},
                            'control_result': {
                                'type': 'control_intent',
                                'status': 'invalid_target',
                                'message': f'Invalid {key} supplied'
                            }
                        }
                    sanitized_target[key] = safe_value
            target = sanitized_target
            control_result = await self.handle_control_intent(action, target, session_id)
            return {
                'mode': 'control_intent',
                'should_trigger_research': False,
                'task_type': TaskType.SIMPLE,
                'confidence': 1.0,
                'reasoning': {'decision': f'Control intent detected: {action}'},
                'control_result': control_result
            }

        # Single-goal mode: if goal not yet set, attempt to detect and launch; else do not auto-trigger
        if SINGLE_GOAL_MODE:
            goal_already_set = False
            # Check explicit metadata first
            if conversation_metadata and isinstance(conversation_metadata, dict):
                goal_already_set = bool(conversation_metadata.get('research_goal')) or bool(conversation_metadata.get('research_locked'))
            # Fallback to internal tracking and active orchestrators
            if not goal_already_set:
                if self._get_session_goal_value(session_id):
                    goal_already_set = True
                elif self.get_active_experiment_for_session(session_id):
                    goal_already_set = True
            if not goal_already_set:
                should_trigger, task_type, confidence, reasoning = task_classifier.should_trigger_research(
                    user_message,
                    confidence_threshold=self.confidence_threshold,
                )
                if should_trigger:
                    try:
                        experiment_id = await self.start_research(
                            goal=user_message,
                            session_id=session_id,
                            research_type='scientific',
                            config=conversation_metadata or {},
                        )
                        # Record goal for single-goal mode
                        self._set_session_goal_value(session_id, user_message)
                        return {
                            'mode': 'research',
                            'should_trigger_research': True,
                            'task_type': task_type,
                            'confidence': confidence,
                            'reasoning': reasoning,
                            'experiment_id': experiment_id,
                            'status': 'research_started',
                            'auto_single_goal': True,
                        }
                    except Exception as e:
                        logger.error(f'Failed to start single-goal research: {e}', exc_info=True)
                        return {
                            'mode': 'normal',
                            'should_trigger_research': False,
                            'task_type': TaskType.SIMPLE,
                            'confidence': 1.0,
                            'reasoning': {'decision': 'Start failed; awaiting user retry'},
                            'error': str(e),
                        }
            # Goal already set or no trigger: no auto start
            return {
                'mode': 'normal',
                'should_trigger_research': False,
                'task_type': TaskType.SIMPLE,
                'confidence': 1.0,
                'reasoning': {'decision': 'Single-goal mode; no auto-trigger'},
            }

        # Normal flow: check if auto-trigger disabled
        if not self.enable_auto_trigger:
            return {
                'mode': 'normal',
                'should_trigger_research': False,
                'task_type': TaskType.SIMPLE,
                'confidence': 1.0,
                'reasoning': {'decision': 'Auto-trigger disabled'},
            }

        # Classify the task
        should_trigger, task_type, confidence, reasoning = task_classifier.should_trigger_research(
            user_message,
            confidence_threshold=self.confidence_threshold
        )

        result = {
            'mode': 'research' if should_trigger else 'normal',
            'should_trigger_research': should_trigger,
            'task_type': task_type,
            'confidence': confidence,
            'reasoning': reasoning,
        }

        # If research should be triggered, start it (non-blocking)
        if should_trigger:
            logger.info(f"Triggering research mode for session {session_id}")
            logger.info(f"Task type: {task_type.value}, confidence: {confidence:.2f}")
            logger.info(f"User message: {user_message[:100]}...")

            try:
                # Start research asynchronously without blocking
                experiment_id = await self.start_research(
                    goal=user_message,
                    session_id=session_id,
                    research_type='scientific',  # Can be made configurable
                    config=conversation_metadata or {}
                )
                result['experiment_id'] = experiment_id
                result['status'] = 'research_started'

                logger.info(f"Research started successfully: experiment_id={experiment_id}")

            except Exception as e:
                # Log error but don't fail - research is optional
                logger.error(f"Failed to start research: {str(e)}", exc_info=True)
                # Keep research mode indicator but mark as failed
                result['status'] = 'research_failed'
                result['error'] = str(e)
                # Don't change mode to 'normal' - let user know research was attempted

        return result

    async def start_research(
        self,
        goal: str,
        session_id: str,
        research_type: str = 'scientific',
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a research experiment.

        Args:
            goal: Research goal
            session_id: Session ID
            research_type: Type of research (scientific, code, roma)
            config: Optional configuration

        Returns:
            experiment_id: ID of the created experiment
            
        Notes:
            This method is designed to work with MultiAgentCoordinator.
            The coordinator can access the orchestrator via:
            middleware.active_orchestrators[experiment_id].orchestrator
            
            The background task runs independently, but coordinator can monitor
            completion by checking if experiment_id is still in active_orchestrators.
        """
        
        logger.info(f"[MIDDLEWARE] start_research called: session_id={session_id}, goal={goal[:100] if goal else 'N/A'}")
        import time
        import uuid

        logger.info(f"[RESEARCH_MIDDLEWARE] start_research called: session_id={session_id}, goal={goal[:100]}")

        # Use the conversation/session id as experiment id so the frontend URL matches
        experiment_id = f"exp_{session_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        logger.info(f"🔬 Starting research for session {session_id}")
        logger.info(f"📋 Experiment ID: {experiment_id}")
        logger.info(f"🎯 Goal: {goal[:100]}...")

        experiment_id = f"exp_{session_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        # Create orchestrator configuration
        config = config or {}
        max_iterations = config.get('max_iterations', 50)
        max_cost = config.get('max_cost', 10.0)
        max_parallel = config.get('max_parallel', 3)

        logger.info(f"[MIDDLEWARE] Config: max_iterations={max_iterations}, max_cost={max_cost}, max_parallel={max_parallel}")
        budget = Budget(
            max_iterations=max_iterations,
            max_cost=max_cost,
            max_tokens=config.get('max_tokens', 100000),
            deadline=None,
        )

        session_mgr = self.get_session_manager()
        event_bus = None
        control_bus = None
        if session_mgr:
            event_bus = session_mgr.event_bus or get_event_bus()
            control_bus = getattr(session_mgr, 'control_bus', None)
        else:
            try:
                event_bus = get_event_bus()
            except Exception:
                logger.debug('Unable to acquire global research event bus', exc_info=True)

        # Try to get LLM instance for intelligent node expansion
        llm = None
        try:
            # Try to get LLM from session manager
            if session_mgr and hasattr(session_mgr, 'llm'):
                llm = session_mgr.llm
                logger.info("[RESEARCH_MIDDLEWARE] Acquired LLM from session manager")
            else:
                # Try to construct LLM using OpenHands' registry
                try:
                    from openhands.llm.llm import LLM
                    from openhands.core.config import LLMConfig
                    logger.info("[RESEARCH_MIDDLEWARE] Attempting to create default LLM instance")
                    
                    # Try to create default LLM config
                    llm_config = LLMConfig()
                    llm = LLM(config=llm_config)
                    logger.info("[RESEARCH_MIDDLEWARE] Successfully created default LLM instance")
                except Exception as e:
                    logger.warning(f"[RESEARCH_MIDDLEWARE] Could not create default LLM: {e}")
                    llm = None
        except Exception as e:
            logger.warning(f"[RESEARCH_MIDDLEWARE] Failed to acquire LLM: {e}")
            llm = None
        
        logger.info(
            f"[RESEARCH_MIDDLEWARE] Creating TreeSearchOrchestrator with max_parallel={max_parallel}, llm={'available' if llm else 'unavailable'}"
        )
        orchestrator = TreeSearchOrchestrator(
            
            # DIAGNOSTIC: Log orchestrator creation (will add after creation completes)
            max_parallel=max_parallel,
            budget=budget,
            event_bus=event_bus,
            control_bus=control_bus,
            llm=llm,
        )
        logger.info(f"✅ TreeSearchOrchestrator created for {experiment_id}")
        logger.info(f"⚙️ Config: max_parallel={max_parallel}, max_iterations={max_iterations}, max_cost={max_cost}")

        logger.info(f"[RESEARCH_MIDDLEWARE] TreeSearchOrchestrator created successfully")

        # Store orchestrator and goal
        record = ExperimentRecord(
            orchestrator=orchestrator,
            goal=goal,
            session_id=session_id,
            max_iterations=max_iterations,
        )
        self._set_active_metadata(experiment_id, record)
        active_count = self._active_experiment_count()
        logger.info(f"📦 Orchestrator stored in active_orchestrators")
        logger.info(f"📊 Total active experiments: {active_count}")

        # Track session goal for single-goal mode
        self._set_session_goal_value(session_id, goal)
        logger.info(f"[RESEARCH_MIDDLEWARE] Stored orchestrator in active_orchestrators, total active: {active_count}")

        if session_mgr:
            try:
                session_mgr.register(experiment_id, orchestrator)
                logger.info(
                    f"[RESEARCH_MIDDLEWARE] Registered experiment {experiment_id} with session manager"
                )
            except Exception:
                logger.exception(
                    f"[RESEARCH_MIDDLEWARE] Failed to register experiment {experiment_id} with session manager"
                )

        # Start research in background, pass experiment_id as research_id
        logger.info(f"[RESEARCH_MIDDLEWARE] Creating background task for experiment {experiment_id}")
        background_task = asyncio.create_task(self._run_research(experiment_id))
        self._set_task_reference(experiment_id, background_task)

        logger.info(f"🚀 Background research task created for {experiment_id}")
        logger.info(f"⏳ Research will run asynchronously in background")
        logger.info(f"[RESEARCH_MIDDLEWARE] Background task created, returning experiment_id")

        return experiment_id

    async def _run_research(
        self,
        experiment_id: str,
    ):
        """Run research in background.

        Args:
            experiment_id: Experiment ID
        """
        # DIAGNOSTIC: Log background task start
        import asyncio
        import threading
        logger.info(f"[DIAGNOSTIC] Background research task started")
        logger.info(f"[DIAGNOSTIC]   Experiment ID: {experiment_id}")
        logger.info(f"[DIAGNOSTIC]   Thread: {threading.current_thread().name}")
        logger.info(f"[DIAGNOSTIC]   Event loop: {id(asyncio.get_event_loop())}")

        logger.info(f"[MIDDLEWARE] _run_research started for {experiment_id}")
        logger.info(f"[MIDDLEWARE] Thread: {threading.current_thread().name}, Event loop: {id(asyncio.get_event_loop())}")

        logger.info(f"🏃 Research execution started for {experiment_id}")

        session_mgr = self.get_session_manager()

        try:
            logger.info(f"[RESEARCH_MIDDLEWARE] _run_research started for {experiment_id}")

            exp_data = self._get_active_metadata(experiment_id)
            if not exp_data:
                logger.error(f"Experiment data not found: {experiment_id}")
                return

            orchestrator = exp_data.orchestrator
            goal = exp_data.goal
            max_iterations = exp_data.max_iterations
            logger.info(f"🎯 Goal: {goal[:100]}...")
            logger.info(f"🔢 Max iterations: {max_iterations}")

            # Run orchestrator
            logger.info(f"🌳 Starting tree search orchestrator for {experiment_id}")
            tree = await orchestrator.run(
                goal=goal,
                max_iterations=max_iterations,
                research_id=experiment_id,
            )

            logger.info(f"✅ Tree search completed for {experiment_id}")
            logger.info(f"📊 Final stats: {tree.stats if hasattr(tree, 'stats') else 'N/A'}")


            logger.info(f"Research completed: {experiment_id}")
            logger.info(f"Tree stats: {tree.stats if hasattr(tree, 'stats') else 'N/A'}")

            if session_mgr and ExperimentStatus:
                try:
                    session_mgr.update_experiment_status(
                        experiment_id, ExperimentStatus.COMPLETE
                    )
                except Exception:
                    logger.exception(
                        f"[RESEARCH_MIDDLEWARE] Failed to mark experiment {experiment_id} complete"
                    )

        except Exception as e:
            logger.error(f"Research failed: {experiment_id}, error: {str(e)}", exc_info=True)

            if session_mgr and ExperimentStatus:
                try:
                    session_mgr.update_experiment_status(
                        experiment_id, ExperimentStatus.FAILED
                    )
                except Exception:
                    logger.exception(
                        f"[RESEARCH_MIDDLEWARE] Failed to mark experiment {experiment_id} failed"
                    )

        finally:
            logger.info(f"[MIDDLEWARE] Cleaning up experiment {experiment_id}")
            # Cleanup
            removed = self._remove_active_metadata(experiment_id)
            if removed:
                logger.info(f"[COORDINATOR] Cleaning up experiment {experiment_id} from active_orchestrators")
                session_id_for_goal = removed.session_id
                if session_id_for_goal:
                    self._clear_session_goal_value(session_id_for_goal)
            self._pop_progress_cache_entry(experiment_id)
            if session_mgr:
                try:
                    session_mgr.unregister(experiment_id)
                except Exception:
                    logger.debug(
                        f"[RESEARCH_MIDDLEWARE] Failed to unregister experiment {experiment_id}",
                        exc_info=True,
                    )

    def get_orchestrator(self, experiment_id: str) -> Optional[TreeSearchOrchestrator]:
        """Get active orchestrator by experiment ID"""
        experiment_id = self._validated_identifier('experiment_id', experiment_id)
        if not experiment_id:
            return None
        exp_data = self._get_active_metadata(experiment_id)
        return exp_data.orchestrator if exp_data else None

    def get_orchestrator_for_tracking(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get orchestrator and metadata for coordinator tracking.
        
        Args:
            experiment_id: Experiment ID

        Returns:
            Dict with orchestrator, goal, session_id, max_iterations, or None if not found
        """
        experiment_id = self._validated_identifier('experiment_id', experiment_id)
        if not experiment_id:
            return None
        return self._get_active_metadata(experiment_id)
    
    def is_experiment_running(self, experiment_id: str) -> bool:
        """
        Check if experiment is still running.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if experiment is in active_orchestrators, False otherwise
        """
        experiment_id = self._validated_identifier('experiment_id', experiment_id)
        if not experiment_id:
            return False
        return self._get_active_metadata(experiment_id) is not None

    def cancel_research(self, experiment_id: str) -> bool:
        """
        Cancel active research.

        Args:
            experiment_id: Experiment ID to cancel

        Returns:
            True if cancelled, False if not found
        """
        experiment_id = self._validated_identifier('experiment_id', experiment_id)
        if not experiment_id:
            return False

        exp_data = self._get_active_metadata(experiment_id)
        if exp_data:
            orchestrator = exp_data['orchestrator']
            cancel_coro = orchestrator.cancel()
            if asyncio.iscoroutine(cancel_coro):
                asyncio.create_task(cancel_coro)
            removed = self._remove_active_metadata(experiment_id)
            logger.info(f"Cancelled research: {experiment_id}")
            self._pop_progress_cache_entry(experiment_id)
            if removed:
                background_task = removed.task
                if background_task and not background_task.done():
                    background_task.cancel()
            if removed and removed.session_id:
                self._clear_session_goal_value(removed.session_id)
            return True
        return False


# Global middleware instance
# Configured via config.py or environment variables
# To enable: export ENABLE_AUTO_RESEARCH_TRIGGER=true
research_middleware = ResearchMiddleware(
    confidence_threshold=RESEARCH_CONFIDENCE_THRESHOLD,
    enable_auto_trigger=ENABLE_AUTO_RESEARCH_TRIGGER,
    poll_interval=RESEARCH_POLL_INTERVAL,
    progress_cache_ttl=PROGRESS_CACHE_TTL,
    coordination_enabled=ENABLE_AGENT_COORDINATION,
)
