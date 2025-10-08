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
from typing import Optional, Dict, Any, Tuple

from ..classifier.task_classifier import task_classifier, TaskType
from ..orchestrator.tree_orchestrator import TreeSearchOrchestrator
from ..uagent_research.models.research_tree import Budget

try:
    from ..config import (
        ENABLE_AUTO_RESEARCH_TRIGGER,
        ENABLE_AGENT_COORDINATION,
        RESEARCH_CONFIDENCE_THRESHOLD,
        RESEARCH_POLL_INTERVAL,
        PROGRESS_CACHE_TTL,
    )
except ImportError:
    # Fallback if config not available
    ENABLE_AUTO_RESEARCH_TRIGGER = False
    ENABLE_AGENT_COORDINATION = True
    RESEARCH_CONFIDENCE_THRESHOLD = 0.7
    RESEARCH_POLL_INTERVAL = 10
    PROGRESS_CACHE_TTL = 2.0

# Enforce single-goal per conversation: do not auto-trigger new research from chat
SINGLE_GOAL_MODE = True

# Import control components
try:
    from ..control.control_bus import ControlMessage
    from ..services.research_session_manager import (
        ResearchSessionManager,
        ExperimentStatus,
    )
    from ..orchestrator.event_bus import get_event_bus
    CONTROL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Control components not available: {e}")
    CONTROL_AVAILABLE = False
    ControlMessage = None
    ResearchSessionManager = None
    ExperimentStatus = None

logger = logging.getLogger(__name__)


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
        self.active_orchestrators: Dict[str, Dict[str, Any]] = {}
        # Track goal by session for single-goal mode
        self._session_goal: Dict[str, str] = {}

        # Session manager for progress queries
        self._session_manager = session_manager

        # Coordination / polling configuration
        self.poll_interval = poll_interval
        self.coordination_enabled = coordination_enabled
        self._progress_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._progress_cache_ttl = progress_cache_ttl

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
        try:
            from ..adapters.base.agent_adapter import adapter_registry
            from ..adapters.deepresearch.adapter import DeepResearchAdapter
            from ..adapters.repomaster.adapter import RepoMasterAdapter
            from ..adapters.codeact.adapter import CodeActAdapter

            # Register adapters with default configs (name comes from class attribute)
            adapter_registry.register(DeepResearchAdapter(config={}))
            adapter_registry.register(RepoMasterAdapter(config={}))
            adapter_registry.register(CodeActAdapter(config={}))

            logger.info("Research adapters registered successfully: deepresearch, repomaster, codeact")
        except Exception as e:
            logger.error(f"Failed to register adapters: {e}", exc_info=True)

    def get_session_manager(self) -> Optional['ResearchSessionManager']:
        """Get or create session manager"""
        if self._session_manager is None and CONTROL_AVAILABLE:
            try:
                from ..control.control_bus import ControlBus
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
        for experiment_id, metadata in self.active_orchestrators.items():
            if isinstance(metadata, dict) and metadata.get('session_id') == session_id:
                return experiment_id

        # Fallback heuristic: experiment identifiers prefixed with exp_{session_id}_
        for experiment_id in self.active_orchestrators.keys():
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
            cached = self._progress_cache.get(experiment_id)
            if cached and (now - cached[1]) <= self._progress_cache_ttl:
                status = cached[0]
            else:
                status = session_mgr.get_status(experiment_id)
                self._progress_cache[experiment_id] = (status, now)

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
                # Remove from active orchestrators
                if experiment_id in self.active_orchestrators:
                    del self.active_orchestrators[experiment_id]
                self._progress_cache.pop(experiment_id, None)

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
                if session_id in self._session_goal:
                    goal_already_set = True
                else:
                    # Also consider background API-started experiments with exp_{session_id}_*
                    if session_id in self.active_orchestrators:
                        goal_already_set = True
                    else:
                        for exp_id in list(self.active_orchestrators.keys()):
                            if exp_id.startswith(f"exp_{session_id}_"):
                                goal_already_set = True
                                break
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
                        self._session_goal[session_id] = user_message
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
        """
        import time
        import uuid

        logger.info(f"[RESEARCH_MIDDLEWARE] start_research called: session_id={session_id}, goal={goal[:100]}")

        # Use the conversation/session id as experiment id so the frontend URL matches
        experiment_id = session_id
        logger.info(f"[RESEARCH_MIDDLEWARE] Using experiment_id = session_id: {experiment_id}")

        # Create orchestrator configuration
        config = config or {}
        max_iterations = config.get('max_iterations', 50)
        max_cost = config.get('max_cost', 10.0)
        max_parallel = config.get('max_parallel', 3)

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

        logger.info(
            f"[RESEARCH_MIDDLEWARE] Creating TreeSearchOrchestrator with max_parallel={max_parallel}"
        )
        orchestrator = TreeSearchOrchestrator(
            max_parallel=max_parallel,
            budget=budget,
            event_bus=event_bus,
            control_bus=control_bus,
        )
        logger.info(f"[RESEARCH_MIDDLEWARE] TreeSearchOrchestrator created successfully")

        # Store orchestrator and goal
        self.active_orchestrators[experiment_id] = {
            'orchestrator': orchestrator,
            'goal': goal,
            'session_id': session_id,
            'max_iterations': max_iterations,
        }
        # Track session goal for single-goal mode
        self._session_goal[session_id] = goal
        logger.info(f"[RESEARCH_MIDDLEWARE] Stored orchestrator in active_orchestrators, total active: {len(self.active_orchestrators)}")

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
        asyncio.create_task(self._run_research(experiment_id))
        logger.info(f"[RESEARCH_MIDDLEWARE] Background task created, returning experiment_id")

        return experiment_id

    async def _run_research(
        self,
        experiment_id: str,
    ):
        """
        Run research in background.

        Args:
            experiment_id: Experiment ID
        """
        session_mgr = self.get_session_manager()

        try:
            logger.info(f"[RESEARCH_MIDDLEWARE] _run_research started for {experiment_id}")

            exp_data = self.active_orchestrators.get(experiment_id)
            if not exp_data:
                logger.error(f"Experiment data not found: {experiment_id}")
                return

            orchestrator = exp_data['orchestrator']
            goal = exp_data['goal']
            max_iterations = exp_data['max_iterations']

            # Run orchestrator
            tree = await orchestrator.run(
                goal=goal,
                max_iterations=max_iterations,
                research_id=experiment_id,
            )

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
            # Cleanup
            if experiment_id in self.active_orchestrators:
                del self.active_orchestrators[experiment_id]
            self._progress_cache.pop(experiment_id, None)
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
        exp_data = self.active_orchestrators.get(experiment_id)
        return exp_data['orchestrator'] if exp_data else None

    def cancel_research(self, experiment_id: str) -> bool:
        """
        Cancel active research.

        Args:
            experiment_id: Experiment ID to cancel

        Returns:
            True if cancelled, False if not found
        """
        exp_data = self.active_orchestrators.get(experiment_id)
        if exp_data:
            orchestrator = exp_data['orchestrator']
            orchestrator.cancel()
            del self.active_orchestrators[experiment_id]
            logger.info(f"Cancelled research: {experiment_id}")
            self._progress_cache.pop(experiment_id, None)
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
