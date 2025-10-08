"""
Code Research Agent - Research coordination capable CodeAct agent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.events.action import Action, MessageAction
from openhands.events.event import EventSource
from openhands.llm.llm_registry import LLMRegistry

from ..engines.code_research import CodeResearchEngine
from ..middleware.research_middleware import research_middleware
from ..services.research_session_manager import ResearchSessionManager

logger = logging.getLogger(__name__)


class CodeResearchAgent(CodeActAgent):
    """Coordinator that keeps repository analysis responsive."""

    VERSION = "2.0"

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        super().__init__(config, llm_registry)

        self.code_engine = CodeResearchEngine(
            llm=self.llm,
            config=getattr(config, 'code_research_config', {}),
        )

        self._coordination_mode = False
        self._experiment_id: Optional[str] = None
        self._experiment_session_id: Optional[str] = None
        self._last_poll_time: float = 0.0
        self._last_user_message: Optional[str] = None
        self._session_manager: Optional[ResearchSessionManager] = None
        self._pending_control_results: List[Dict[str, Any]] = []

        default_poll = getattr(research_middleware, 'poll_interval', 10)
        self._poll_interval = getattr(config, 'research_poll_interval', default_poll)
        logger.info(
            "CodeResearchAgent initialized with coordination poll interval %ss",
            self._poll_interval,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(self, state: State) -> Action:
        if not getattr(research_middleware, 'coordination_enabled', True):
            return super().step(state)

        if self._coordination_mode:
            pending_result = self._dequeue_control_result()
            if pending_result:
                return MessageAction(content=pending_result, thought="Control result")

            latest_user_message = self._extract_latest_user_message(state)
            if latest_user_message and latest_user_message != self._last_user_message:
                self._last_user_message = latest_user_message
                user_response = self._handle_user_query_during_research(
                    latest_user_message, state
                )
                if user_response:
                    return user_response

            if self._should_poll():
                return self._poll_research_progress()

            return MessageAction(
                content="Repository analysis is underway. I'll surface the next update shortly.",
                thought="Awaiting next poll",
            )

        if self._is_code_analysis_task(state):
            return self._start_coordination(state)

        return super().step(state)

    def reset(self) -> None:
        super().reset()
        self._coordination_mode = False
        self._experiment_id = None
        self._experiment_session_id = None
        self._last_poll_time = 0.0
        self._last_user_message = None
        self._pending_control_results.clear()
        logger.info("CodeResearchAgent reset coordination state")

    # ------------------------------------------------------------------
    # Coordination helpers
    # ------------------------------------------------------------------
    def _is_code_analysis_task(self, state: State) -> bool:
        code_keywords = [
            'analyze code',
            'understand repository',
            'find implementation',
            'code structure',
            'architecture',
            'how does',
            'where is',
            'explain code',
            'trace',
            'dependency',
            'code review',
            'code quality',
        ]

        if self._coordination_mode:
            return True

        if state.history:
            latest_events = state.history.get_events_as_list()
            for event in reversed(latest_events[-10:]):
                if getattr(event, 'source', None) != EventSource.USER:
                    continue
                message = getattr(event, 'message', None)
                if not message:
                    continue
                lowered = message.lower()
                if any(keyword in lowered for keyword in code_keywords):
                    return True

        return False

    def _start_coordination(self, state: State) -> Action:
        goal = self._extract_query(state)
        session_id = self._resolve_session_id(state)

        try:
            experiment_id = self._run_coroutine_sync(
                research_middleware.start_research(
                    goal=goal,
                    session_id=session_id,
                    research_type='code',
                    config={},
                )
            )
        except Exception as exc:  # pragma: no cover - defensive log path
            logger.exception("Failed to start code research experiment")
            return MessageAction(
                content=(
                    "I tried to launch an in-depth repository analysis but hit an "
                    f"error: {exc}. I'll fall back to standard reasoning."
                ),
                thought="Research launch failed",
            )

        if not experiment_id:
            return MessageAction(
                content="Repository analysis request queued. I'll provide updates soon.",
                thought="Research queued",
            )

        self._coordination_mode = True
        self._experiment_id = experiment_id
        self._experiment_session_id = session_id
        self._last_poll_time = time.time()
        self._last_user_message = None
        logger.info(
            "Code research coordination started", extra={'experiment_id': experiment_id}
        )

        return MessageAction(
            content=(
                "Starting a deep repository analysis now. Expect updates every "
                f"{self._poll_interval} seconds. You can always ask for a status "
                "check or issue commands like pause or cancel."
            ),
            thought="Research started",
        )

    def _poll_research_progress(self) -> Action:
        manager = self._ensure_session_manager()
        if not manager or not self._experiment_id:
            self._coordination_mode = False
            return MessageAction(
                content="I can't reach the analysis tracker anymore. Returning to normal flow.",
                thought="Coordination disabled",
            )

        try:
            status = manager.get_status(self._experiment_id)
        except KeyError:
            self._coordination_mode = False
            self._experiment_id = None
            return MessageAction(
                content="The analysis session is no longer active. Let's continue manually.",
                thought="Experiment missing",
            )

        self._last_poll_time = time.time()
        formatted = self._format_progress_message(status)
        experiment_status = status.get('status')

        if experiment_status in {'complete', 'failed', 'cancelled'}:
            self._coordination_mode = False
            self._experiment_id = None
            completion_note = (
                "Code analysis completed!" if experiment_status == 'complete'
                else "Code analysis stopped before completion."
            )
            return MessageAction(
                content=f"{completion_note}\n\n{formatted}",
                thought="Research finished",
            )

        return MessageAction(
            content=formatted,
            thought="Repository analysis progress",
        )

    def _handle_user_query_during_research(
        self, message: str, state: State
    ) -> Optional[Action]:
        if research_middleware.detect_progress_query(message):
            return self._poll_research_progress()

        intent = research_middleware.detect_control_intent(message)
        if intent:
            action, target = intent
            result = self._execute_control_command(action, target, state)
            if isinstance(result, dict):
                status = result.get('status', 'pending')
                msg = result.get('message') or f"Sent '{action}' command to the analysis session."
                return MessageAction(content=msg, thought=f"Control intent {status}")
            if isinstance(result, str):
                return MessageAction(content=result, thought="Control intent ack")
            return MessageAction(
                content=f"Control command '{action}' queued. I'll confirm once it's processed.",
                thought="Control intent pending",
            )

        return MessageAction(
            content=(
                "Analysis is running in the background. Ask 'how's progress?' for a "
                "status update or issue commands like 'pause research' as needed."
            ),
            thought="General coordination guidance",
        )

    def _execute_control_command(
        self, action: str, target: Dict[str, Any], state: State
    ) -> Dict[str, Any] | str:
        session_id = self._experiment_session_id or self._resolve_session_id(state)

        async def _send() -> Dict[str, Any]:
            return await research_middleware.handle_control_intent(
                action=action,
                target=target,
                session_id=session_id,
            )

        try:
            return self._run_coroutine_sync(_send())
        except Exception as exc:  # pragma: no cover - defensive log path
            logger.exception("Failed to send control command")
            return {
                'status': 'error',
                'message': f"Control command '{action}' failed: {exc}",
            }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _ensure_session_manager(self) -> Optional[ResearchSessionManager]:
        if self._session_manager is None:
            self._session_manager = research_middleware.get_session_manager()
        return self._session_manager

    def _format_progress_message(self, status: Dict[str, Any]) -> str:
        stats = status.get('stats', {})
        total = stats.get('total_nodes', 0)
        completed = stats.get('completed', 0)
        running = stats.get('running', 0)
        failed = stats.get('failed', 0)
        total_cost = stats.get('total_cost', 0.0)

        lines = [
            "[Code Analysis Progress]",
            f"Status: {status.get('status', 'unknown')}",
            f"Nodes: {completed}/{total} complete, {running} running, {failed} failed",
        ]

        if total_cost:
            lines.append(f"Cost: ${total_cost:.3f}")

        active_branches = status.get('active_branches', [])
        if active_branches:
            lines.append(
                "Active branches: " + ", ".join(active_branches[:3]) + (
                    f" (+{len(active_branches) - 3} more)"
                    if len(active_branches) > 3
                    else ""
                )
            )

        adapters = status.get('adapters', {})
        if adapters:
            adapter_status = []
            for name, info in adapters.items():
                running_count = info.get('running_count')
                if running_count:
                    adapter_status.append(f"{name}({running_count})")
            if adapter_status:
                lines.append("Active adapters: " + ", ".join(adapter_status))

        return "\n".join(lines)

    def _extract_query(self, state: State) -> str:
        if state.history:
            latest_events = state.history.get_events_as_list()
            for event in reversed(latest_events):
                if getattr(event, 'source', None) != EventSource.USER:
                    continue
                message = getattr(event, 'message', None)
                if message:
                    return message
        return "Analyze the repository"

    def _extract_latest_user_message(self, state: State) -> Optional[str]:
        if not state.history:
            return None
        for event in reversed(state.history.get_events_as_list()):
            if getattr(event, 'source', None) == EventSource.USER:
                message = getattr(event, 'message', None)
                if message:
                    return message
        return None

    def _resolve_session_id(self, state: State) -> str:
        if state.session_id:
            return state.session_id
        return state.extra_data.get('session_id') if state.extra_data else ''

    def _should_poll(self) -> bool:
        return (time.time() - self._last_poll_time) >= self._poll_interval

    def _dequeue_control_result(self) -> Optional[str]:
        if not self._pending_control_results:
            return None
        result = self._pending_control_results.pop(0)
        if isinstance(result, dict):
            message = result.get('message')
            if message:
                return message
        return str(result)

    def _run_coroutine_sync(self, coro):
        try:
            return asyncio.run(coro)
        except RuntimeError:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError as inner_exc:
                logger.error("Unable to acquire running loop: %s", inner_exc)
                return None

            future = loop.create_task(coro)

            def _capture(fut: asyncio.Future) -> None:
                try:
                    result = fut.result()
                except Exception as exc:  # pragma: no cover - background error path
                    logger.exception("Background coroutine failed")
                    result = {
                        'status': 'error',
                        'message': f"Background task failed: {exc}",
                    }
                self._pending_control_results.append(result)

            future.add_done_callback(_capture)
            return None


# Register agent type
AGENT_CLS = CodeResearchAgent
