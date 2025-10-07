"""Pytest fixtures and test doubles for the UAgent research extension.

This module provides shared fixtures used across the integration test suites. The
goal is to supply lightweight, fully controllable stand-ins for the core
OpenHands services (LLM, runtime, event stream, etc.) so that higher-level
components can be exercised deterministically.
"""

from __future__ import annotations

from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PACKAGE_DIR = _PACKAGE_ROOT / 'uagent_research'
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

try:
    import openhands.runtime as _openhands_runtime
    sys.modules.setdefault('openhands.runtime.runtime', _openhands_runtime)
except ImportError:  # pragma: no cover - optional dependency
    pass

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest

import importlib

tests_pkg = importlib.import_module(__package__)
_load_module = getattr(tests_pkg, '_load_module')
_PACKAGE_DIR = tests_pkg._PACKAGE_DIR  # type: ignore[attr-defined]

_load_module('uagent_research', _PACKAGE_DIR / '__init__.py')
_load_module('uagent_research.models', _PACKAGE_DIR / 'models' / '__init__.py')
_load_module('uagent_research.models.base', _PACKAGE_DIR / 'models' / 'base.py')

from openhands.core.config import AgentConfig
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation

from uagent_research.models.base import close_database, init_database


class MockLLM:
    """A controllable LLM stand-in for testing."""

    def __init__(self) -> None:
        # Populate minimal attributes referenced by production code
        self.service_id = 'mock-llm'
        self.config = SimpleNamespace(model='mock-llm', temperature=0.0)

        self._response_queue: List[str] = []
        self.call_history: List[Dict[str, Any]] = []

    def _serialise_response(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content)

    def queue_response(self, content: Any) -> None:
        """Append a response that will be returned on the next call."""

        self._response_queue.append(self._serialise_response(content))

    def set_next_response(self, content: Any) -> None:
        """Replace the queue with a single response."""

        self._response_queue = [self._serialise_response(content)]

    def set_responses(self, *responses: Any) -> None:
        """Convenience method to seed multiple responses at once."""

        self._response_queue = [self._serialise_response(r) for r in responses]

    def reset(self) -> None:
        """Clear recorded state between tests."""

        self._response_queue.clear()
        self.call_history.clear()

    async def completion(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        """Return the next queued response."""

        self.call_history.append({'args': args, 'kwargs': kwargs})

        if self._response_queue:
            content = self._response_queue.pop(0)
        else:
            content = json.dumps({'result': 'ok'})

        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class MockRuntime:
    """Minimal runtime that records executed actions."""

    def __init__(self) -> None:
        self.executed_actions: List[CmdRunAction] = []
        self._command_outputs: Dict[str, CmdOutputObservation] = {}
        self.default_output = 'mock runtime output'

    def set_command_output(
        self,
        command: str,
        *,
        content: str | None = None,
        exit_code: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        observation = CmdOutputObservation(
            content=content or '',
            command=command,
            exit_code=exit_code,
            metadata=metadata or {},
        )
        self._command_outputs[command] = observation

    def get_executed_commands(self) -> List[str]:
        return [getattr(action, 'command', '') for action in self.executed_actions]

    def reset(self) -> None:
        self.executed_actions.clear()
        self._command_outputs.clear()

    async def run_action(self, action: CmdRunAction) -> CmdOutputObservation:
        self.executed_actions.append(action)
        command = getattr(action, 'command', '')
        if command in self._command_outputs:
            return self._command_outputs[command]

        return CmdOutputObservation(
            content=self.default_output,
            command=command,
            exit_code=0,
            metadata={},
        )


class MockEventStream:
    """In-memory event recorder with awaitable helpers."""

    def __init__(self) -> None:
        self._events: List[Any] = []
        self._notification = asyncio.Event()
        self._cursor = 0

    async def add_event(self, event: Any) -> None:
        self._events.append(event)
        self._notification.set()

    def get_events(self) -> List[Any]:
        return list(self._events)

    def get_events_by_type(self, event_type: type) -> List[Any]:
        return [event for event in self._events if isinstance(event, event_type)]

    def clear(self) -> None:
        self._events.clear()
        self._cursor = 0

    async def wait_for_event(
        self,
        predicate: Optional[Callable[[Any], bool]] = None,
        *,
        timeout: float = 2.0,
        poll_interval: float = 0.05,
    ) -> Any:
        predicate = predicate or (lambda _: True)

        async def _poll() -> Any:
            index = self._cursor
            while True:
                while index < len(self._events):
                    event = self._events[index]
                    index += 1
                    if predicate(event):
                        self._cursor = index
                        return event

                self._notification.clear()
                try:
                    await asyncio.wait_for(self._notification.wait(), poll_interval)
                except asyncio.TimeoutError:
                    pass

        return await asyncio.wait_for(_poll(), timeout=timeout)


class MockHistory:
    """History container exposing ``get_events_as_list`` like the real View."""

    def __init__(self) -> None:
        self._events: List[SimpleNamespace] = []

    def add_message(self, message: str, *, source: str = 'user') -> None:
        self._events.append(SimpleNamespace(message=message, source=source))

    def get_events_as_list(self) -> List[SimpleNamespace]:
        return list(self._events)

    def __bool__(self) -> bool:
        return bool(self._events)


class MockState:
    """Minimal drop-in replacement for ``openhands.controller.state.State``."""

    def __init__(self) -> None:
        self.history = MockHistory()

    def add_user_message(self, message: str) -> None:
        self.history.add_message(message, source='user')


class MockLLMRegistry:
    """Simple registry that always returns the supplied mock LLM."""

    def __init__(self, llm: MockLLM) -> None:
        self._llm = llm

    def get_llm(self, service_id: str, config: Any | None = None) -> MockLLM:
        return self._llm

    def get_active_llm(self) -> MockLLM:
        return self._llm

    def get_llm_from_agent_config(
        self, service_id: str, agent_config: AgentConfig
    ) -> MockLLM:
        return self._llm

    def get_router(self, agent_config: AgentConfig) -> MockLLM:
        return self._llm


@pytest.fixture
def mock_llm() -> MockLLM:
    llm = MockLLM()
    yield llm
    llm.reset()


@pytest.fixture
def mock_runtime() -> MockRuntime:
    runtime = MockRuntime()
    yield runtime
    runtime.reset()


@pytest.fixture
def mock_event_stream() -> MockEventStream:
    return MockEventStream()


@pytest.fixture
def mock_state() -> MockState:
    return MockState()


@pytest.fixture
def mock_llm_registry(mock_llm: MockLLM) -> MockLLMRegistry:
    return MockLLMRegistry(mock_llm)


@pytest.fixture(scope='function')
async def test_db() -> None:
    """Initialise an in-memory SQLite database for each test."""

    await init_database('sqlite+aiosqlite:///:memory:')
    try:
        yield
    finally:
        await close_database()
