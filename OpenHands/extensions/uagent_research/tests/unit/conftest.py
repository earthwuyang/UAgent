"""Shared pytest fixtures for unit tests in uagent research."""

import asyncio
from typing import Any, Dict, List

import pytest

from OpenHands.extensions.uagent_research.control.control_bus import ControlBus, ControlMessage


@pytest.fixture
def sample_control_message() -> Any:
    """Factory for creating ControlMessage instances with default values."""

    def _factory(**overrides: Dict[str, Any]) -> ControlMessage:
        defaults: Dict[str, Any] = {"action": "pause"}
        return ControlMessage(**{**defaults, **overrides})

    return _factory


@pytest.fixture
def control_bus() -> ControlBus:
    """Provide a fresh ControlBus for each test."""
    return ControlBus()


@pytest.fixture
def experiment_id() -> str:
    """Return canonical experiment identifier used by unit tests."""
    return "test-exp-123"


@pytest.fixture
async def subscriber_helper():
    """Collect messages from a ControlBus subscription for testing."""

    async def _helper(bus: ControlBus, experiment: str, count: int = 1, timeout: float = 5.0) -> List[ControlMessage]:
        messages: List[ControlMessage] = []

        async def _subscriber():
            async for message in bus.subscribe(experiment):
                messages.append(message)
                if len(messages) >= count:
                    bus.unsubscribe_all(experiment)
                    break

        task = asyncio.create_task(_subscriber())
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            bus.unsubscribe_all(experiment)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return messages

    return _helper

