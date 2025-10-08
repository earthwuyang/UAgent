"""Tests for CodeActAdapter steering support."""

from __future__ import annotations

import asyncio
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.codeact.adapter import CodeActAdapter
from uagent_research.models.research_tree import Task, Context
from uagent_research.models.events import StepEvent, CompleteEvent


class DummySession:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.cancelled = False

    async def send_user_message(self, message: str) -> None:
        self.messages.append(message)

    async def cancel(self) -> None:
        self.cancelled = True


class TestableCodeActAdapter(CodeActAdapter):
    """CodeActAdapter variant with simplified execution for testing."""

    def __init__(self) -> None:
        super().__init__(config={})
        self._events_emitted: list[str] = []
        self._session_factory = DummySession

    async def run(self, task: Task, context: Context):  # type: ignore[override]
        self._current_session = self._session_factory()
        step = StepEvent(branch_id=context.branch_id, node_id=task.id, action="start")
        step.adapter_name = self.name  # type: ignore[attr-defined]
        yield step
        await asyncio.sleep(0)
        complete = CompleteEvent(
            branch_id=context.branch_id,
            node_id=task.id,
            summary="done",
            artifacts=[],
            success=True,
        )
        complete.adapter_name = self.name  # type: ignore[attr-defined]
        yield complete


@pytest.mark.asyncio
async def test_send_message_active_session():
    adapter = CodeActAdapter(config={})
    session = DummySession()
    adapter._current_session = session  # type: ignore[attr-defined]

    await adapter.send_message("Focus on unit tests")

    assert session.messages == ["Focus on unit tests"]


@pytest.mark.asyncio
async def test_send_message_no_session(caplog):
    adapter = CodeActAdapter(config={})

    with caplog.at_level("WARNING"):
        await adapter.send_message("Unused message")

    assert any("no active session" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_send_message_during_run():
    adapter = TestableCodeActAdapter()
    task = Task(id="node-1", goal="Run sample", context="")
    context = Context(branch_id="branch-1", parent_nodes=[])

    async def consume():
        async for _ in adapter.run(task, context):
            await adapter.send_message("Add logging")

    await consume()

    assert adapter._current_session.messages == ["Add logging"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_multiple_steering_messages():
    adapter = TestableCodeActAdapter()
    adapter._current_session = DummySession()

    await adapter.send_message("First")
    await adapter.send_message("Second")

    assert adapter._current_session.messages == ["First", "Second"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_send_message_after_cancel():
    adapter = TestableCodeActAdapter()
    adapter._current_session = DummySession()

    await adapter.cancel()
    adapter._current_session = DummySession()
    await adapter.send_message("After cancel")

    assert adapter._current_session.messages == ["After cancel"]  # type: ignore[attr-defined]
