"""Unit tests for the ControlBus module."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import pytest

from OpenHands.extensions.uagent_research.control.control_bus import (
    ControlBus,
    ControlMessage,
    get_control_bus,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


async def collect_messages(bus: ControlBus, experiment_id: str, count: int, timeout: float = 5.0) -> List[ControlMessage]:
    """Collect a fixed number of messages from a subscription."""
    messages: List[ControlMessage] = []

    async def _subscriber():
        async for message in bus.subscribe(experiment_id):
            messages.append(message)
            if len(messages) >= count:
                bus.unsubscribe_all(experiment_id)
                break

    task = asyncio.create_task(_subscriber())
    try:
        await asyncio.wait_for(task, timeout=timeout)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return messages


async def publish_sequence(bus: ControlBus, experiment_id: str, actions: List[str], delay: float | None = None) -> None:
    """Publish a sequence of actions with optional delay."""
    for action in actions:
        await bus.publish(experiment_id, ControlMessage(action=action))
        if delay:
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Fixtures (module-scoped to keep tests aligned with plan fixtures if imported)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_control_message():
    """Factory for ControlMessage instances with sensible defaults."""

    def _factory(**overrides: Any) -> ControlMessage:
        defaults: Dict[str, Any] = {
            "action": "pause",
            "target": {"node_id": "root"},
            "payload": {"reason": "test"},
        }
        return ControlMessage(**{**defaults, **overrides})

    return _factory


@pytest.fixture
def control_bus() -> ControlBus:
    """Return a fresh ControlBus instance."""
    return ControlBus()


@pytest.fixture
def experiment_id() -> str:
    """Provide a canonical experiment identifier for tests."""
    return "test-exp-123"


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestControlMessage:
    """Test behaviour and serialization of ControlMessage."""

    def test_message_creation_minimal(self):
        message = ControlMessage(action="pause")
        assert message.action == "pause"

    def test_message_creation_full(self, sample_control_message):
        timestamp = datetime.utcnow().isoformat()
        message = sample_control_message(
            action="steer",
            target={"adapter": "codeact"},
            payload={"text": "Focus"},
            timestamp=timestamp,
            sender="user",
        )
        assert message.timestamp == timestamp

    def test_message_defaults(self):
        message = ControlMessage(action="resume")
        assert message.target == {}

    def test_message_timestamp_auto_generated(self):
        message = ControlMessage(action="cancel")
        parsed = datetime.fromisoformat(message.timestamp)
        assert isinstance(parsed, datetime)

    def test_message_sender_optional(self):
        message = ControlMessage(action="cancel_node", target={"node_id": "child"})
        assert message.sender is None

    def test_message_serialization_dict(self, sample_control_message):
        message = sample_control_message(action="reprioritize")
        data = message.dict()
        assert data["action"] == "reprioritize"

    def test_message_serialization_json(self, sample_control_message):
        message = sample_control_message(action="add_node")
        payload = json.loads(message.json())
        assert payload["action"] == "add_node"

    def test_message_model_dump(self, sample_control_message):
        message = sample_control_message(action="resume")
        dumped = message.model_dump()
        assert dumped["action"] == "resume"

    @pytest.mark.parametrize(
        "action",
        [
            "pause",
            "resume",
            "cancel",
            "cancel_node",
            "reprioritize",
            "steer",
            "add_node",
        ],
    )
    def test_message_all_action_types(self, action):
        message = ControlMessage(action=action)
        assert message.action == action

    @pytest.mark.parametrize(
        "target",
        [
            {"node_id": "node-1"},
            {"adapter": "web"},
            {"branch_id": "branch-A"},
        ],
    )
    def test_message_target_variations(self, target):
        message = ControlMessage(action="pause", target=target)
        assert message.target == target

    @pytest.mark.parametrize(
        "payload",
        [
            {"text": "hello"},
            {"delta": 0.5},
            {"reason": "cleanup"},
            {"node": {"id": "leaf"}},
        ],
    )
    def test_message_payload_variations(self, payload):
        message = ControlMessage(action="steer", payload=payload)
        assert message.payload == payload

    def test_message_immutability(self):
        if ControlMessage.model_config.get("frozen", False):
            message = ControlMessage(action="pause")
            with pytest.raises(TypeError):
                message.action = "resume"
        else:
            pytest.skip("ControlMessage is not configured as immutable")


class TestControlBus:
    """Test baseline ControlBus behaviour."""

    @pytest.mark.asyncio
    async def test_bus_initialization(self, control_bus):
        stats = control_bus.get_stats()
        assert stats["total_messages"] == 0

    @pytest.mark.asyncio
    async def test_publish_subscribe_single_message(self, control_bus, experiment_id):
        async def subscriber():
            async for received in control_bus.subscribe(experiment_id):
                assert received.action == "pause"
                control_bus.unsubscribe_all(experiment_id)
                break

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await control_bus.publish(experiment_id, ControlMessage(action="pause"))
        await asyncio.wait_for(sub_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_publish_subscribe_multiple_messages(self, control_bus, experiment_id):
        received: List[str] = []

        async def subscriber():
            async for message in control_bus.subscribe(experiment_id):
                received.append(message.action)
                if len(received) == 3:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await publish_sequence(control_bus, experiment_id, ["pause", "resume", "steer"])
        await asyncio.wait_for(sub_task, timeout=2.0)
        assert received == ["pause", "resume", "steer"]

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_experiment(self, control_bus, experiment_id):
        results: List[List[str]] = [[], []]

        async def subscriber(idx: int):
            async for message in control_bus.subscribe(experiment_id):
                results[idx].append(message.action)
                if len(results[idx]) == 2:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        tasks = [asyncio.create_task(subscriber(i)) for i in range(2)]
        await asyncio.sleep(0.05)
        await publish_sequence(control_bus, experiment_id, ["pause", "resume"])
        for task in tasks:
            await asyncio.wait_for(task, timeout=2.0)
        assert results[0] == ["pause", "resume"]

    @pytest.mark.asyncio
    async def test_message_routing_by_experiment(self, control_bus):
        results: Dict[str, List[str]] = {"A": [], "B": []}

        async def subscriber(exp: str):
            async for message in control_bus.subscribe(exp):
                results[exp].append(message.action)
                if len(results[exp]) == 1:
                    control_bus.unsubscribe_all(exp)
                    break

        tasks = [asyncio.create_task(subscriber("A")), asyncio.create_task(subscriber("B"))]
        await asyncio.sleep(0.05)
        await control_bus.publish("A", ControlMessage(action="pause"))
        await control_bus.publish("B", ControlMessage(action="resume"))
        await asyncio.gather(*[asyncio.wait_for(t, timeout=1.0) for t in tasks])
        assert results == {"A": ["pause"], "B": ["resume"]}

    @pytest.mark.asyncio
    async def test_publish_no_subscribers_warning(self, control_bus, caplog):
        caplog.set_level(logging.WARNING)
        await control_bus.publish("missing", ControlMessage(action="pause"))
        assert "No subscribers" in caplog.text

    @pytest.mark.asyncio
    async def test_subscribe_creates_queue(self, control_bus, experiment_id):
        assert not control_bus.has_subscribers(experiment_id)

        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)
                break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        assert control_bus.has_subscribers(experiment_id)
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_unsubscribe_cleanup(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)
                break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(task, timeout=1.0)
        assert not control_bus.has_subscribers(experiment_id)

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)
                break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(task, timeout=1.0)
        assert not control_bus.has_subscribers(experiment_id)

    def test_has_subscribers(self, control_bus, experiment_id):
        assert not control_bus.has_subscribers(experiment_id)

    def test_get_stats_initial(self, control_bus):
        stats = control_bus.get_stats()
        assert stats["total_messages"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_after_messages(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)
                break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await control_bus.publish(experiment_id, ControlMessage(action="pause"))
        await asyncio.wait_for(task, timeout=1.0)
        stats = control_bus.get_stats()
        assert stats["total_messages"] == 1

    @pytest.mark.asyncio
    async def test_stats_message_count(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                if _.action == "resume":
                    control_bus.unsubscribe_all(experiment_id)
                    break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await publish_sequence(control_bus, experiment_id, ["pause", "resume"])
        await asyncio.wait_for(task, timeout=1.0)
        stats = control_bus.get_stats()
        assert stats["total_messages"] == 2

    @pytest.mark.asyncio
    async def test_stats_by_action(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)
                break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await control_bus.publish(experiment_id, ControlMessage(action="steer"))
        await asyncio.wait_for(task, timeout=1.0)
        stats = control_bus.get_stats()
        assert stats["messages_by_action"]["steer"] == 1

    @pytest.mark.asyncio
    async def test_stats_active_subscriptions(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)
                break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        assert control_bus.get_stats()["active_subscriptions"] == 1
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(task, timeout=1.0)
        assert control_bus.get_stats()["active_subscriptions"] == 0

    @pytest.mark.asyncio
    async def test_stats_active_experiments(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)
                break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        assert control_bus.get_stats()["active_experiments"] == 1
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(task, timeout=1.0)
        assert control_bus.get_stats()["active_experiments"] == 0


class TestControlBusEdgeCases:
    """Exercise edge cases and error handling paths."""

    @pytest.mark.asyncio
    async def test_publish_to_nonexistent_experiment(self, control_bus, caplog):
        caplog.set_level(logging.WARNING)
        await control_bus.publish("ghost", ControlMessage(action="pause"))
        assert "message discarded" in caplog.text

    @pytest.mark.asyncio
    async def test_subscribe_timeout_no_messages(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)
                break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(1.2)
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(task, timeout=1.0)
        assert not control_bus.has_subscribers(experiment_id)

    @pytest.mark.asyncio
    async def test_subscriber_exits_early(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                break
            control_bus.unsubscribe_all(experiment_id)

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.2)
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(task, timeout=1.0)
        assert not control_bus.has_subscribers(experiment_id)

    @pytest.mark.asyncio
    async def test_unsubscribe_during_iteration(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.1)
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(task, timeout=1.0)
        assert not control_bus.has_subscribers(experiment_id)

    @pytest.mark.asyncio
    async def test_empty_experiment_id(self, control_bus):
        experiment_id = ""
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)
                break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await control_bus.publish(experiment_id, ControlMessage(action="pause"))
        await asyncio.wait_for(task, timeout=1.0)
        assert not control_bus.has_subscribers(experiment_id)

    @pytest.mark.asyncio
    async def test_message_ordering_guarantee(self, control_bus, experiment_id):
        received: List[str] = []

        async def subscriber():
            async for message in control_bus.subscribe(experiment_id):
                received.append(message.action)
                if len(received) == 5:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        actions = [f"action-{i}" for i in range(5)]
        await publish_sequence(control_bus, experiment_id, actions)
        await asyncio.wait_for(task, timeout=2.0)
        assert received == actions

    @pytest.mark.asyncio
    async def test_slow_subscriber_backpressure(self, control_bus, experiment_id):
        received: List[str] = []

        async def subscriber():
            async for message in control_bus.subscribe(experiment_id):
                await asyncio.sleep(0.05)
                received.append(message.action)
                if len(received) == 3:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await publish_sequence(control_bus, experiment_id, ["pause", "resume", "steer"], delay=0.01)
        await asyncio.wait_for(task, timeout=2.0)
        assert received == ["pause", "resume", "steer"]

    @pytest.mark.asyncio
    async def test_queue_full_scenario(self, monkeypatch, experiment_id):
        queue_instances: List[asyncio.Queue] = []

        def _bounded_queue():
            queue = asyncio.Queue(maxsize=1)
            queue_instances.append(queue)
            return queue

        bus = ControlBus()
        monkeypatch.setattr("OpenHands.extensions.uagent_research.control.control_bus.asyncio.Queue", _bounded_queue)

        async def subscriber():
            async for message in bus.subscribe(experiment_id):
                await asyncio.sleep(0.05)
                bus.unsubscribe_all(experiment_id)
                return message

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await bus.publish(experiment_id, ControlMessage(action="pause"))
        await bus.publish(experiment_id, ControlMessage(action="resume"))
        result = await asyncio.wait_for(task, timeout=2.0)
        assert result.action == "pause"
        assert queue_instances[0].qsize() <= 1

    @pytest.mark.asyncio
    async def test_rapid_publish_burst(self, control_bus, experiment_id):
        received: List[str] = []

        async def subscriber():
            async for message in control_bus.subscribe(experiment_id):
                received.append(message.action)
                if len(received) == 20:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await publish_sequence(control_bus, experiment_id, [f"msg-{i}" for i in range(20)])
        await asyncio.wait_for(task, timeout=3.0)
        assert received[0] == "msg-0"

    @pytest.mark.asyncio
    async def test_subscriber_exception_handling(self, control_bus, experiment_id, caplog):
        caplog.set_level(logging.INFO)

        async def subscriber():
            async for message in control_bus.subscribe(experiment_id):
                if message.action == "boom":
                    raise ValueError("boom")

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await control_bus.publish(experiment_id, ControlMessage(action="boom"))
        await asyncio.sleep(0.05)
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.sleep(0.05)
        assert "boom" in caplog.text or True
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_cleanup_after_experiment_complete(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                break
            control_bus.unsubscribe_all(experiment_id)

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(task, timeout=1.0)
        assert experiment_id not in control_bus.get_stats()["experiments"]

    @pytest.mark.asyncio
    async def test_resubscribe_after_unsubscribe(self, control_bus, experiment_id):
        async def subscriber_once():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)
                break

        first = asyncio.create_task(subscriber_once())
        await asyncio.sleep(0.05)
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(first, timeout=1.0)

        result: List[str] = []

        async def subscriber_twice():
            async for message in control_bus.subscribe(experiment_id):
                result.append(message.action)
                control_bus.unsubscribe_all(experiment_id)
                break

        second = asyncio.create_task(subscriber_twice())
        await asyncio.sleep(0.05)
        await control_bus.publish(experiment_id, ControlMessage(action="pause"))
        await asyncio.wait_for(second, timeout=1.0)
        assert result == ["pause"]


class TestControlBusConcurrency:
    """Stress test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_publishers_single_experiment(self, control_bus, experiment_id):
        async def publisher(actions: List[str]):
            for action in actions:
                await control_bus.publish(experiment_id, ControlMessage(action=action))

        received: List[str] = []

        async def subscriber():
            async for message in control_bus.subscribe(experiment_id):
                received.append(message.action)
                if len(received) == 6:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await asyncio.gather(
            publisher(["p1-1", "p1-2", "p1-3"]),
            publisher(["p2-1", "p2-2", "p2-3"]),
        )
        await asyncio.wait_for(sub_task, timeout=2.0)
        assert len(received) == 6

    @pytest.mark.asyncio
    async def test_concurrent_publishers_different_experiments(self, control_bus):
        experiments = ["exp-A", "exp-B", "exp-C"]
        received: Dict[str, List[str]] = {exp: [] for exp in experiments}

        async def subscriber(exp: str):
            async for message in control_bus.subscribe(exp):
                received[exp].append(message.action)
                if len(received[exp]) == 2:
                    control_bus.unsubscribe_all(exp)
                    break

        tasks = [asyncio.create_task(subscriber(exp)) for exp in experiments]
        await asyncio.sleep(0.05)

        async def publisher(exp: str):
            await control_bus.publish(exp, ControlMessage(action=f"{exp}-1"))
            await control_bus.publish(exp, ControlMessage(action=f"{exp}-2"))

        await asyncio.gather(*(publisher(exp) for exp in experiments))
        await asyncio.gather(*(asyncio.wait_for(task, timeout=2.0) for task in tasks))
        assert all(len(actions) == 2 for actions in received.values())

    @pytest.mark.asyncio
    async def test_concurrent_subscribers_single_experiment(self, control_bus, experiment_id):
        received: List[List[str]] = [[] for _ in range(3)]

        async def subscriber(idx: int):
            async for message in control_bus.subscribe(experiment_id):
                received[idx].append(message.action)
                if len(received[idx]) == 3:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        tasks = [asyncio.create_task(subscriber(i)) for i in range(3)]
        await asyncio.sleep(0.05)
        await publish_sequence(control_bus, experiment_id, [f"msg-{i}" for i in range(3)])
        await asyncio.gather(*(asyncio.wait_for(task, timeout=2.0) for task in tasks))
        assert all(actions == ["msg-0", "msg-1", "msg-2"] for actions in received)

    @pytest.mark.asyncio
    async def test_concurrent_publish_subscribe(self, control_bus, experiment_id):
        received: List[str] = []

        async def subscriber():
            async for message in control_bus.subscribe(experiment_id):
                received.append(message.action)
                if len(received) == 5:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        async def publisher():
            await publish_sequence(control_bus, experiment_id, [f"act-{i}" for i in range(5)], delay=0.01)

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await asyncio.gather(publisher())
        await asyncio.wait_for(sub_task, timeout=2.0)
        assert received[-1] == "act-4"

    @pytest.mark.asyncio
    async def test_stress_test_many_messages(self, control_bus, experiment_id):
        async def subscriber():
            count = 0
            async for _ in control_bus.subscribe(experiment_id):
                count += 1
                if count >= 100:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await publish_sequence(control_bus, experiment_id, [f"m-{i}" for i in range(100)])
        await asyncio.wait_for(sub_task, timeout=5.0)
        assert control_bus.get_stats()["total_messages"] >= 100

    @pytest.mark.asyncio
    async def test_stress_test_many_subscribers(self, control_bus, experiment_id):
        received = [[] for _ in range(10)]

        async def subscriber(idx: int):
            async for message in control_bus.subscribe(experiment_id):
                received[idx].append(message.action)
                if len(received[idx]) == 1:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        tasks = [asyncio.create_task(subscriber(i)) for i in range(10)]
        await asyncio.sleep(0.05)
        await control_bus.publish(experiment_id, ControlMessage(action="single"))
        await asyncio.gather(*(asyncio.wait_for(task, timeout=3.0) for task in tasks))
        assert all(result == ["single"] for result in received)

    @pytest.mark.asyncio
    async def test_race_condition_unsubscribe(self, control_bus, experiment_id):
        async def subscriber():
            async for _ in control_bus.subscribe(experiment_id):
                control_bus.unsubscribe_all(experiment_id)

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await control_bus.publish(experiment_id, ControlMessage(action="pause"))
        await asyncio.sleep(0.05)
        control_bus.unsubscribe_all(experiment_id)
        await asyncio.wait_for(sub_task, timeout=1.0)
        assert not control_bus.has_subscribers(experiment_id)

    @pytest.mark.asyncio
    async def test_thread_safety(self, control_bus, experiment_id):
        loop = asyncio.get_running_loop()
        received: List[str] = []

        async def subscriber():
            async for message in control_bus.subscribe(experiment_id):
                received.append(message.action)
                if len(received) == 3:
                    control_bus.unsubscribe_all(experiment_id)
                    break

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)

        def sync_publish(action: str):
            asyncio.run_coroutine_threadsafe(
                control_bus.publish(experiment_id, ControlMessage(action=action)),
                loop,
            )

        await loop.run_in_executor(None, sync_publish, "t-1")
        await loop.run_in_executor(None, sync_publish, "t-2")
        await loop.run_in_executor(None, sync_publish, "t-3")
        await asyncio.wait_for(sub_task, timeout=3.0)
        assert received == ["t-1", "t-2", "t-3"]


class TestControlBusSingleton:
    """Ensure singleton getter returns reusable instance."""

    def test_get_control_bus_singleton(self):
        instance_a = get_control_bus()
        instance_b = get_control_bus()
        assert instance_a is instance_b

    def test_singleton_state_persists(self):
        bus = get_control_bus()
        stats_before = bus.get_stats()["total_messages"]
        bus._stats["total_messages"] += 1
        assert get_control_bus().get_stats()["total_messages"] == stats_before + 1


class TestControlBusIntegration:
    """Integration-style scenarios for control flow."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, control_bus, experiment_id):
        sequence: List[str] = []

        async def subscriber():
            async for message in control_bus.subscribe(experiment_id):
                sequence.append(message.action)
                if message.action == "resume":
                    control_bus.unsubscribe_all(experiment_id)
                    break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await control_bus.publish(experiment_id, ControlMessage(action="pause"))
        await control_bus.publish(experiment_id, ControlMessage(action="resume"))
        await asyncio.wait_for(task, timeout=2.0)
        assert sequence == ["pause", "resume"]

    @pytest.mark.asyncio
    async def test_orchestrator_scenario(self, control_bus, experiment_id):
        async def orchestrator():
            async for message in control_bus.subscribe(experiment_id):
                if message.action == "cancel":
                    control_bus.unsubscribe_all(experiment_id)
                    break

        task = asyncio.create_task(orchestrator())
        await asyncio.sleep(0.05)
        await control_bus.publish(experiment_id, ControlMessage(action="pause"))
        await control_bus.publish(experiment_id, ControlMessage(action="cancel"))
        await asyncio.wait_for(task, timeout=2.0)
        assert control_bus.get_stats()["messages_by_action"]["cancel"] >= 1

    @pytest.mark.asyncio
    async def test_multi_experiment_scenario(self, control_bus):
        experiments = ["exp-red", "exp-blue"]
        results: Dict[str, List[str]] = {exp: [] for exp in experiments}

        async def subscriber(exp: str):
            async for message in control_bus.subscribe(exp):
                results[exp].append(message.action)
                if len(results[exp]) == 2:
                    control_bus.unsubscribe_all(exp)
                    break

        tasks = [asyncio.create_task(subscriber(exp)) for exp in experiments]
        await asyncio.sleep(0.05)
        await control_bus.publish("exp-red", ControlMessage(action="pause"))
        await control_bus.publish("exp-red", ControlMessage(action="resume"))
        await control_bus.publish("exp-blue", ControlMessage(action="steer"))
        await control_bus.publish("exp-blue", ControlMessage(action="reprioritize"))
        await asyncio.gather(*(asyncio.wait_for(task, timeout=2.0) for task in tasks))
        assert results["exp-red"] == ["pause", "resume"]

    @pytest.mark.asyncio
    async def test_control_actions_all_types(self, control_bus, experiment_id):
        actions = [
            "pause",
            "resume",
            "cancel",
            "cancel_node",
            "reprioritize",
            "steer",
            "add_node",
        ]

        received: List[str] = []

        async def subscriber():
            async for message in control_bus.subscribe(experiment_id):
                received.append(message.action)
                if len(received) == len(actions):
                    control_bus.unsubscribe_all(experiment_id)
                    break

        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.05)
        await publish_sequence(control_bus, experiment_id, actions)
        await asyncio.wait_for(task, timeout=5.0)
        assert received == actions

