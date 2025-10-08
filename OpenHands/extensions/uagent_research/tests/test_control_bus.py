"""Unit tests for the ControlBus message routing system."""

from __future__ import annotations

import asyncio
from typing import Iterable, List

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.control_bus import ControlBus, ControlMessage, get_control_bus


async def collect_messages(bus: ControlBus, experiment_id: str, count: int) -> List[ControlMessage]:
    """Subscribe to the control bus and collect ``count`` messages."""

    received: List[ControlMessage] = []
    done = asyncio.Event()

    async def _subscriber():
        async for message in bus.subscribe(experiment_id):
            received.append(message)
            if len(received) >= count:
                done.set()
                break

    task = asyncio.create_task(_subscriber())
    await asyncio.wait_for(done.wait(), timeout=5)
    bus.unsubscribe_all(experiment_id)
    await asyncio.wait_for(task, timeout=5)
    return received


async def publish_messages(bus: ControlBus, experiment_id: str, messages: Iterable[ControlMessage]) -> None:
    for message in messages:
        await bus.publish(experiment_id, message)


def test_control_message_creation():
    message = ControlMessage(action="pause")
    assert message.action == "pause"
    assert message.target == {}
    assert message.payload == {}
    assert message.timestamp
    assert message.sender is None

    resume = ControlMessage(action="resume", sender="user")
    assert resume.action == "resume"
    assert resume.sender == "user"

    cancel_node = ControlMessage(
        action="cancel_node",
        target={"node_id": "idea-2"},
        payload={"reason": "redundant"},
    )
    assert cancel_node.target == {"node_id": "idea-2"}
    assert cancel_node.payload == {"reason": "redundant"}

    steer = ControlMessage(
        action="steer",
        target={"adapter": "codeact"},
        payload={"text": "Focus on accuracy"},
    )
    assert steer.target["adapter"] == "codeact"
    assert steer.payload["text"] == "Focus on accuracy"


def test_control_bus_initialization():
    bus = ControlBus()
    stats = bus.get_stats()
    assert stats["total_messages"] == 0
    assert stats["active_subscriptions"] == 0
    assert stats["experiments"] == []
    assert not bus.has_subscribers("any-exp")


@pytest.mark.asyncio
async def test_publish_subscribe_basic():
    bus = ControlBus()
    results: List[ControlMessage] = []

    async def subscriber():
        async for message in bus.subscribe("exp-basic"):
            results.append(message)
            if len(results) >= 3:
                break

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.05)

    await bus.publish("exp-basic", ControlMessage(action="pause"))
    await bus.publish("exp-basic", ControlMessage(action="resume"))
    await bus.publish("exp-basic", ControlMessage(action="cancel"))

    await asyncio.wait_for(task, timeout=5)
    bus.unsubscribe_all("exp-basic")

    assert [msg.action for msg in results] == ["pause", "resume", "cancel"]
    stats = bus.get_stats()
    assert stats["total_messages"] == 3
    assert stats["messages_by_action"]["pause"] == 1
    assert stats["active_subscriptions"] == 0


@pytest.mark.asyncio
async def test_multiple_subscribers():
    bus = ControlBus()
    received = [[] for _ in range(3)]

    async def make_subscriber(index: int):
        async for message in bus.subscribe("exp-multi"):
            received[index].append(message)
            if len(received[index]) >= 2:
                break

    tasks = [asyncio.create_task(make_subscriber(i)) for i in range(3)]
    await asyncio.sleep(0.05)

    for action in ("pause", "resume"):
        await bus.publish("exp-multi", ControlMessage(action=action))

    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    bus.unsubscribe_all("exp-multi")

    for bucket in received:
        assert [msg.action for msg in bucket] == ["pause", "resume"]

    stats = bus.get_stats()
    assert stats["total_messages"] == 2
    assert stats["active_subscriptions"] == 0


@pytest.mark.asyncio
async def test_publish_no_subscribers(caplog):
    bus = ControlBus()

    with caplog.at_level("WARNING"):
        await bus.publish("exp-none", ControlMessage(action="pause"))

    assert any("No subscribers" in record.message for record in caplog.records)
    stats = bus.get_stats()
    assert stats["total_messages"] == 0


@pytest.mark.asyncio
async def test_unsubscribe_cleanup():
    bus = ControlBus()
    tracker: List[ControlMessage] = []

    async def subscriber():
        async for message in bus.subscribe("exp-clean"):
            tracker.append(message)
            break

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.05)
    await bus.publish("exp-clean", ControlMessage(action="pause"))
    await asyncio.wait_for(task, timeout=5)

    assert bus.has_subscribers("exp-clean") or "exp-clean" in bus.get_stats()["experiments"]
    bus.unsubscribe_all("exp-clean")
    assert not bus.has_subscribers("exp-clean")
    stats = bus.get_stats()
    assert stats["active_subscriptions"] == 0
    assert "exp-clean" not in stats["experiments"]


@pytest.mark.asyncio
async def test_unsubscribe_all():
    bus = ControlBus()

    async def subscriber():
        async for _ in bus.subscribe("exp-unsub"):
            await asyncio.sleep(0.01)

    tasks = [asyncio.create_task(subscriber()) for _ in range(3)]
    await asyncio.sleep(0.05)

    bus.unsubscribe_all("exp-unsub")
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)

    stats = bus.get_stats()
    assert not bus.has_subscribers("exp-unsub")
    assert "exp-unsub" not in stats["experiments"]
    assert stats["active_subscriptions"] == 0
    assert bus.get_stats()["active_subscriptions"] == 0


@pytest.mark.asyncio
async def test_has_subscribers():
    bus = ControlBus()
    assert not bus.has_subscribers("exp-check")

    async def subscriber():
        async for _ in bus.subscribe("exp-check"):
            break

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.05)
    assert bus.has_subscribers("exp-check")
    bus.unsubscribe_all("exp-check")
    await asyncio.wait_for(task, timeout=5)
    assert not bus.has_subscribers("exp-check")


@pytest.mark.asyncio
async def test_message_routing_by_experiment():
    bus = ControlBus()
    results_1: List[str] = []
    results_2: List[str] = []

    async def subscriber(exp_id: str, bucket: List[str]):
        async for message in bus.subscribe(exp_id):
            bucket.append(message.action)
            break

    task1 = asyncio.create_task(subscriber("exp-1", results_1))
    task2 = asyncio.create_task(subscriber("exp-2", results_2))
    await asyncio.sleep(0.05)

    await bus.publish("exp-1", ControlMessage(action="pause"))
    await asyncio.sleep(0.05)
    await bus.publish("exp-2", ControlMessage(action="resume"))

    await asyncio.wait_for(asyncio.gather(task1, task2), timeout=5)
    bus.unsubscribe_all("exp-1")
    bus.unsubscribe_all("exp-2")

    assert results_1 == ["pause"]
    assert results_2 == ["resume"]


@pytest.mark.asyncio
async def test_stats_tracking():
    bus = ControlBus()
    collector: List[ControlMessage] = []

    async def subscriber():
        async for message in bus.subscribe("exp-stats"):
            collector.append(message)
            if len(collector) >= 5:
                break

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.05)

    actions = ["pause", "resume", "cancel", "steer", "reprioritize"]
    for action in actions:
        await bus.publish("exp-stats", ControlMessage(action=action))

    await asyncio.wait_for(task, timeout=5)
    pre_stats = bus.get_stats()
    assert "exp-stats" in pre_stats["experiments"]
    bus.unsubscribe_all("exp-stats")

    stats = bus.get_stats()
    assert stats["total_messages"] == 5
    for action in actions:
        assert stats["messages_by_action"][action] == 1
    assert "exp-stats" not in stats["experiments"]


@pytest.mark.asyncio
async def test_control_message_types():
    bus = ControlBus()
    received: List[ControlMessage] = []

    async def subscriber():
        async for message in bus.subscribe("exp-types"):
            received.append(message)
            if len(received) >= 7:
                break

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.05)

    messages = [
        ControlMessage(action="pause"),
        ControlMessage(action="resume"),
        ControlMessage(action="cancel"),
        ControlMessage(action="cancel_node", target={"node_id": "idea-2"}),
        ControlMessage(action="steer", target={"adapter": "codeact"}, payload={"text": "Focus"}),
        ControlMessage(action="reprioritize", target={"adapter": "codeact"}, payload={"delta": 0.2}),
        ControlMessage(action="add_node", payload={"node": {"title": "New idea"}}),
    ]

    await publish_messages(bus, "exp-types", messages)
    await asyncio.wait_for(task, timeout=5)
    bus.unsubscribe_all("exp-types")

    assert [msg.action for msg in received] == [m.action for m in messages]


def test_singleton_pattern():
    first = get_control_bus()
    second = get_control_bus()
    assert first is second


@pytest.mark.asyncio
async def test_concurrent_publish_subscribe():
    bus = ControlBus()
    received: List[ControlMessage] = []

    async def subscriber():
        async for message in bus.subscribe("exp-concurrent"):
            received.append(message)
            if len(received) >= 10:
                break

    async def publisher():
        for idx in range(10):
            await bus.publish("exp-concurrent", ControlMessage(action=f"action-{idx}"))

    sub_task = asyncio.create_task(subscriber())
    pub_task = asyncio.create_task(publisher())
    await asyncio.wait_for(asyncio.gather(sub_task, pub_task), timeout=5)
    bus.unsubscribe_all("exp-concurrent")

    assert len(received) == 10
    assert received[0].action == "action-0"


@pytest.mark.asyncio
async def test_subscription_timeout():
    bus = ControlBus()
    result: List[ControlMessage] = []

    async def subscriber():
        async for message in bus.subscribe("exp-timeout"):
            result.append(message)
            break

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(2)
    await bus.publish("exp-timeout", ControlMessage(action="pause"))
    await asyncio.wait_for(task, timeout=5)
    bus.unsubscribe_all("exp-timeout")

    assert [msg.action for msg in result] == ["pause"]


@pytest.mark.asyncio
async def test_sender_field():
    bus = ControlBus()
    received: List[str] = []

    async def subscriber():
        async for message in bus.subscribe("exp-sender"):
            received.append(message.sender or "")
            if len(received) >= 3:
                break

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.05)

    senders = ["user", "main_agent", "orchestrator"]
    for sender in senders:
        await bus.publish("exp-sender", ControlMessage(action="pause", sender=sender))

    await asyncio.wait_for(task, timeout=5)
    bus.unsubscribe_all("exp-sender")

    assert received == senders


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
