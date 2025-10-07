"""
ControlBus - Typed command routing for research orchestration

Enables bidirectional communication between:
- Main agent / UI → Orchestrator
- Main agent / UI → Subagents
- Orchestrator → Subagents

Supports runtime control actions:
- pause/resume/cancel entire experiments
- cancel specific branches/nodes
- reprioritize nodes or adapters
- add new research directions
- steer subagents with natural language
"""

import asyncio
import logging
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ControlMessage(BaseModel):
    """
    Control command message for research orchestration.

    Examples:
        # Pause experiment
        ControlMessage(action="pause")

        # Cancel specific node
        ControlMessage(
            action="cancel_node",
            target={"node_id": "idea-2"},
            payload={}
        )

        # Steer adapter
        ControlMessage(
            action="steer",
            target={"adapter": "codeact"},
            payload={"text": "Focus on DARTS, ignore ENAS"}
        )

        # Reprioritize
        ControlMessage(
            action="reprioritize",
            target={"adapter": "codeact"},
            payload={"delta": 0.2}
        )
    """

    action: str = Field(
        ...,
        description="Control action: pause, resume, cancel, cancel_node, reprioritize, add_node, steer"
    )
    target: Dict[str, str] = Field(
        default_factory=dict,
        description="Target selector: experiment_id, branch_id, adapter, node_id, node_type"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific data"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Command timestamp"
    )
    sender: Optional[str] = Field(
        default=None,
        description="Who sent this command (user, main_agent, orchestrator)"
    )


class ControlBus:
    """
    Typed command bus for research control.

    Provides pub/sub mechanism for control messages scoped by experiment_id.
    Multiple subscribers can listen to same experiment (orchestrator + subagents).

    Thread-safe and async-safe using asyncio.Queue per experiment.

    Example:
        # Create control bus
        bus = ControlBus()

        # Subscriber (orchestrator)
        async for cmd in bus.subscribe("exp-123"):
            if cmd.action == "pause":
                orchestrator.pause()

        # Publisher (main agent / UI)
        await bus.publish("exp-123", ControlMessage(action="pause"))
    """

    def __init__(self):
        """Initialize control bus with per-experiment queues"""
        # experiment_id -> list of queues (multiple subscribers)
        self._subscribers: Dict[str, list[asyncio.Queue]] = {}

        # Track active experiments
        self._active_experiments: set[str] = set()

        # Stats
        self._stats = {
            "total_messages": 0,
            "messages_by_action": {},
            "active_subscriptions": 0,
        }

        logger.info("ControlBus initialized")

    async def publish(self, experiment_id: str, message: ControlMessage):
        """
        Publish control message to all subscribers of an experiment.

        Args:
            experiment_id: Target experiment ID
            message: Control message to send

        Example:
            await bus.publish(
                "exp-123",
                ControlMessage(
                    action="cancel_node",
                    target={"node_id": "idea-2"}
                )
            )
        """
        if experiment_id not in self._subscribers:
            logger.warning(
                f"No subscribers for experiment {experiment_id}, "
                f"message discarded: {message.action}"
            )
            return

        # Update stats
        self._stats["total_messages"] += 1
        self._stats["messages_by_action"][message.action] = (
            self._stats["messages_by_action"].get(message.action, 0) + 1
        )

        # Publish to all subscribers
        queues = self._subscribers[experiment_id]
        for queue in queues:
            await queue.put(message)

        logger.info(
            f"Published control message: {message.action} "
            f"to {len(queues)} subscribers for experiment {experiment_id}"
        )

    async def subscribe(self, experiment_id: str) -> AsyncIterator[ControlMessage]:
        """
        Subscribe to control messages for an experiment.

        Yields control messages as they arrive.
        Blocks until messages are available or experiment ends.

        Args:
            experiment_id: Experiment to subscribe to

        Yields:
            ControlMessage instances

        Example:
            async for cmd in bus.subscribe("exp-123"):
                if cmd.action == "pause":
                    # Handle pause
                    break
        """
        # Create queue for this subscriber
        queue: asyncio.Queue[ControlMessage] = asyncio.Queue()

        # Register queue
        if experiment_id not in self._subscribers:
            self._subscribers[experiment_id] = []

        self._subscribers[experiment_id].append(queue)
        self._active_experiments.add(experiment_id)
        self._stats["active_subscriptions"] += 1

        logger.info(
            f"New subscription to experiment {experiment_id}, "
            f"total subscribers: {len(self._subscribers[experiment_id])}"
        )

        try:
            while experiment_id in self._active_experiments:
                try:
                    # Wait for message with timeout to check if experiment ended
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield message

                except asyncio.TimeoutError:
                    # No message within timeout, continue loop
                    continue

        finally:
            # Cleanup on unsubscribe
            if experiment_id in self._subscribers:
                if queue in self._subscribers[experiment_id]:
                    self._subscribers[experiment_id].remove(queue)

                # Remove experiment if no more subscribers
                if not self._subscribers[experiment_id]:
                    del self._subscribers[experiment_id]

            self._stats["active_subscriptions"] -= 1

            logger.info(
                f"Unsubscribed from experiment {experiment_id}, "
                f"remaining subscribers: "
                f"{len(self._subscribers.get(experiment_id, []))}"
            )

    def unsubscribe_all(self, experiment_id: str):
        """
        Remove all subscriptions for an experiment.

        Call this when experiment completes/cancels.

        Args:
            experiment_id: Experiment to cleanup
        """
        if experiment_id in self._active_experiments:
            self._active_experiments.remove(experiment_id)

        if experiment_id in self._subscribers:
            count = len(self._subscribers[experiment_id])
            del self._subscribers[experiment_id]
            logger.info(
                f"Removed all {count} subscriptions for experiment {experiment_id}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get control bus statistics.

        Returns:
            Stats dict with message counts and active subscriptions
        """
        return {
            **self._stats,
            "active_experiments": len(self._active_experiments),
            "experiments": list(self._active_experiments),
        }

    def has_subscribers(self, experiment_id: str) -> bool:
        """Check if experiment has any active subscribers"""
        return experiment_id in self._subscribers and len(self._subscribers[experiment_id]) > 0


# Global control bus instance (singleton)
_control_bus: Optional[ControlBus] = None


def get_control_bus() -> ControlBus:
    """Get global control bus instance (singleton pattern)"""
    global _control_bus
    if _control_bus is None:
        _control_bus = ControlBus()
    return _control_bus


# Example usage and tests
async def test_control_bus():
    """Test control bus functionality"""
    bus = ControlBus()

    # Create subscriber task
    async def subscriber_task():
        count = 0
        async for msg in bus.subscribe("test-exp"):
            logger.info(f"Received: {msg.action}")
            count += 1
            if count >= 3:
                break

    # Start subscriber
    sub_task = asyncio.create_task(subscriber_task())

    # Give subscriber time to start
    await asyncio.sleep(0.1)

    # Publish messages
    await bus.publish("test-exp", ControlMessage(action="pause"))
    await bus.publish("test-exp", ControlMessage(action="resume"))
    await bus.publish("test-exp", ControlMessage(
        action="steer",
        target={"adapter": "codeact"},
        payload={"text": "Focus on performance"}
    ))

    # Wait for subscriber to finish
    await sub_task

    # Check stats
    stats = bus.get_stats()
    logger.info(f"Stats: {stats}")

    assert stats["total_messages"] == 3
    assert stats["messages_by_action"]["pause"] == 1
    assert stats["messages_by_action"]["resume"] == 1
    assert stats["messages_by_action"]["steer"] == 1

    logger.info("✓ All control bus tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_control_bus())
