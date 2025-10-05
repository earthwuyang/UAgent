"""
Event Bus - Real-time Event Streaming

Streams research events to frontend via SSE/WebSocket with:
- Event coalescing (batch similar events)
- Backpressure handling (drop old events if buffer full)
- Multiple subscriber support
- Type-safe event routing
"""

import asyncio
import logging
from typing import Dict, Set, Optional, AsyncIterator, List, Callable
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass
import json

from ..uagent_research.models.events import ResearchEvent, EventType

logger = logging.getLogger(__name__)


@dataclass
class EventSubscription:
    """Subscription configuration"""
    subscriber_id: str
    event_types: Optional[Set[EventType]] = None  # None = all types
    branch_ids: Optional[Set[str]] = None  # None = all branches
    buffer_size: int = 100  # Max events to buffer
    coalesce_window_ms: int = 100  # Coalesce events within this window


class EventBus:
    """
    Central event bus for research execution.

    Features:
    - Multiple subscribers with filtering
    - Event coalescing (combine rapid similar events)
    - Backpressure handling (drop old events if subscriber slow)
    - SSE/WebSocket streaming
    - Graceful shutdown

    Example:
        bus = EventBus()

        # Subscribe to events
        async for event in bus.subscribe(
            subscriber_id="frontend-1",
            event_types={EventType.STEP, EventType.COMPLETE}
        ):
            print(f"Received: {event.type}")

        # Publish events
        await bus.publish(StepEvent(...))
    """

    def __init__(self, max_buffer_size: int = 1000):
        """
        Initialize event bus.

        Args:
            max_buffer_size: Maximum events to buffer per subscriber
        """
        self.max_buffer_size = max_buffer_size

        # Subscriber management
        self._subscribers: Dict[str, EventSubscription] = {}
        self._event_queues: Dict[str, asyncio.Queue] = {}
        self._active_subscribers: Set[str] = set()

        # Event coalescing
        self._coalesce_buffers: Dict[str, List[ResearchEvent]] = defaultdict(list)
        self._coalesce_tasks: Dict[str, asyncio.Task] = {}

        # Statistics
        self.stats = {
            "events_published": 0,
            "events_delivered": 0,
            "events_dropped": 0,
            "events_coalesced": 0,
            "active_subscribers": 0,
        }

        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        subscriber_id: str,
        event_types: Optional[Set[EventType]] = None,
        branch_ids: Optional[Set[str]] = None,
        buffer_size: int = 100,
        coalesce_window_ms: int = 100,
    ) -> AsyncIterator[ResearchEvent]:
        """
        Subscribe to research events.

        Args:
            subscriber_id: Unique subscriber identifier
            event_types: Filter by event types (None = all)
            branch_ids: Filter by branch IDs (None = all)
            buffer_size: Max events to buffer
            coalesce_window_ms: Coalesce events within this window

        Yields:
            ResearchEvent objects matching filters

        Example:
            async for event in bus.subscribe("frontend-1"):
                if event.type == EventType.COMPLETE:
                    print("Task completed!")
        """
        async with self._lock:
            # Create subscription
            subscription = EventSubscription(
                subscriber_id=subscriber_id,
                event_types=event_types,
                branch_ids=branch_ids,
                buffer_size=buffer_size,
                coalesce_window_ms=coalesce_window_ms,
            )

            self._subscribers[subscriber_id] = subscription
            self._event_queues[subscriber_id] = asyncio.Queue(maxsize=buffer_size)
            self._active_subscribers.add(subscriber_id)

            self.stats["active_subscribers"] = len(self._active_subscribers)

            logger.info(
                f"Subscriber '{subscriber_id}' registered "
                f"(filters: types={event_types}, branches={branch_ids})"
            )

        try:
            queue = self._event_queues[subscriber_id]

            while subscriber_id in self._active_subscribers:
                try:
                    # Wait for event with timeout to allow graceful shutdown
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)

                    self.stats["events_delivered"] += 1
                    yield event

                except asyncio.TimeoutError:
                    # No events, continue waiting
                    continue

        finally:
            # Cleanup on unsubscribe
            await self.unsubscribe(subscriber_id)

    async def unsubscribe(self, subscriber_id: str):
        """
        Unsubscribe from events.

        Args:
            subscriber_id: Subscriber to remove
        """
        async with self._lock:
            if subscriber_id in self._active_subscribers:
                self._active_subscribers.discard(subscriber_id)
                self._subscribers.pop(subscriber_id, None)
                self._event_queues.pop(subscriber_id, None)

                # Cancel coalesce task if exists
                if subscriber_id in self._coalesce_tasks:
                    self._coalesce_tasks[subscriber_id].cancel()
                    self._coalesce_tasks.pop(subscriber_id)

                self.stats["active_subscribers"] = len(self._active_subscribers)

                logger.info(f"Subscriber '{subscriber_id}' unregistered")

    async def publish(self, event: ResearchEvent):
        """
        Publish event to all matching subscribers.

        Args:
            event: Research event to publish

        Example:
            await bus.publish(StepEvent(
                branch_id="branch-1",
                description="Analyzing results...",
                artifacts=[]
            ))
        """
        self.stats["events_published"] += 1

        async with self._lock:
            for subscriber_id in list(self._active_subscribers):
                # Check if subscriber matches filters
                if not self._matches_subscription(event, subscriber_id):
                    continue

                subscription = self._subscribers[subscriber_id]

                # Coalescing enabled?
                if subscription.coalesce_window_ms > 0:
                    await self._add_to_coalesce_buffer(subscriber_id, event)
                else:
                    await self._deliver_event(subscriber_id, event)

    def _matches_subscription(self, event: ResearchEvent, subscriber_id: str) -> bool:
        """
        Check if event matches subscriber filters.

        Args:
            event: Event to check
            subscriber_id: Subscriber ID

        Returns:
            True if event matches filters
        """
        subscription = self._subscribers.get(subscriber_id)
        if not subscription:
            return False

        # Check event type filter
        if subscription.event_types and event.type not in subscription.event_types:
            return False

        # Check branch filter
        if subscription.branch_ids and event.branch_id not in subscription.branch_ids:
            return False

        return True

    async def _add_to_coalesce_buffer(self, subscriber_id: str, event: ResearchEvent):
        """
        Add event to coalescing buffer.

        Args:
            subscriber_id: Subscriber ID
            event: Event to buffer
        """
        self._coalesce_buffers[subscriber_id].append(event)

        # Start coalesce timer if not running
        if subscriber_id not in self._coalesce_tasks:
            subscription = self._subscribers[subscriber_id]
            delay = subscription.coalesce_window_ms / 1000.0

            self._coalesce_tasks[subscriber_id] = asyncio.create_task(
                self._flush_coalesce_buffer(subscriber_id, delay)
            )

    async def _flush_coalesce_buffer(self, subscriber_id: str, delay: float):
        """
        Flush coalescing buffer after delay.

        Args:
            subscriber_id: Subscriber ID
            delay: Delay in seconds
        """
        try:
            await asyncio.sleep(delay)

            async with self._lock:
                events = self._coalesce_buffers.pop(subscriber_id, [])
                self._coalesce_tasks.pop(subscriber_id, None)

                if not events:
                    return

                # Coalesce similar events
                coalesced = self._coalesce_events(events)

                # Deliver coalesced events
                for event in coalesced:
                    await self._deliver_event(subscriber_id, event)

                self.stats["events_coalesced"] += len(events) - len(coalesced)

        except asyncio.CancelledError:
            pass

    def _coalesce_events(self, events: List[ResearchEvent]) -> List[ResearchEvent]:
        """
        Coalesce similar events to reduce noise.

        Strategy:
        - Keep all COMPLETE, ERROR, SUMMARY, CRITIQUE events
        - Coalesce rapid STEP events (keep first and last)
        - Coalesce rapid OBSERVATION events (keep last)

        Args:
            events: Events to coalesce

        Returns:
            Coalesced event list
        """
        if len(events) <= 1:
            return events

        # Group by type
        by_type = defaultdict(list)
        for event in events:
            by_type[event.type].append(event)

        result = []

        # Always keep important events
        for event_type in [EventType.COMPLETE, EventType.ERROR, EventType.SUMMARY, EventType.CRITIQUE]:
            result.extend(by_type[event_type])

        # Coalesce STEP events (keep first and last)
        step_events = by_type[EventType.STEP]
        if step_events:
            if len(step_events) == 1:
                result.extend(step_events)
            else:
                result.append(step_events[0])  # First
                result.append(step_events[-1])  # Last

        # Coalesce OBSERVATION events (keep last)
        obs_events = by_type[EventType.OBSERVATION]
        if obs_events:
            result.append(obs_events[-1])

        # Keep other event types as-is
        for event_type in by_type:
            if event_type not in [EventType.STEP, EventType.OBSERVATION, EventType.COMPLETE,
                                   EventType.ERROR, EventType.SUMMARY, EventType.CRITIQUE]:
                result.extend(by_type[event_type])

        # Sort by timestamp
        result.sort(key=lambda e: e.timestamp)

        return result

    async def _deliver_event(self, subscriber_id: str, event: ResearchEvent):
        """
        Deliver event to subscriber queue.

        Args:
            subscriber_id: Subscriber ID
            event: Event to deliver
        """
        queue = self._event_queues.get(subscriber_id)
        if not queue:
            return

        try:
            # Try to put event in queue (non-blocking)
            queue.put_nowait(event)

        except asyncio.QueueFull:
            # Backpressure: drop oldest event
            try:
                queue.get_nowait()  # Remove oldest
                queue.put_nowait(event)  # Add new
                self.stats["events_dropped"] += 1

                logger.warning(
                    f"Subscriber '{subscriber_id}' queue full, dropped oldest event"
                )

            except Exception as e:
                logger.error(f"Error handling backpressure for '{subscriber_id}': {e}")

    async def broadcast(self, events: List[ResearchEvent]):
        """
        Broadcast multiple events.

        Args:
            events: List of events to publish
        """
        for event in events:
            await self.publish(event)

    def get_stats(self) -> Dict[str, int]:
        """
        Get event bus statistics.

        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()

    async def close(self):
        """Close event bus and cleanup resources"""
        logger.info("Closing event bus...")

        # Unsubscribe all
        for subscriber_id in list(self._active_subscribers):
            await self.unsubscribe(subscriber_id)

        logger.info(f"Event bus closed (stats: {self.stats})")


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create global event bus"""
    global _event_bus

    if _event_bus is None:
        _event_bus = EventBus()

    return _event_bus


async def close_event_bus():
    """Close global event bus"""
    global _event_bus

    if _event_bus:
        await _event_bus.close()
        _event_bus = None


# Example usage
async def test_event_bus():
    """Test event bus"""
    from ..uagent_research.models.events import StepEvent, CompleteEvent

    bus = EventBus()

    # Publisher task
    async def publisher():
        for i in range(10):
            await bus.publish(
                StepEvent(
                    branch_id="branch-1",
                    description=f"Step {i}",
                    artifacts=[],
                )
            )
            await asyncio.sleep(0.05)  # 50ms between events

        await bus.publish(
            CompleteEvent(
                branch_id="branch-1",
                summary="All steps completed",
                artifacts=[],
            )
        )

    # Subscriber task
    async def subscriber():
        event_count = 0
        async for event in bus.subscribe("test-subscriber", coalesce_window_ms=100):
            event_count += 1
            print(f"Received: {event.type} - {getattr(event, 'description', getattr(event, 'summary', ''))}")

            if event.type == EventType.COMPLETE:
                break

        print(f"Total events received: {event_count}")
        print(f"Bus stats: {bus.get_stats()}")

    # Run publisher and subscriber
    await asyncio.gather(
        publisher(),
        subscriber(),
    )

    await bus.close()


if __name__ == "__main__":
    asyncio.run(test_event_bus())
