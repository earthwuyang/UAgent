"""
Event Bus - Real-time Event Streaming

Streams research events to frontend via SSE/WebSocket with:
- Event coalescing (batch similar events)
- Backpressure handling (drop old events if buffer full)
- Multiple subscriber support
- Type-safe event routing
- Event storage for retrieval
"""

import asyncio
import logging
import os
import time
from typing import Dict, Set, Optional, AsyncIterator, List, Callable, Any
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass
import json

from ..uagent_research.models.events import ResearchEvent, EventType, StepEvent
try:
    from ..api.websocket_routes import broadcast_tree_update
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger = None  # Will be set later
    if logger:
        logger.warning("WebSocket routes not available for event broadcasting")


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
    - Event storage for retrieval via REST API
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
        
        # Retrieve stored events
        result = bus.get_events("exp_123", since_version=10, limit=50)
    """

    def __init__(self, max_buffer_size: int = 1000, heartbeat_interval: int = int(os.getenv('RESEARCH_HEARTBEAT_INTERVAL', '5'))):
        """
        Initialize event bus.

        Args:
            max_buffer_size: Maximum events to buffer per subscriber
            heartbeat_interval: Heartbeat interval in seconds (0 to disable).
                              Defaults to RESEARCH_HEARTBEAT_INTERVAL env var (default: 5)
        """
        self.max_buffer_size = max_buffer_size
        self.heartbeat_interval = heartbeat_interval
        
        # Validate and adjust heartbeat interval
        if self.heartbeat_interval < 0:
            self.heartbeat_interval = 0  # Disable heartbeats
        elif self.heartbeat_interval > 10:
            logger.warning(f"Heartbeat interval {self.heartbeat_interval}s exceeds recommended maximum of 10s, capping at 10s")
            self.heartbeat_interval = 10

        # Subscriber management
        self._subscribers: Dict[str, EventSubscription] = {}
        self._event_queues: Dict[str, asyncio.Queue] = {}
        self._active_subscribers: Set[str] = set()

        # Event coalescing
        self._coalesce_buffers: Dict[str, List[ResearchEvent]] = defaultdict(list)
        self._coalesce_tasks: Dict[str, asyncio.Task] = {}

        # Heartbeat tracking
        self._last_event_time: Dict[str, float] = {}  # branch_id -> timestamp
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}  # branch_id -> task
        self._active_branches: Set[str] = set()

        # Event storage for retrieval via /events endpoint
        self._event_logs: Dict[str, deque] = {}  # experiment_id -> deque of (version, timestamp, event_dict)
        self._event_versions: Dict[str, int] = defaultdict(int)  # experiment_id -> current version
        self._max_log_size = 1000  # Max events per experiment

        # Statistics
        self.stats = {
            "events_published": 0,
            "events_delivered": 0,
            "events_dropped": 0,
            "events_coalesced": 0,
            "active_subscribers": 0,
            "heartbeats_sent": 0,
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
        Publish event to all matching subscribers and store in event log.

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

        # Extract experiment ID - check explicit field first, then parse from branch_id
        experiment_id = getattr(event, 'experiment_id', None)
        if not experiment_id and hasattr(event, 'branch_id') and event.branch_id:
            # Try to extract experiment ID from branch_id prefix
            parts = event.branch_id.split('-')
            if parts:
                experiment_id = parts[0]  # Use first part as experiment ID
        
        # Store event in log
        if experiment_id:
            if experiment_id not in self._event_logs:
                self._event_logs[experiment_id] = deque(maxlen=self._max_log_size)
            
            # Increment version
            self._event_versions[experiment_id] += 1
            version = self._event_versions[experiment_id]
            
            # Serialize event to dict
            event_dict = {
                "type": event.type.value if hasattr(event.type, 'value') else str(event.type),
                "branch_id": getattr(event, 'branch_id', None),
                "data": self._serialize_event(event)
            }
            
            # Store as tuple: (version, timestamp, event_dict)
            timestamp = event.timestamp if hasattr(event, 'timestamp') else datetime.utcnow().isoformat()
            self._event_logs[experiment_id].append((version, timestamp, event_dict))

        # Update heartbeat tracking
        if hasattr(event, 'branch_id') and event.branch_id:
            branch_id = event.branch_id
            self._last_event_time[branch_id] = time.time()
            self._active_branches.add(branch_id)

            # Start heartbeat supervisor if not running and heartbeats enabled
            if (self.heartbeat_interval > 0 and
                branch_id not in self._heartbeat_tasks):
                task = asyncio.create_task(
                    self._heartbeat_supervisor(branch_id)
                )
                self._heartbeat_tasks[branch_id] = task

        # Bridge to WebSocket if available and experiment_id is present
        if WEBSOCKET_AVAILABLE and experiment_id:
            try:
                # Create event message for WebSocket clients
                event_message = {
                    "type": "event",
                    "event_type": event.type.value if hasattr(event.type, 'value') else str(event.type),
                    "experiment_id": experiment_id,
                    "branch_id": getattr(event, 'branch_id', None),
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": self._serialize_event(event)
                }
                
                # Schedule broadcast without blocking (fire and forget)
                asyncio.create_task(broadcast_tree_update(experiment_id, event_message))
            except Exception as e:
                # Log but don't fail event publishing if WebSocket broadcast fails
                logger.debug(f"Failed to broadcast event to WebSocket: {e}")

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
    
    def _serialize_event(self, event: ResearchEvent) -> dict:
        """
        Serialize event to dictionary.
        
        Args:
            event: Event to serialize
        
        Returns:
            Dictionary representation
        """
        # Try using event's dict method if available
        if hasattr(event, 'dict'):
            try:
                return event.dict()
            except:
                pass
        
        # Manual serialization
        result = {}
        for attr in dir(event):
            if not attr.startswith('_') and not callable(getattr(event, attr)):
                value = getattr(event, attr)
                # Skip type and timestamp as they're stored separately
                if attr not in ['type', 'timestamp']:
                    try:
                        # Try to serialize, skip if not JSON-serializable
                        json.dumps(value)
                        result[attr] = value
                    except (TypeError, ValueError):
                        result[attr] = str(value)
        
        return result

    def get_events(
        self,
        experiment_id: str,
        since_version: int = 0,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get events for experiment since version.
        
        Args:
            experiment_id: Experiment ID to retrieve events for
            since_version: Only return events with version > this value
            limit: Maximum number of events to return
        
        Returns:
            Dictionary with:
            - events: List of event dicts with version, timestamp, type, data
            - current_version: Latest version number
            - earliest_available_version: Oldest version still in buffer
            - has_more: True if more events available beyond limit
            - has_gap: True if since_version < earliest_available_version (data loss)
        """
        if experiment_id not in self._event_logs:
            return {
                "events": [],
                "current_version": 0,
                "earliest_available_version": 0,
                "has_more": False,
                "has_gap": False
            }
        
        event_log = self._event_logs[experiment_id]
        current_version = self._event_versions[experiment_id]
        
        # Determine earliest available version (oldest event in buffer)
        earliest_available_version = event_log[0][0] if event_log else 0
        
        # Check for gap (requested version older than available)
        has_gap = since_version > 0 and since_version < earliest_available_version
        
        # Filter events by version
        filtered_events = [
            event_tuple for event_tuple in event_log
            if event_tuple[0] > since_version
        ]
        
        # Apply limit
        has_more = len(filtered_events) > limit
        filtered_events = filtered_events[:limit]
        
        # Serialize events
        events = []
        for version, timestamp, event_dict in filtered_events:
            events.append({
                "version": version,
                "timestamp": timestamp,
                **event_dict
            })
        
        return {
            "events": events,
            "current_version": current_version,
            "earliest_available_version": earliest_available_version,
            "has_more": has_more,
            "has_gap": has_gap
        }

    def clear_event_log(self, experiment_id: str):
        """
        Clear event log for an experiment.
        
        Args:
            experiment_id: Experiment ID to clear logs for
        """
        if experiment_id in self._event_logs:
            del self._event_logs[experiment_id]
            logger.info(f"Cleared event log for experiment {experiment_id}")
        
        if experiment_id in self._event_versions:
            del self._event_versions[experiment_id]

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

    async def _heartbeat_supervisor(self, branch_id: str):
        """
        Emit heartbeat events when branch is idle.

        Args:
            branch_id: Branch to monitor
        """
        logger.debug(
            f"Heartbeat supervisor started for {branch_id} "
            f"(interval={self.heartbeat_interval}s)"
        )

        try:
            while branch_id in self._active_branches:
                await asyncio.sleep(self.heartbeat_interval)

                # Check if branch has been idle
                last_time = self._last_event_time.get(branch_id, 0)
                idle_time = time.time() - last_time

                if idle_time >= self.heartbeat_interval:
                    # Emit heartbeat event
                    heartbeat = StepEvent(
                        branch_id=branch_id,
                        node_id="heartbeat",
                        action="heartbeat",
                        reasoning=f"Branch {branch_id} still active (idle for {idle_time:.1f}s)"
                    )

                    # Publish heartbeat (bypass normal publish to avoid recursion)
                    subscriber_count = 0

                    async with self._lock:
                        subscriber_count = len(self._active_subscribers)
                        for subscriber_id in list(self._active_subscribers):
                            if self._matches_subscription(heartbeat, subscriber_id):
                                await self._deliver_event(subscriber_id, heartbeat)

                    self.stats["heartbeats_sent"] += 1
                    logger.debug(
                        f"Emitting heartbeat for {branch_id} "
                        f"(idle={idle_time:.1f}s) to {subscriber_count} subscribers"
                    )

        except asyncio.CancelledError:
            logger.debug(f"Heartbeat supervisor cancelled for branch {branch_id}")
        except Exception as e:
            logger.error(
                f"Error in heartbeat supervisor for {branch_id}: {e}",
                exc_info=True
            )

    def stop_branch_heartbeat(self, branch_id: str):
        """
        Stop heartbeat for a branch (when branch completes).

        Args:
            branch_id: Branch to stop heartbeat for
        """
        last_event_ts = self._last_event_time.get(branch_id, time.time())
        duration = time.time() - last_event_ts

        self._active_branches.discard(branch_id)
        self._last_event_time.pop(branch_id, None)

        # Cancel heartbeat task
        if branch_id in self._heartbeat_tasks:
            task = self._heartbeat_tasks.pop(branch_id)
            if not task.done():
                task.cancel()

        logger.info(
            f"Stopped heartbeat for {branch_id} "
            f"(active ~{duration:.1f}s, sent {self.stats['heartbeats_sent']})"
        )

    async def broadcast(self, events: List[ResearchEvent]):
        """
        Broadcast multiple events.

        Args:
            events: List of events to publish
        """
        for event in events:
            await self.publish(event)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics.

        Returns:
            Dictionary of statistics
        """
        s = self.stats.copy()
        s["active_heartbeats"] = len(self._heartbeat_tasks)
        s["heartbeat_branches"] = list(self._active_branches)
        s["stored_experiment_logs"] = len(self._event_logs)
        return s

    async def close(self):
        """Close event bus and cleanup resources"""
        logger.info("Closing event bus...")

        # Cancel all heartbeat tasks
        for branch_id in list(self._heartbeat_tasks.keys()):
            self.stop_branch_heartbeat(branch_id)

        # Clear all event logs
        self._event_logs.clear()
        self._event_versions.clear()

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
