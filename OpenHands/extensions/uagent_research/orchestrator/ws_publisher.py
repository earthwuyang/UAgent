"""
WebSocket Publisher - Bridge EventBus to WebSocket

Subscribes to EventBus and publishes events to WebSocket clients.
Maps ResearchEvent → WebSocket messages for frontend consumption.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime
import json

from ..orchestrator.event_bus import EventBus
from ..uagent_research.models.events import (
    ResearchEvent,
    EventType,
    PlanEvent,
    StepEvent,
    ToolCallEvent,
    ObservationEvent,
    SummaryEvent,
    CompleteEvent,
    ErrorEvent,
    CritiqueEvent,
)
from ..uagent_research.models.research_tree import ResearchTree

logger = logging.getLogger(__name__)


class WebSocketMessage:
    """WebSocket message format (ROMA-compatible)"""

    @staticmethod
    def tree_snapshot(tree: ResearchTree, version: int) -> Dict[str, Any]:
        """Full tree snapshot"""
        return {
            "type": "tree_snapshot",
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "nodes": [
                    {
                        "id": node.id,
                        "type": node.type.value,
                        "title": node.title,
                        "content": node.content,
                        "status": node.status.value,
                        "visits": node.visits,
                        "prior": node.prior,
                        "avg_value": node.avg_value,
                        "cost": node.cost,
                        "tokens_used": node.tokens_used,
                        "created_at": node.created_at.isoformat()
                        if node.created_at
                        else None,
                        "completed_at": node.completed_at.isoformat()
                        if node.completed_at
                        else None,
                    }
                    for node in tree.nodes.values()
                ],
                "edges": [
                    {"parent_id": parent_id, "child_id": child_id}
                    for parent_id, children in tree.children.items()
                    for child_id in children
                ],
                "stats": tree.stats or {},
            },
        }

    @staticmethod
    def node_added(
        node_id: str, parent_id: Optional[str], node_data: Dict[str, Any], version: int
    ) -> Dict[str, Any]:
        """Node added to tree"""
        return {
            "type": "node_added",
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "data": {"node_id": node_id, "parent_id": parent_id, "node": node_data},
        }

    @staticmethod
    def node_updated(
        node_id: str, updates: Dict[str, Any], version: int
    ) -> Dict[str, Any]:
        """Node updated"""
        return {
            "type": "node_updated",
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "data": {"node_id": node_id, "updates": updates},
        }

    @staticmethod
    def edge_added(
        parent_id: str, child_id: str, version: int
    ) -> Dict[str, Any]:
        """Edge added to tree"""
        return {
            "type": "edge_added",
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "data": {"parent_id": parent_id, "child_id": child_id},
        }

    @staticmethod
    def stats_updated(stats: Dict[str, Any], version: int) -> Dict[str, Any]:
        """Tree statistics updated"""
        return {
            "type": "stats_updated",
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "data": {"stats": stats},
        }

    @staticmethod
    def event_log(event: ResearchEvent, version: int) -> Dict[str, Any]:
        """Research event log"""
        return {
            "type": "event_log",
            "version": version,
            "timestamp": event.timestamp.isoformat(),
            "data": {
                "event_type": event.type.value,
                "branch_id": event.branch_id,
                "node_id": event.node_id,
                "message": WebSocketPublisher._format_event_message(event),
                "artifacts": [
                    {
                        "type": artifact.type.value,
                        "content": artifact.content[:500],  # Limit size
                        "metadata": artifact.metadata,
                    }
                    for artifact in getattr(event, "artifacts", [])
                ],
            },
        }

    @staticmethod
    def error(error_msg: str, version: int) -> Dict[str, Any]:
        """Error message"""
        return {
            "type": "error",
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "data": {"error": error_msg},
        }

    @staticmethod
    def complete(summary: str, version: int) -> Dict[str, Any]:
        """Completion message"""
        return {
            "type": "complete",
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "data": {"summary": summary},
        }


class WebSocketPublisher:
    """
    Bridges EventBus to WebSocket connections.

    Subscribes to EventBus and publishes formatted messages to WebSocket clients.
    Supports filtering by experiment_id/session_id.

    Example:
        publisher = WebSocketPublisher(
            event_bus=bus,
            ws_manager=manager,
            experiment_id="exp-123"
        )

        await publisher.start()
    """

    def __init__(
        self,
        event_bus: EventBus,
        ws_manager: Any,  # WebSocket manager (from fastapi)
        experiment_id: str,
        session_id: Optional[str] = None,
    ):
        """
        Initialize WebSocket publisher.

        Args:
            event_bus: EventBus to subscribe to
            ws_manager: WebSocket connection manager
            experiment_id: Experiment ID to filter events
            session_id: Optional session ID
        """
        self.event_bus = event_bus
        self.ws_manager = ws_manager
        self.experiment_id = experiment_id
        self.session_id = session_id

        self.version = 0  # Monotonic version counter
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start publishing events"""
        if self._running:
            logger.warning("WebSocket publisher already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._publish_loop())

        logger.info(
            f"WebSocket publisher started for experiment {self.experiment_id}"
        )

    async def stop(self):
        """Stop publishing events"""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"WebSocket publisher stopped for experiment {self.experiment_id}"
        )

    async def _publish_loop(self):
        """Main publish loop"""
        try:
            # Subscribe to EventBus
            subscriber_id = f"ws-{self.experiment_id}"

            async for event in self.event_bus.subscribe(
                subscriber_id=subscriber_id,
                # Filter by branch (experiment/session)
                branch_ids={self.experiment_id}
                if not self.session_id
                else {self.session_id},
            ):
                if not self._running:
                    break

                # Convert event to WebSocket message
                ws_message = self._convert_event(event)

                if ws_message:
                    # Broadcast to all connected clients for this experiment
                    await self._broadcast(ws_message)

        except Exception as e:
            logger.error(f"WebSocket publish loop error: {e}", exc_info=True)

    def _convert_event(self, event: ResearchEvent) -> Optional[Dict[str, Any]]:
        """
        Convert ResearchEvent to WebSocket message.

        Args:
            event: Research event

        Returns:
            WebSocket message dict or None
        """
        self.version += 1

        # Map event types to WebSocket messages
        if event.type == EventType.PLAN:
            # Log plan
            return WebSocketMessage.event_log(event, self.version)

        elif event.type == EventType.STEP:
            # Log step
            return WebSocketMessage.event_log(event, self.version)

        elif event.type == EventType.TOOL_CALL:
            # Log tool call
            return WebSocketMessage.event_log(event, self.version)

        elif event.type == EventType.OBSERVATION:
            # Log observation
            return WebSocketMessage.event_log(event, self.version)

        elif event.type == EventType.SUMMARY:
            # Log summary
            return WebSocketMessage.event_log(event, self.version)

        elif event.type == EventType.CRITIQUE:
            # Log critique
            return WebSocketMessage.event_log(event, self.version)

        elif event.type == EventType.COMPLETE:
            # Node completed - update node status
            if event.node_id:
                return WebSocketMessage.node_updated(
                    node_id=event.node_id,
                    updates={
                        "status": "complete",
                        "completed_at": event.timestamp.isoformat(),
                    },
                    version=self.version,
                )

        elif event.type == EventType.ERROR:
            # Error - log and possibly update node
            return WebSocketMessage.error(
                error_msg=getattr(event, "error", "Unknown error"),
                version=self.version,
            )

        return None

    async def _broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message to all connected WebSocket clients.

        Args:
            message: Message to broadcast
        """
        try:
            # Convert to JSON
            json_message = json.dumps(message)

            # Broadcast via WebSocket manager
            # (This assumes ws_manager has a broadcast method)
            if hasattr(self.ws_manager, "broadcast"):
                await self.ws_manager.broadcast(
                    message=json_message, experiment_id=self.experiment_id
                )
            else:
                logger.warning("WebSocket manager has no broadcast method")

        except Exception as e:
            logger.error(f"WebSocket broadcast error: {e}", exc_info=True)

    async def publish_tree_snapshot(self, tree: ResearchTree):
        """
        Publish full tree snapshot.

        Args:
            tree: Research tree
        """
        self.version += 1

        message = WebSocketMessage.tree_snapshot(tree, self.version)

        await self._broadcast(message)

    async def publish_node_added(
        self, node_id: str, parent_id: Optional[str], node_data: Dict[str, Any]
    ):
        """
        Publish node added event.

        Args:
            node_id: Node ID
            parent_id: Parent node ID
            node_data: Node data
        """
        self.version += 1

        message = WebSocketMessage.node_added(
            node_id, parent_id, node_data, self.version
        )

        await self._broadcast(message)

    async def publish_stats_updated(self, stats: Dict[str, Any]):
        """
        Publish stats updated event.

        Args:
            stats: Statistics
        """
        self.version += 1

        message = WebSocketMessage.stats_updated(stats, self.version)

        await self._broadcast(message)

    @staticmethod
    def _format_event_message(event: ResearchEvent) -> str:
        """Format event as human-readable message"""
        if isinstance(event, PlanEvent):
            return f"Plan: {event.plan}"
        elif isinstance(event, StepEvent):
            return f"Step {event.step_number}: {event.description}"
        elif isinstance(event, ToolCallEvent):
            return f"Tool call: {event.tool_name}"
        elif isinstance(event, ObservationEvent):
            return f"Observation: {event.observation}"
        elif isinstance(event, SummaryEvent):
            return f"Summary: {event.summary[:100]}..."
        elif isinstance(event, CompleteEvent):
            return f"Complete: {event.summary}"
        elif isinstance(event, ErrorEvent):
            return f"Error: {event.error}"
        elif isinstance(event, CritiqueEvent):
            return f"Critique: {event.critique[:100]}..."
        else:
            return str(event)


# Example usage
async def test_ws_publisher():
    """Test WebSocket publisher"""
    from ..orchestrator.event_bus import EventBus
    from ..uagent_research.models.events import StepEvent

    # Mock WebSocket manager
    class MockWSManager:
        async def broadcast(self, message: str, experiment_id: str):
            print(f"Broadcasting to {experiment_id}: {message[:100]}...")

    bus = EventBus()
    manager = MockWSManager()

    publisher = WebSocketPublisher(
        event_bus=bus, ws_manager=manager, experiment_id="test-exp"
    )

    await publisher.start()

    # Publish some events
    await bus.publish(
        StepEvent(
            branch_id="test-exp",
            node_id="node-1",
            step_number=1,
            description="Testing WebSocket publisher",
            artifacts=[],
        )
    )

    await asyncio.sleep(1)

    await publisher.stop()
    await bus.close()


if __name__ == "__main__":
    asyncio.run(test_ws_publisher())
