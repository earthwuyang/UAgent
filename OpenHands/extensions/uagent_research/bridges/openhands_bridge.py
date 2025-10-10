"""
OpenHandsEventBridge - Convert OpenHands events to ResearchEvents

Maps OpenHands CodeActAgent events to UAgent research event model for
seamless integration with the research orchestrator.
"""

import asyncio
import logging
import uuid
from typing import AsyncIterator, Optional

from openhands.events import EventSource
from openhands.events.action import (
    MessageAction,
    CmdRunAction,
    IPythonRunCellAction,
    FileReadAction,
    FileEditAction,
    BrowseInteractiveAction,
    AgentFinishAction,
    AgentThinkAction,
)
from openhands.events.observation import (
    CmdOutputObservation,
    IPythonRunCellObservation,
    FileReadObservation,
    FileEditObservation,
    BrowserOutputObservation,
    ErrorObservation,
)
from openhands.events.stream import EventStreamSubscriber

from extensions.uagent_research.uagent_research.models.events import (
    ResearchEvent,
    StepEvent,
    PlanEvent,
    ToolCallEvent,
    ObservationEvent,
    ErrorEvent,
    CompleteEvent,
    ArtifactType,
    Artifact,
)

logger = logging.getLogger(__name__)


class OpenHandsEventBridge:
    """
    Bridge OpenHands EventStream to ResearchEvent stream.

    Subscribes to OpenHands events from a CodeActAgent session and
    converts them to UAgent research events for the orchestrator.

    Example:
        bridge = OpenHandsEventBridge(
            event_stream=session.event_stream,
            branch_id="idea-0-hyp-0",
            node_id="task-123"
        )

        async for research_event in bridge.stream():
            await event_bus.publish(research_event)
    """

    def __init__(
        self,
        event_stream,
        branch_id: str,
        node_id: str
    ):
        """
        Initialize event bridge.

        Args:
            event_stream: OpenHands EventStream instance
            branch_id: Research tree branch ID
            node_id: Research tree node ID
        """
        self.event_stream = event_stream
        self.branch_id = branch_id
        self.node_id = node_id

        self._event_count = 0
        logger.info(
            f"OpenHandsEventBridge initialized for branch={branch_id}, node={node_id}"
        )

    async def stream(self) -> AsyncIterator[ResearchEvent]:
        """
        Stream bridged research events.

        Subscribes to OpenHands event stream and yields converted
        research events.

        Yields:
            ResearchEvent instances
        """
        # Subscribe to OpenHands event stream
        # Note: Actual subscription mechanism depends on EventStream API
        # This is a simplified version

        logger.info(
            f"Starting event bridge stream for branch={self.branch_id}"
        )

        try:
            # Iterate through OpenHands events
            # In practice, this would use event_stream.subscribe() or similar
            async for oh_event in self._subscribe_to_stream():
                research_event = self._map_event(oh_event)

                if research_event:
                    self._event_count += 1
                    yield research_event

        except Exception as e:
            logger.error(f"Error in event bridge stream: {e}", exc_info=True)
            yield ErrorEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                message=f"Event bridge error: {str(e)}"
            )

    async def _subscribe_to_stream(self):
        """
        Subscribe to the OpenHands EventStream and yield raw events.

        Uses an asyncio.Queue to shuttle events from the background
        callback thread into the async generator.
        """
        if not self.event_stream:
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        callback_id = f'research_bridge_{self.branch_id}_{self.node_id}_{uuid.uuid4().hex}'

        def _on_event(event):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                logger.debug('Failed to enqueue OpenHands event in research bridge', exc_info=True)

        try:
            self.event_stream.subscribe(
                EventStreamSubscriber.SERVER,
                _on_event,
                callback_id,
            )
        except Exception as exc:
            logger.error(
                'Unable to subscribe research bridge to event stream: %s',
                exc,
                exc_info=True,
            )
            return

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            try:
                self.event_stream.unsubscribe(
                    EventStreamSubscriber.SERVER, callback_id
                )
            except Exception:
                logger.debug(
                    'Failed to unsubscribe research bridge from event stream',
                    exc_info=True,
                )
            try:
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except RuntimeError:
                pass

    def _map_event(self, oh_event) -> Optional[ResearchEvent]:
        """
        Map OpenHands event to ResearchEvent.

        Args:
            oh_event: OpenHands Event instance

        Returns:
            Corresponding ResearchEvent or None if unmapped
        """
        event_type = type(oh_event).__name__

        # Agent thinking/planning
        if isinstance(oh_event, AgentThinkAction):
            return PlanEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                steps=[oh_event.thought] if oh_event.thought else [],
                reasoning=oh_event.thought or "Agent thinking..."
            )

        # Agent messages
        elif isinstance(oh_event, MessageAction):
            if oh_event.source == EventSource.AGENT:
                # Agent's response or commentary
                return StepEvent(
                    branch_id=self.branch_id,
                    node_id=self.node_id,
                    action=oh_event.content[:200] if oh_event.content else "...",
                    reasoning=oh_event.content or ""
                )

        # Command execution
        elif isinstance(oh_event, CmdRunAction):
            return ToolCallEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                tool="bash",
                args={"command": oh_event.command}
            )

        # IPython code execution
        elif isinstance(oh_event, IPythonRunCellAction):
            return ToolCallEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                tool="ipython",
                args={"code": oh_event.code}
            )

        # File operations
        elif isinstance(oh_event, FileReadAction):
            return ToolCallEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                tool="file_read",
                args={"path": oh_event.path}
            )

        elif isinstance(oh_event, FileEditAction):
            return ToolCallEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                tool="file_edit",
                args={
                    "path": oh_event.path,
                    "content_preview": oh_event.content[:100] if oh_event.content else ""
                }
            )

        # Browser interaction
        elif isinstance(oh_event, BrowseInteractiveAction):
            return ToolCallEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                tool="browser",
                args={
                    "url": getattr(oh_event, 'url', ''),
                    "action": getattr(oh_event, 'browser_actions', '')
                }
            )

        # Observations (tool results)
        elif isinstance(oh_event, CmdOutputObservation):
            return ObservationEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                result={
                    "output": oh_event.content,
                    "exit_code": oh_event.exit_code
                },
                success=oh_event.exit_code == 0
            )

        elif isinstance(oh_event, IPythonRunCellObservation):
            return ObservationEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                result={"output": oh_event.content},
                success=True  # IPython doesn't have exit code
            )

        elif isinstance(oh_event, FileReadObservation):
            return ObservationEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                result={
                    "path": oh_event.path,
                    "content_preview": oh_event.content[:500] if oh_event.content else ""
                },
                success=True
            )

        elif isinstance(oh_event, BrowserOutputObservation):
            return ObservationEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                result={
                    "url": oh_event.url,
                    "content_preview": oh_event.content[:500] if oh_event.content else ""
                },
                success=oh_event.error is False
            )

        # Error observations
        elif isinstance(oh_event, ErrorObservation):
            return ErrorEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                message=oh_event.content
            )

        # Agent completion
        elif isinstance(oh_event, AgentFinishAction):
            # Extract artifacts from finish action
            artifacts = []

            # Try to extract output content as artifact
            output_content = oh_event.outputs.get("content", "") if oh_event.outputs else ""

            if output_content:
                artifacts.append(
                    Artifact(
                        kind=ArtifactType.SUMMARY,
                        locator="agent_output",
                        content=output_content
                    )
                )

            return CompleteEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                summary=output_content[:200] if output_content else "Task completed",
                artifacts=artifacts,
                success=True
            )

        # Unmapped event - log and ignore
        else:
            logger.debug(
                f"Unmapped OpenHands event type: {event_type} "
                f"(branch={self.branch_id}, node={self.node_id})"
            )
            return None

    def get_stats(self) -> dict:
        """Get bridge statistics"""
        return {
            "branch_id": self.branch_id,
            "node_id": self.node_id,
            "events_mapped": self._event_count
        }


# Example usage
async def test_event_bridge():
    """Test OpenHandsEventBridge"""
    # This would require a real EventStream instance
    # For now, just demonstrate the API

    class MockEventStream:
        async def subscribe(self):
            # Yield some mock events
            from openhands.events.action import MessageAction
            yield MessageAction(
                content="Running benchmark...",
                source=EventSource.AGENT
            )

    bridge = OpenHandsEventBridge(
        event_stream=MockEventStream(),
        branch_id="test-branch",
        node_id="test-node"
    )

    logger.info(" OpenHandsEventBridge created successfully")

    # In real usage:
    # async for event in bridge.stream():
    #     print(f"Research event: {event}")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_event_bridge())
