"""
WebSocket Manager for Real-Time Research Progress Streaming
Provides infrastructure for live research process visualization
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of real-time events"""
    RESEARCH_STARTED = "research_started"
    RESEARCH_PROGRESS = "research_progress"
    RESEARCH_COMPLETED = "research_completed"
    RESEARCH_ERROR = "research_error"

    ENGINE_STATUS = "engine_status"
    ENGINE_STARTED = "engine_started"
    ENGINE_COMPLETED = "engine_completed"

    OPENHANDS_OUTPUT = "openhands_output"
    OPENHANDS_STATUS = "openhands_status"

    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"

    JOURNAL_ENTRY = "journal_entry"
    METRICS_UPDATE = "metrics_update"

    TREE_NODE_ADDED = "tree_node_added"
    TREE_NODE_UPDATED = "tree_node_updated"


@dataclass
class ProgressEvent:
    """Structured event for real-time progress updates"""
    event_type: EventType
    session_id: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str  # engine name or component
    progress_percentage: Optional[float] = None
    message: Optional[str] = None


@dataclass
class ResearchJournalEntry:
    """Entry in the real-time research journal"""
    timestamp: datetime
    session_id: str
    engine: str
    phase: str
    message: str
    metadata: Dict[str, Any]
    level: str = "info"  # info, warning, error, success


@dataclass
class EngineStatus:
    """Real-time status of research engines"""
    engine_name: str
    status: str  # idle, running, completed, error
    current_task: Optional[str]
    progress_percentage: float
    start_time: Optional[datetime]
    estimated_completion: Optional[datetime]
    metrics: Dict[str, Any]


class WebSocketConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        # Active connections by session/goal ID
        self.research_connections: Dict[str, Set[WebSocket]] = {}
        self.engine_status_connections: Set[WebSocket] = set()
        self.metrics_connections: Set[WebSocket] = set()
        self.openhands_connections: Dict[str, Set[WebSocket]] = {}
        self.llm_stream_connections: Dict[str, Set[WebSocket]] = {}

        # Event storage for replay
        self.event_history: Dict[str, List[ProgressEvent]] = {}
        self.journal_entries: Dict[str, List[ResearchJournalEntry]] = {}
        self.llm_conversations: Dict[str, List[Dict[str, Any]]] = {}

        # Current engine statuses
        self.engine_statuses: Dict[str, EngineStatus] = {}

    async def connect_research(self, websocket: WebSocket, session_id: str):
        """Connect to research progress updates for a specific session"""
        await websocket.accept()

        if session_id not in self.research_connections:
            self.research_connections[session_id] = set()
        self.research_connections[session_id].add(websocket)

        # Send event history for this session
        if session_id in self.event_history:
            for event in self.event_history[session_id]:
                # Serialize event manually to handle enum properly
                event_dict = asdict(event)
                event_dict["event_type"] = event.event_type.value  # Use enum value instead of name
                await self._send_to_websocket(websocket, {
                    "type": "event_replay",
                    "event": event_dict
                })

        logger.info(f"WebSocket connected for research session: {session_id}")

    async def connect_engine_status(self, websocket: WebSocket):
        """Connect to engine status updates"""
        await websocket.accept()
        self.engine_status_connections.add(websocket)

        # Send current engine statuses
        for engine_status in self.engine_statuses.values():
            await self._send_to_websocket(websocket, {
                "type": "engine_status",
                "status": asdict(engine_status)
            })

        logger.info("WebSocket connected for engine status updates")

    async def connect_metrics(self, websocket: WebSocket):
        """Connect to live metrics updates"""
        await websocket.accept()
        self.metrics_connections.add(websocket)
        logger.info("WebSocket connected for metrics updates")

    async def connect_openhands(self, websocket: WebSocket, session_id: str):
        """Connect to OpenHands output streaming"""
        await websocket.accept()

        if session_id not in self.openhands_connections:
            self.openhands_connections[session_id] = set()
        self.openhands_connections[session_id].add(websocket)

        logger.info(f"WebSocket connected for OpenHands session: {session_id}")

    async def connect_llm_stream(self, websocket: WebSocket, session_id: str):
        """Connect to LLM interaction streaming"""
        await websocket.accept()

        if session_id not in self.llm_stream_connections:
            self.llm_stream_connections[session_id] = set()
        self.llm_stream_connections[session_id].add(websocket)

        # Replay stored conversation events
        if session_id in self.llm_conversations:
            for event in self.llm_conversations[session_id]:
                await self._send_to_websocket(websocket, event)

        logger.info(f"WebSocket connected for LLM stream session: {session_id}")

    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        # Remove from all connection sets
        for connections in self.research_connections.values():
            connections.discard(websocket)

        self.engine_status_connections.discard(websocket)
        self.metrics_connections.discard(websocket)

        for connections in self.openhands_connections.values():
            connections.discard(websocket)

        for connections in self.llm_stream_connections.values():
            connections.discard(websocket)

        logger.info("WebSocket disconnected")

    def store_llm_event(self, session_id: str, message: Dict[str, Any]):
        """Persist LLM stream event for later replay"""
        if session_id not in self.llm_conversations:
            self.llm_conversations[session_id] = []

        self.llm_conversations[session_id].append(message)

        # Prevent unbounded growth
        if len(self.llm_conversations[session_id]) > 2000:
            self.llm_conversations[session_id] = self.llm_conversations[session_id][-1000:]

    async def broadcast_research_event(self, session_id: str, event: ProgressEvent):
        """Broadcast research progress event to connected clients"""
        # Store event in history
        if session_id not in self.event_history:
            self.event_history[session_id] = []
        self.event_history[session_id].append(event)

        # Limit history size
        if len(self.event_history[session_id]) > 1000:
            self.event_history[session_id] = self.event_history[session_id][-500:]

        # Broadcast to connected clients
        if session_id in self.research_connections:
            # Serialize event manually to handle enum properly
            event_dict = asdict(event)
            event_dict["event_type"] = event.event_type.value  # Use enum value instead of name
            message = {
                "type": "research_event",
                "event": event_dict
            }
            await self._broadcast_to_connections(
                self.research_connections[session_id],
                message
            )

    async def broadcast_journal_entry(self, session_id: str, entry: ResearchJournalEntry):
        """Broadcast journal entry to connected clients"""
        # Store journal entry
        if session_id not in self.journal_entries:
            self.journal_entries[session_id] = []
        self.journal_entries[session_id].append(entry)

        # Limit journal size
        if len(self.journal_entries[session_id]) > 2000:
            self.journal_entries[session_id] = self.journal_entries[session_id][-1000:]

        # Broadcast to research connections
        if session_id in self.research_connections:
            message = {
                "type": "journal_entry",
                "entry": asdict(entry)
            }
            await self._broadcast_to_connections(
                self.research_connections[session_id],
                message
            )

    async def update_engine_status(self, engine_status: EngineStatus):
        """Update and broadcast engine status"""
        self.engine_statuses[engine_status.engine_name] = engine_status

        message = {
            "type": "engine_status",
            "status": asdict(engine_status)
        }
        await self._broadcast_to_connections(self.engine_status_connections, message)

    async def broadcast_openhands_output(self, session_id: str, output: str, output_type: str = "stdout"):
        """Broadcast OpenHands output to connected clients"""
        if session_id in self.openhands_connections:
            message = {
                "type": "openhands_output",
                "session_id": session_id,
                "output": output,
                "output_type": output_type,
                "timestamp": datetime.now().isoformat()
            }
            await self._broadcast_to_connections(
                self.openhands_connections[session_id],
                message
            )

    async def broadcast_openhands_event(self, session_id: str, event_payload: Dict[str, Any]):
        """Broadcast raw OpenHands agent events to connected clients."""
        if session_id not in self.openhands_connections:
            return

        message = {
            "type": "openhands_event",
            "session_id": session_id,
            "event": event_payload,
            "timestamp": datetime.now().isoformat(),
        }
        await self._broadcast_to_connections(
            self.openhands_connections[session_id],
            message,
        )

    async def broadcast_metrics_update(self, metrics: Dict[str, Any]):
        """Broadcast live metrics update"""
        message = {
            "type": "metrics_update",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        await self._broadcast_to_connections(self.metrics_connections, message)

    async def _broadcast_to_connections(self, connections: Set[WebSocket], message: Dict[str, Any]):
        """Broadcast message to a set of WebSocket connections"""
        if not connections:
            return

        # Create list copy to avoid modification during iteration
        connection_list = list(connections)

        for websocket in connection_list:
            try:
                await self._send_to_websocket(websocket, message)
            except WebSocketDisconnect:
                # Remove disconnected websocket
                connections.discard(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                connections.discard(websocket)

    async def _send_to_websocket(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to individual WebSocket"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            raise


class ResearchProgressTracker:
    """Tracks and broadcasts research progress events"""

    def __init__(self, websocket_manager: WebSocketConnectionManager):
        self.websocket_manager = websocket_manager

    async def log_research_started(self, session_id: str, request: str, engine: str):
        """Log research start event"""
        event = ProgressEvent(
            event_type=EventType.RESEARCH_STARTED,
            session_id=session_id,
            timestamp=datetime.now(),
            data={
                "request": request,
                "engine": engine,
                "status": "started"
            },
            source=engine,
            progress_percentage=0.0,
            message=f"Research started using {engine} engine"
        )

        await self.websocket_manager.broadcast_research_event(session_id, event)

        # Add journal entry
        journal_entry = ResearchJournalEntry(
            timestamp=datetime.now(),
            session_id=session_id,
            engine=engine,
            phase="initialization",
            message=f"Research session started: {request}",
            metadata={"request": request},
            level="info"
        )

        await self.websocket_manager.broadcast_journal_entry(session_id, journal_entry)

    async def log_research_completed(self, session_id: str, engine: str, result_summary: str, metadata: Dict[str, Any] = None):
        """Log research completion event"""
        event = ProgressEvent(
            event_type=EventType.RESEARCH_COMPLETED,
            session_id=session_id,
            timestamp=datetime.now(),
            data={
                "engine": engine,
                "result_summary": result_summary,
                "metadata": metadata or {},
                "status": "completed"
            },
            source=engine,
            progress_percentage=100.0,
            message=f"Research completed successfully using {engine} engine"
        )

        await self.websocket_manager.broadcast_research_event(session_id, event)

        # Add journal entry
        journal_entry = ResearchJournalEntry(
            timestamp=datetime.now(),
            session_id=session_id,
            engine=engine,
            phase="completion",
            message=f"Research completed: {result_summary}",
            metadata=metadata or {},
            level="success"
        )

        await self.websocket_manager.broadcast_journal_entry(session_id, journal_entry)

    async def log_research_progress(self, session_id: str, engine: str, phase: str,
                                  progress: float, message: str, metadata: Dict[str, Any] = None):
        """Log research progress event"""
        event = ProgressEvent(
            event_type=EventType.RESEARCH_PROGRESS,
            session_id=session_id,
            timestamp=datetime.now(),
            data={
                "engine": engine,
                "phase": phase,
                "metadata": metadata or {}
            },
            source=engine,
            progress_percentage=progress,
            message=message
        )

        await self.websocket_manager.broadcast_research_event(session_id, event)

        # Add journal entry
        journal_entry = ResearchJournalEntry(
            timestamp=datetime.now(),
            session_id=session_id,
            engine=engine,
            phase=phase,
            message=message,
            metadata=metadata or {},
            level="info"
        )

        await self.websocket_manager.broadcast_journal_entry(session_id, journal_entry)

    async def log_engine_coordination(self, session_id: str, primary_engine: str,
                                    coordinated_engines: List[str], phase: str):
        """Log multi-engine coordination event"""
        event = ProgressEvent(
            event_type=EventType.ENGINE_STATUS,
            session_id=session_id,
            timestamp=datetime.now(),
            data={
                "primary_engine": primary_engine,
                "coordinated_engines": coordinated_engines,
                "phase": phase,
                "coordination_type": "multi_engine"
            },
            source="coordinator",
            message=f"Coordinating {len(coordinated_engines)} engines for {phase}"
        )

        await self.websocket_manager.broadcast_research_event(session_id, event)

    async def log_openhands_execution(self, session_id: str, command: str,
                                    workspace_path: str, phase: str):
        """Log OpenHands execution start"""
        journal_entry = ResearchJournalEntry(
            timestamp=datetime.now(),
            session_id=session_id,
            engine="openhands",
            phase=phase,
            message=f"Executing: {command}",
            metadata={
                "command": command,
                "workspace": workspace_path,
                "execution_type": "code"
            },
            level="info"
        )

        await self.websocket_manager.broadcast_journal_entry(session_id, journal_entry)

    async def log_error(self, session_id: str, engine: str, error: str, phase: str):
        """Log error event"""
        event = ProgressEvent(
            event_type=EventType.RESEARCH_ERROR,
            session_id=session_id,
            timestamp=datetime.now(),
            data={
                "engine": engine,
                "error": error,
                "phase": phase
            },
            source=engine,
            message=f"Error in {engine}: {error}"
        )

        await self.websocket_manager.broadcast_research_event(session_id, event)

        # Add journal entry
        journal_entry = ResearchJournalEntry(
            timestamp=datetime.now(),
            session_id=session_id,
            engine=engine,
            phase=phase,
            message=f"Error: {error}",
            metadata={"error_details": error},
            level="error"
        )

        await self.websocket_manager.broadcast_journal_entry(session_id, journal_entry)


# Global instances
websocket_manager = WebSocketConnectionManager()
progress_tracker = ResearchProgressTracker(websocket_manager)
