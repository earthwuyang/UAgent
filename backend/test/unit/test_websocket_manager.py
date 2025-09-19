"""
Unit tests for WebSocket Manager
Tests WebSocket connection management and event broadcasting
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from app.core.websocket_manager import (
    WebSocketConnectionManager,
    ResearchProgressTracker,
    ProgressEvent,
    ResearchJournalEntry,
    EngineStatus,
    EventType
)


@pytest.fixture
def websocket_manager():
    """Create WebSocket manager instance"""
    return WebSocketConnectionManager()


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket"""
    mock = Mock(spec=WebSocket)
    mock.accept = AsyncMock()
    mock.send_text = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def progress_tracker(websocket_manager):
    """Create progress tracker instance"""
    return ResearchProgressTracker(websocket_manager)


class TestWebSocketConnectionManager:
    """Test WebSocket connection management"""

    @pytest.mark.asyncio
    async def test_connect_research(self, websocket_manager, mock_websocket):
        """Test research WebSocket connection"""
        session_id = "test_session"

        await websocket_manager.connect_research(mock_websocket, session_id)

        # Should accept WebSocket and add to connections
        mock_websocket.accept.assert_called_once()
        assert session_id in websocket_manager.research_connections
        assert mock_websocket in websocket_manager.research_connections[session_id]

    @pytest.mark.asyncio
    async def test_connect_research_with_history(self, websocket_manager, mock_websocket):
        """Test research connection with existing event history"""
        session_id = "test_session"

        # Add some event history
        event = ProgressEvent(
            event_type=EventType.RESEARCH_STARTED,
            session_id=session_id,
            timestamp=datetime.now(),
            data={"test": "data"},
            source="test_engine"
        )
        websocket_manager.event_history[session_id] = [event]

        await websocket_manager.connect_research(mock_websocket, session_id)

        # Should send event history replay
        mock_websocket.send_text.assert_called()
        sent_data = mock_websocket.send_text.call_args[0][0]
        assert "event_replay" in sent_data

    @pytest.mark.asyncio
    async def test_connect_engine_status(self, websocket_manager, mock_websocket):
        """Test engine status WebSocket connection"""
        await websocket_manager.connect_engine_status(mock_websocket)

        mock_websocket.accept.assert_called_once()
        assert mock_websocket in websocket_manager.engine_status_connections

    @pytest.mark.asyncio
    async def test_connect_engine_status_with_existing_statuses(self, websocket_manager, mock_websocket):
        """Test engine status connection with existing engine statuses"""
        # Add existing engine status
        engine_status = EngineStatus(
            engine_name="test_engine",
            status="running",
            current_task="test task",
            progress_percentage=50.0,
            start_time=datetime.now(),
            estimated_completion=None,
            metrics={}
        )
        websocket_manager.engine_statuses["test_engine"] = engine_status

        await websocket_manager.connect_engine_status(mock_websocket)

        # Should send existing engine statuses
        mock_websocket.send_text.assert_called()

    @pytest.mark.asyncio
    async def test_connect_llm_stream(self, websocket_manager, mock_websocket):
        """Test LLM stream WebSocket connection"""
        session_id = "test_session"

        await websocket_manager.connect_llm_stream(mock_websocket, session_id)

        mock_websocket.accept.assert_called_once()
        assert session_id in websocket_manager.llm_stream_connections
        assert mock_websocket in websocket_manager.llm_stream_connections[session_id]

    @pytest.mark.asyncio
    async def test_disconnect(self, websocket_manager, mock_websocket):
        """Test WebSocket disconnection"""
        session_id = "test_session"

        # Add websocket to multiple connection types
        await websocket_manager.connect_research(mock_websocket, session_id)
        await websocket_manager.connect_engine_status(mock_websocket)
        await websocket_manager.connect_llm_stream(mock_websocket, session_id)

        # Disconnect
        await websocket_manager.disconnect(mock_websocket)

        # Should be removed from all connection sets
        assert mock_websocket not in websocket_manager.research_connections[session_id]
        assert mock_websocket not in websocket_manager.engine_status_connections
        assert mock_websocket not in websocket_manager.llm_stream_connections[session_id]

    @pytest.mark.asyncio
    async def test_broadcast_research_event(self, websocket_manager, mock_websocket):
        """Test research event broadcasting"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        event = ProgressEvent(
            event_type=EventType.RESEARCH_PROGRESS,
            session_id=session_id,
            timestamp=datetime.now(),
            data={"engine": "test_engine", "phase": "test_phase"},
            source="test_engine",
            progress_percentage=50.0,
            message="Test progress message"
        )

        await websocket_manager.broadcast_research_event(session_id, event)

        # Should store event in history
        assert session_id in websocket_manager.event_history
        assert event in websocket_manager.event_history[session_id]

        # Should broadcast to connected clients
        mock_websocket.send_text.assert_called()
        sent_data = mock_websocket.send_text.call_args_list[-1][0][0]
        assert "research_event" in sent_data

    @pytest.mark.asyncio
    async def test_broadcast_journal_entry(self, websocket_manager, mock_websocket):
        """Test journal entry broadcasting"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        entry = ResearchJournalEntry(
            timestamp=datetime.now(),
            session_id=session_id,
            engine="test_engine",
            phase="test_phase",
            message="Test journal message",
            metadata={"test": "data"},
            level="info"
        )

        await websocket_manager.broadcast_journal_entry(session_id, entry)

        # Should store entry in journal
        assert session_id in websocket_manager.journal_entries
        assert entry in websocket_manager.journal_entries[session_id]

        # Should broadcast to connected clients
        mock_websocket.send_text.assert_called()
        sent_data = mock_websocket.send_text.call_args_list[-1][0][0]
        assert "journal_entry" in sent_data

    @pytest.mark.asyncio
    async def test_update_engine_status(self, websocket_manager, mock_websocket):
        """Test engine status update and broadcasting"""
        await websocket_manager.connect_engine_status(mock_websocket)

        engine_status = EngineStatus(
            engine_name="test_engine",
            status="completed",
            current_task=None,
            progress_percentage=100.0,
            start_time=datetime.now(),
            estimated_completion=None,
            metrics={"success": True}
        )

        await websocket_manager.update_engine_status(engine_status)

        # Should store engine status
        assert websocket_manager.engine_statuses["test_engine"] == engine_status

        # Should broadcast to connected clients
        mock_websocket.send_text.assert_called()

    @pytest.mark.asyncio
    async def test_broadcast_openhands_output(self, websocket_manager, mock_websocket):
        """Test OpenHands output broadcasting"""
        session_id = "test_session"
        await websocket_manager.connect_openhands(mock_websocket, session_id)

        output = "Test command output"
        output_type = "stdout"

        await websocket_manager.broadcast_openhands_output(session_id, output, output_type)

        # Should broadcast to connected clients
        mock_websocket.send_text.assert_called()
        sent_data = mock_websocket.send_text.call_args[0][0]
        assert "openhands_output" in sent_data
        assert output in sent_data

    @pytest.mark.asyncio
    async def test_broadcast_to_nonexistent_session(self, websocket_manager):
        """Test broadcasting to session with no connections"""
        session_id = "nonexistent_session"

        event = ProgressEvent(
            event_type=EventType.RESEARCH_STARTED,
            session_id=session_id,
            timestamp=datetime.now(),
            data={},
            source="test_engine"
        )

        # Should not raise error
        await websocket_manager.broadcast_research_event(session_id, event)

        # Event should still be stored in history
        assert session_id in websocket_manager.event_history
        assert event in websocket_manager.event_history[session_id]

    @pytest.mark.asyncio
    async def test_event_history_limit(self, websocket_manager, mock_websocket):
        """Test event history size limiting"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        # Add more than 1000 events
        for i in range(1200):
            event = ProgressEvent(
                event_type=EventType.RESEARCH_PROGRESS,
                session_id=session_id,
                timestamp=datetime.now(),
                data={"index": i},
                source="test_engine"
            )
            await websocket_manager.broadcast_research_event(session_id, event)

        # Should be limited to 500 most recent events
        assert len(websocket_manager.event_history[session_id]) == 500
        # Should contain the most recent events
        assert websocket_manager.event_history[session_id][-1].data["index"] == 1199

    @pytest.mark.asyncio
    async def test_journal_entry_limit(self, websocket_manager, mock_websocket):
        """Test journal entry size limiting"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        # Add more than 2000 journal entries
        for i in range(2200):
            entry = ResearchJournalEntry(
                timestamp=datetime.now(),
                session_id=session_id,
                engine="test_engine",
                phase="test_phase",
                message=f"Test message {i}",
                metadata={"index": i}
            )
            await websocket_manager.broadcast_journal_entry(session_id, entry)

        # Should be limited to 1000 most recent entries
        assert len(websocket_manager.journal_entries[session_id]) == 1000

    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, websocket_manager, mock_websocket):
        """Test WebSocket error handling during broadcasting"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        # Mock send_text to raise WebSocketDisconnect
        mock_websocket.send_text.side_effect = WebSocketDisconnect(code=1001)

        event = ProgressEvent(
            event_type=EventType.RESEARCH_STARTED,
            session_id=session_id,
            timestamp=datetime.now(),
            data={},
            source="test_engine"
        )

        # Should not raise error
        await websocket_manager.broadcast_research_event(session_id, event)

        # WebSocket should be removed from connections
        assert mock_websocket not in websocket_manager.research_connections[session_id]

    @pytest.mark.asyncio
    async def test_multiple_connections_same_session(self, websocket_manager):
        """Test multiple WebSocket connections for same session"""
        session_id = "test_session"
        mock_ws1 = Mock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()

        mock_ws2 = Mock(spec=WebSocket)
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_text = AsyncMock()

        # Connect both WebSockets to same session
        await websocket_manager.connect_research(mock_ws1, session_id)
        await websocket_manager.connect_research(mock_ws2, session_id)

        assert len(websocket_manager.research_connections[session_id]) == 2

        # Broadcast event
        event = ProgressEvent(
            event_type=EventType.RESEARCH_STARTED,
            session_id=session_id,
            timestamp=datetime.now(),
            data={},
            source="test_engine"
        )

        await websocket_manager.broadcast_research_event(session_id, event)

        # Both WebSockets should receive the event
        mock_ws1.send_text.assert_called()
        mock_ws2.send_text.assert_called()


class TestResearchProgressTracker:
    """Test research progress tracking functionality"""

    @pytest.mark.asyncio
    async def test_log_research_started(self, progress_tracker, websocket_manager, mock_websocket):
        """Test logging research start event"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        request = "Test research request"
        engine = "test_engine"

        await progress_tracker.log_research_started(session_id, request, engine)

        # Should create research event
        assert session_id in websocket_manager.event_history
        events = websocket_manager.event_history[session_id]
        assert len(events) == 1
        assert events[0].event_type == EventType.RESEARCH_STARTED
        assert events[0].progress_percentage == 0.0

        # Should create journal entry
        assert session_id in websocket_manager.journal_entries
        entries = websocket_manager.journal_entries[session_id]
        assert len(entries) == 1
        assert entries[0].phase == "initialization"
        assert entries[0].level == "info"

    @pytest.mark.asyncio
    async def test_log_research_completed(self, progress_tracker, websocket_manager, mock_websocket):
        """Test logging research completion event"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        engine = "test_engine"
        result_summary = "Research completed successfully"
        metadata = {"sources_count": 5, "confidence_score": 0.95}

        await progress_tracker.log_research_completed(session_id, engine, result_summary, metadata)

        # Should create completion event
        events = websocket_manager.event_history[session_id]
        assert len(events) == 1
        assert events[0].event_type == EventType.RESEARCH_COMPLETED
        assert events[0].progress_percentage == 100.0
        assert events[0].data["status"] == "completed"

        # Should create journal entry
        entries = websocket_manager.journal_entries[session_id]
        assert len(entries) == 1
        assert entries[0].phase == "completion"
        assert entries[0].level == "success"

    @pytest.mark.asyncio
    async def test_log_research_progress(self, progress_tracker, websocket_manager, mock_websocket):
        """Test logging research progress event"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        engine = "test_engine"
        phase = "data_collection"
        progress = 75.0
        message = "Collecting data from sources"
        metadata = {"sources_processed": 3, "total_sources": 4}

        await progress_tracker.log_research_progress(session_id, engine, phase, progress, message, metadata)

        # Should create progress event
        events = websocket_manager.event_history[session_id]
        assert len(events) == 1
        assert events[0].event_type == EventType.RESEARCH_PROGRESS
        assert events[0].progress_percentage == progress
        assert events[0].data["phase"] == phase

        # Should create journal entry
        entries = websocket_manager.journal_entries[session_id]
        assert len(entries) == 1
        assert entries[0].phase == phase
        assert entries[0].message == message

    @pytest.mark.asyncio
    async def test_log_error(self, progress_tracker, websocket_manager, mock_websocket):
        """Test logging error event"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        engine = "test_engine"
        error = "Network connection failed"
        phase = "data_collection"

        await progress_tracker.log_error(session_id, engine, error, phase)

        # Should create error event
        events = websocket_manager.event_history[session_id]
        assert len(events) == 1
        assert events[0].event_type == EventType.RESEARCH_ERROR
        assert events[0].data["error"] == error

        # Should create error journal entry
        entries = websocket_manager.journal_entries[session_id]
        assert len(entries) == 1
        assert entries[0].level == "error"
        assert entries[0].message == f"Error: {error}"

    @pytest.mark.asyncio
    async def test_log_engine_coordination(self, progress_tracker, websocket_manager, mock_websocket):
        """Test logging engine coordination event"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        primary_engine = "scientific_research"
        coordinated_engines = ["deep_research", "code_research"]
        phase = "literature_review"

        await progress_tracker.log_engine_coordination(session_id, primary_engine, coordinated_engines, phase)

        # Should create engine status event
        events = websocket_manager.event_history[session_id]
        assert len(events) == 1
        assert events[0].event_type == EventType.ENGINE_STATUS
        assert events[0].source == "coordinator"
        assert events[0].data["primary_engine"] == primary_engine
        assert events[0].data["coordinated_engines"] == coordinated_engines

    @pytest.mark.asyncio
    async def test_log_openhands_execution(self, progress_tracker, websocket_manager, mock_websocket):
        """Test logging OpenHands execution"""
        session_id = "test_session"
        await websocket_manager.connect_research(mock_websocket, session_id)

        command = "python test_script.py"
        workspace_path = "/tmp/test_workspace"
        phase = "code_execution"

        await progress_tracker.log_openhands_execution(session_id, command, workspace_path, phase)

        # Should create journal entry
        entries = websocket_manager.journal_entries[session_id]
        assert len(entries) == 1
        assert entries[0].engine == "openhands"
        assert entries[0].phase == phase
        assert command in entries[0].message
        assert entries[0].metadata["command"] == command
        assert entries[0].metadata["workspace"] == workspace_path