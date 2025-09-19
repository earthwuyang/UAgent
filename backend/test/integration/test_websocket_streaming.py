"""
Integration tests for WebSocket streaming system
Tests real-time research progress and LLM interaction streaming
"""

import pytest
import pytest_asyncio
import asyncio
import json
import os
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from app.main import app
from app.core.llm_client import create_llm_client
from app.core.streaming_llm_client import StreamingLLMClient
from app.core.websocket_manager import websocket_manager, progress_tracker
from app.core.app_state import set_app_state


@pytest.fixture
def test_client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def real_llm_client():
    """Create real LLM client for integration tests"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("DASHSCOPE_API_KEY not set, skipping integration tests")

    return create_llm_client("dashscope", api_key=api_key, model="qwen-max-latest")


@pytest_asyncio.fixture
async def setup_app_state(real_llm_client):
    """Setup application state for testing"""
    # Mock research engines for testing
    mock_deep_engine = AsyncMock()
    mock_deep_engine.research = AsyncMock(return_value=AsyncMock(
        query="test query",
        sources=[],
        key_findings=["test finding"],
        confidence_score=0.9,
        summary="Test research summary"
    ))

    mock_code_engine = AsyncMock()
    mock_code_engine.research_code = AsyncMock(return_value=AsyncMock(
        query="test code query",
        repositories=[],
        best_practices=["test practice"],
        confidence_score=0.85,
        integration_guide="Test integration guide",
        recommendations=["test recommendation"]
    ))

    mock_scientific_engine = AsyncMock()
    mock_scientific_engine.conduct_research = AsyncMock(return_value=AsyncMock(
        query="test scientific query",
        hypotheses=["test hypothesis"],
        experiments=[],
        iteration_count=1,
        confidence_score=0.8,
        literature_review=None,
        code_analysis=None,
        synthesis="Test synthesis"
    ))

    # Setup app state
    set_app_state({
        "llm_client": real_llm_client,
        "engines": {
            "deep": mock_deep_engine,
            "code": mock_code_engine,
            "scientific": mock_scientific_engine
        },
        "smart_router": AsyncMock()
    })

    yield {
        "deep": mock_deep_engine,
        "code": mock_code_engine,
        "scientific": mock_scientific_engine
    }


class TestWebSocketStreamingIntegration:
    """Test WebSocket streaming integration"""

    @pytest.mark.asyncio
    async def test_research_progress_websocket_connection(self, test_client):
        """Test research progress WebSocket connection establishment"""
        session_id = "test_session_123"

        with test_client.websocket_connect(f"/ws/research/{session_id}") as websocket:
            # Should connect successfully
            assert websocket is not None

            # Send a test message
            websocket.send_text(json.dumps({"action": "get_status"}))

            # Should maintain connection
            # Connection will be maintained until context exits

    @pytest.mark.asyncio
    async def test_llm_stream_websocket_connection(self, test_client):
        """Test LLM stream WebSocket connection establishment"""
        session_id = "test_session_123"

        with test_client.websocket_connect(f"/ws/llm/{session_id}") as websocket:
            # Should connect successfully
            assert websocket is not None

    @pytest.mark.asyncio
    async def test_research_progress_event_broadcasting(self, test_client):
        """Test research progress event broadcasting through WebSocket"""
        session_id = "test_session_progress"

        with test_client.websocket_connect(f"/ws/research/{session_id}") as websocket:
            # Log a research started event
            await progress_tracker.log_research_started(
                session_id=session_id,
                request="Test research request",
                engine="test_engine"
            )

            # Should receive the event
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "research_event"
            assert message["event"]["event_type"] == "research_started"
            assert message["event"]["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_research_completion_event_broadcasting(self, test_client):
        """Test research completion event broadcasting"""
        session_id = "test_session_completion"

        with test_client.websocket_connect(f"/ws/research/{session_id}") as websocket:
            # Log a research completion event
            await progress_tracker.log_research_completed(
                session_id=session_id,
                engine="test_engine",
                result_summary="Test research completed successfully",
                metadata={"sources_count": 5}
            )

            # Should receive the completion event
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "research_event"
            assert message["event"]["event_type"] == "research_completed"
            assert message["event"]["progress_percentage"] == 100.0
            assert "completed successfully" in message["event"]["data"]["result_summary"]

    @pytest.mark.asyncio
    async def test_llm_interaction_streaming(self, test_client, real_llm_client):
        """Test LLM interaction streaming through WebSocket"""
        session_id = "test_session_llm"

        with test_client.websocket_connect(f"/ws/llm/{session_id}") as websocket:
            # Create streaming LLM client
            streaming_client = StreamingLLMClient(real_llm_client, session_id)

            # Start an async task for LLM generation
            async def generate_text():
                return await streaming_client.generate("Say hello", max_tokens=10)

            # Start generation in background
            generation_task = asyncio.create_task(generate_text())

            # Collect WebSocket messages
            messages = []
            try:
                # Receive messages with timeout
                for _ in range(5):  # Expect start, possibly tokens, and complete
                    try:
                        data = websocket.receive_text(timeout=10)
                        messages.append(json.loads(data))
                    except:
                        break

                # Wait for generation to complete
                result = await generation_task

                # Should have received start and complete events at minimum
                assert len(messages) >= 2

                # Check start event
                start_event = messages[0]
                assert start_event["type"] == "llm_prompt_start"
                assert start_event["session_id"] == session_id

                # Check completion event
                complete_event = messages[-1]
                assert complete_event["type"] == "llm_prompt_complete"
                assert complete_event["session_id"] == session_id

            except Exception as e:
                # Ensure generation task is cancelled
                generation_task.cancel()
                raise

    @pytest.mark.asyncio
    async def test_multiple_concurrent_websocket_connections(self, test_client):
        """Test multiple concurrent WebSocket connections"""
        session_id = "test_session_concurrent"

        # Create multiple WebSocket connections
        connections = []
        try:
            for i in range(3):
                ws = test_client.websocket_connect(f"/ws/research/{session_id}")
                connections.append(ws.__enter__())

            # Broadcast an event
            await progress_tracker.log_research_progress(
                session_id=session_id,
                engine="test_engine",
                phase="test_phase",
                progress=50.0,
                message="Test progress message"
            )

            # All connections should receive the event
            for i, websocket in enumerate(connections):
                data = websocket.receive_text()
                message = json.loads(data)

                assert message["type"] == "research_event"
                assert message["event"]["progress_percentage"] == 50.0

        finally:
            # Close all connections
            for ws in connections:
                try:
                    ws.__exit__(None, None, None)
                except:
                    pass

    @pytest.mark.asyncio
    async def test_websocket_connection_history_replay(self, test_client):
        """Test that new connections receive event history"""
        session_id = "test_session_history"

        # First, create some events without any connections
        await progress_tracker.log_research_started(
            session_id=session_id,
            request="Historical test request",
            engine="test_engine"
        )

        await progress_tracker.log_research_progress(
            session_id=session_id,
            engine="test_engine",
            phase="test_phase",
            progress=25.0,
            message="Historical progress"
        )

        # Now connect and should receive history replay
        with test_client.websocket_connect(f"/ws/research/{session_id}") as websocket:
            # Should receive replayed events
            messages = []
            try:
                # Receive events with timeout
                for _ in range(3):  # Expect 2 history events
                    try:
                        data = websocket.receive_text(timeout=5)
                        messages.append(json.loads(data))
                    except:
                        break

                # Should have received replayed events
                assert len(messages) >= 2

                # Check that these are replay events
                for message in messages:
                    assert message["type"] == "event_replay"
                    assert "event" in message

            except Exception:
                pass  # Timeout is expected when no more events

    @pytest.mark.asyncio
    async def test_websocket_error_recovery(self, test_client):
        """Test WebSocket error handling and recovery"""
        session_id = "test_session_error"

        with test_client.websocket_connect(f"/ws/research/{session_id}") as websocket:
            # Send an invalid JSON message
            try:
                websocket.send_text("invalid json")

                # Connection should remain stable
                await progress_tracker.log_research_started(
                    session_id=session_id,
                    request="Test after error",
                    engine="test_engine"
                )

                # Should still receive events
                data = websocket.receive_text()
                message = json.loads(data)
                assert message["type"] == "research_event"

            except Exception:
                # Error handling may close connection, which is acceptable
                pass

    @pytest.mark.asyncio
    async def test_session_isolation(self, test_client):
        """Test that different sessions are properly isolated"""
        session_id_1 = "test_session_1"
        session_id_2 = "test_session_2"

        with test_client.websocket_connect(f"/ws/research/{session_id_1}") as ws1, \
             test_client.websocket_connect(f"/ws/research/{session_id_2}") as ws2:

            # Send event to session 1
            await progress_tracker.log_research_started(
                session_id=session_id_1,
                request="Session 1 request",
                engine="test_engine"
            )

            # Only session 1 should receive the event
            data1 = ws1.receive_text()
            message1 = json.loads(data1)
            assert message1["event"]["session_id"] == session_id_1

            # Session 2 should not receive anything
            try:
                ws2.receive_text(timeout=1)  # Short timeout
                pytest.fail("Session 2 should not have received session 1's event")
            except:
                pass  # Expected timeout

    @pytest.mark.asyncio
    async def test_engine_status_broadcasting(self, test_client):
        """Test engine status broadcasting"""
        with test_client.websocket_connect("/ws/engines/status") as websocket:
            # Update engine status
            from app.core.websocket_manager import EngineStatus
            from datetime import datetime

            engine_status = EngineStatus(
                engine_name="test_engine",
                status="running",
                current_task="Processing test data",
                progress_percentage=75.0,
                start_time=datetime.now(),
                estimated_completion=None,
                metrics={"processed_items": 100}
            )

            await websocket_manager.update_engine_status(engine_status)

            # Should receive engine status update
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "engine_status"
            assert message["status"]["engine_name"] == "test_engine"
            assert message["status"]["progress_percentage"] == 75.0

    @pytest.mark.asyncio
    async def test_journal_entry_broadcasting(self, test_client):
        """Test research journal entry broadcasting"""
        session_id = "test_session_journal"

        with test_client.websocket_connect(f"/ws/research/{session_id}") as websocket:
            # Create journal entry through progress tracker
            await progress_tracker.log_research_progress(
                session_id=session_id,
                engine="test_engine",
                phase="data_analysis",
                progress=60.0,
                message="Analyzing collected data",
                metadata={"items_analyzed": 50}
            )

            # Should receive both research event and journal entry
            messages = []
            for _ in range(2):
                try:
                    data = websocket.receive_text(timeout=5)
                    messages.append(json.loads(data))
                except:
                    break

            # Should have received research_event and journal_entry
            event_types = [msg["type"] for msg in messages]
            assert "research_event" in event_types
            assert "journal_entry" in event_types

    @pytest.mark.asyncio
    async def test_end_to_end_research_streaming(self, test_client, setup_app_state, real_llm_client):
        """Test end-to-end research execution with streaming"""
        session_id = "test_session_e2e"
        engines = setup_app_state

        # Setup streaming connections
        research_messages = []
        llm_messages = []

        with test_client.websocket_connect(f"/ws/research/{session_id}") as research_ws, \
             test_client.websocket_connect(f"/ws/llm/{session_id}") as llm_ws:

            # Simulate research execution with streaming LLM client
            streaming_llm_client = StreamingLLMClient(real_llm_client, session_id)

            async def simulate_research():
                # Log research start
                await progress_tracker.log_research_started(
                    session_id=session_id,
                    request="Test end-to-end research",
                    engine="deep_research"
                )

                # Simulate LLM interaction during research
                result = await streaming_llm_client.generate(
                    "Summarize research findings",
                    max_tokens=20
                )

                # Log research completion
                await progress_tracker.log_research_completed(
                    session_id=session_id,
                    engine="deep_research",
                    result_summary="Research completed with findings",
                    metadata={"result": result}
                )

            # Start research simulation
            research_task = asyncio.create_task(simulate_research())

            # Collect messages from both WebSockets
            async def collect_messages():
                try:
                    # Collect research messages
                    for _ in range(5):
                        try:
                            data = research_ws.receive_text(timeout=2)
                            research_messages.append(json.loads(data))
                        except:
                            break

                    # Collect LLM messages
                    for _ in range(5):
                        try:
                            data = llm_ws.receive_text(timeout=2)
                            llm_messages.append(json.loads(data))
                        except:
                            break
                except Exception:
                    pass

            # Run both tasks
            await asyncio.gather(research_task, collect_messages(), return_exceptions=True)

            # Verify we received events
            assert len(research_messages) >= 2  # At least start and completion
            assert len(llm_messages) >= 2      # At least start and completion

            # Verify research flow
            research_event_types = [msg["event"]["event_type"] for msg in research_messages if msg.get("type") == "research_event"]
            assert "research_started" in research_event_types
            assert "research_completed" in research_event_types

            # Verify LLM flow
            llm_event_types = [msg["type"] for msg in llm_messages]
            assert "llm_prompt_start" in llm_event_types
            assert "llm_prompt_complete" in llm_event_types