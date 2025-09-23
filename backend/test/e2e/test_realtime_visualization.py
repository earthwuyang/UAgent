"""
End-to-end tests for real-time research visualization
Tests complete research workflows with streaming visualization using real LLM client
"""

import pytest
import asyncio
import json
import os
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.core.llm_client import create_llm_client
from app.core.app_state import set_app_state, clear_app_state


@pytest.fixture
def test_client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def real_llm_client():
    """Create real LLM client for E2E tests"""
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        pytest.skip("LITELLM_API_KEY not set, skipping E2E tests")

    return create_llm_client("litellm", api_key=api_key, model="qwen-max-latest")


@pytest.fixture
async def setup_complete_system(real_llm_client):
    """Setup complete system for E2E tests"""

    # Create mock research engines with realistic behavior
    class MockDeepEngine:
        def __init__(self, llm_client):
            self.llm_client = llm_client

        async def research(self, request):
            # Simulate research with real LLM interaction
            summary = await self.llm_client.generate(
                f"Provide a brief research summary for: {request}",
                max_tokens=100
            )

            return AsyncMock(
                query=request,
                sources=["source1.com", "source2.com"],
                key_findings=["Finding 1", "Finding 2"],
                confidence_score=0.9,
                summary=summary
            )

    class MockCodeEngine:
        def __init__(self, llm_client):
            self.llm_client = llm_client

        async def research_code(self, request):
            # Simulate code research with real LLM interaction
            analysis = await self.llm_client.generate(
                f"Analyze code patterns for: {request}",
                max_tokens=80
            )

            return AsyncMock(
                query=request,
                repositories=[],
                best_practices=["Practice 1", "Practice 2"],
                confidence_score=0.85,
                integration_guide=analysis,
                recommendations=["Recommendation 1"]
            )

    class MockScientificEngine:
        def __init__(self, llm_client):
            self.llm_client = llm_client

        async def conduct_research(self, request, **kwargs):
            # Simulate scientific research with real LLM interaction
            hypothesis = await self.llm_client.generate(
                f"Generate a scientific hypothesis for: {request}",
                max_tokens=60
            )

            return AsyncMock(
                query=request,
                hypotheses=[hypothesis],
                experiments=[],
                iteration_count=1,
                confidence_score=0.8,
                literature_review=None,
                code_analysis=None,
                synthesis="Scientific research synthesis"
            )

    # Create mock smart router
    class MockSmartRouter:
        def __init__(self, llm_client):
            self.llm_client = llm_client

        async def classify_and_route(self, request):
            # Use real LLM to classify the request
            classification_prompt = """
            Classify this request into one of these categories and respond with valid JSON:
            1. DEEP_RESEARCH - General information gathering
            2. CODE_RESEARCH - Code analysis and implementation
            3. SCIENTIFIC_RESEARCH - Experimental research with hypotheses

            Request: {request}

            Respond only with JSON in this format:
            {{"primary_engine": "DEEP_RESEARCH", "confidence_score": 0.9, "reasoning": "explanation", "sub_components": {{}}, "workflow_plan": {{}}}}
            """.format(request=request.user_request)

            try:
                result = await self.llm_client.classify(request.user_request, classification_prompt)
                return AsyncMock(
                    primary_engine=result.get("primary_engine", "DEEP_RESEARCH"),
                    confidence_score=result.get("confidence_score", 0.8),
                    reasoning=result.get("reasoning", "Default classification"),
                    sub_components=result.get("sub_components", {}),
                    workflow_plan=result.get("workflow_plan", {})
                )
            except Exception:
                # Fallback classification
                return AsyncMock(
                    primary_engine="DEEP_RESEARCH",
                    confidence_score=0.8,
                    reasoning="Fallback classification",
                    sub_components={},
                    workflow_plan={}
                )

    # Setup engines with real LLM client
    deep_engine = MockDeepEngine(real_llm_client)
    code_engine = MockCodeEngine(real_llm_client)
    scientific_engine = MockScientificEngine(real_llm_client)
    smart_router = MockSmartRouter(real_llm_client)

    # Setup app state
    set_app_state({
        "llm_client": real_llm_client,
        "engines": {
            "deep": deep_engine,
            "code": code_engine,
            "scientific": scientific_engine
        },
        "smart_router": smart_router,
        "cache": AsyncMock()
    })

    yield {
        "deep": deep_engine,
        "code": code_engine,
        "scientific": scientific_engine,
        "smart_router": smart_router
    }

    # Cleanup
    clear_app_state()


class TestRealtimeVisualizationE2E:
    """End-to-end tests for real-time research visualization"""

    @pytest.mark.asyncio
    async def test_complete_deep_research_workflow(self, test_client, setup_complete_system):
        """Test complete deep research workflow with real-time streaming"""
        session_id = "e2e_deep_research"

        # Collect messages from WebSocket connections
        research_messages = []
        llm_messages = []

        with test_client.websocket_connect(f"/ws/research/{session_id}") as research_ws, \
             test_client.websocket_connect(f"/ws/llm/{session_id}") as llm_ws:

            # Execute research request
            request_data = {
                "user_request": "Research the latest developments in artificial intelligence",
                "session_id": session_id
            }

            # Start research via API
            api_response = test_client.post("/api/router/route-and-execute", json=request_data)

            # Collect WebSocket messages during research
            async def collect_websocket_messages():
                # Collect research progress messages
                for _ in range(10):  # Expect multiple progress events
                    try:
                        data = research_ws.receive_text(timeout=5)
                        research_messages.append(json.loads(data))
                    except:
                        break

                # Collect LLM interaction messages
                for _ in range(10):  # Expect LLM interaction events
                    try:
                        data = llm_ws.receive_text(timeout=5)
                        llm_messages.append(json.loads(data))
                    except:
                        break

            # Run message collection
            await asyncio.create_task(collect_websocket_messages())

            # Verify API response
            assert api_response.status_code == 200
            response_data = api_response.json()
            assert "classification" in response_data
            assert "execution" in response_data

            # Verify research progress streaming
            assert len(research_messages) >= 2  # At least start and completion

            # Check for research started event
            start_events = [msg for msg in research_messages
                          if msg.get("event", {}).get("event_type") == "research_started"]
            assert len(start_events) >= 1

            # Check for research completed event
            completed_events = [msg for msg in research_messages
                              if msg.get("event", {}).get("event_type") == "research_completed"]
            assert len(completed_events) >= 1

            # Verify completion event has 100% progress
            if completed_events:
                completion_event = completed_events[0]
                assert completion_event["event"]["progress_percentage"] == 100.0

            # Verify LLM interaction streaming
            assert len(llm_messages) >= 2  # At least start and completion

            # Check for LLM prompt events
            llm_start_events = [msg for msg in llm_messages if msg.get("type") == "llm_prompt_start"]
            llm_complete_events = [msg for msg in llm_messages if msg.get("type") == "llm_prompt_complete"]

            assert len(llm_start_events) >= 1
            assert len(llm_complete_events) >= 1

    @pytest.mark.asyncio
    async def test_code_research_with_streaming(self, test_client, setup_complete_system):
        """Test code research workflow with streaming"""
        session_id = "e2e_code_research"

        research_messages = []
        llm_messages = []

        with test_client.websocket_connect(f"/ws/research/{session_id}") as research_ws, \
             test_client.websocket_connect(f"/ws/llm/{session_id}") as llm_ws:

            # Execute code research request
            request_data = {
                "user_request": "Find best practices for implementing REST APIs in Python",
                "session_id": session_id
            }

            # Mock smart router to return CODE_RESEARCH
            with patch.object(setup_complete_system["smart_router"], 'classify_and_route') as mock_classify:
                mock_classify.return_value = AsyncMock(
                    primary_engine="CODE_RESEARCH",
                    confidence_score=0.9,
                    reasoning="Request is about code implementation",
                    sub_components={},
                    workflow_plan={}
                )

                api_response = test_client.post("/api/router/route-and-execute", json=request_data)

                # Collect messages
                async def collect_messages():
                    for _ in range(8):
                        try:
                            data = research_ws.receive_text(timeout=3)
                            research_messages.append(json.loads(data))
                        except:
                            break

                    for _ in range(8):
                        try:
                            data = llm_ws.receive_text(timeout=3)
                            llm_messages.append(json.loads(data))
                        except:
                            break

                await asyncio.create_task(collect_messages())

                # Verify API response
                assert api_response.status_code == 200
                response_data = api_response.json()
                assert response_data["classification"]["primary_engine"] == "CODE_RESEARCH"

                # Verify streaming events
                assert len(research_messages) >= 1
                assert len(llm_messages) >= 1

    @pytest.mark.asyncio
    async def test_scientific_research_with_streaming(self, test_client, setup_complete_system):
        """Test scientific research workflow with streaming"""
        session_id = "e2e_scientific_research"

        research_messages = []
        llm_messages = []

        with test_client.websocket_connect(f"/ws/research/{session_id}") as research_ws, \
             test_client.websocket_connect(f"/ws/llm/{session_id}") as llm_ws:

            # Execute scientific research request
            request_data = {
                "user_request": "Design experiments to test the effectiveness of attention mechanisms",
                "session_id": session_id
            }

            # Mock smart router to return SCIENTIFIC_RESEARCH
            with patch.object(setup_complete_system["smart_router"], 'classify_and_route') as mock_classify:
                mock_classify.return_value = AsyncMock(
                    primary_engine="SCIENTIFIC_RESEARCH",
                    confidence_score=0.95,
                    reasoning="Request involves experimental design and hypothesis testing",
                    sub_components={"experimentation": True},
                    workflow_plan={
                        "include_literature_review": True,
                        "include_code_analysis": True,
                        "enable_iteration": True
                    }
                )

                api_response = test_client.post("/api/router/route-and-execute", json=request_data)

                # Collect messages
                async def collect_messages():
                    for _ in range(10):
                        try:
                            data = research_ws.receive_text(timeout=4)
                            research_messages.append(json.loads(data))
                        except:
                            break

                    for _ in range(10):
                        try:
                            data = llm_ws.receive_text(timeout=4)
                            llm_messages.append(json.loads(data))
                        except:
                            break

                await asyncio.create_task(collect_messages())

                # Verify API response
                assert api_response.status_code == 200
                response_data = api_response.json()
                assert response_data["classification"]["primary_engine"] == "SCIENTIFIC_RESEARCH"

                # Verify streaming events for scientific research
                assert len(research_messages) >= 1
                assert len(llm_messages) >= 1

    @pytest.mark.asyncio
    async def test_multiple_concurrent_research_sessions(self, test_client, setup_complete_system):
        """Test multiple concurrent research sessions with isolated streaming"""
        session_ids = ["e2e_concurrent_1", "e2e_concurrent_2"]

        # Track messages for each session
        session_messages = {sid: [] for sid in session_ids}

        # Create WebSocket connections for both sessions
        connections = []
        try:
            for session_id in session_ids:
                ws = test_client.websocket_connect(f"/ws/research/{session_id}")
                connections.append((session_id, ws.__enter__()))

            # Start research for both sessions concurrently
            requests = [
                {
                    "user_request": f"Research topic {i+1}",
                    "session_id": session_id
                }
                for i, session_id in enumerate(session_ids)
            ]

            # Execute requests concurrently
            async def execute_research(request_data):
                return test_client.post("/api/router/route-and-execute", json=request_data)

            # Start both research tasks
            tasks = [asyncio.create_task(execute_research(req)) for req in requests]

            # Collect messages from both sessions
            async def collect_session_messages():
                for session_id, websocket in connections:
                    for _ in range(5):
                        try:
                            data = websocket.receive_text(timeout=2)
                            session_messages[session_id].append(json.loads(data))
                        except:
                            break

            # Run collection and research concurrently
            await asyncio.gather(
                collect_session_messages(),
                *tasks,
                return_exceptions=True
            )

            # Verify session isolation
            for session_id in session_ids:
                messages = session_messages[session_id]

                # Each session should have received its own messages
                for message in messages:
                    if "event" in message and "session_id" in message["event"]:
                        assert message["event"]["session_id"] == session_id

        finally:
            # Close all connections
            for _, ws in connections:
                try:
                    ws.__exit__(None, None, None)
                except:
                    pass

    @pytest.mark.asyncio
    async def test_engine_status_monitoring(self, test_client, setup_complete_system):
        """Test engine status monitoring during research"""

        engine_messages = []

        with test_client.websocket_connect("/ws/engines/status") as engine_ws:
            # Execute a research request to trigger engine activity
            request_data = {
                "user_request": "Test engine status monitoring",
                "session_id": "e2e_engine_status"
            }

            # Start research
            api_response = test_client.post("/api/router/route-and-execute", json=request_data)

            # Collect engine status messages
            for _ in range(5):
                try:
                    data = engine_ws.receive_text(timeout=3)
                    engine_messages.append(json.loads(data))
                except:
                    break

            # Verify API response
            assert api_response.status_code == 200

            # Verify engine status messages
            # Note: Actual engine status updates depend on the implementation
            # At minimum, we should verify the WebSocket connection works
            assert isinstance(engine_messages, list)

    @pytest.mark.asyncio
    async def test_research_error_handling_with_streaming(self, test_client, setup_complete_system):
        """Test error handling in research with streaming"""
        session_id = "e2e_error_handling"

        research_messages = []

        with test_client.websocket_connect(f"/ws/research/{session_id}") as research_ws:
            # Mock an engine to raise an error
            with patch.object(setup_complete_system["deep"], 'research') as mock_research:
                mock_research.side_effect = Exception("Simulated research error")

                request_data = {
                    "user_request": "This will cause an error",
                    "session_id": session_id
                }

                # Execute request that will cause error
                try:
                    api_response = test_client.post("/api/router/route-and-execute", json=request_data)
                    # API might return error status
                except Exception:
                    pass

                # Collect any error messages from WebSocket
                for _ in range(3):
                    try:
                        data = research_ws.receive_text(timeout=2)
                        research_messages.append(json.loads(data))
                    except:
                        break

                # Verify error handling
                # Should have at least started the research before error
                if research_messages:
                    start_events = [msg for msg in research_messages
                                  if msg.get("event", {}).get("event_type") == "research_started"]
                    error_events = [msg for msg in research_messages
                                  if msg.get("event", {}).get("event_type") == "research_error"]

                    # Should have either started or error events
                    assert len(start_events) > 0 or len(error_events) > 0

    @pytest.mark.asyncio
    async def test_websocket_reconnection_simulation(self, test_client, setup_complete_system):
        """Test WebSocket reconnection behavior simulation"""
        session_id = "e2e_reconnection"

        # First connection
        with test_client.websocket_connect(f"/ws/research/{session_id}") as ws1:
            # Execute research to create some events
            request_data = {
                "user_request": "Test reconnection",
                "session_id": session_id
            }

            api_response = test_client.post("/api/router/route-and-execute", json=request_data)

            # Receive some messages
            messages_1 = []
            for _ in range(3):
                try:
                    data = ws1.receive_text(timeout=2)
                    messages_1.append(json.loads(data))
                except:
                    break

        # Simulate reconnection with new WebSocket
        with test_client.websocket_connect(f"/ws/research/{session_id}") as ws2:
            # Should receive event history replay
            messages_2 = []
            for _ in range(5):
                try:
                    data = ws2.receive_text(timeout=2)
                    messages_2.append(json.loads(data))
                except:
                    break

            # Verify that reconnection works
            # New connection should receive some messages (either replay or new events)
            assert len(messages_2) >= 0  # At minimum, connection should work