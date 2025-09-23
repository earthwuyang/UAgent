"""
Unit tests for StreamingLLMClient using real DashScope Qwen client
Tests the LLM interaction streaming wrapper functionality with real API calls
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch

from app.core.streaming_llm_client import StreamingLLMClient
from app.core.llm_client import create_llm_client


@pytest.fixture
def real_llm_client():
    """Create real DashScope LLM client"""
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        pytest.skip("LITELLM_API_KEY not set, skipping real LLM tests")

    return create_llm_client("litellm", api_key=api_key, model="qwen-max-latest")


@pytest.fixture
def mock_websocket_manager():
    """Mock WebSocket manager"""
    with patch('app.core.streaming_llm_client.websocket_manager') as mock:
        mock.llm_stream_connections = {"test_session": {Mock()}}
        mock._broadcast_to_connections = AsyncMock()
        yield mock


class TestStreamingLLMClientReal:
    """Test StreamingLLMClient functionality with real LLM client"""

    def test_init_without_session(self, real_llm_client):
        """Test initialization without session ID"""
        client = StreamingLLMClient(real_llm_client)

        assert client.llm_client == real_llm_client
        assert client.session_id is None

    def test_init_with_session(self, real_llm_client):
        """Test initialization with session ID"""
        session_id = "test_session"
        client = StreamingLLMClient(real_llm_client, session_id)

        assert client.llm_client == real_llm_client
        assert client.session_id == session_id

    def test_with_session(self, real_llm_client):
        """Test creating new instance with session ID"""
        client = StreamingLLMClient(real_llm_client)
        new_client = client.with_session("new_session")

        assert new_client.llm_client == real_llm_client
        assert new_client.session_id == "new_session"
        assert client.session_id is None  # Original unchanged

    @pytest.mark.asyncio
    async def test_real_llm_classify_without_session(self, real_llm_client, mock_websocket_manager):
        """Test real LLM classification without session ID (no broadcasting)"""
        client = StreamingLLMClient(real_llm_client)

        # Use a simple classification prompt
        request = "What is machine learning?"
        prompt = """Classify this request into one of these categories:
        1. DEEP_RESEARCH - General information gathering
        2. CODE_RESEARCH - Code analysis and implementation
        3. SCIENTIFIC_RESEARCH - Experimental research with hypotheses

        Respond with valid JSON: {"engine": "DEEP_RESEARCH", "confidence": 0.9, "reasoning": "explanation"}"""

        result = await client.classify(request, prompt)

        # Should return valid classification result
        assert isinstance(result, dict)
        assert "engine" in result
        assert result["engine"] in ["DEEP_RESEARCH", "CODE_RESEARCH", "SCIENTIFIC_RESEARCH"]

        # Should not broadcast when no session
        mock_websocket_manager._broadcast_to_connections.assert_not_called()

    @pytest.mark.asyncio
    async def test_real_llm_classify_with_session(self, real_llm_client, mock_websocket_manager):
        """Test real LLM classification with session ID (with broadcasting)"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        request = "How to implement a neural network?"
        prompt = """Classify this request into one of these categories:
        1. DEEP_RESEARCH - General information gathering
        2. CODE_RESEARCH - Code analysis and implementation
        3. SCIENTIFIC_RESEARCH - Experimental research with hypotheses

        Respond with valid JSON: {"engine": "CODE_RESEARCH", "confidence": 0.8, "reasoning": "implementation focused"}"""

        result = await client.classify(request, prompt)

        # Should return valid classification result
        assert isinstance(result, dict)
        assert "engine" in result

        # Should broadcast start and complete events
        assert mock_websocket_manager._broadcast_to_connections.call_count == 2

        # Check start event
        start_call = mock_websocket_manager._broadcast_to_connections.call_args_list[0]
        start_message = start_call[0][1]
        assert start_message["type"] == "llm_prompt_start"
        assert start_message["session_id"] == "test_session"
        assert "Classification: How to implement" in start_message["prompt"]

        # Check completion event
        complete_call = mock_websocket_manager._broadcast_to_connections.call_args_list[1]
        complete_message = complete_call[0][1]
        assert complete_message["type"] == "llm_prompt_complete"
        assert complete_message["session_id"] == "test_session"

    @pytest.mark.asyncio
    async def test_real_llm_generate_without_session(self, real_llm_client, mock_websocket_manager):
        """Test real LLM generation without session ID (no broadcasting)"""
        client = StreamingLLMClient(real_llm_client)

        prompt = "Explain what machine learning is in one sentence."

        result = await client.generate(prompt, max_tokens=100)

        # Should return valid text result
        assert isinstance(result, str)
        assert len(result) > 0

        # Should not broadcast when no session
        mock_websocket_manager._broadcast_to_connections.assert_not_called()

    @pytest.mark.asyncio
    async def test_real_llm_generate_with_session(self, real_llm_client, mock_websocket_manager):
        """Test real LLM generation with session ID (with broadcasting)"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        prompt = "What is the capital of France?"

        result = await client.generate(prompt, max_tokens=50)

        # Should return valid text result
        assert isinstance(result, str)
        assert len(result) > 0

        # Should broadcast start and complete events
        assert mock_websocket_manager._broadcast_to_connections.call_count == 2

        # Check that events contain session information
        calls = mock_websocket_manager._broadcast_to_connections.call_args_list
        for call in calls:
            message = call[0][1]
            assert message["session_id"] == "test_session"

    @pytest.mark.asyncio
    async def test_real_llm_stream_generate_without_session(self, real_llm_client, mock_websocket_manager):
        """Test real LLM streaming generation without session ID (no broadcasting)"""
        client = StreamingLLMClient(real_llm_client)

        prompt = "Count from 1 to 3."

        tokens = []
        async for token in client.stream_generate(prompt, max_tokens=20):
            tokens.append(token)

        # Should return valid tokens
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)

        # Should not broadcast when no session
        mock_websocket_manager._broadcast_to_connections.assert_not_called()

    @pytest.mark.asyncio
    async def test_real_llm_stream_generate_with_session(self, real_llm_client, mock_websocket_manager):
        """Test real LLM streaming generation with session ID (with broadcasting)"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        prompt = "Say hello."

        tokens = []
        async for token in client.stream_generate(prompt, max_tokens=10):
            tokens.append(token)

        # Should return valid tokens
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)

        # Should broadcast start, tokens, and complete events
        expected_calls = 1 + len(tokens) + 1  # start + tokens + complete
        assert mock_websocket_manager._broadcast_to_connections.call_count == expected_calls

        # Check start event
        start_call = mock_websocket_manager._broadcast_to_connections.call_args_list[0]
        start_message = start_call[0][1]
        assert start_message["type"] == "llm_prompt_start"

        # Check token events
        token_calls = mock_websocket_manager._broadcast_to_connections.call_args_list[1:-1]
        for i, call in enumerate(token_calls):
            message = call[0][1]
            assert message["type"] == "llm_token"
            assert message["session_id"] == "test_session"
            assert message["token"] == tokens[i]

        # Check completion event
        complete_call = mock_websocket_manager._broadcast_to_connections.call_args_list[-1]
        complete_message = complete_call[0][1]
        assert complete_message["type"] == "llm_prompt_complete"

    @pytest.mark.asyncio
    async def test_real_llm_error_handling(self, real_llm_client, mock_websocket_manager):
        """Test error handling with real LLM client"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        # Test with invalid prompt that might cause issues
        try:
            # Use extremely long prompt that might cause token limit error
            long_prompt = "Explain this: " + "x" * 10000
            result = await client.generate(long_prompt, max_tokens=100)

            # If it succeeds, that's fine - we're testing error handling capability
            assert isinstance(result, str)

        except Exception as e:
            # If it fails, check that error event was broadcast
            error_calls = [call for call in mock_websocket_manager._broadcast_to_connections.call_args_list
                          if call[0][1].get("type") == "llm_error"]
            assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_concurrent_real_llm_operations(self, real_llm_client, mock_websocket_manager):
        """Test concurrent LLM operations with real client"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        # Run multiple operations concurrently
        tasks = [
            client.generate("What is 2+2?", max_tokens=10),
            client.generate("What is the color of the sky?", max_tokens=10),
            client.classify("Research machine learning",
                           """Classify as: {"engine": "DEEP_RESEARCH", "confidence": 0.9}""")
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should complete all tasks
        assert len(results) == 3

        # Results should be valid (either strings or dicts)
        for result in results:
            if not isinstance(result, Exception):
                assert isinstance(result, (str, dict))

        # Should have broadcasted events for all operations
        assert mock_websocket_manager._broadcast_to_connections.call_count >= 6  # At least 2 events per operation

    @pytest.mark.asyncio
    async def test_prompt_truncation_in_broadcast(self, real_llm_client, mock_websocket_manager):
        """Test that long prompts are properly truncated in broadcast messages"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        # Use a prompt longer than 200 characters
        long_prompt = "This is a very long prompt that exceeds 200 characters. " * 5
        assert len(long_prompt) > 200

        try:
            await client.generate(long_prompt, max_tokens=10)
        except Exception:
            pass  # Ignore LLM errors, we're testing broadcast message formatting

        # Check that broadcast message has truncated prompt
        if mock_websocket_manager._broadcast_to_connections.call_count > 0:
            start_call = mock_websocket_manager._broadcast_to_connections.call_args_list[0]
            start_message = start_call[0][1]
            assert len(start_message["prompt"]) <= 203  # 200 + "..."

    @pytest.mark.asyncio
    async def test_no_session_connections_real_llm(self, real_llm_client, mock_websocket_manager):
        """Test behavior when session has no WebSocket connections with real LLM"""
        mock_websocket_manager.llm_stream_connections = {}  # No connections

        client = StreamingLLMClient(real_llm_client, "test_session")

        result = await client.generate("Hello", max_tokens=5)

        # Should still work and return valid result
        assert isinstance(result, str)
        assert len(result) > 0

        # Should not attempt to broadcast when no connections exist
        mock_websocket_manager._broadcast_to_connections.assert_not_called()

    @pytest.mark.asyncio
    async def test_message_formatting_with_real_responses(self, real_llm_client, mock_websocket_manager):
        """Test message formatting with real LLM responses"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        result = await client.generate("Say exactly: Hello World", max_tokens=10)

        # Check broadcast message formatting
        calls = mock_websocket_manager._broadcast_to_connections.call_args_list

        # Should have at least start and complete events
        assert len(calls) >= 2

        # Check start message format
        start_message = calls[0][0][1]
        assert start_message["type"] == "llm_prompt_start"
        assert "timestamp" in start_message
        assert start_message["engine"] == "qwen"

        # Check complete message format
        complete_message = calls[-1][0][1]
        assert complete_message["type"] == "llm_prompt_complete"
        assert "timestamp" in complete_message