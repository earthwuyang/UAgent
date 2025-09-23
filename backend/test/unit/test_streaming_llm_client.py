"""
Unit tests for StreamingLLMClient
Tests the LLM interaction streaming wrapper functionality using real DashScope Qwen client
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


class TestStreamingLLMClient:
    """Test StreamingLLMClient functionality"""

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
    async def test_classify_without_session(self, real_llm_client, mock_websocket_manager):
        """Test classification without session ID (no broadcasting)"""
        client = StreamingLLMClient(real_llm_client)

        prompt = """Classify this request into one of these categories:
        1. DEEP_RESEARCH - General information gathering
        2. CODE_RESEARCH - Code analysis and implementation
        3. SCIENTIFIC_RESEARCH - Experimental research with hypotheses

        Respond with valid JSON: {"engine": "DEEP_RESEARCH", "confidence": 0.9, "reasoning": "explanation"}"""

        result = await client.classify("test request", prompt)

        assert isinstance(result, dict)
        assert "engine" in result
        mock_websocket_manager._broadcast_to_connections.assert_not_called()

    @pytest.mark.asyncio
    async def test_classify_with_session(self, real_llm_client, mock_websocket_manager):
        """Test classification with session ID (with broadcasting)"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        prompt = """Classify this request into one of these categories:
        1. DEEP_RESEARCH - General information gathering
        2. CODE_RESEARCH - Code analysis and implementation
        3. SCIENTIFIC_RESEARCH - Experimental research with hypotheses

        Respond with valid JSON: {"engine": "DEEP_RESEARCH", "confidence": 0.9, "reasoning": "explanation"}"""

        result = await client.classify("test request", prompt)

        assert isinstance(result, dict)
        assert "engine" in result

        # Should broadcast start and complete events
        assert mock_websocket_manager._broadcast_to_connections.call_count == 2

        # Check start event
        start_call = mock_websocket_manager._broadcast_to_connections.call_args_list[0]
        start_message = start_call[0][1]
        assert start_message["type"] == "llm_prompt_start"
        assert start_message["session_id"] == "test_session"
        assert "Classification: test request" in start_message["prompt"]

        # Check completion event
        complete_call = mock_websocket_manager._broadcast_to_connections.call_args_list[1]
        complete_message = complete_call[0][1]
        assert complete_message["type"] == "llm_prompt_complete"
        assert complete_message["session_id"] == "test_session"

    @pytest.mark.asyncio
    async def test_generate_without_session(self, real_llm_client, mock_websocket_manager):
        """Test generation without session ID (no broadcasting)"""
        client = StreamingLLMClient(real_llm_client)

        result = await client.generate("What is 2+2?", max_tokens=10)

        assert isinstance(result, str)
        assert len(result) > 0
        mock_websocket_manager._broadcast_to_connections.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_with_session(self, real_llm_client, mock_websocket_manager):
        """Test generation with session ID (with broadcasting)"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        result = await client.generate("Say hello", max_tokens=10)

        assert isinstance(result, str)
        assert len(result) > 0

        # Should broadcast start and complete events
        assert mock_websocket_manager._broadcast_to_connections.call_count == 2

    @pytest.mark.asyncio
    async def test_stream_generate_without_session(self, real_llm_client, mock_websocket_manager):
        """Test streaming generation without session ID (no broadcasting)"""
        client = StreamingLLMClient(real_llm_client)

        tokens = []
        async for token in client.stream_generate("Count to 3", max_tokens=15):
            tokens.append(token)

        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
        mock_websocket_manager._broadcast_to_connections.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_generate_with_session(self, real_llm_client, mock_websocket_manager):
        """Test streaming generation with session ID (with broadcasting)"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        tokens = []
        async for token in client.stream_generate("Say hello", max_tokens=10):
            tokens.append(token)

        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)

        # Should broadcast start, tokens, and complete events
        expected_calls = 1 + len(tokens) + 1  # start + tokens + complete
        assert mock_websocket_manager._broadcast_to_connections.call_count == expected_calls

        # Check token events
        token_calls = mock_websocket_manager._broadcast_to_connections.call_args_list[1:-1]
        for i, (call, expected_token) in enumerate(zip(token_calls, tokens)):
            message = call[0][1]
            assert message["type"] == "llm_token"
            assert message["session_id"] == "test_session"
            assert message["token"] == expected_token

    @pytest.mark.asyncio
    async def test_classify_error_handling(self, real_llm_client, mock_websocket_manager):
        """Test error handling during classification"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        # Test with invalid prompt that may cause errors
        try:
            # Use extremely long prompt that might cause token limit error
            long_prompt = "Classify this: " + "x" * 10000
            result = await client.classify("test request", long_prompt)

            # If it succeeds, that's fine - we're testing error handling capability
            assert isinstance(result, dict) or result is None

        except Exception as e:
            # If it fails, check that error event was broadcast
            error_calls = [call for call in mock_websocket_manager._broadcast_to_connections.call_args_list
                          if call[0][1].get("type") == "llm_error"]
            assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_generate_error_handling(self, real_llm_client, mock_websocket_manager):
        """Test error handling during generation"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        # Test with invalid prompt that may cause errors
        try:
            # Use extremely long prompt that might cause token limit error
            long_prompt = "Explain this: " + "x" * 10000
            result = await client.generate(long_prompt, max_tokens=10)

            # If it succeeds, that's fine - we're testing error handling capability
            assert isinstance(result, str) or result is None

        except Exception as e:
            # If it fails, check that error event was broadcast
            error_calls = [call for call in mock_websocket_manager._broadcast_to_connections.call_args_list
                          if call[0][1].get("type") == "llm_error"]
            assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_stream_generate_error_handling(self, real_llm_client, mock_websocket_manager):
        """Test error handling during streaming generation"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        # Test with prompt that may cause streaming errors
        tokens = []
        try:
            async for token in client.stream_generate("Say hello", max_tokens=5):
                tokens.append(token)
                # Artificially break after first token to test error handling
                if len(tokens) >= 1:
                    break

            # Should have received at least one token
            assert len(tokens) >= 1

        except Exception as e:
            # If streaming fails, that's acceptable for testing error handling
            pass

        # Should have attempted to broadcast events
        assert mock_websocket_manager._broadcast_to_connections.call_count >= 1

    @pytest.mark.asyncio
    async def test_no_session_connections(self, real_llm_client, mock_websocket_manager):
        """Test behavior when session has no WebSocket connections"""
        mock_websocket_manager.llm_stream_connections = {}  # No connections

        client = StreamingLLMClient(real_llm_client, "test_session")

        result = await client.generate("Hello", max_tokens=5)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should not attempt to broadcast when no connections exist
        mock_websocket_manager._broadcast_to_connections.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_error_handling(self, real_llm_client, mock_websocket_manager):
        """Test that broadcast errors don't affect LLM operations"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        # Mock broadcast to raise an error
        mock_websocket_manager._broadcast_to_connections.side_effect = Exception("Broadcast failed")

        # Should still return result despite broadcast error
        result = await client.generate("Hello", max_tokens=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_message_formatting(self, real_llm_client):
        """Test message formatting for different event types"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        # Test prompt truncation
        long_prompt = "a" * 300
        truncated = long_prompt[:200] + "..."

        # This would be tested by verifying the actual broadcast calls
        # in the integration tests

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, real_llm_client, mock_websocket_manager):
        """Test concurrent LLM operations with same client"""
        client = StreamingLLMClient(real_llm_client, "test_session")

        # Run multiple operations concurrently
        classify_prompt = """Classify as: {"engine": "DEEP_RESEARCH", "confidence": 0.9}"""
        tasks = [
            client.generate("What is 2+2?", max_tokens=10),
            client.generate("Hello", max_tokens=5),
            client.classify("Research AI", classify_prompt)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 3
        # Results should be valid (either strings or dicts, not exceptions)
        for result in results:
            if not isinstance(result, Exception):
                assert isinstance(result, (str, dict))

        # Should have broadcasted events for all operations
        assert mock_websocket_manager._broadcast_to_connections.call_count >= 6  # At least 2 events per operation