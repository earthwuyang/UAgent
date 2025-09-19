"""Streaming-aware LLM client wrapper for WebSocket broadcasting"""

import json
import logging
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime

from .llm_client import LLMClient
from .websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


class StreamingLLMClient:
    """LLM client wrapper that broadcasts interactions to WebSocket connections"""

    def __init__(self, llm_client: LLMClient, session_id: Optional[str] = None):
        """Initialize streaming LLM client

        Args:
            llm_client: The underlying LLM client
            session_id: Session ID for WebSocket broadcasting
        """
        self.llm_client = llm_client
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)

    async def classify(self, request: str, prompt: str) -> Dict[str, Any]:
        """Classify a request and broadcast to WebSocket if session_id is set"""
        if self.session_id:
            await self._broadcast_prompt_start(f"Classification: {request[:100]}...")

        try:
            result = await self.llm_client.classify(request, prompt)

            if self.session_id:
                await self._broadcast_prompt_complete(f"Classification result: {result.get('engine', 'Unknown')}")

            return result
        except Exception as e:
            if self.session_id:
                await self._broadcast_error(str(e))
            raise

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text and broadcast to WebSocket if session_id is set"""
        logger.info(f"[DEBUG] StreamingLLMClient.generate called for session {self.session_id}")
        if self.session_id:
            logger.info(f"[DEBUG] Broadcasting prompt start for session {self.session_id}")
            await self._broadcast_prompt_start(prompt[:200] + "..." if len(prompt) > 200 else prompt)

        try:
            logger.info(f"[DEBUG] Calling underlying LLM client for session {self.session_id}")
            result = await self.llm_client.generate(prompt, **kwargs)
            logger.info(f"[DEBUG] LLM generation completed for session {self.session_id}")

            if self.session_id:
                logger.info(f"[DEBUG] Broadcasting prompt complete for session {self.session_id}")
                await self._broadcast_prompt_complete(result[:200] + "..." if len(result) > 200 else result)

            return result
        except Exception as e:
            logger.error(f"[DEBUG] LLM generation error for session {self.session_id}: {e}")
            if self.session_id:
                await self._broadcast_error(str(e))
            raise

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream generate text and broadcast to WebSocket if session_id is set"""
        if self.session_id:
            await self._broadcast_prompt_start(prompt[:200] + "..." if len(prompt) > 200 else prompt)

        try:
            response_text = ""
            async for chunk in self.llm_client.stream_generate(prompt, **kwargs):
                response_text += chunk
                if self.session_id:
                    await self._broadcast_token(chunk)
                yield chunk

            if self.session_id:
                await self._broadcast_prompt_complete(response_text[:200] + "..." if len(response_text) > 200 else response_text)

        except Exception as e:
            if self.session_id:
                await self._broadcast_error(str(e))
            raise

    async def _broadcast_prompt_start(self, prompt: str):
        """Broadcast prompt start event"""
        logger.info(f"[DEBUG] _broadcast_prompt_start called for session {self.session_id}")
        if not self.session_id:
            return

        message = {
            "type": "llm_prompt_start",
            "session_id": self.session_id,
            "prompt": prompt,
            "engine": "qwen",
            "timestamp": datetime.now().isoformat()
        }

        websocket_manager.store_llm_event(self.session_id, message)

        if self.session_id not in websocket_manager.llm_stream_connections:
            logger.warning(f"[DEBUG] No WebSocket connections for session {self.session_id}")
            return

        try:
            connections = websocket_manager.llm_stream_connections[self.session_id]
            logger.info(f"[DEBUG] Broadcasting prompt start to {len(connections)} connections for session {self.session_id}")
            await websocket_manager._broadcast_to_connections(connections, message)
            logger.info(f"[DEBUG] Prompt start broadcast completed for session {self.session_id}")
        except Exception as e:
            logger.warning(f"Failed to broadcast prompt start for session {self.session_id}: {e}")

    async def _broadcast_token(self, token: str):
        """Broadcast token event"""
        if not self.session_id:
            return

        message = {
            "type": "llm_token",
            "session_id": self.session_id,
            "token": token,
            "timestamp": datetime.now().isoformat()
        }

        websocket_manager.store_llm_event(self.session_id, message)

        if self.session_id not in websocket_manager.llm_stream_connections:
            return

        try:
            connections = websocket_manager.llm_stream_connections[self.session_id]
            await websocket_manager._broadcast_to_connections(connections, message)
        except Exception as e:
            logger.warning(f"Failed to broadcast token for session {self.session_id}: {e}")

    async def _broadcast_prompt_complete(self, response: str):
        """Broadcast prompt completion event"""
        if not self.session_id:
            return

        message = {
            "type": "llm_prompt_complete",
            "session_id": self.session_id,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

        websocket_manager.store_llm_event(self.session_id, message)

        if self.session_id not in websocket_manager.llm_stream_connections:
            return

        try:
            connections = websocket_manager.llm_stream_connections[self.session_id]
            await websocket_manager._broadcast_to_connections(connections, message)
        except Exception as e:
            logger.warning(f"Failed to broadcast completion for session {self.session_id}: {e}")

    async def _broadcast_error(self, error: str):
        """Broadcast error event"""
        if not self.session_id:
            return

        message = {
            "type": "llm_error",
            "session_id": self.session_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

        websocket_manager.store_llm_event(self.session_id, message)

        if self.session_id not in websocket_manager.llm_stream_connections:
            return

        try:
            connections = websocket_manager.llm_stream_connections[self.session_id]
            await websocket_manager._broadcast_to_connections(connections, message)
        except Exception as e:
            logger.warning(f"Failed to broadcast error for session {self.session_id}: {e}")

    def with_session(self, session_id: str) -> 'StreamingLLMClient':
        """Create a new instance with a specific session ID"""
        return StreamingLLMClient(self.llm_client, session_id)
