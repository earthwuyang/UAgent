"""Adapter for connecting to an external OpenHands application via Socket.IO."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx
import socketio

from ..core.websocket_manager import websocket_manager


logger = logging.getLogger(__name__)


class OpenHandsAppError(Exception):
    """Raised when interactions with the OpenHands app fail."""


class OpenHandsAppSession:
    """Single OpenHands conversation bridged into a UAgent session."""

    def __init__(
        self,
        *,
        base_url: str,
        session_id: str,
        conversation_id: str,
        latest_event_id: int = -1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.latest_event_id = latest_event_id

        self._sio = socketio.AsyncClient(transports=["websocket"], reconnection=True)
        self._sio.on("oh_event", self._handle_event)
        self._sio.on("connect", self._handle_connect)
        self._sio.on("disconnect", self._handle_disconnect)

        self._connected = False
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._connected:
            return

        async with self._connect_lock:
            if self._connected:
                return

            query_params = {
                "conversation_id": self.conversation_id,
                "latest_event_id": str(self.latest_event_id),
            }
            logger.info(
                "Connecting to OpenHands app %s for session %s (conversation %s)",
                self.base_url,
                self.session_id,
                self.conversation_id,
            )
            await self._sio.connect(
                self.base_url,
                socketio_path="/socket.io",
                transports=["websocket"],
                query=query_params,
            )
            self._connected = True

    async def send_user_message(self, message: str) -> None:
        if not message:
            raise ValueError("Message must be a non-empty string")
        await self.connect()
        payload = {"type": "message", "source": "user", "message": message}
        await self._sio.emit("oh_user_action", payload)

    async def send_action(self, payload: Dict[str, Any]) -> None:
        await self.connect()
        await self._sio.emit("oh_user_action", payload)

    async def close(self) -> None:
        if not self._connected:
            return
        try:
            await self._sio.disconnect()
        finally:
            self._connected = False

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        event_id = event.get("event_id")
        if isinstance(event_id, int):
            self.latest_event_id = event_id

        await websocket_manager.broadcast_openhands_event(self.session_id, event)

        message_text = event.get("message")
        if message_text:
            source = event.get("source", "agent")
            await websocket_manager.broadcast_openhands_output(
                self.session_id,
                message_text,
                output_type=str(source),
            )

        action_payload = event.get("result")
        if isinstance(action_payload, dict):
            for key in ("stdout", "stderr"):
                text = action_payload.get(key)
                if isinstance(text, str) and text.strip():
                    await websocket_manager.broadcast_openhands_output(
                        self.session_id,
                        text,
                        output_type=key,
                    )

    async def _handle_connect(self) -> None:
        logger.info(
            "Connected to OpenHands app for session %s (conversation %s)",
            self.session_id,
            self.conversation_id,
        )

    async def _handle_disconnect(self) -> None:
        logger.info(
            "Disconnected from OpenHands app for session %s (conversation %s)",
            self.session_id,
            self.conversation_id,
        )
        self._connected = False


class OpenHandsAppClient:
    """Client manager for the external OpenHands application."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=request_timeout)
        self._sessions: Dict[str, OpenHandsAppSession] = {}
        self._lock = asyncio.Lock()

    async def ensure_app_ready(self) -> None:
        """Check that the OpenHands app is reachable."""
        response = await self._http.get(f"{self.base_url}/")
        response.raise_for_status()

    async def create_conversation(self, title: Optional[str] = None) -> str:
        payload = {"title": title or "UAgent Session"}
        url = f"{self.base_url}/api/conversations"
        response = await self._http.post(url, json=payload)
        if response.status_code // 100 != 2:
            raise OpenHandsAppError(
                f"Failed to create conversation (status {response.status_code}): {response.text}"
            )

        data = response.json()
        conversation_id = (
            data.get("conversation_id")
            or data.get("id")
            or data.get("conversation", {}).get("id")
        )
        if not conversation_id:
            raise OpenHandsAppError("OpenHands response did not include a conversation id")
        return str(conversation_id)

    async def start_session(
        self,
        *,
        session_id: str,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> OpenHandsAppSession:
        async with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

            conv_id = conversation_id or await self.create_conversation(title=title)
            session = OpenHandsAppSession(
                base_url=self.base_url,
                session_id=session_id,
                conversation_id=conv_id,
            )
            await session.connect()
            self._sessions[session_id] = session
            return session

    async def send_message(self, session_id: str, message: str) -> None:
        session = await self._get_active_session(session_id)
        await session.send_user_message(message)

    async def send_action(self, session_id: str, payload: Dict[str, Any]) -> None:
        session = await self._get_active_session(session_id)
        await session.send_action(payload)

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            await session.close()

    async def shutdown(self) -> None:
        sessions = list(self._sessions.keys())
        for session_id in sessions:
            await self.close_session(session_id)
        await self._http.aclose()

    async def _get_active_session(self, session_id: str) -> OpenHandsAppSession:
        session = self._sessions.get(session_id)
        if not session:
            raise OpenHandsAppError(f"OpenHands session {session_id} is not active")
        return session

    def list_sessions(self) -> Dict[str, OpenHandsAppSession]:
        """Return a snapshot of active sessions."""
        return dict(self._sessions)
