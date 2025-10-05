"""
WebSocket routes for real-time research progress updates
"""

import asyncio
import json
import logging
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research/ws", tags=["research-websocket"])


# Connection manager for WebSocket clients
class ConnectionManager:
    """Manages WebSocket connections for experiment updates"""

    def __init__(self):
        # Map of experiment_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map of session_id -> set of WebSocket connections
        self.session_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, experiment_id: str = None, session_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()

        if experiment_id:
            if experiment_id not in self.active_connections:
                self.active_connections[experiment_id] = set()
            self.active_connections[experiment_id].add(websocket)
            logger.info(f"WebSocket connected to experiment {experiment_id}")

        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = set()
            self.session_connections[session_id].add(websocket)
            logger.info(f"WebSocket connected to session {session_id}")

    def disconnect(self, websocket: WebSocket, experiment_id: str = None, session_id: str = None):
        """Remove a WebSocket connection"""
        if experiment_id and experiment_id in self.active_connections:
            self.active_connections[experiment_id].discard(websocket)
            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]
            logger.info(f"WebSocket disconnected from experiment {experiment_id}")

        if session_id and session_id in self.session_connections:
            self.session_connections[session_id].discard(websocket)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
            logger.info(f"WebSocket disconnected from session {session_id}")

    async def send_experiment_update(self, experiment_id: str, message: dict):
        """Send update to all clients watching an experiment"""
        if experiment_id not in self.active_connections:
            return

        # Add timestamp
        message["timestamp"] = datetime.utcnow().isoformat()

        # Send to all connected clients
        disconnected = set()
        for websocket in self.active_connections[experiment_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.active_connections[experiment_id].discard(websocket)

    async def send_session_update(self, session_id: str, message: dict):
        """Send update to all clients watching a session"""
        if session_id not in self.session_connections:
            return

        # Add timestamp
        message["timestamp"] = datetime.utcnow().isoformat()

        # Send to all connected clients
        disconnected = set()
        for websocket in self.session_connections[session_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.session_connections[session_id].discard(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        message["timestamp"] = datetime.utcnow().isoformat()

        all_websockets = set()
        for connections in self.active_connections.values():
            all_websockets.update(connections)
        for connections in self.session_connections.values():
            all_websockets.update(connections)

        disconnected = set()
        for websocket in all_websockets:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(websocket)

        # Clean up (will be removed by individual disconnect calls)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/experiment/{experiment_id}")
async def experiment_websocket(websocket: WebSocket, experiment_id: str):
    """
    WebSocket endpoint for real-time experiment updates.

    Clients can connect to receive live progress updates for a specific experiment.

    Message format:
    {
        "type": "progress" | "status" | "log" | "result" | "error",
        "experiment_id": "exp_123",
        "data": { ... },
        "timestamp": "2024-10-04T12:00:00Z"
    }
    """
    logger.info(f"[WebSocket] New connection request for experiment {experiment_id}")
    await manager.connect(websocket, experiment_id=experiment_id)
    logger.info(f"[WebSocket] Connected to experiment {experiment_id}")

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "experiment_id": experiment_id,
            "message": f"Connected to experiment {experiment_id}",
            "timestamp": datetime.utcnow().isoformat()
        })

        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for messages from client (e.g., ping/pong)
                data = await websocket.receive_text()

                # Handle ping
                if data == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in experiment WebSocket: {e}")
                break

    finally:
        manager.disconnect(websocket, experiment_id=experiment_id)


@router.websocket("/session/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time session updates.

    Clients can connect to receive updates for all experiments in a session.

    Message format:
    {
        "type": "experiment_started" | "experiment_completed" | "idea_generated" | ...,
        "session_id": "session_123",
        "data": { ... },
        "timestamp": "2024-10-04T12:00:00Z"
    }
    """
    logger.info(f"[WebSocket] New connection request for session {session_id}")
    await manager.connect(websocket, session_id=session_id)
    logger.info(f"[WebSocket] Connected to session {session_id}")

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": f"Connected to session {session_id}",
            "timestamp": datetime.utcnow().isoformat()
        })

        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_text()

                # Handle ping
                if data == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in session WebSocket: {e}")
                break

    finally:
        manager.disconnect(websocket, session_id=session_id)


# Export manager for use by research engines
__all__ = ["router", "manager"]
