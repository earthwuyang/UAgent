"""
WebSocket routes for real-time research progress updates

Phase 3 Enhancement:
- Bidirectional control (client can send control commands via WebSocket)
- Control command validation and routing to ControlBus
- Real-time acknowledgement of control commands
"""

import asyncio
import json
import logging
from typing import Any, Dict, Set, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime

# Import control components
try:
    from ...control.control_bus import ControlMessage
    from ...services.research_session_manager import (
        ResearchSessionManager,
        get_global_session_manager,
    )
    CONTROL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Control components not available: {e}")
    CONTROL_AVAILABLE = False
    ControlMessage = None
    ResearchSessionManager = Any  # type: ignore[assignment]
    get_global_session_manager = None

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

# Global session manager for control commands
_session_manager: Optional[ResearchSessionManager] = None


def get_session_manager() -> Optional[ResearchSessionManager]:
    """Get or create global session manager"""
    global _session_manager

    if not CONTROL_AVAILABLE:
        return None

    if _session_manager is None:
        try:
            _session_manager = get_global_session_manager()
            logger.info("✅ WebSocket using global ResearchSessionManager singleton")
        except Exception as e:
            logger.error(f"Failed to initialize session manager: {e}")
            return None

    return _session_manager


async def handle_control_command(
    websocket: WebSocket,
    experiment_id: str,
    command: dict
) -> dict:
    """
    Handle control command from WebSocket client.

    Args:
        websocket: WebSocket connection
        experiment_id: Experiment ID
        command: Control command dict

    Returns:
        Response dict

    Command format:
    {
        "type": "control",
        "action": "pause" | "resume" | "cancel" | "cancel_node" | "reprioritize" | "steer" | "add_node",
        "target": { ... },  # Optional target specification
        "payload": { ... }  # Optional payload
    }

    Response format:
    {
        "type": "control_ack",
        "action": "pause",
        "status": "success" | "error",
        "message": "...",
        "timestamp": "2024-10-04T12:00:00Z"
    }
    """
    session_mgr = get_session_manager()

    if not session_mgr or not CONTROL_AVAILABLE:
        return {
            "type": "control_ack",
            "status": "error",
            "message": "Control system not available"
        }

    try:
        # Extract control parameters
        action = command.get("action")
        target = command.get("target", {})
        payload = command.get("payload", {})

        if not action:
            return {
                "type": "control_ack",
                "status": "error",
                "message": "Missing 'action' field in control command"
            }

        # Validate action
        valid_actions = ["pause", "resume", "cancel", "cancel_node", "reprioritize", "steer", "add_node"]
        if action not in valid_actions:
            return {
                "type": "control_ack",
                "action": action,
                "status": "error",
                "message": f"Invalid action '{action}'. Valid actions: {', '.join(valid_actions)}"
            }

        # Create control message
        control_msg = ControlMessage(
            action=action,
            target=target,
            payload=payload,
            sender="websocket"
        )

        # Send control command
        await session_mgr.send_control(experiment_id, control_msg)

        logger.info(f"WebSocket control command sent: action={action}, experiment={experiment_id}")

        return {
            "type": "control_ack",
            "action": action,
            "status": "success",
            "message": f"Control command '{action}' sent successfully"
        }

    except Exception as e:
        logger.error(f"Error handling control command: {e}", exc_info=True)
        return {
            "type": "control_ack",
            "status": "error",
            "message": f"Error processing control command: {str(e)}"
        }


@router.websocket("/experiment/{experiment_id}")
async def experiment_websocket(websocket: WebSocket, experiment_id: str):
    """
    WebSocket endpoint for real-time experiment updates and control.

    Phase 3 Enhancement: Bidirectional communication
    - Server → Client: progress updates, status changes, logs
    - Client → Server: control commands (pause, resume, cancel, etc.)

    Server → Client message format:
    {
        "type": "progress" | "status" | "log" | "result" | "error" | "control_ack",
        "experiment_id": "exp_123",
        "data": { ... },
        "timestamp": "2024-10-04T12:00:00Z"
    }

    Client → Server message format:
    {
        "type": "control",
        "action": "pause" | "resume" | "cancel" | ...,
        "target": { ... },
        "payload": { ... }
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
            "control_enabled": CONTROL_AVAILABLE,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()

                # Try to parse as JSON
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    # Handle simple text messages (ping, etc.)
                    if data == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON format",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue

                # Handle control commands
                if isinstance(message, dict) and message.get("type") == "control":
                    logger.info(f"[WebSocket] Received control command for experiment {experiment_id}: {message.get('action')}")

                    # Process control command
                    response = await handle_control_command(websocket, experiment_id, message)

                    # Add timestamp and send acknowledgement
                    response["timestamp"] = datetime.utcnow().isoformat()
                    await websocket.send_json(response)

                # Handle ping
                elif isinstance(message, dict) and message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })

                # Unknown message type
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message.get('type') if isinstance(message, dict) else 'invalid'}",
                        "timestamp": datetime.utcnow().isoformat()
                    })

            except WebSocketDisconnect:
                logger.info(f"[WebSocket] Client disconnected from experiment {experiment_id}")
                break
            except Exception as e:
                logger.error(f"Error in experiment WebSocket: {e}", exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Server error: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except:
                    pass
                break

    finally:
        manager.disconnect(websocket, experiment_id=experiment_id)


@router.websocket("/session/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time session updates.

    Phase 3 Enhancement: Bidirectional communication
    - Server → Client: experiment lifecycle events, progress updates
    - Client → Server: control commands (routed to active experiment)

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
            "control_enabled": CONTROL_AVAILABLE,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_text()

                # Try to parse as JSON
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    # Handle simple text messages
                    if data == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON format",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue

                # Handle control commands (requires experiment_id in message)
                if isinstance(message, dict) and message.get("type") == "control":
                    experiment_id = message.get("experiment_id")

                    if not experiment_id:
                        await websocket.send_json({
                            "type": "control_ack",
                            "status": "error",
                            "message": "Control command requires 'experiment_id' field",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue

                    logger.info(f"[WebSocket] Received control command for session {session_id}, experiment {experiment_id}: {message.get('action')}")

                    # Process control command
                    response = await handle_control_command(websocket, experiment_id, message)
                    response["timestamp"] = datetime.utcnow().isoformat()
                    await websocket.send_json(response)

                # Handle ping
                elif isinstance(message, dict) and message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })

                # Unknown message type
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message.get('type') if isinstance(message, dict) else 'invalid'}",
                        "timestamp": datetime.utcnow().isoformat()
                    })

            except WebSocketDisconnect:
                logger.info(f"[WebSocket] Client disconnected from session {session_id}")
                break
            except Exception as e:
                logger.error(f"Error in session WebSocket: {e}", exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Server error: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except:
                    pass
                break

    finally:
        manager.disconnect(websocket, session_id=session_id)


# Export manager for use by research engines
__all__ = ["router", "manager"]
