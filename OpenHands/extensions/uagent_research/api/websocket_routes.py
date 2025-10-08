"""WebSocket routes for real-time research tree updates."""

import asyncio
import json
import logging
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

ws_router = APIRouter(prefix="/api/research", tags=["research-websocket"])

# Connection manager for WebSocket clients
class ConnectionManager:
    """Manages WebSocket connections for research tree updates."""

    def __init__(self):
        # experiment_id -> set of active WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, experiment_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()

        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = set()

        self.active_connections[experiment_id].add(websocket)
        logger.info(
            f"✅ WebSocket client connected for experiment {experiment_id}. "
            f"Total connections for this experiment: {len(self.active_connections[experiment_id])}"
        )

        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "experiment_id": experiment_id,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Send initial tree snapshot if available
        try:
            from .research_routes import _active_trees
            if experiment_id in _active_trees:
                tree_snapshot = _active_trees[experiment_id]
                await websocket.send_json({
                    "type": "tree_snapshot",
                    **tree_snapshot
                })
                logger.info(f"✅ Sent initial tree snapshot to new client for {experiment_id}")
            else:
                logger.debug(f"No active tree found for {experiment_id} to send to new client")
        except Exception as e:
            logger.warning(f"Could not send initial tree snapshot: {e}")

    def disconnect(self, websocket: WebSocket, experiment_id: str):
        """Remove a WebSocket connection."""
        if experiment_id in self.active_connections:
            self.active_connections[experiment_id].discard(websocket)

            # Clean up empty sets
            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]

            logger.info(f"WebSocket disconnected for experiment {experiment_id}")

    async def broadcast(self, message: dict, experiment_id: str):
        """
        Broadcast a message to all connected clients for an experiment.

        Args:
            message: The message dict to send (will be JSON-encoded)
            experiment_id: The experiment to broadcast to
        """
        targets: Dict[str, Set[WebSocket]]
        if experiment_id == "*":
            if not self.active_connections:
                logger.debug("No active connections for wildcard broadcast")
                return
            targets = dict(self.active_connections)
        else:
            if experiment_id not in self.active_connections:
                logger.debug(f"No active connections for experiment {experiment_id}")
                return
            targets = {experiment_id: self.active_connections[experiment_id]}

        for target_id, websockets in list(targets.items()):
            connections = list(websockets)
            disconnected = []

            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    disconnected.append(connection)

            for connection in disconnected:
                self.disconnect(connection, target_id)


# Global connection manager instance
manager = ConnectionManager()


@ws_router.websocket("/ws/experiment/{experiment_id}")
async def websocket_experiment_endpoint(
    websocket: WebSocket,
    experiment_id: str
):
    """
    WebSocket endpoint for real-time research tree updates.

    Clients connect to this endpoint to receive live updates as the research
    tree evolves. Messages follow the ROMA-compatible format:

    {
        "type": "tree_snapshot" | "node_added" | "node_updated" | "edge_added" | "stats_updated",
        "version": int,
        "timestamp": str,
        "experiment_id": str,
        "data": { ... }
    }

    Args:
        websocket: The WebSocket connection
        experiment_id: The experiment ID to subscribe to
    """
    await manager.connect(websocket, experiment_id)

    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Receive messages from client (for potential control messages)
            data = await websocket.receive_text()

            # Parse client message
            try:
                message = json.loads(data)
                logger.debug(f"Received from client: {message}")

                # Handle client messages (e.g., control requests)
                # For now, just acknowledge
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": asyncio.get_event_loop().time()
                    })

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client: {data}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, experiment_id)
        logger.info(f"Client disconnected from experiment {experiment_id}")
    except Exception as e:
        logger.error(f"WebSocket error for experiment {experiment_id}: {e}")
        manager.disconnect(websocket, experiment_id)


# Helper function for orchestrator to send updates
async def broadcast_tree_update(experiment_id: str, message: dict):
    """Broadcast a tree update to all connected clients."""
    if not experiment_id:
        logger.warning("❌ Cannot broadcast: experiment_id is empty")
        return
    
    if experiment_id not in manager.active_connections:
        logger.debug(f"ℹ️ No WebSocket clients connected for {experiment_id}")
        return
    
    client_count = len(manager.active_connections[experiment_id])
    logger.info(f"📡 Broadcasting tree update to {client_count} client(s) for {experiment_id}")
    
    try:
        await manager.broadcast(message, experiment_id)
        logger.debug(f"✅ Broadcast completed for {experiment_id}")
    except Exception as e:
        logger.error(f"❌ Broadcast failed for {experiment_id}: {e}", exc_info=True)
