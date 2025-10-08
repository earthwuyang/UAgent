"""WebSocket routes for real-time research tree updates."""

import asyncio
import json
import logging
import traceback
from datetime import datetime
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
        # Track connection metadata for diagnostics
        self.connection_metadata: Dict[str, dict] = {}
        logger.info("🔧 ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, experiment_id: str):
        """Accept and register a new WebSocket connection."""
        try:
            logger.info(f"🔌 Attempting to accept WebSocket connection for experiment {experiment_id}")
            await websocket.accept()
            logger.debug(f"✅ WebSocket accepted for {experiment_id}")

            if experiment_id not in self.active_connections:
                self.active_connections[experiment_id] = set()
                logger.debug(f"📝 Created new connection set for {experiment_id}")

            self.active_connections[experiment_id].add(websocket)
            
            # Store connection metadata
            conn_id = id(websocket)
            self.connection_metadata[str(conn_id)] = {
                "experiment_id": experiment_id,
                "connected_at": datetime.now().isoformat(),
                "messages_sent": 0,
                "errors": []
            }
            
            logger.info(
                f"✅ WebSocket client connected for experiment {experiment_id}. "
                f"Total connections for this experiment: {len(self.active_connections[experiment_id])}, "
                f"Connection ID: {conn_id}"
            )

            # Send initial connection confirmation
            try:
                await websocket.send_json({
                    "type": "connected",
                    "experiment_id": experiment_id,
                    "timestamp": datetime.now().isoformat(),
                    "connection_id": str(conn_id)
                })
                logger.debug(f"✅ Sent connection confirmation to client {conn_id}")
            except Exception as e:
                logger.error(f"❌ Failed to send connection confirmation: {e}", exc_info=True)
            
            # Send initial tree snapshot if available
            try:
                logger.debug(f"🔍 Checking for active tree snapshot for {experiment_id}")
                from .research_routes import _active_trees
                if experiment_id in _active_trees:
                    tree_snapshot = _active_trees[experiment_id]
                    await websocket.send_json({
                        "type": "tree_snapshot",
                        **tree_snapshot
                    })
                    logger.info(f"✅ Sent initial tree snapshot to new client {conn_id} for {experiment_id}")
                    self.connection_metadata[str(conn_id)]["messages_sent"] += 1
                else:
                    logger.debug(f"ℹ️ No active tree found for {experiment_id} to send to new client {conn_id}")
            except Exception as e:
                error_msg = f"Could not send initial tree snapshot: {e}"
                logger.warning(error_msg, exc_info=True)
                self.connection_metadata[str(conn_id)]["errors"].append({
                    "timestamp": datetime.now().isoformat(),
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                })
                
        except Exception as e:
            logger.error(f"❌ Failed to establish WebSocket connection for {experiment_id}: {e}", exc_info=True)
            raise

    def disconnect(self, websocket: WebSocket, experiment_id: str):
        """Remove a WebSocket connection."""
        try:
            conn_id = id(websocket)
            logger.info(f"🔌 Disconnecting WebSocket {conn_id} for experiment {experiment_id}")
            
            if experiment_id in self.active_connections:
                self.active_connections[experiment_id].discard(websocket)

                # Clean up empty sets
                if not self.active_connections[experiment_id]:
                    del self.active_connections[experiment_id]
                    logger.debug(f"🧹 Removed empty connection set for {experiment_id}")

                # Log connection metadata before cleanup
                if str(conn_id) in self.connection_metadata:
                    metadata = self.connection_metadata[str(conn_id)]
                    logger.info(
                        f"📊 Connection {conn_id} stats - Messages sent: {metadata['messages_sent']}, "
                        f"Errors: {len(metadata['errors'])}, "
                        f"Duration: {metadata['connected_at']}"
                    )
                    del self.connection_metadata[str(conn_id)]

                logger.info(f"✅ WebSocket {conn_id} disconnected for experiment {experiment_id}")
            else:
                logger.warning(f"⚠️ Attempted to disconnect unknown experiment {experiment_id}")
                
        except Exception as e:
            logger.error(f"❌ Error during disconnect: {e}", exc_info=True)

    async def broadcast(self, message: dict, experiment_id: str):
        """
        Broadcast a message to all connected clients for an experiment.

        Args:
            message: The message dict to send (will be JSON-encoded)
            experiment_id: The experiment to broadcast to
        """
        try:
            logger.debug(f"📡 Starting broadcast for experiment {experiment_id}, message type: {message.get('type', 'unknown')}")
            
            targets: Dict[str, Set[WebSocket]]
            if experiment_id == "*":
                if not self.active_connections:
                    logger.debug("ℹ️ No active connections for wildcard broadcast")
                    return
                targets = dict(self.active_connections)
                logger.debug(f"📡 Wildcard broadcast to {len(targets)} experiments")
            else:
                if experiment_id not in self.active_connections:
                    logger.debug(f"ℹ️ No active connections for experiment {experiment_id}")
                    return
                targets = {experiment_id: self.active_connections[experiment_id]}

            total_sent = 0
            total_failed = 0
            
            for target_id, websockets in list(targets.items()):
                connections = list(websockets)
                disconnected = []

                for connection in connections:
                    try:
                        conn_id = str(id(connection))
                        await connection.send_json(message)
                        total_sent += 1
                        
                        # Update metadata
                        if conn_id in self.connection_metadata:
                            self.connection_metadata[conn_id]["messages_sent"] += 1
                            
                        logger.debug(f"✅ Sent message to connection {conn_id}")
                    except Exception as e:
                        total_failed += 1
                        conn_id = str(id(connection))
                        error_msg = f"Error sending to client {conn_id}: {e}"
                        logger.error(error_msg, exc_info=True)
                        
                        # Track error in metadata
                        if conn_id in self.connection_metadata:
                            self.connection_metadata[conn_id]["errors"].append({
                                "timestamp": datetime.now().isoformat(),
                                "error": error_msg,
                                "traceback": traceback.format_exc()
                            })
                        
                        disconnected.append(connection)

                for connection in disconnected:
                    self.disconnect(connection, target_id)

            logger.info(f"📊 Broadcast complete for {experiment_id}: {total_sent} sent, {total_failed} failed")
            
        except Exception as e:
            logger.error(f"❌ Critical error during broadcast: {e}", exc_info=True)

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about all connections."""
        return {
            "total_experiments": len(self.active_connections),
            "total_connections": sum(len(conns) for conns in self.active_connections.values()),
            "experiments": {
                exp_id: len(conns) 
                for exp_id, conns in self.active_connections.items()
            },
            "connection_metadata": dict(self.connection_metadata)
        }


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
    logger.info(f"🔌 New WebSocket connection request for experiment {experiment_id}")
    
    try:
        await manager.connect(websocket, experiment_id)
        logger.info(f"✅ WebSocket connection established for {experiment_id}")

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive messages from client (for potential control messages)
                data = await websocket.receive_text()
                logger.debug(f"📨 Received from client for {experiment_id}: {data[:100]}...")

                # Parse client message
                try:
                    message = json.loads(data)
                    logger.debug(f"📨 Parsed message from client: {message}")

                    # Handle client messages (e.g., control requests)
                    if message.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        })
                        logger.debug(f"🏓 Sent pong response for {experiment_id}")
                    elif message.get("type") == "diagnostics":
                        diagnostics = manager.get_diagnostics()
                        await websocket.send_json({
                            "type": "diagnostics_response",
                            "data": diagnostics,
                            "timestamp": datetime.now().isoformat()
                        })
                        logger.info(f"📊 Sent diagnostics to client for {experiment_id}")
                    else:
                        logger.debug(f"ℹ️ Unhandled message type from client: {message.get('type')}")

                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Invalid JSON from client for {experiment_id}: {data[:100]}... Error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON",
                        "timestamp": datetime.now().isoformat()
                    })

            except WebSocketDisconnect:
                logger.info(f"🔌 Client initiated disconnect for experiment {experiment_id}")
                raise  # Re-raise to be caught by outer except

    except WebSocketDisconnect:
        manager.disconnect(websocket, experiment_id)
        logger.info(f"✅ Client cleanly disconnected from experiment {experiment_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error for experiment {experiment_id}: {e}", exc_info=True)
        manager.disconnect(websocket, experiment_id)


# Helper function for orchestrator to send updates
async def broadcast_tree_update(experiment_id: str, message: dict):
    """Broadcast a tree update to all connected clients."""
    try:
        if not experiment_id:
            logger.warning("❌ Cannot broadcast: experiment_id is empty")
            return
        
        if experiment_id not in manager.active_connections:
            logger.debug(f"ℹ️ No WebSocket clients connected for {experiment_id}")
            return
        
        client_count = len(manager.active_connections[experiment_id])
        logger.info(f"📡 Broadcasting tree update to {client_count} client(s) for {experiment_id}, message type: {message.get('type', 'unknown')}")
        
        await manager.broadcast(message, experiment_id)
        logger.debug(f"✅ Broadcast completed for {experiment_id}")
        
    except Exception as e:
        logger.error(f"❌ Broadcast failed for {experiment_id}: {e}", exc_info=True)


@ws_router.get("/ws/diagnostics")
async def get_websocket_diagnostics():
    """Get diagnostic information about WebSocket connections."""
    try:
        logger.info("📊 WebSocket diagnostics requested")
        diagnostics = manager.get_diagnostics()
        logger.debug(f"📊 Diagnostics: {diagnostics}")
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "websocket_diagnostics": diagnostics
        }
    except Exception as e:
        logger.error(f"❌ Failed to get WebSocket diagnostics: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
