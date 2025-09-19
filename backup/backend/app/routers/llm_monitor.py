"""
Real-time LLM Communication Monitor Router
WebSocket endpoint for streaming LLM request/response data
"""

import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse

from ..core.llm_client import llm_client

logger = logging.getLogger(__name__)

router = APIRouter()

# Active WebSocket connections for LLM monitoring
active_connections: Set[WebSocket] = set()

class LLMMonitor:
    """Manages real-time LLM communication monitoring"""

    def __init__(self):
        self.is_monitoring = False
        self.setup_llm_listener()

    def setup_llm_listener(self):
        """Set up the LLM client listener"""
        llm_client.add_message_listener(self.broadcast_llm_event)

    async def broadcast_llm_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast LLM events to all connected WebSocket clients"""
        if not active_connections:
            return

        message = {
            "type": "llm_event",
            "event_type": event_type,
            "data": data,
            "broadcast_timestamp": datetime.now().isoformat()
        }

        # Create list copy to avoid modification during iteration
        connections_copy = list(active_connections)

        for websocket in connections_copy:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                # Remove dead connection
                active_connections.discard(websocket)

# Global monitor instance
llm_monitor = LLMMonitor()

@router.websocket("/ws/llm-monitor")
async def websocket_llm_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time LLM communication monitoring"""
    await websocket.accept()
    active_connections.add(websocket)

    # Send welcome message
    welcome_message = {
        "type": "connection",
        "message": "Connected to LLM Monitor",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(active_connections)
    }

    try:
        await websocket.send_text(json.dumps(welcome_message))

        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for messages from client (for potential commands)
                data = await websocket.receive_text()
                client_message = json.loads(data)

                # Handle client commands
                if client_message.get("command") == "ping":
                    pong_message = {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(pong_message))

                elif client_message.get("command") == "status":
                    status_message = {
                        "type": "status",
                        "active_connections": len(active_connections),
                        "monitoring": llm_monitor.is_monitoring,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(status_message))

            except asyncio.TimeoutError:
                # Send periodic heartbeat
                heartbeat = {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(heartbeat))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected from LLM monitor")
    except Exception as e:
        logger.error(f"WebSocket error in LLM monitor: {e}")
    finally:
        active_connections.discard(websocket)
        logger.info(f"WebSocket removed. Active connections: {len(active_connections)}")

@router.get("/llm-monitor/status")
async def get_monitor_status():
    """Get current status of LLM monitoring"""
    return JSONResponse({
        "active_connections": len(active_connections),
        "monitoring": llm_monitor.is_monitoring,
        "timestamp": datetime.now().isoformat()
    })

@router.get("/llm-monitor/test")
async def test_llm_monitor():
    """Test the LLM monitoring by sending a test message"""
    try:
        # Send a test request to the LLM
        test_response = await llm_client.generate_response(
            prompt="Hello! This is a test message for monitoring. Please respond with 'Test successful'.",
            system_prompt="You are a test assistant. Respond briefly and confirm the test.",
            temperature=0.1,
            max_tokens=50
        )

        return JSONResponse({
            "message": "Test message sent to LLM",
            "llm_response": test_response,
            "active_connections": len(active_connections),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in LLM monitor test: {e}")
        return JSONResponse({
            "error": f"Test failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, status_code=500)