"""
WebSocket Router for Real-Time Research Progress Streaming
Implements all WebSocket endpoints for live research visualization
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocketState

from app.core.websocket_manager import websocket_manager, progress_tracker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/research/{session_id}")
async def websocket_research_progress(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time research progress updates
    Streams progress events, journal entries, and status updates for a specific research session
    """
    try:
        await websocket_manager.connect_research(websocket, session_id)
        logger.info(f"Research progress WebSocket connected for session: {session_id}")

        # Keep connection alive and handle client messages
        while True:
            try:
                # Listen for client messages (like pause/resume commands)
                message = await websocket.receive_text()
                logger.debug(f"Received message from client {session_id}: {message}")

                # Handle client commands if needed
                await handle_client_command(session_id, message)

            except WebSocketDisconnect:
                logger.info(f"Research progress WebSocket disconnected for session: {session_id}")
                break
            except Exception as e:
                logger.error(f"Error in research WebSocket for session {session_id}: {e}")
                break

    except Exception as e:
        logger.error(f"Failed to establish research WebSocket for session {session_id}: {e}")
    finally:
        await websocket_manager.disconnect(websocket)


@router.websocket("/engines/status")
async def websocket_engine_status(websocket: WebSocket):
    """
    WebSocket endpoint for real-time engine status updates
    Streams status updates for all research engines (Deep, Code, Scientific)
    """
    try:
        await websocket_manager.connect_engine_status(websocket)
        logger.info("Engine status WebSocket connected")

        # Keep connection alive
        while True:
            try:
                # Listen for ping messages to keep connection alive
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text('{"type": "ping"}')
            except WebSocketDisconnect:
                logger.info("Engine status WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error in engine status WebSocket: {e}")
                break

    except Exception as e:
        logger.error(f"Failed to establish engine status WebSocket: {e}")
    finally:
        await websocket_manager.disconnect(websocket)


@router.websocket("/openhands/{session_id}")
async def websocket_openhands_output(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time OpenHands execution output
    Streams live code execution output, debugging info, and workspace changes
    """
    try:
        await websocket_manager.connect_openhands(websocket, session_id)
        logger.info(f"OpenHands output WebSocket connected for session: {session_id}")

        # Keep connection alive and handle client messages
        while True:
            try:
                message = await websocket.receive_text()
                logger.debug(f"Received OpenHands message from client {session_id}: {message}")

                # Handle client commands for OpenHands control
                await handle_openhands_command(session_id, message)

            except WebSocketDisconnect:
                logger.info(f"OpenHands output WebSocket disconnected for session: {session_id}")
                break
            except Exception as e:
                logger.error(f"Error in OpenHands WebSocket for session {session_id}: {e}")
                break

    except Exception as e:
        logger.error(f"Failed to establish OpenHands WebSocket for session {session_id}: {e}")
    finally:
        await websocket_manager.disconnect(websocket)


@router.websocket("/research/{session_id}/journal")
async def websocket_research_journal(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time research journal updates
    Streams research log entries with filtering and search capabilities
    """
    try:
        await websocket_manager.connect_research(websocket, session_id)
        logger.info(f"Research journal WebSocket connected for session: {session_id}")

        # Send existing journal entries
        if session_id in websocket_manager.journal_entries:
            for entry in websocket_manager.journal_entries[session_id]:
                await websocket.send_text(f'{{"type": "journal_entry", "entry": {entry}}}')

        # Keep connection alive and handle filtering commands
        while True:
            try:
                message = await websocket.receive_text()
                logger.debug(f"Received journal message from client {session_id}: {message}")

                # Handle journal filtering commands
                await handle_journal_command(session_id, message, websocket)

            except WebSocketDisconnect:
                logger.info(f"Research journal WebSocket disconnected for session: {session_id}")
                break
            except Exception as e:
                logger.error(f"Error in research journal WebSocket for session {session_id}: {e}")
                break

    except Exception as e:
        logger.error(f"Failed to establish research journal WebSocket for session {session_id}: {e}")
    finally:
        await websocket_manager.disconnect(websocket)


@router.websocket("/metrics/live")
async def websocket_live_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for live performance and progress metrics
    Streams system performance, research metrics, and usage statistics
    """
    try:
        await websocket_manager.connect_metrics(websocket)
        logger.info("Live metrics WebSocket connected")

        # Send initial metrics
        initial_metrics = await get_current_metrics()
        await websocket.send_text(f'{{"type": "initial_metrics", "metrics": {initial_metrics}}}')

        # Keep connection alive and send periodic metrics
        while True:
            try:
                # Wait for client message or timeout for periodic updates
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                    logger.debug(f"Received metrics message from client: {message}")
                except asyncio.TimeoutError:
                    # Send periodic metrics update
                    current_metrics = await get_current_metrics()
                    await websocket_manager.broadcast_metrics_update(current_metrics)

            except WebSocketDisconnect:
                logger.info("Live metrics WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error in live metrics WebSocket: {e}")
                break

    except Exception as e:
        logger.error(f"Failed to establish live metrics WebSocket: {e}")
    finally:
        await websocket_manager.disconnect(websocket)


async def handle_client_command(session_id: str, message: str):
    """Handle client commands for research session control"""
    try:
        import json
        command = json.loads(message)

        if command.get("action") == "pause":
            logger.info(f"Pause command received for session {session_id}")
            # TODO: Implement pause functionality

        elif command.get("action") == "resume":
            logger.info(f"Resume command received for session {session_id}")
            # TODO: Implement resume functionality

        elif command.get("action") == "cancel":
            logger.info(f"Cancel command received for session {session_id}")
            # TODO: Implement cancel functionality

    except Exception as e:
        logger.error(f"Failed to handle client command: {e}")


async def handle_openhands_command(session_id: str, message: str):
    """Handle client commands for OpenHands control"""
    try:
        import json
        command = json.loads(message)

        if command.get("action") == "interrupt":
            logger.info(f"Interrupt command received for OpenHands session {session_id}")
            # TODO: Implement OpenHands interrupt functionality

        elif command.get("action") == "restart":
            logger.info(f"Restart command received for OpenHands session {session_id}")
            # TODO: Implement OpenHands restart functionality

    except Exception as e:
        logger.error(f"Failed to handle OpenHands command: {e}")


async def handle_journal_command(session_id: str, message: str, websocket: WebSocket):
    """Handle journal filtering and search commands"""
    try:
        import json
        command = json.loads(message)

        if command.get("action") == "filter":
            # Filter journal entries based on criteria
            filter_criteria = command.get("criteria", {})
            filtered_entries = await filter_journal_entries(session_id, filter_criteria)

            await websocket.send_text(json.dumps({
                "type": "filtered_journal",
                "entries": filtered_entries
            }))

        elif command.get("action") == "search":
            # Search journal entries
            search_term = command.get("term", "")
            search_results = await search_journal_entries(session_id, search_term)

            await websocket.send_text(json.dumps({
                "type": "search_results",
                "results": search_results
            }))

    except Exception as e:
        logger.error(f"Failed to handle journal command: {e}")


async def filter_journal_entries(session_id: str, criteria: Dict[str, Any]) -> list:
    """Filter journal entries based on criteria"""
    if session_id not in websocket_manager.journal_entries:
        return []

    entries = websocket_manager.journal_entries[session_id]
    filtered = []

    for entry in entries:
        # Filter by engine
        if "engine" in criteria and entry.engine != criteria["engine"]:
            continue

        # Filter by level
        if "level" in criteria and entry.level != criteria["level"]:
            continue

        # Filter by phase
        if "phase" in criteria and entry.phase != criteria["phase"]:
            continue

        filtered.append(entry)

    return [entry.__dict__ for entry in filtered]


async def search_journal_entries(session_id: str, search_term: str) -> list:
    """Search journal entries for a term"""
    if session_id not in websocket_manager.journal_entries:
        return []

    entries = websocket_manager.journal_entries[session_id]
    results = []

    search_term_lower = search_term.lower()

    for entry in entries:
        if (search_term_lower in entry.message.lower() or
            search_term_lower in entry.engine.lower() or
            search_term_lower in entry.phase.lower()):
            results.append(entry.__dict__)

    return results


async def get_current_metrics() -> Dict[str, Any]:
    """Get current system and research metrics"""
    return {
        "timestamp": "2024-01-01T00:00:00Z",
        "system": {
            "active_sessions": len(websocket_manager.research_connections),
            "total_connections": sum(len(conns) for conns in websocket_manager.research_connections.values()),
            "engine_statuses": {name: status.status for name, status in websocket_manager.engine_statuses.items()}
        },
        "research": {
            "total_sessions": len(websocket_manager.event_history),
            "completed_sessions": 0,  # TODO: Calculate from event history
            "average_session_duration": "00:15:30",  # TODO: Calculate from metrics
        },
        "performance": {
            "cpu_usage": 25.5,  # TODO: Get from system monitoring
            "memory_usage": 512.0,  # TODO: Get from system monitoring
            "response_time_avg": 0.85  # TODO: Calculate from metrics
        }
    }