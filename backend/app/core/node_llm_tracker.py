"""
Node-specific LLM Communication Tracker
Tracks and stores LLM communications for each research node
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class NodeLLMTracker:
    """Tracks LLM communications for each research node"""

    def __init__(self):
        # Storage: node_id -> list of messages
        self._node_messages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def add_message(self, node_id: str, event_type: str, data: Dict[str, Any]):
        """Add a new LLM message for a specific node"""
        if not node_id:
            return

        async with self._lock:
            message = {
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }

            self._node_messages[node_id].append(message)

            # Keep only the last 100 messages per node to prevent memory issues
            if len(self._node_messages[node_id]) > 100:
                self._node_messages[node_id] = self._node_messages[node_id][-100:]

            logger.debug(f"Added {event_type} message for node {node_id[:8]}")

    async def get_node_messages(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all LLM messages for a specific node"""
        async with self._lock:
            return list(self._node_messages.get(node_id, []))

    async def get_node_message_count(self, node_id: str) -> int:
        """Get the number of messages for a specific node"""
        async with self._lock:
            return len(self._node_messages.get(node_id, []))

    async def clear_node_messages(self, node_id: str):
        """Clear all messages for a specific node"""
        async with self._lock:
            if node_id in self._node_messages:
                del self._node_messages[node_id]
                logger.info(f"Cleared messages for node {node_id[:8]}")

    async def get_all_nodes_with_messages(self) -> List[str]:
        """Get list of all node IDs that have LLM messages"""
        async with self._lock:
            return list(self._node_messages.keys())

    async def get_node_summary(self, node_id: str) -> Dict[str, Any]:
        """Get a summary of LLM activity for a node"""
        async with self._lock:
            messages = self._node_messages.get(node_id, [])

            if not messages:
                return {
                    "node_id": node_id,
                    "total_messages": 0,
                    "requests": 0,
                    "responses": 0,
                    "errors": 0,
                    "first_message": None,
                    "last_message": None
                }

            requests = [m for m in messages if m["event_type"] == "request"]
            responses = [m for m in messages if m["event_type"] == "response"]
            errors = [m for m in messages if m["event_type"] == "response" and not m["data"].get("success", True)]

            return {
                "node_id": node_id,
                "total_messages": len(messages),
                "requests": len(requests),
                "responses": len(responses),
                "errors": len(errors),
                "first_message": messages[0]["timestamp"] if messages else None,
                "last_message": messages[-1]["timestamp"] if messages else None,
                "success_rate": (len(responses) - len(errors)) / max(len(responses), 1) * 100 if responses else 0
            }


# Global tracker instance
node_llm_tracker = NodeLLMTracker()


async def llm_message_listener(event_type: str, data: Dict[str, Any]):
    """Listener function to capture LLM messages and store them per node"""
    node_id = data.get("node_id")
    if node_id:
        await node_llm_tracker.add_message(node_id, event_type, data)