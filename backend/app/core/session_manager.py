"""Research session manager for tracking running and completed sessions"""

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SessionRecord:
    """Internal representation of a research session"""
    session_id: str
    request: str
    classification: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, error
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    task_name: Optional[str] = None


class ResearchSessionManager:
    """Manages lifecycle and metadata for research sessions"""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, session_id: str, request: str, classification: Dict[str, Any]) -> SessionRecord:
        """Create a new session entry"""
        async with self._lock:
            record = SessionRecord(
                session_id=session_id,
                request=request,
                classification=classification,
                status="pending"
            )
            self._sessions[session_id] = record
            return record

    async def attach_task(self, session_id: str, task: asyncio.Task) -> None:
        """Associate an asyncio task with a session"""
        async with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return
            record.task_name = task.get_name()
            record.updated_at = datetime.utcnow()
            self._tasks[session_id] = task

    async def set_status(self, session_id: str, status: str) -> None:
        """Update session status"""
        async with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return
            record.status = status
            record.updated_at = datetime.utcnow()

    async def set_result(self, session_id: str, result: Dict[str, Any]) -> None:
        """Store final result for a session"""
        async with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return
            record.result = result
            record.status = "completed"
            record.updated_at = datetime.utcnow()
            record.error = None

    async def set_error(self, session_id: str, error: str) -> None:
        """Mark session as errored"""
        async with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return
            record.error = error
            record.status = "error"
            record.updated_at = datetime.utcnow()

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with metadata"""
        async with self._lock:
            return [self._serialize(record) for record in self._sessions.values()]

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single session record"""
        async with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return None
            return self._serialize(record)

    async def get_session_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return only the stored result for a session"""
        async with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return None
            return record.result

    async def shutdown(self) -> None:
        """Cancel all running tasks and clear state"""
        async with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _serialize(self, record: SessionRecord) -> Dict[str, Any]:
        """Serialize a session record for API consumption"""
        return {
            "session_id": record.session_id,
            "request": record.request,
            "classification": record.classification,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "result": record.result,
            "error": record.error,
            "task_name": record.task_name,
        }
