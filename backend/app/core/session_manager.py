"""Research session manager for tracking running and completed sessions"""

import asyncio
import contextlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
import json


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
    completed_at: Optional[datetime] = None


class ResearchSessionManager:
    """Manages lifecycle and metadata for research sessions"""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._index_path = (Path(__file__).resolve().parents[2] / "data" / "session_index.json")
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        # Load existing sessions to enable resume after restart
        self._load_index()

    def _load_index(self) -> None:
        try:
            if self._index_path.exists():
                raw = self._index_path.read_text(encoding="utf-8")
                data = json.loads(raw) if raw.strip() else {}
                if isinstance(data, dict):
                    self._sessions = {
                        k: SessionRecord(
                            session_id=v.get("session_id", k),
                            request=v.get("request", ""),
                            classification=v.get("classification", {}),
                            status=v.get("status", "pending"),
                            created_at=datetime.fromisoformat(v.get("created_at")) if v.get("created_at") else datetime.utcnow(),
                            updated_at=datetime.fromisoformat(v.get("updated_at")) if v.get("updated_at") else datetime.utcnow(),
                            result=v.get("result"),
                            error=v.get("error"),
                            task_name=v.get("task_name"),
                            completed_at=datetime.fromisoformat(v.get("completed_at")) if v.get("completed_at") else None,
                        )
                        for k, v in data.items()
                        if isinstance(v, dict)
                    }
        except Exception:
            # Ignore corrupt index; start fresh
            self._sessions = self._sessions or {}

    def _save_index(self) -> None:
        try:
            serializable: Dict[str, Any] = {}
            for sid, rec in self._sessions.items():
                d = asdict(rec)
                d["created_at"] = rec.created_at.isoformat()
                d["updated_at"] = rec.updated_at.isoformat()
                d["completed_at"] = rec.completed_at.isoformat() if rec.completed_at else None
                serializable[sid] = d
            self._index_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        except Exception:
            # Best effort persistence
            pass

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
            self._save_index()
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
            self._save_index()

    async def set_status(self, session_id: str, status: str) -> None:
        """Update session status"""
        async with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return
            record.status = status
            record.updated_at = datetime.utcnow()
            if status in {"completed", "error"}:
                record.completed_at = record.completed_at or record.updated_at
            self._save_index()

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
            record.completed_at = record.updated_at
            self._save_index()

    async def set_error(self, session_id: str, error: str) -> None:
        """Mark session as errored"""
        async with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return
            record.error = error
            record.status = "error"
            record.updated_at = datetime.utcnow()
            record.completed_at = record.updated_at
            self._save_index()

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
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "is_active": record.status in {"pending", "running"},
            "duration_seconds": (
                (record.completed_at or datetime.utcnow()) - record.created_at
            ).total_seconds(),
        }
