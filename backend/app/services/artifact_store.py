"""File artifact persistence for experiments and analyses."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


class ArtifactStore:
    """Store experiment artifacts on disk with simple provenance metadata."""

    def __init__(self, root: str | Path = "artifacts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_text(self, session_id: str, name: str, content: str) -> Path:
        return self._write_artifact(session_id, name, content.encode("utf-8"), suffix=".txt")

    def save_bytes(self, session_id: str, name: str, content: bytes, suffix: str = "") -> Path:
        return self._write_artifact(session_id, name, content, suffix=suffix)

    def record_metadata(self, session_id: str, artifact_path: Path, metadata: Optional[Dict[str, object]] = None) -> None:
        meta_dir = artifact_path.parent
        meta_file = meta_dir / "metadata.jsonl"
        record = {
            "artifact": artifact_path.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata or {},
        }
        with meta_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def list_artifacts(self, session_id: str) -> List[Path]:
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return []
        return [path for path in session_dir.iterdir() if path.is_file() and path.name != "metadata.jsonl"]

    def _write_artifact(self, session_id: str, name: str, payload: bytes, suffix: str = "") -> Path:
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(" ", "_")
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"{safe_name}_{timestamp}{suffix}"
        artifact_path = session_dir / filename

        with artifact_path.open("wb") as fh:
            fh.write(payload)
        LOGGER.info("Saved artifact %s", artifact_path)
        return artifact_path

    def _session_dir(self, session_id: str) -> Path:
        return self.root / session_id


__all__ = ["ArtifactStore"]
