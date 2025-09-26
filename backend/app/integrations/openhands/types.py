from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ActionEnvelope:
    id: str
    tool: str
    args: Dict[str, Any]
    timeout_sec: int = 90
    cwd: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionError:
    type: str
    message: str


@dataclass
class ActionResult:
    id: str
    tool: str
    success: bool
    exit_code: Optional[int]
    duration_ms: int
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    artifact_paths: List[str] = field(default_factory=list)
    error: Optional[ActionError] = None

