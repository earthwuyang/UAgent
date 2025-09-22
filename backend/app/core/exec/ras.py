from __future__ import annotations

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

StepKind = Literal[
    "fetch_repo",
    "detect_buildsystem",
    "build",
    "run_commands",
    "collect_artifacts",
    "evaluate_artifacts",
    "write_report",
    "code_edit",
    "multi_agent_debate",
]


class Step(BaseModel):
    id: str
    kind: StepKind
    with_: Dict[str, object] = Field(alias="with")


class ResearchActionSpec(BaseModel):
    version: int = 1
    run: Dict[str, object]
    steps: List[Step]
    matrix: Optional[Dict[str, List[object]]] = None
    assertions: Optional[List[Dict[str, object]]] = None
