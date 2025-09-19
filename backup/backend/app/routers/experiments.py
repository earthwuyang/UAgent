from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List

from ..core.orchestrator import ExperimentOrchestrator


router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentIdea(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class ExperimentPlan(BaseModel):
    root: ExperimentIdea
    max_depth: int = 2
    parallelism: int = 4


def default_runner(idea: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder: simulate a computation
    score = len(idea.get("name", "")) + len(str(idea.get("params", {})))
    return {"score": score, "summary": f"Ran {idea.get('name')}"}


def default_brancher(idea: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Simple branch: create up to two variants based on score
    score = result.get("score", 0)
    if score < 5:
        return [{"name": idea["name"] + "-A", "params": idea.get("params", {})}, {"name": idea["name"] + "-B", "params": idea.get("params", {})}]
    if score < 12:
        return [{"name": idea["name"] + "-A", "params": idea.get("params", {})}]
    return []


def default_reducer(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"best": None, "count": 0}
    best = max(results, key=lambda r: r.get("score", 0))
    return {"best": best, "count": len(results)}


@router.post("/run")
def run_experiments(plan: ExperimentPlan) -> Dict[str, Any]:
    orch = ExperimentOrchestrator(
        runner=default_runner,
        brancher=default_brancher,
        reducer=default_reducer,
        rollback=None,
        max_workers=plan.parallelism,
    )
    summary = orch.run(plan.root.model_dump(), max_depth=plan.max_depth)
    return {"summary": summary}

