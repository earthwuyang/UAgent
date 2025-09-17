import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.orchestrator import ExperimentOrchestrator


def _runner(idea):
    return {"score": len(idea["name"]) }


def _brancher(idea, result):
    if result["score"] < 3:
        return [{"name": idea["name"] + "x"}]
    return []


def _reducer(results):
    return {"n": len(results)}


def test_orchestrator_runs_tree():
    orch = ExperimentOrchestrator(_runner, _brancher, _reducer, max_workers=2)
    out = orch.run({"name": "a"}, max_depth=3)
    assert out["n"] >= 1
