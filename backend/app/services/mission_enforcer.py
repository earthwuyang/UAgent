from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple


class MissionEnforcer:
    """Simple finish gate for research missions.

    Ensures that the proxy ledger shows a minimum number of successful query
    executions and that a measurements file exists under experiments/<id>/.
    """

    def __init__(self, workspace_dir: Path, experiment_id: str, min_success_queries: int = 3) -> None:
        self.ws = workspace_dir
        self.experiment_id = experiment_id
        self.min_success_queries = min_success_queries

    def _count_proxy_success(self) -> int:
        fp = self.ws / "logs" / "proxy_calls.jsonl"
        if not fp.exists():
            return 0
        n = 0
        try:
            for line in fp.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("phase") == "proxy_exec" and rec.get("status") == "SUCCEEDED":
                    n += 1
        except Exception:
            return 0
        return n

    def _has_measurements(self) -> bool:
        mdir = self.ws / f"experiments/{self.experiment_id}/measurements"
        return mdir.exists() and any(mdir.glob("*.jsonl"))

    def can_finish(self) -> Tuple[bool, str]:
        success_q = self._count_proxy_success()
        if success_q < self.min_success_queries:
            return False, f"Insufficient successful proxy executions: {success_q}/{self.min_success_queries}"
        if not self._has_measurements():
            return False, "No measurements saved under experiments/<id>/measurements/*.jsonl"
        return True, "ready"

    def assert_finish_or_raise(self) -> None:
        ok, why = self.can_finish()
        if not ok:
            raise RuntimeError(why)

