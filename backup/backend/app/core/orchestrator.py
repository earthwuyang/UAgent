from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional
import time
import uuid


@dataclass
class ExperimentNode:
    id: str
    parent_id: Optional[str]
    idea: Dict[str, Any]
    status: str = "pending"  # pending|running|succeeded|failed
    result: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)


class ExperimentOrchestrator:
    """
    Minimal tree-of-experiments orchestrator.

    - Accepts a root idea and a runner callable.
    - Expands into parallel sub-experiments via a branching function.
    - Rolls back failed nodes via a user-supplied rollback function.
    - Aggregates results with a reducer.
    """

    def __init__(
        self,
        runner: Callable[[Dict[str, Any]], Dict[str, Any]],
        brancher: Callable[[Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]],
        reducer: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
        rollback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_workers: int = 4,
    ) -> None:
        self.runner = runner
        self.brancher = brancher
        self.reducer = reducer
        self.rollback = rollback
        self.max_workers = max_workers
        self.nodes: Dict[str, ExperimentNode] = {}

    def _add_node(self, parent_id: Optional[str], idea: Dict[str, Any]) -> ExperimentNode:
        node = ExperimentNode(id=str(uuid.uuid4()), parent_id=parent_id, idea=idea)
        self.nodes[node.id] = node
        return node

    def run(self, root_idea: Dict[str, Any], max_depth: int = 2) -> Dict[str, Any]:
        root = self._add_node(None, root_idea)
        frontier = [root]

        for depth in range(max_depth):
            if not frontier:
                break
            next_frontier: List[ExperimentNode] = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self._execute_node, n): n for n in frontier}
                for fut in as_completed(futures):
                    node = futures[fut]
                    try:
                        result = fut.result()
                        node.status = "succeeded"
                        node.result = result
                        # branch
                        children_ideas = self.brancher(node.idea, result)
                        for child_idea in children_ideas:
                            next_frontier.append(self._add_node(node.id, child_idea))
                    except Exception:
                        node.status = "failed"
                        if self.rollback:
                            try:
                                self.rollback(node.idea)
                            except Exception:
                                pass
            frontier = next_frontier

        # aggregate from all succeeded leaves
        leaf_results: List[Dict[str, Any]] = [n.result for n in self.nodes.values() if n.status == "succeeded" and n.result]
        return self.reducer(leaf_results)

    def _execute_node(self, node: ExperimentNode) -> Dict[str, Any]:
        node.status = "running"
        return self.runner(node.idea)

