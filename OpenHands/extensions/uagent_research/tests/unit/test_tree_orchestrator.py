import asyncio
import time
from typing import List, Tuple

import pytest

from extensions.uagent_research.orchestrator import tree_orchestrator
from extensions.uagent_research.orchestrator.tree_orchestrator import TreeSearchOrchestrator
from extensions.uagent_research.adapters.base.agent_adapter import AgentAdapter, adapter_registry
from extensions.uagent_research.uagent_research.models.events import CompleteEvent
from extensions.uagent_research.uagent_research.models.research_tree import (
    Context,
    ResearchNode,
    ResearchTree,
    NodeStatus,
    NodeType,
)
from extensions.uagent_research.uagent_research.models.research_tree import Task


class _DummyEventBus:
    async def publish(self, event):  # pragma: no cover - simple stub
        return None

    def stop_branch_heartbeat(self, node_id):  # pragma: no cover - simple stub
        return None

    def clear_event_log(self, research_id):  # pragma: no cover - simple stub
        return None


class _DummyRouter:
    def route(self, task, context):
        return "dummy"


class _DummyAdapter(AgentAdapter):
    name = "dummy"

    def __init__(self, delay: float, recorder: List[Tuple[str, float]]):
        super().__init__(name="dummy")
        self.delay = delay
        self.recorder = recorder

    async def run(self, task: Task, context: Context):
        start = time.perf_counter()
        self.recorder.append(("start", start))
        await asyncio.sleep(self.delay)
        self.recorder.append(("finish", time.perf_counter()))
        yield CompleteEvent(
            branch_id=task.id,
            node_id=task.id,
            artifacts=[],
            summary="done",
            success=True,
        )

    async def cancel(self):
        await super().cancel()


@pytest.fixture
def restore_adapter_registry():
    original = dict(adapter_registry._adapters)
    try:
        yield
    finally:
        adapter_registry._adapters = original


@pytest.mark.asyncio
async def test_orchestrator_initializes_with_intelligent_expansion(monkeypatch):
    monkeypatch.setattr(tree_orchestrator, "ENABLE_INTELLIGENT_EXPANSION", True)

    class DummyIdeaService:
        def __init__(self, llm, config=None):
            self.llm = llm

        async def generate_ideas(self, *args, **kwargs):
            return []

        async def generate_hypotheses(self, *args, **kwargs):
            return []

        async def generate_experiments(self, *args, **kwargs):
            return []

    monkeypatch.setattr(tree_orchestrator, "IdeaGenerationService", DummyIdeaService)

    orchestrator = TreeSearchOrchestrator(
        llm=object(),
        event_bus=_DummyEventBus(),
    )

    assert orchestrator.use_intelligent_expansion is True
    assert isinstance(orchestrator.idea_service, DummyIdeaService)


@pytest.mark.asyncio
async def test_parallel_execution_respects_max_parallel(monkeypatch, restore_adapter_registry):
    monkeypatch.setattr(tree_orchestrator, "ENABLE_INTELLIGENT_EXPANSION", False)

    recorder: List[Tuple[str, float]] = []
    adapter_registry._adapters.pop("dummy", None)
    adapter_registry.register(_DummyAdapter(delay=0.2, recorder=recorder))

    orchestrator = TreeSearchOrchestrator(
        max_parallel=2,
        router=_DummyRouter(),
        event_bus=_DummyEventBus(),
    )
    orchestrator._publish_tree_to_api = lambda: None  # Avoid side effects in test

    # Prepare tree with root and two child nodes
    tree = ResearchTree(research_id="test-exp")
    root = ResearchNode(
        id="root",
        type=NodeType.ROOT,
        title="root",
        content="root",
        status=NodeStatus.COMPLETE,
    )
    tree.add_node(root)

    child1 = ResearchNode(
        id="node-1",
        type=NodeType.IDEA,
        title="First node",
        content="search something",
        status=NodeStatus.PENDING,
        prior=0.5,
    )
    child2 = ResearchNode(
        id="node-2",
        type=NodeType.IDEA,
        title="Second node",
        content="search something else",
        status=NodeStatus.PENDING,
        prior=0.5,
    )
    tree.add_node(child1, parent_id="root")
    tree.add_node(child2, parent_id="root")

    orchestrator.tree = tree

    start = time.perf_counter()
    await orchestrator._execute_children_parallel([child1, child2])
    elapsed = time.perf_counter() - start

    # With max_parallel=2 and each task sleeping 0.2s, elapsed time should be well below sequential (0.4s)
    assert elapsed < 0.35, f"Parallel execution took too long: {elapsed:.3f}s"

    # Ensure both nodes reached completion
    assert child1.status == NodeStatus.COMPLETE
    assert child2.status == NodeStatus.COMPLETE

    starts = [t for label, t in recorder if label == "start"]
    finishes = [t for label, t in recorder if label == "finish"]
    assert len(starts) == len(finishes) == 2
    # Verify starts are close together (within 0.15s), indicating concurrent execution
    assert abs(starts[0] - starts[1]) < 0.15
