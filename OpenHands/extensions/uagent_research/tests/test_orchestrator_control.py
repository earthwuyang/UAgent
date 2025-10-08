"""Integration tests for TreeSearchOrchestrator control loop."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.control_bus import ControlBus, ControlMessage
from orchestrator.event_bus import EventBus
from orchestrator.tree_orchestrator import TreeSearchOrchestrator
from adapters.base.agent_adapter import AgentAdapter, adapter_registry
from router.skill_router import SkillRouter
from uagent_research.models.research_tree import (
    ResearchNode,
    NodeType,
    NodeStatus,
    Task,
    Context,
)
from uagent_research.models.events import StepEvent, CompleteEvent


class StubRouter(SkillRouter):
    """Router that always selects the stub adapter."""

    def route(self, task: Task, context: Context) -> str:  # type: ignore[override]
        return "stub"


class RecordingAdapter(AgentAdapter):
    """Minimal adapter that records execution and supports steering."""

    name = "stub"

    def __init__(self) -> None:
        super().__init__(name=self.name)
        self.messages: Dict[str, list[str]] = {}
        self.active: Dict[str, asyncio.Task] = {}
        self.sleep_duration = 0.2

    async def run(self, task: Task, context: Context):  # type: ignore[override]
        self.active[task.id] = asyncio.current_task()
        try:
            step = StepEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                action=f"Processing {task.goal}",
            )
            step.adapter_name = self.name  # type: ignore[attr-defined]
            step.cost = 0.02  # type: ignore[attr-defined]
            step.tokens = 10  # type: ignore[attr-defined]
            yield step

            # Simulate work
            await asyncio.sleep(self.sleep_duration)

            complete = CompleteEvent(
                branch_id=context.branch_id,
                node_id=task.id,
                summary=f"Completed {task.goal}",
                artifacts=[],
                success=True,
            )
            complete.adapter_name = self.name  # type: ignore[attr-defined]
            yield complete
        except asyncio.CancelledError:
            raise
        finally:
            self.active.pop(task.id, None)

    async def cancel(self):  # type: ignore[override]
        self._cancelled = True
        for task in list(self.active.values()):
            task.cancel()

    async def send_message(self, message: str) -> None:  # type: ignore[override]
        # Attach message to all active nodes
        for node_id in self.active.keys():
            self.messages.setdefault(node_id, []).append(message)


@pytest.fixture
def stub_adapter():
    original = adapter_registry._adapters.copy()  # type: ignore[attr-defined]
    adapter_registry._adapters.clear()  # type: ignore[attr-defined]
    adapter = RecordingAdapter()
    adapter_registry.register(adapter)
    yield adapter
    adapter_registry._adapters.clear()  # type: ignore[attr-defined]
    adapter_registry._adapters.update(original)  # type: ignore[attr-defined]


@pytest.fixture
async def control_bus():
    bus = ControlBus()
    yield bus
    for exp in list(bus.get_stats().get("experiments", [])):
        bus.unsubscribe_all(exp)


@pytest.fixture
async def event_bus():
    bus = EventBus(heartbeat_interval=0)
    yield bus
    await bus.close()


@pytest.fixture
async def orchestrator(stub_adapter, control_bus, event_bus):  # pylint: disable=unused-argument
    orch = TreeSearchOrchestrator(
        max_parallel=1,
        router=StubRouter(),
        event_bus=event_bus,
        control_bus=control_bus,
    )
    yield orch
    await orch.cancel()


async def wait_for(predicate, timeout: float = 2.0, interval: float = 0.01):
    """Poll predicate until truthy or timeout."""
    end_time = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end_time:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("Condition not met within timeout")


@pytest.mark.asyncio
async def test_running_tasks_populated(orchestrator):
    run_task = asyncio.create_task(orchestrator.run("Investigate control loop", max_iterations=1))

    await wait_for(lambda: bool(orchestrator._running_tasks))
    assert all(isinstance(task, asyncio.Task) for task in orchestrator._running_tasks.values())

    await run_task
    assert orchestrator._running_tasks == {}


@pytest.mark.asyncio
async def test_pause_resume_flow(orchestrator, control_bus):
    run_task = asyncio.create_task(orchestrator.run("Explore pause/resume", max_iterations=3))

    await wait_for(lambda: orchestrator.tree is not None)
    research_id = orchestrator.tree.research_id

    await wait_for(lambda: orchestrator.stats["iterations"] >= 1)
    await control_bus.publish(research_id, ControlMessage(action="pause"))
    await wait_for(lambda: orchestrator._paused is True)
    paused_iterations = orchestrator.stats["iterations"]
    await asyncio.sleep(0.2)
    assert orchestrator.stats["iterations"] == paused_iterations

    await control_bus.publish(research_id, ControlMessage(action="resume"))
    await wait_for(lambda: orchestrator._paused is False)

    await run_task
    assert orchestrator.stats["iterations"] >= paused_iterations


@pytest.mark.asyncio
async def test_cancel_experiment(orchestrator, control_bus):
    run_task = asyncio.create_task(orchestrator.run("Cancel experiment", max_iterations=5))

    await wait_for(lambda: orchestrator.tree is not None)
    research_id = orchestrator.tree.research_id
    await wait_for(lambda: bool(orchestrator._running_tasks))

    await control_bus.publish(research_id, ControlMessage(action="cancel"))
    await run_task

    assert orchestrator._cancelled is True
    assert orchestrator._running_tasks == {}


@pytest.mark.asyncio
async def test_cancel_node(orchestrator, control_bus):
    run_task = asyncio.create_task(orchestrator.run("Cancel node", max_iterations=3))

    await wait_for(lambda: orchestrator.tree is not None)
    research_id = orchestrator.tree.research_id
    await wait_for(lambda: bool(orchestrator._running_tasks))
    node_id = next(iter(orchestrator._running_tasks.keys()))

    await control_bus.publish(
        research_id,
        ControlMessage(action="cancel_node", target={"node_id": node_id}),
    )

    await run_task
    assert orchestrator.tree.nodes[node_id].status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_reprioritize_by_adapter(orchestrator, control_bus):
    run_task = asyncio.create_task(orchestrator.run("Reprioritize nodes", max_iterations=1))
    await wait_for(lambda: orchestrator.tree is not None)
    research_id = orchestrator.tree.research_id
    await run_task

    manual_node = ResearchNode(
        id="manual-node",
        type=NodeType.IDEA,
        title="Manual",
        content="Investigate",
        status=NodeStatus.PENDING,
        prior=0.1,
    )
    manual_node.adapter = "stub"
    orchestrator.tree.add_node(manual_node, parent_id="root")

    await control_bus.publish(
        research_id,
        ControlMessage(
            action="reprioritize",
            target={"adapter": "stub"},
            payload={"delta": 0.2},
        ),
    )

    assert manual_node.prior == pytest.approx(0.3, rel=1e-2)


@pytest.mark.asyncio
async def test_steer_to_adapter(orchestrator, control_bus, stub_adapter):
    run_task = asyncio.create_task(orchestrator.run("Steer adapter", max_iterations=2))
    await wait_for(lambda: orchestrator.tree is not None)
    research_id = orchestrator.tree.research_id
    await wait_for(lambda: bool(orchestrator._running_tasks))
    node_id = next(iter(orchestrator._running_tasks.keys()))

    await control_bus.publish(
        research_id,
        ControlMessage(
            action="steer",
            target={"node_id": node_id},
            payload={"text": "Focus on performance"},
        ),
    )

    await wait_for(lambda: stub_adapter.messages.get(node_id))
    await run_task

    assert "Focus on performance" in stub_adapter.messages[node_id][0]


@pytest.mark.asyncio
async def test_add_node_manual(orchestrator, control_bus):
    run_task = asyncio.create_task(orchestrator.run("Manual node", max_iterations=2))
    await wait_for(lambda: orchestrator.tree is not None)
    research_id = orchestrator.tree.research_id

    await control_bus.publish(
        research_id,
        ControlMessage(
            action="add_node",
            payload={
                "parent_id": "root",
                "node": {
                    "type": "IDEA",
                    "title": "Manual Direction",
                    "content": "Consider alternative approach",
                    "prior": 0.4,
                },
            },
        ),
    )

    await wait_for(lambda: any(n.id.startswith("manual-") for n in orchestrator.tree.nodes.values()))
    manual_nodes = [n for n in orchestrator.tree.nodes.values() if n.id.startswith("manual-")]
    manual_node = manual_nodes[0]

    await run_task
    assert manual_node.status in {NodeStatus.COMPLETE, NodeStatus.RUNNING, NodeStatus.FAILED}


@pytest.mark.asyncio
async def test_node_adapter_field_set(orchestrator):
    run_task = asyncio.create_task(orchestrator.run("Adapter metadata", max_iterations=2))
    await wait_for(lambda: bool(orchestrator._running_tasks))
    node_id = next(iter(orchestrator._running_tasks.keys()))
    await wait_for(lambda: orchestrator.tree.nodes[node_id].adapter is not None)
    await run_task
    assert orchestrator.tree.nodes[node_id].adapter == "stub"


@pytest.mark.asyncio
async def test_concurrent_commands(orchestrator, control_bus, stub_adapter):
    run_task = asyncio.create_task(orchestrator.run("Concurrent commands", max_iterations=3))
    await wait_for(lambda: orchestrator.tree is not None)
    research_id = orchestrator.tree.research_id

    await wait_for(lambda: bool(orchestrator._running_tasks))
    node_id = next(iter(orchestrator._running_tasks.keys()))

    await asyncio.gather(
        control_bus.publish(research_id, ControlMessage(action="pause")),
        control_bus.publish(
            research_id,
            ControlMessage(
                action="steer",
                target={"node_id": node_id},
                payload={"text": "Handle edge cases"},
            ),
        ),
        control_bus.publish(research_id, ControlMessage(action="resume")),
    )

    await wait_for(lambda: stub_adapter.messages.get(node_id))
    await run_task

    assert orchestrator._paused is False
    assert orchestrator._cancelled is False
    assert "Handle edge cases" in stub_adapter.messages[node_id][0]
