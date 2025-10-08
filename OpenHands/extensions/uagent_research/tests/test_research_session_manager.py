"""Unit tests for the ResearchSessionManager lifecycle component."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.control_bus import ControlMessage
from services import research_session_manager as rsm_module
from services.research_session_manager import (
    AdapterStatus,
    BranchStatus,
    ExperimentState,
    ExperimentStatus,
    ResearchSessionManager,
    get_session_manager,
)
from uagent_research.models.events import StepEvent


@pytest.mark.asyncio
async def test_initialization(mock_event_bus, mock_control_bus):
    manager = ResearchSessionManager(event_bus=mock_event_bus, control_bus=mock_control_bus)
    assert manager.experiments == {}
    assert manager.event_bus is mock_event_bus
    assert manager.control_bus is mock_control_bus
    assert manager._event_subscription_task is not None
    assert not manager._event_subscription_task.done()

    await manager.close()


@pytest.mark.asyncio
async def test_initialization_without_buses():
    manager = ResearchSessionManager()
    assert manager.experiments == {}
    assert manager.event_bus is None
    assert manager.control_bus is None
    assert manager._event_subscription_task is None
    await manager.close()


@pytest.mark.asyncio
async def test_register_experiment(mock_research_session_manager, mock_orchestrator):
    manager = mock_research_session_manager
    manager.register("exp-1", mock_orchestrator)
    assert "exp-1" in manager.experiments
    state = manager.experiments["exp-1"]
    assert state.status == ExperimentStatus.RUNNING
    assert state.orchestrator is mock_orchestrator


@pytest.mark.asyncio
async def test_register_duplicate_experiment(mock_research_session_manager, mock_orchestrator, caplog):
    manager = mock_research_session_manager
    manager.register("exp-dup", mock_orchestrator)
    with caplog.at_level("WARNING"):
        manager.register("exp-dup", mock_orchestrator)
    assert "exp-dup" in manager.experiments
    assert len(manager.experiments) == 1
    assert any("already registered" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_unregister_experiment(mock_research_session_manager, mock_orchestrator):
    manager = mock_research_session_manager
    manager.register("exp-unreg", mock_orchestrator)
    assert "exp-unreg" in manager.experiments
    manager.unregister("exp-unreg")
    assert "exp-unreg" not in manager.experiments


@pytest.mark.asyncio
async def test_get_status_basic(mock_research_session_manager, mock_orchestrator):
    manager = mock_research_session_manager
    manager.register("exp-status", mock_orchestrator)
    status = manager.get_status("exp-status")

    assert status["experiment_id"] == "exp-status"
    assert status["status"] == ExperimentStatus.RUNNING.value
    assert status["stats"] == {
        "total_nodes": 0,
        "completed": 0,
        "failed": 0,
        "running": 0,
        "pending": 0,
        "total_cost": 0.0,
        "total_tokens": 0,
    }
    assert status["adapters"] == {}
    assert status["active_branches"] == []
    assert status["created_at"]
    assert status["last_update"]


@pytest.mark.asyncio
async def test_get_status_not_found(mock_research_session_manager):
    manager = mock_research_session_manager
    with pytest.raises(KeyError) as exc:
        manager.get_status("missing")
    assert "Experiment missing not found" in str(exc.value)


@pytest.mark.asyncio
async def test_list_active(mock_research_session_manager, mock_orchestrator):
    manager = mock_research_session_manager
    manager.register("exp-a", mock_orchestrator)
    manager.register("exp-b", mock_orchestrator)
    manager.register("exp-c", mock_orchestrator)

    manager.update_experiment_status("exp-b", ExperimentStatus.PAUSED)
    manager.update_experiment_status("exp-c", ExperimentStatus.COMPLETE)

    active = sorted(manager.list_active())
    assert active == ["exp-a", "exp-b"]


@pytest.mark.asyncio
async def test_send_control(mock_control_bus, mock_event_bus, mock_orchestrator):
    bus = mock_control_bus
    manager = ResearchSessionManager(event_bus=mock_event_bus, control_bus=bus)
    manager.register("exp-send", mock_orchestrator)

    received: List[ControlMessage] = []

    async def subscriber():
        async for message in bus.subscribe("exp-send"):
            received.append(message)
            break

    sub_task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.05)
    await manager.send_control("exp-send", ControlMessage(action="pause"))
    await asyncio.wait_for(sub_task, timeout=5)
    bus.unsubscribe_all("exp-send")

    assert received and received[0].action == "pause"
    await manager.close()


@pytest.mark.asyncio
async def test_send_control_no_bus(mock_orchestrator, caplog):
    manager = ResearchSessionManager(event_bus=None, control_bus=None)
    manager.register("exp-send", mock_orchestrator)

    with caplog.at_level("ERROR"):
        await manager.send_control("exp-send", ControlMessage(action="pause"))

    assert any("ControlBus not configured" in record.message for record in caplog.records)
    await manager.close()


@pytest.mark.asyncio
async def test_send_control_experiment_not_found(mock_control_bus, mock_event_bus, caplog):
    manager = ResearchSessionManager(event_bus=mock_event_bus, control_bus=mock_control_bus)
    with caplog.at_level("WARNING"):
        await manager.send_control("missing", ControlMessage(action="pause"))
    assert any("Experiment missing not found" in record.message for record in caplog.records)
    await manager.close()


@pytest.mark.asyncio
async def test_update_experiment_status(mock_research_session_manager, mock_orchestrator):
    manager = mock_research_session_manager
    manager.register("exp-update", mock_orchestrator)
    manager.update_experiment_status("exp-update", ExperimentStatus.PAUSED)
    status = manager.get_status("exp-update")
    assert status["status"] == ExperimentStatus.PAUSED.value


@pytest.mark.asyncio
async def test_handle_step_event(mock_research_session_manager, mock_orchestrator, sample_research_events):
    manager = mock_research_session_manager
    manager.register("exp-step", mock_orchestrator)
    event = sample_research_events["step"](adapter_name="codeact", action="Running tests")
    setattr(event, "experiment_id", "exp-step")

    await manager._handle_event(event)
    status = manager.get_status("exp-step")
    adapter_status = status["adapters"]["codeact"]
    assert adapter_status["status"] == "running"
    assert adapter_status["current_step"] == "Running tests"
    assert status["active_branches"]
    branch = status["active_branches"][0]
    assert branch["branch_id"] == event.branch_id
    assert branch["status"] == "running"
    assert branch["progress"] == "Running tests"


@pytest.mark.asyncio
async def test_handle_complete_event(mock_research_session_manager, mock_orchestrator, sample_research_events):
    manager = mock_research_session_manager
    manager.register("exp-complete", mock_orchestrator)
    state = manager.experiments["exp-complete"]
    state.running_nodes = 1
    event = sample_research_events["complete"]()
    setattr(event, "experiment_id", "exp-complete")

    await manager._handle_event(event)
    status = manager.get_status("exp-complete")
    assert status["stats"]["completed"] == 1
    assert status["stats"]["running"] == 0
    assert status["active_branches"][0]["status"] == "complete"
    assert status["active_branches"][0]["progress"] == event.summary


@pytest.mark.asyncio
async def test_handle_error_event(mock_research_session_manager, mock_orchestrator, sample_research_events):
    manager = mock_research_session_manager
    manager.register("exp-error", mock_orchestrator)
    state = manager.experiments["exp-error"]
    state.running_nodes = 1
    event = sample_research_events["error"](message="Failure")
    setattr(event, "experiment_id", "exp-error")

    await manager._handle_event(event)
    status = manager.get_status("exp-error")
    assert status["stats"]["failed"] == 1
    assert status["stats"]["running"] == 0
    assert status["active_branches"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_adapter_cost_tracking(mock_research_session_manager, mock_orchestrator, sample_research_events):
    manager = mock_research_session_manager
    manager.register("exp-cost", mock_orchestrator)

    event1 = sample_research_events["step"](cost=0.05)
    event2 = sample_research_events["step"](cost=0.03)
    for event in (event1, event2):
        setattr(event, "experiment_id", "exp-cost")
        await manager._handle_event(event)

    status = manager.get_status("exp-cost")
    adapter = status["adapters"]["codeact"]
    assert adapter["cost"] == pytest.approx(0.08)
    assert status["stats"]["total_cost"] == pytest.approx(0.08)


@pytest.mark.asyncio
async def test_adapter_token_tracking(mock_research_session_manager, mock_orchestrator, sample_research_events):
    manager = mock_research_session_manager
    manager.register("exp-token", mock_orchestrator)

    event1 = sample_research_events["step"](tokens=1000)
    event2 = sample_research_events["step"](tokens=500)
    for event in (event1, event2):
        setattr(event, "experiment_id", "exp-token")
        await manager._handle_event(event)

    status = manager.get_status("exp-token")
    assert status["stats"]["total_tokens"] == 1500
    assert status["adapters"]["codeact"]["tokens"] == 1500


@pytest.mark.asyncio
async def test_multiple_adapters(mock_research_session_manager, mock_orchestrator, sample_research_events):
    manager = mock_research_session_manager
    manager.register("exp-multi", mock_orchestrator)

    events = [
        sample_research_events["step"](adapter_name="codeact"),
        sample_research_events["step"](adapter_name="deepresearch"),
        sample_research_events["step"](adapter_name="repomaster"),
    ]
    for event in events:
        setattr(event, "experiment_id", "exp-multi")
        await manager._handle_event(event)

    status = manager.get_status("exp-multi")
    assert sorted(status["adapters"].keys()) == ["codeact", "deepresearch", "repomaster"]


@pytest.mark.asyncio
async def test_get_experiment_id_from_event(mock_research_session_manager, mock_orchestrator):
    manager = mock_research_session_manager
    manager.register("exp-event", mock_orchestrator)

    event = StepEvent(branch_id="branch-1", node_id="node-1", action="Test")
    setattr(event, "experiment_id", "exp-event")
    assert manager._get_experiment_id_from_event(event) == "exp-event"

    branch_event = StepEvent(branch_id="branch-1", node_id="node-1", action="Test again")
    assert manager._get_experiment_id_from_event(branch_event) == "exp-event"


@pytest.mark.asyncio
async def test_singleton_pattern():
    first = get_session_manager()
    second = get_session_manager()
    assert first is second
    await first.close()
    rsm_module._session_manager = None


@pytest.mark.asyncio
async def test_close(mock_event_bus, mock_control_bus):
    manager = ResearchSessionManager(event_bus=mock_event_bus, control_bus=mock_control_bus)
    assert manager._event_subscription_task is not None
    await manager.close()
    assert manager._event_subscription_task.cancelled()


@pytest.mark.asyncio
async def test_event_subscription_integration(mock_control_bus, mock_event_bus, mock_orchestrator, sample_research_events):
    manager = ResearchSessionManager(event_bus=mock_event_bus, control_bus=mock_control_bus)
    manager.register("exp-integration", mock_orchestrator)

    event = sample_research_events["step"](adapter_name="codeact", tokens=321)
    setattr(event, "experiment_id", "exp-integration")
    await mock_event_bus.publish(event)
    await asyncio.sleep(0.05)

    status = manager.get_status("exp-integration")
    assert status["adapters"]["codeact"]["tokens"] == 321

    await manager.close()


@pytest.mark.asyncio
async def test_status_aggregation_full_workflow(mock_research_session_manager, mock_orchestrator, sample_research_events):
    manager = mock_research_session_manager
    manager.register("exp-workflow", mock_orchestrator)

    step1 = sample_research_events["step"](branch_id="branch-codeact", adapter_name="codeact", action="Starting")
    step2 = sample_research_events["step"](branch_id="branch-deep", adapter_name="deepresearch", action="Searching")
    complete_event = sample_research_events["complete"](branch_id="branch-codeact")
    error_event = sample_research_events["error"](branch_id="branch-deep")

    for event in (step1, step2, complete_event, error_event):
        setattr(event, "experiment_id", "exp-workflow")
        await manager._handle_event(event)

    status = manager.get_status("exp-workflow")
    assert status["stats"]["completed"] == 1
    assert status["stats"]["failed"] == 1
    assert sorted(status["adapters"].keys()) == ["codeact", "deepresearch"]
    assert status["last_update"]
    branches = {b["branch_id"]: b for b in status["active_branches"]}
    assert branches["branch-codeact"]["status"] == "complete"
    assert branches["branch-deep"]["status"] == "failed"


def test_adapter_status_dataclass():
    adapter = AdapterStatus(adapter_name="codeact")
    assert adapter.status == "idle"
    adapter.status = "running"
    adapter.total_cost += 0.1
    assert adapter.total_cost == pytest.approx(0.1)


def test_branch_status_dataclass():
    branch = BranchStatus(branch_id="branch", title="Title", adapter="codeact", status="running")
    assert branch.branch_id == "branch"
    assert branch.status == "running"


def test_experiment_state_dataclass():
    state = ExperimentState(experiment_id="exp")
    assert state.status == ExperimentStatus.INITIALIZING
    assert state.total_nodes == 0
    assert state.total_cost == 0.0
    assert state.last_update


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
