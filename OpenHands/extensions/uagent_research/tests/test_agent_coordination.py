import time
from typing import Dict, Optional
from unittest.mock import MagicMock

import pytest

from extensions.uagent_research.middleware.research_middleware import research_middleware
from extensions.uagent_research.uagent_research.agents.code_research_agent import (
    CodeResearchAgent,
)
from extensions.uagent_research.uagent_research.agents.scientific_research_agent import (
    ScientificResearchAgent,
)
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.events.action import MessageAction
from openhands.events.event import EventSource


class DummyHistory:
    def __init__(self, messages):
        self._messages = messages

    def get_events_as_list(self):
        return self._messages


class DummyEvent:
    def __init__(self, message: str):
        self.source = EventSource.USER
        self.message = message


@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfig()


@pytest.fixture
def llm_registry():
    dummy_llm = MagicMock()
    dummy_llm.config.disable_vision = False
    dummy_llm.vision_is_active.return_value = True

    registry = MagicMock()
    registry.get_router.return_value = dummy_llm
    registry.get_llm_from_agent_config.return_value = dummy_llm
    return registry


def make_state(message: str, session_id: str = 'session-1') -> State:
    state = State()
    state.session_id = session_id
    state.history = DummyHistory([DummyEvent(message)])
    return state


def make_status(status: str = 'running') -> Dict[str, object]:
    return {
        'status': status,
        'stats': {
            'total_nodes': 10,
            'completed': 4,
            'running': 3,
            'failed': 1,
            'total_cost': 1.23,
        },
        'active_branches': ['root', 'idea-1'],
        'adapters': {
            'codeact': {'running_count': 1},
        },
    }


def test_agent_enters_coordination_mode_on_research_task(agent_config, llm_registry, monkeypatch):
    agent = ScientificResearchAgent(agent_config, llm_registry)
    monkeypatch.setattr(agent, '_run_coroutine_sync', lambda coro: 'exp-test')

    state = make_state('Please conduct an experiment comparing algorithms.')
    result = agent.step(state)

    assert agent._coordination_mode is True
    assert agent._experiment_id == 'exp-test'
    assert isinstance(result, MessageAction)
    assert 'Starting a scientific research experiment' not in result.content
    assert 'Launching a scientific research experiment' in result.content


def test_agent_stays_in_normal_mode_for_simple_tasks(agent_config, llm_registry, monkeypatch):
    agent = ScientificResearchAgent(agent_config, llm_registry)
    sentinel_action = MessageAction(content='normal', thought='base')
    step_spy = MagicMock(return_value=sentinel_action)
    monkeypatch.setattr(CodeActAgent, 'step', step_spy)

    state = make_state('Write a hello world function in Python.')
    agent._coordination_mode = False
    result = agent.step(state)

    step_spy.assert_called_once_with(state)
    assert result is sentinel_action
    assert agent._coordination_mode is False


def test_agent_polls_progress_periodically(agent_config, llm_registry):
    agent = ScientificResearchAgent(agent_config, llm_registry)
    agent._coordination_mode = True
    agent._experiment_id = 'exp-1'
    agent._experiment_session_id = 'session-1'
    agent._last_user_message = 'status'
    agent._last_poll_time = time.time() - 30

    manager = MagicMock()
    manager.get_status.return_value = make_status('running')
    agent._session_manager = manager

    state = make_state('status')
    result = agent.step(state)

    manager.get_status.assert_called_once_with('exp-1')
    assert 'Research Progress' in result.content
    assert agent._last_poll_time <= time.time()


def test_agent_skips_poll_if_too_soon(agent_config, llm_registry):
    agent = ScientificResearchAgent(agent_config, llm_registry)
    agent._coordination_mode = True
    agent._experiment_id = 'exp-1'
    agent._experiment_session_id = 'session-1'
    agent._last_user_message = 'status'
    agent._last_poll_time = time.time()

    manager = MagicMock()
    agent._session_manager = manager

    state = make_state('status')
    result = agent.step(state)

    manager.get_status.assert_not_called()
    assert 'Research is still running' in result.content


def test_agent_responds_to_progress_query_immediately(agent_config, llm_registry, monkeypatch):
    agent = ScientificResearchAgent(agent_config, llm_registry)
    agent._coordination_mode = True
    agent._experiment_id = 'exp-1'
    agent._experiment_session_id = 'session-1'

    progress_action = MessageAction(content='progress summary', thought='progress')
    poll_mock = MagicMock(return_value=progress_action)
    monkeypatch.setattr(agent, '_poll_research_progress', poll_mock)

    state = make_state("how's progress?")
    result = agent.step(state)

    poll_mock.assert_called_once()
    assert result is progress_action


def test_agent_handles_control_commands(agent_config, llm_registry, monkeypatch):
    agent = ScientificResearchAgent(agent_config, llm_registry)
    agent._coordination_mode = True
    agent._experiment_id = 'exp-1'
    agent._experiment_session_id = 'session-1'

    control_result = {'status': 'success', 'message': 'Paused research'}
    monkeypatch.setattr(agent, '_execute_control_command', lambda action, target, state: control_result)

    state = make_state('please pause research now')
    result = agent.step(state)

    assert 'Paused research' in result.content


def test_agent_exits_coordination_when_research_completes(agent_config, llm_registry):
    agent = ScientificResearchAgent(agent_config, llm_registry)
    agent._coordination_mode = True
    agent._experiment_id = 'exp-1'
    agent._experiment_session_id = 'session-1'
    agent._last_user_message = 'status'
    agent._last_poll_time = time.time() - 60

    manager = MagicMock()
    manager.get_status.return_value = make_status('complete')
    agent._session_manager = manager

    state = make_state('status')
    result = agent.step(state)

    assert agent._coordination_mode is False
    assert agent._experiment_id is None
    assert 'Research completed' in result.content


def test_agent_exits_coordination_on_failure(agent_config, llm_registry):
    agent = ScientificResearchAgent(agent_config, llm_registry)
    agent._coordination_mode = True
    agent._experiment_id = 'exp-1'
    agent._experiment_session_id = 'session-1'
    agent._last_user_message = 'status'
    agent._last_poll_time = time.time() - 60

    manager = MagicMock()
    manager.get_status.return_value = make_status('failed')
    agent._session_manager = manager

    state = make_state('status')
    result = agent.step(state)

    assert agent._coordination_mode is False
    assert agent._experiment_id is None
    assert 'Research ended before completion' in result.content


def test_agent_reset_clears_coordination_state(agent_config, llm_registry):
    agent = ScientificResearchAgent(agent_config, llm_registry)
    agent._coordination_mode = True
    agent._experiment_id = 'exp-1'
    agent._experiment_session_id = 'session-1'
    agent._last_user_message = 'status'
    agent._pending_control_results.append({'message': 'pending'})

    agent.reset()

    assert agent._coordination_mode is False
    assert agent._experiment_id is None
    assert agent._pending_control_results == []


def test_code_agent_uses_code_research_type(agent_config, llm_registry, monkeypatch):
    captured: Dict[str, Optional[str]] = {}

    async def fake_start(goal: str, session_id: str, research_type: str, config: Dict) -> str:
        captured['goal'] = goal
        captured['session_id'] = session_id
        captured['research_type'] = research_type
        return 'exp-code'

    monkeypatch.setattr(research_middleware, 'start_research', fake_start)

    agent = CodeResearchAgent(agent_config, llm_registry)
    state = make_state('Analyze code structure for this repository.')
    result = agent.step(state)

    assert agent._coordination_mode is True
    assert captured['research_type'] == 'code'
    assert 'Starting a deep repository analysis' in result.content


def test_full_coordination_workflow(agent_config, llm_registry):
    agent = ScientificResearchAgent(agent_config, llm_registry)
    agent._run_coroutine_sync = lambda coro: 'exp-42'

    state_start = make_state('Design an experiment to compare models.')
    start_action = agent.step(state_start)
    assert agent._coordination_mode is True
    assert 'Launching a scientific research experiment' in start_action.content

    manager = MagicMock()
    manager.get_status.side_effect = [
        make_status('running'),
        make_status('complete'),
    ]
    agent._session_manager = manager
    agent._experiment_session_id = 'session-1'
    agent._experiment_id = 'exp-42'
    agent._last_user_message = 'design request'
    agent._last_poll_time = time.time() - 60

    state_poll = make_state('design request')
    progress_action = agent.step(state_poll)
    assert 'Research Progress' in progress_action.content

    agent._last_user_message = None
    state_control = make_state('pause research')
    agent._execute_control_command = lambda action, target, state: {'status': 'success', 'message': 'Paused research'}
    control_action = agent.step(state_control)
    assert 'Paused research' in control_action.content

    agent._last_user_message = None
    state_complete = make_state("how's progress?")
    final_action = agent.step(state_complete)
    assert agent._coordination_mode is False
    assert 'Research completed' in final_action.content
