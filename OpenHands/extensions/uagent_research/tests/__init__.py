"""Integration test utilities for the UAgent research extension.

This package bundles a common toolbox that all integration suites can rely on.
The helpers defined here keep individual test modules focused on behaviour
rather than scaffolding, and they mirror the data structures used by the
extension in production.

Utilities provided:

``create_sample_hypothesis``
    Return a ready-to-use :class:`ResearchHypothesis` instance.

``create_sample_experiment_plan``
    Build an :class:`ExperimentPlan` linked to a hypothesis.

``create_sample_tree_node``
    Produce a :class:`ResearchNode` with sensible defaults for tree tests.

``create_sample_experiment``
    Construct an :class:`Experiment` ORM instance for database-oriented tests.

``assert_event_sequence``
    Validate that a sequence of events matches expected Python types.

``wait_for_condition`` / ``wait_for_status``
    Asynchronous polling helpers that simplify waiting for background tasks.

Example::

    from tests import create_sample_hypothesis, assert_event_sequence

    async def test_example(mock_event_stream):
        hypothesis = create_sample_hypothesis("Compare models")
        ...
        await mock_event_stream.wait_for_event()
        assert_event_sequence(mock_event_stream.get_events(), [MessageAction])

"""

from __future__ import annotations

import asyncio
from datetime import datetime
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable, Iterable, Sequence, Type

_EXT_ROOT = Path(__file__).resolve().parent.parent
if str(_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXT_ROOT))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PACKAGE_DIR = _EXT_ROOT / 'uagent_research'
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

try:  # pragma: no cover - optional when OpenHands is available
    import openhands.runtime as _openhands_runtime

    sys.modules.setdefault('openhands.runtime.runtime', _openhands_runtime)
except ImportError:  # pragma: no cover - only when OpenHands is absent
    pass

__all__ = [
    'create_sample_hypothesis',
    'create_sample_experiment_plan',
    'create_sample_tree_node',
    'create_sample_experiment',
    'assert_event_sequence',
    'wait_for_condition',
    'wait_for_status',
]


import types
from dataclasses import dataclass


def _ensure_package(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


if 'openhands' not in sys.modules:
    _ensure_package('openhands')
if 'openhands.llm' not in sys.modules:
    _ensure_package('openhands.llm')
if 'openhands.runtime' not in sys.modules:
    _ensure_package('openhands.runtime')
if 'openhands.events' not in sys.modules:
    _ensure_package('openhands.events')
if 'openhands.events.action' not in sys.modules:
    _ensure_package('openhands.events.action')
if 'openhands.events.observation' not in sys.modules:
    _ensure_package('openhands.events.observation')
if 'openhands.events.stream' not in sys.modules:
    _ensure_package('openhands.events.stream')

if 'openhands.llm.llm' not in sys.modules:
    llm_module = types.ModuleType('openhands.llm.llm')

    class LLM:  # pragma: no cover - stub for imports
        pass

    llm_module.LLM = LLM
    sys.modules['openhands.llm.llm'] = llm_module

if 'openhands.runtime.runtime' not in sys.modules:
    runtime_module = types.ModuleType('openhands.runtime.runtime')

    class Runtime:  # pragma: no cover - stub for imports
        async def run_action(self, action):  # noqa: ANN001
            raise NotImplementedError

    runtime_module.Runtime = Runtime
    sys.modules['openhands.runtime.runtime'] = runtime_module

action_module = sys.modules.get('openhands.events.action', types.ModuleType('openhands.events.action'))
if action_module.__name__ not in sys.modules:
    sys.modules['openhands.events.action'] = action_module

if not hasattr(action_module, 'CmdRunAction'):
    @dataclass
    class CmdRunAction:  # pragma: no cover
        command: str
        thought: str | None = None

    action_module.CmdRunAction = CmdRunAction

if not hasattr(action_module, 'MessageAction'):
    @dataclass
    class MessageAction:  # pragma: no cover
        content: str
        thought: str | None = None

    action_module.MessageAction = MessageAction

obs_module = sys.modules.get('openhands.events.observation', types.ModuleType('openhands.events.observation'))
if obs_module.__name__ not in sys.modules:
    sys.modules['openhands.events.observation'] = obs_module

if not hasattr(obs_module, 'CmdOutputObservation'):
    @dataclass
    class CmdOutputObservation:  # pragma: no cover
        content: str
        command: str
        exit_code: int = 0
        metadata: dict[str, Any] | None = None

    obs_module.CmdOutputObservation = CmdOutputObservation

stream_module = sys.modules.get('openhands.events.stream', types.ModuleType('openhands.events.stream'))
if stream_module.__name__ not in sys.modules:
    sys.modules['openhands.events.stream'] = stream_module

if not hasattr(stream_module, 'EventStream'):
    class EventStream:  # pragma: no cover
        pass

    stream_module.EventStream = EventStream

import importlib.util


def _load_module(name: str, location: Path):
    spec = importlib.util.spec_from_file_location(
        name, location, submodule_search_locations=[str(location.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module

# Load core extension modules so tests can import them without installation
_load_module('uagent_research', _PACKAGE_DIR / '__init__.py')
_load_module('uagent_research.models', _PACKAGE_DIR / 'models' / '__init__.py')
_load_module('uagent_research.models.base', _PACKAGE_DIR / 'models' / 'base.py')
_load_module('uagent_research.models.experiment', _PACKAGE_DIR / 'models' / 'experiment.py')
_load_module('uagent_research.models.research_session', _PACKAGE_DIR / 'models' / 'research_session.py')
_load_module('uagent_research.models.idea', _PACKAGE_DIR / 'models' / 'idea.py')
_load_module('uagent_research.models.hypothesis', _PACKAGE_DIR / 'models' / 'hypothesis.py')

def _import_scientific_module():
    return import_module('uagent_research.engines.scientific_research')


def _import_models_module():
    return import_module('uagent_research.models')


def _import_research_tree_module():
    return import_module('uagent_research.models.research_tree')


def create_sample_hypothesis(statement: str = 'Testable hypothesis') -> Any:
    """Return a minimal hypothesis object suitable for unit or integration tests."""

    module = _import_scientific_module()
    ResearchHypothesis = module.ResearchHypothesis
    HypothesisStatus = module.HypothesisStatus

    return ResearchHypothesis(
        id='hyp-test',
        statement=statement,
        reasoning='Because science demands evidence.',
        testable_predictions=['prediction 1', 'prediction 2'],
        success_criteria={'metric': 'accuracy', 'threshold': '>=0.9'},
        status=HypothesisStatus.PENDING,
    )


def create_sample_experiment_plan(
    hypothesis_id: str = 'hyp-test',
    *,
    code_to_execute: str | None = "print('running experiment')",
) -> Any:
    """Create a simple experiment plan attached to *hypothesis_id*."""

    ExperimentPlan = _import_scientific_module().ExperimentPlan

    return ExperimentPlan(
        id=f'plan-{hypothesis_id}',
        hypothesis_id=hypothesis_id,
        title='Evaluate sample experiment',
        description='Run a mocked experiment workflow.',
        methodology='1. Prepare data\n2. Run evaluation\n3. Collect results',
        expected_outcomes=['Outcome A', 'Outcome B'],
        code_to_execute=code_to_execute,
        data_requirements=['sample dataset'],
    )


def create_sample_tree_node(
    node_id: str = 'node-test',
    *,
    parent_id: str | None = 'root',
    node_type: Any | None = None,
    status: Any | None = None,
) -> Any:
    """Return a research tree node with representative defaults."""

    module = _import_research_tree_module()
    ResearchNode = module.ResearchNode
    NodeType = module.NodeType
    NodeStatus = module.NodeStatus

    node_type = node_type or NodeType.IDEA
    status = status or NodeStatus.PENDING

    return ResearchNode(
        id=node_id,
        type=node_type,
        title='Sample node',
        content='Investigate the sample hypothesis.',
        status=status,
        parent_id=parent_id,
    )


def create_sample_experiment(
    experiment_id: str = 'exp_test',
    *,
    session_id: str = 'session_test',
    goal: str = 'Run sample experiment',
    experiment_type: Any | None = None,
    status: Any | None = None,
) -> Any:
    """Instantiate an ORM experiment with baseline values."""

    models = _import_models_module()
    Experiment = models.Experiment
    ExperimentStatus = models.ExperimentStatus
    ExperimentType = models.ExperimentType

    experiment_type = experiment_type or ExperimentType.SCIENTIFIC
    status = status or ExperimentStatus.PENDING

    experiment = Experiment(
        id=experiment_id,
        session_id=session_id,
        experiment_type=experiment_type,
        goal=goal,
    )
    experiment.status = status
    experiment.created_at = datetime.utcnow()
    experiment.progress_percentage = 0.0
    experiment.steps_completed = 0
    experiment.total_steps = 0
    return experiment


def assert_event_sequence(events: Sequence[Any], expected_types: Iterable[Type[Any]]) -> None:
    """Assert that *events* align with *expected_types* one-to-one."""

    expected = list(expected_types)
    if len(events) != len(expected):
        raise AssertionError(
            f'Expected {len(expected)} events, received {len(events)}'
        )

    for index, (event, expected_type) in enumerate(zip(events, expected, strict=True)):
        if not isinstance(event, expected_type):
            raise AssertionError(
                f'Event #{index} expected {expected_type.__name__}, '
                f'got {type(event).__name__}'
            )


def _ensure_awaitable(result: Any) -> Awaitable[Any]:
    if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
        return result  # type: ignore[return-value]

    async def _wrapper() -> Any:
        return result

    return _wrapper()


def wait_for_condition(
    predicate: Callable[[], Awaitable[bool] | bool],
    *,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> Awaitable[None]:
    """Poll *predicate* until it returns ``True`` or *timeout* expires."""

    async def _wait() -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            result = await _ensure_awaitable(predicate())
            if result:
                return
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError('Condition not met within timeout window')
            await asyncio.sleep(interval)

    return _wait()


def wait_for_status(
    fetcher: Callable[[], Awaitable[Any] | Any],
    expected_status: Any,
    *,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> Awaitable[Any]:
    """Wait until *fetcher* yields *expected_status* and return the status."""

    async def _wait() -> Any:
        async def _predicate() -> bool:
            return await _ensure_awaitable(fetcher()) == expected_status

        await wait_for_condition(_predicate, timeout=timeout, interval=interval)
        return await _ensure_awaitable(fetcher())

    return _wait()

