"""Integration tests for research engines."""

from __future__ import annotations

import json
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PACKAGE_DIR = _PACKAGE_ROOT / 'uagent_research'
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

try:
    import openhands.runtime as _openhands_runtime
    sys.modules.setdefault('openhands.runtime.runtime', _openhands_runtime)
except ImportError:  # pragma: no cover
    pass

import pytest

from openhands.events.action import MessageAction

from . import (
    assert_event_sequence,
    create_sample_experiment_plan,
    create_sample_hypothesis,
)

import importlib.util

def _load_engine_module(module_name: str, relative_name: str):
    module_path = _PACKAGE_DIR / 'engines' / relative_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module

scientific_module = _load_engine_module(
    'uagent_research.engines.scientific_research',
    'scientific_research.py',
)
code_module = _load_engine_module(
    'uagent_research.engines.code_research',
    'code_research.py',
)

ScientificResearchEngine = scientific_module.ScientificResearchEngine
CodeResearchEngine = code_module.CodeResearchEngine


@pytest.mark.asyncio
async def test_scientific_engine_full_workflow(mock_llm, mock_runtime, mock_event_stream):
    engine = ScientificResearchEngine(
        llm=mock_llm,
        config={'validation_strict': True, 'simulation_detection': True},
    )

    mock_llm.set_responses(
        {
            'hypotheses': [
                {
                    'id': 'hyp1',
                    'statement': 'Algorithm A outperforms B',
                    'reasoning': 'A uses better heuristics',
                    'testable_predictions': ['A is faster', 'A uses less CPU'],
                    'success_criteria': {'metric': 'speed', 'threshold': '>=1.1x'},
                }
            ]
        },
        {
            'title': 'Benchmark algorithms',
            'description': 'Run performance benchmarks',
            'methodology': 'Execute benchmark script',
            'code_to_execute': 'print("benchmark completed")',
            'expected_outcomes': ['A faster than B'],
            'data_requirements': ['benchmark data'],
        },
        {
            'hypothesis_evaluations': [
                {
                    'hypothesis_id': 'hyp1',
                    'status': 'supported',
                    'evidence': ['Benchmark showed improvement'],
                    'confidence': 0.9,
                    'recommendation': 'Adopt algorithm A',
                }
            ],
            'overall_findings': 'Algorithm A is superior',
            'limitations': [],
        },
    )

    result = await engine.run_experiment(
        goal='Compare algorithm A and B',
        runtime=mock_runtime,
        event_stream=mock_event_stream,
        session_id='session-1',
    )

    assert result['status'] == 'completed'
    assert len(result['hypotheses']) == 1
    assert mock_runtime.executed_actions, 'Experiment should execute code via runtime'

    events = mock_event_stream.get_events()
    assert events, 'Progress events should be emitted'
    payloads = [json.loads(event.content) for event in events if isinstance(event, MessageAction)]
    assert any(evt['type'] == 'experiment_started' for evt in payloads)
    assert any(evt['type'] == 'experiment_completed' for evt in payloads)


@pytest.mark.asyncio
async def test_scientific_engine_hypothesis_generation(mock_llm):
    engine = ScientificResearchEngine(llm=mock_llm)
    mock_llm.set_next_response(
        {
            'hypotheses': [
                {
                    'id': 'hypA',
                    'statement': 'Hypothesis A',
                    'reasoning': 'Because science',
                    'testable_predictions': ['prediction'],
                    'success_criteria': {'metric': 'accuracy', 'threshold': '>=0.8'},
                },
                {
                    'id': 'hypB',
                    'statement': 'Hypothesis B',
                    'reasoning': 'Because curiosity',
                    'testable_predictions': ['prediction'],
                    'success_criteria': {'metric': 'precision', 'threshold': '>=0.7'},
                },
            ]
        }
    )

    hypotheses = await engine._generate_hypotheses('Investigate behaviour')
    assert len(hypotheses) == 2
    assert hypotheses[0].statement == 'Hypothesis A'


@pytest.mark.asyncio
async def test_scientific_engine_experiment_design(mock_llm):
    engine = ScientificResearchEngine(llm=mock_llm)
    hypothesis = create_sample_hypothesis('Test impact')

    mock_llm.set_next_response(
        {
            'title': 'Test impact',
            'description': 'Execute sample experiment',
            'methodology': 'step 1 -> step 2',
            'code_to_execute': 'print("run")',
            'expected_outcomes': ['observed impact'],
            'data_requirements': [],
        }
    )

    plans = await engine._design_experiments([hypothesis])
    assert len(plans) == 1
    assert plans[0].hypothesis_id == hypothesis.id


@pytest.mark.asyncio
async def test_scientific_engine_execution(mock_runtime, mock_event_stream):
    engine = ScientificResearchEngine(llm=None)  # LLM unused in execution phase
    plan = create_sample_experiment_plan(code_to_execute='print("success")')

    command = "python3 -c 'print(\"success\")'"
    mock_runtime.set_command_output(command, content='success\n', exit_code=0)

    results = await engine._execute_experiments(
        [plan],
        runtime=mock_runtime,
        event_stream=mock_event_stream,
        experiment_id='exp-42',
    )

    assert results[0]['success'] is True

    events = mock_event_stream.get_events()
    assert_event_sequence(events, [MessageAction, MessageAction])
    payloads = [json.loads(event.content) for event in events]
    assert payloads[0]['type'] == 'step_started'
    assert payloads[1]['type'] == 'step_completed'


@pytest.mark.asyncio
async def test_scientific_engine_result_analysis(mock_llm):
    engine = ScientificResearchEngine(llm=mock_llm)
    hypothesis = create_sample_hypothesis('Effect size')
    results = [{'plan_id': 'plan_hyp-test', 'success': True, 'output': 'data'}]

    mock_llm.set_next_response(
        {
            'hypothesis_evaluations': [
                {
                    'hypothesis_id': hypothesis.id,
                    'status': 'supported',
                    'evidence': ['data indicates effect'],
                    'confidence': 0.95,
                    'recommendation': 'Publish',
                }
            ],
            'overall_findings': 'Effect confirmed',
            'limitations': [],
        }
    )

    analysis = await engine._analyze_results([hypothesis], results)
    assert analysis['overall_findings'] == 'Effect confirmed'
    assert hypothesis.status.name == 'SUPPORTED'


@pytest.mark.asyncio
async def test_scientific_engine_validation_anti_simulation():
    engine = ScientificResearchEngine(llm=None, config={'simulation_detection': True})
    valid = await engine._validate_findings({'overall_findings': 'Real data'})
    invalid = await engine._validate_findings({'overall_findings': 'Simulated mock data'})

    assert valid is True
    assert invalid is False


@pytest.mark.asyncio
async def test_scientific_engine_runtime_failure(mock_llm, mock_runtime, mock_event_stream):
    engine = ScientificResearchEngine(llm=mock_llm)
    mock_llm.set_responses(
        {
            'hypotheses': [
                {
                    'id': 'hyp1',
                    'statement': 'Hypothesis',
                    'reasoning': 'Reason',
                    'testable_predictions': ['prediction'],
                    'success_criteria': {'metric': 'metric', 'threshold': '1'},
                }
            ]
        },
        {
            'title': 'Failing experiment',
            'description': 'This will fail',
            'methodology': 'Attempt execution',
            'code_to_execute': 'print("ok")',
            'expected_outcomes': ['failure'],
            'data_requirements': [],
        }
    )

    async def failing_run_action(action):  # noqa: ANN001
        raise RuntimeError('execution failed')

    mock_runtime.run_action = failing_run_action  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        await engine.run_experiment(
            goal='Trigger failure',
            runtime=mock_runtime,
            event_stream=mock_event_stream,
            session_id='session-fail',
        )

    events = mock_event_stream.get_events()
    assert events
    payload = json.loads(events[-1].content)
    assert payload['type'] == 'experiment_failed'
    assert payload['error']['type'] == 'RuntimeError'


# ---------------------------------------------------------------------------
# Code research engine tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_engine_full_workflow(tmp_path, mock_llm, mock_runtime):
    engine = CodeResearchEngine(llm=mock_llm)
    workspace = tmp_path

    (workspace / 'auth.py').write_text('def authenticate(user):\n    return True')
    (workspace / 'app.py').write_text('def main():\n    pass')

    files_command = f"find {workspace} -type f -name '*.py' | head -100"
    dirs_command = f"find {workspace} -maxdepth 3 -type d"
    mock_runtime.set_command_output(files_command, content='\n'.join(str(workspace / n) for n in ['auth.py', 'app.py']))
    mock_runtime.set_command_output(dirs_command, content=str(workspace))

    mock_llm.set_responses(
        ["authentication", "login"],
        {
            'summary': 'Auth module',
            'key_components': ['authenticate'],
            'relevance': 'Handles authentication',
        },
        {
            'summary': 'App module',
            'key_components': ['main'],
            'relevance': 'Entrypoint',
        },
        'Authentication uses token-based checks.',
    )

    grep_auth = f"grep -r -l 'authentication' {workspace} --include='*.py' 2>/dev/null | head -10"
    grep_login = f"grep -r -l 'login' {workspace} --include='*.py' 2>/dev/null | head -10"
    mock_runtime.set_command_output(grep_auth, content=str(workspace / 'auth.py'))
    mock_runtime.set_command_output(grep_login, content=str(workspace / 'auth.py'))

    mock_runtime.set_command_output(f"cat {workspace / 'auth.py'}", content='def authenticate(user):\n    return True')
    mock_runtime.set_command_output(f"cat {workspace / 'app.py'}", content='def main():\n    pass')

    result = await engine.analyze_repository(
        query='How does authentication work?',
        workspace=str(workspace),
        runtime=mock_runtime,
    )

    assert result['structure']['file_count'] == 2
    assert result['relevant_files'], 'Should identify relevant files'
    assert 'Authentication uses token-based checks.' in result['answer']


@pytest.mark.asyncio
async def test_code_engine_structure_analysis(tmp_path, mock_runtime):
    engine = CodeResearchEngine(llm=None)
    workspace = tmp_path

    files_command = f"find {workspace} -type f -name '*.py' | head -100"
    dirs_command = f"find {workspace} -maxdepth 3 -type d"
    mock_runtime.set_command_output(files_command, content=str(workspace / 'mod.py'))
    mock_runtime.set_command_output(dirs_command, content=str(workspace))

    structure = await engine._analyze_structure(Path(workspace), mock_runtime)
    assert structure['file_count'] == 1
    assert structure['directory_count'] == 1


@pytest.mark.asyncio
async def test_code_engine_file_search(tmp_path, mock_llm, mock_runtime):
    engine = CodeResearchEngine(llm=mock_llm)
    workspace = Path(tmp_path)
    structure = {
        'python_files': [str(workspace / 'auth.py')],
        'directories': [str(workspace)],
        'file_count': 1,
        'directory_count': 1,
    }

    mock_llm.set_next_response(["auth", "token"])

    grep_auth = f"grep -r -l 'auth' {workspace} --include='*.py' 2>/dev/null | head -10"
    mock_runtime.set_command_output(grep_auth, content=str(workspace / 'auth.py'))
    mock_runtime.set_command_output(
        f"grep -r -l 'token' {workspace} --include='*.py' 2>/dev/null | head -10",
        content='',
    )

    relevant_files = await engine._find_relevant_files(
        'Explain authentication',
        structure,
        workspace,
        mock_runtime,
    )

    assert relevant_files
    assert relevant_files[0]['keyword'] == 'auth'


@pytest.mark.asyncio
async def test_code_engine_code_analysis(tmp_path, mock_llm, mock_runtime):
    engine = CodeResearchEngine(llm=mock_llm)
    workspace = Path(tmp_path)
    file_path = workspace / 'auth.py'
    relevant_files = [{'path': str(file_path), 'keyword': 'auth', 'relevance_score': 1.0}]

    mock_runtime.set_command_output(f"cat {file_path}", content='def authenticate():\n    pass')
    mock_llm.set_next_response(
        {
            'summary': 'Auth helper',
            'key_components': ['authenticate'],
            'relevance': 'Handles auth',
        }
    )

    analysis = await engine._analyze_code('Explain auth', relevant_files, workspace, mock_runtime)
    assert analysis['total_analyzed'] == 1
    assert analysis['analyzed_files'][0]['analysis']['summary'] == 'Auth helper'


@pytest.mark.asyncio
async def test_code_engine_answer_synthesis(mock_llm):
    engine = CodeResearchEngine(llm=mock_llm)
    mock_llm.set_next_response('Comprehensive answer about the codebase.')

    answer = await engine._synthesize_answer(
        query='How does auth work?',
        structure={'file_count': 2, 'directory_count': 1},
        relevant_files=[{'path': 'auth.py'}],
        code_analysis={'analyzed_files': [{'file': 'auth.py', 'analysis': {}}]},
    )

    assert 'Comprehensive answer' in answer


@pytest.mark.asyncio
async def test_code_engine_error_handling(tmp_path, mock_llm, mock_runtime):
    engine = CodeResearchEngine(llm=mock_llm)
    workspace = tmp_path

    (workspace / 'empty.py').write_text('')
    files_command = f"find {workspace} -type f -name '*.py' | head -100"
    dirs_command = f"find {workspace} -maxdepth 3 -type d"
    mock_runtime.set_command_output(files_command, content=str(workspace / 'empty.py'))
    mock_runtime.set_command_output(dirs_command, content=str(workspace))

    mock_llm.set_responses(
        ['keyword'],
        {
            'summary': 'Empty file',
            'key_components': [],
            'relevance': 'Minimal',
        },
        'Summary answer',
    )

    mock_runtime.set_command_output(f"grep -r -l 'keyword' {workspace} --include='*.py' 2>/dev/null | head -10", content=str(workspace / 'empty.py'))
    mock_runtime.set_command_output(f"cat {workspace / 'empty.py'}", content='')

    result = await engine.analyze_repository(
        query='Any info?',
        workspace=str(workspace),
        runtime=mock_runtime,
    )

    assert result['answer'] == 'Summary answer'
