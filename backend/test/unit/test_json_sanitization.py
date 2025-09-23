"""Unit tests for JSON sanitization in research engines."""

import pytest

from app.core.research_engines.scientific_research import (
    HypothesisGenerator,
    ExperimentDesigner,
    ResearchHypothesis,
)


class _StubLLM:
    """Minimal async LLM stub that returns canned responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    async def generate(self, *_, **__):
        if not self._responses:
            raise AssertionError("Stub LLM was called more times than responses provided")
        idx = min(self._index, len(self._responses) - 1)
        self._index += 1
        return self._responses[idx]


@pytest.mark.asyncio
async def test_hypothesis_generation_handles_wrapped_json():
    """HypothesisGenerator should parse JSON with hard line wraps inside values."""

    raw_response = (
        "[\n"
        "  {\n"
        "    \"statement\": \"Zero-copy feature extraction keeps overhead low even when the response\n"
        "router prints hard-wrapped lines\",\n"
        "    \"reasoning\": \"Literal newlines inside a JSON string should not break parsing.",\n"
        "    \"testable_predictions\": [\n"
        "      \"Profiling shows ≤ 1 % CPU overhead\"\n"
        "    ],\n"
        "    \"success_criteria\": {\"overhead_pct\": \"≤ 1\"},\n"
        "    \"variables\": {\"independent\": [\"zcfe_enabled\"], \"dependent\": [\"cpu_overhead\"]}\n"
        "  }\n"
        "]"
    )

    generator = HypothesisGenerator(_StubLLM([raw_response]))
    hypotheses = await generator.generate_hypotheses("Zero-copy data capture?", None, None)

    assert len(hypotheses) == 1
    assert "Zero-copy feature extraction" in hypotheses[0].statement
    assert hypotheses[0].testable_predictions == ["Profiling shows ≤ 1 % CPU overhead"]


@pytest.mark.asyncio
async def test_experiment_design_handles_wrapped_json():
    """ExperimentDesigner should parse JSON with wrapped strings."""

    wrapped_json = (
        "{\n"
        "  \"name\": \"Latency measurement plan\",\n"
        "  \"description\": \"Design verifies that instrumentation\n"
        "can tolerate wrapped output.\",\n"
        "  \"methodology\": \"Collect latencies and compare to baseline\",\n"
        "  \"variables\": {\"independent\": [\"mode\"], \"dependent\": [\"latency_ms\"]},\n"
        "  \"controls\": [],\n"
        "  \"data_collection_plan\": {\"samples\": 10},\n"
        "  \"analysis_plan\": \"Compute mean latency\",\n"
        "  \"expected_duration\": \"5m\",\n"
        "  \"resource_requirements\": {\"cpu\": \"2 cores\"},\n"
        "  \"code_requirements\": [],\n"
        "  \"dependencies\": []\n"
        "}"\n"
    )

    designer = ExperimentDesigner(_StubLLM([wrapped_json]))
    hypothesis = ResearchHypothesis(
        id="hyp-001",
        statement="Test",
        reasoning="",
        testable_predictions=[],
        success_criteria={},
        variables={},
    )

    design = await designer.design_experiment(hypothesis)

    assert design.name == "Latency measurement plan"
    assert "Instrumentation" in design.description or "instrumentation" in design.description.lower()
    assert design.methodology.startswith("Collect latencies")
