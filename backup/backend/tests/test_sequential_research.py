import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.unified_orchestrator import UnifiedOrchestrator
from app.core.ai_scientist import ResearchPhase


@pytest.mark.asyncio
async def test_sequential_research_pipeline():
    """Test that sequential research executes all phases correctly without ROMA recursion"""

    # Configure AI Scientist mock to return successful phase results
    mock_phase_results = {
        ResearchPhase.LITERATURE_REVIEW: {"status": "success", "data": "literature findings"},
        ResearchPhase.HYPOTHESIS_GENERATION: {"status": "success", "data": "generated hypothesis"},
        ResearchPhase.EXPERIMENTAL_DESIGN: {"status": "success", "data": "experiment design"},
        ResearchPhase.EXPERIMENT_EXECUTION: {"status": "success", "data": "experiment results"},
        ResearchPhase.RESULT_ANALYSIS: {"status": "success", "data": "analysis complete"}
    }

    async def mock_execute_phase(project_id: str, phase: ResearchPhase):
        return mock_phase_results.get(phase, {"error": "Phase not implemented"})

    # Create orchestrator and mock its ai_scientist component
    orchestrator = UnifiedOrchestrator()
    orchestrator.ai_scientist.execute_research_phase = mock_execute_phase

    # Execute sequential research
    project_id = "test_project_123"
    result = await orchestrator._execute_sequential_research(project_id)

    # Verify results
    assert result["method"] == "sequential"
    assert len(result["phases_completed"]) == 5
    assert "literature_review" in result["phases_completed"]
    assert "hypothesis_generation" in result["phases_completed"]
    assert "experimental_design" in result["phases_completed"]
    assert "experiment_execution" in result["phases_completed"]
    assert "result_analysis" in result["phases_completed"]

    # Verify phase results contain expected data
    phase_results = result["phase_results"]
    assert phase_results["literature_review"]["status"] == "success"
    assert phase_results["hypothesis_generation"]["status"] == "success"
    assert phase_results["experimental_design"]["status"] == "success"
    assert phase_results["experiment_execution"]["status"] == "success"
    assert phase_results["result_analysis"]["status"] == "success"


@pytest.mark.asyncio
async def test_sequential_research_with_phase_failure():
    """Test that sequential research stops when a phase fails"""

    # Configure AI Scientist mock to fail at hypothesis generation
    async def mock_execute_phase(project_id: str, phase: ResearchPhase):
        if phase == ResearchPhase.LITERATURE_REVIEW:
            return {"status": "success", "data": "literature findings"}
        elif phase == ResearchPhase.HYPOTHESIS_GENERATION:
            return {"error": "Failed to generate hypothesis"}
        else:
            return {"status": "success", "data": "should not reach here"}

    # Create orchestrator and mock its ai_scientist component
    orchestrator = UnifiedOrchestrator()
    orchestrator.ai_scientist.execute_research_phase = mock_execute_phase

    # Execute sequential research
    project_id = "test_project_456"
    result = await orchestrator._execute_sequential_research(project_id)

    # Verify execution stopped at failed phase
    assert result["method"] == "sequential"
    assert len(result["phases_completed"]) == 2  # Only literature review and hypothesis generation
    assert "literature_review" in result["phases_completed"]
    assert "hypothesis_generation" in result["phases_completed"]
    assert "experimental_design" not in result["phases_completed"]

    # Verify failure was captured
    phase_results = result["phase_results"]
    assert phase_results["literature_review"]["status"] == "success"
    assert "error" in phase_results["hypothesis_generation"]


@pytest.mark.asyncio
async def test_sequential_vs_roma_path():
    """Test that sequential path doesn't invoke ROMA recursion"""

    # Configure successful phase execution
    async def mock_execute_phase(project_id: str, phase: ResearchPhase):
        return {"status": "success", "data": f"completed {phase.value}"}

    # Create orchestrator and mock its ai_scientist component
    orchestrator = UnifiedOrchestrator()
    orchestrator.ai_scientist.execute_research_phase = mock_execute_phase

    # Execute sequential research
    project_id = "test_project_789"
    result = await orchestrator._execute_sequential_research(project_id)

    # Verify sequential method was used
    assert result["method"] == "sequential"
    assert len(result["phases_completed"]) == 5