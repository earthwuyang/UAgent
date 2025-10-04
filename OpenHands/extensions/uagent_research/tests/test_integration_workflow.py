"""
Integration tests for complete research workflows
"""

import asyncio
import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add extension to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from uagent_research.models.base import init_database, close_database, get_session
from uagent_research.models import (
    Experiment,
    ExperimentType,
    ExperimentStatus,
    ResearchSession,
    SessionMode,
    Idea,
    Hypothesis,
)
from sqlalchemy import select


@pytest.fixture
async def test_db():
    """Initialize test database"""
    await init_database("sqlite+aiosqlite:///:memory:")
    yield
    await close_database()


@pytest.mark.asyncio
async def test_complete_research_workflow(test_db):
    """Test complete research workflow from session creation to experiment completion"""
    print("\n=== Testing Complete Research Workflow ===\n")

    async for session in get_session():
        # Step 1: Create research session
        research_session = ResearchSession(
            id="session_test_1",
            user_id="user_123",
            mode=SessionMode.RESEARCH,
            title="Test Research: Algorithm Comparison",
            description="Testing complete workflow",
            tags=["test", "algorithms"],
        )
        session.add(research_session)
        await session.commit()
        await session.refresh(research_session)

        print(f"✅ Created research session: {research_session.id}")
        assert research_session.id == "session_test_1"
        assert research_session.mode == SessionMode.RESEARCH

        # Step 2: Create idea
        idea = Idea(
            id="idea_test_1",
            session_id=research_session.id,
            title="Compare sorting algorithms",
            description="Test quicksort vs mergesort performance",
            topic="Algorithms",
            novelty_score=0.7,
            feasibility_score=0.9,
            impact_score=0.6,
        )
        session.add(idea)
        await session.commit()
        await session.refresh(idea)

        print(f"✅ Created idea: {idea.title}")
        assert idea.session_id == research_session.id

        # Step 3: Create hypothesis
        hypothesis = Hypothesis(
            id="hyp_test_1",
            idea_id=idea.id,
            session_id=research_session.id,
            statement="Quicksort is faster than mergesort for random data",
            null_hypothesis="No significant difference in performance",
            testability_score=0.95,
        )
        session.add(hypothesis)
        await session.commit()
        await session.refresh(hypothesis)

        print(f"✅ Created hypothesis: {hypothesis.statement}")
        assert hypothesis.idea_id == idea.id
        assert not hypothesis.tested

        # Step 4: Create experiment
        experiment = Experiment(
            id="exp_test_1",
            session_id=research_session.id,
            experiment_type=ExperimentType.SCIENTIFIC,
            goal="Test sorting algorithm performance",
            status=ExperimentStatus.PENDING,
        )
        session.add(experiment)
        await session.commit()
        await session.refresh(experiment)

        print(f"✅ Created experiment: {experiment.id}")
        assert experiment.status == ExperimentStatus.PENDING

        # Step 5: Simulate experiment execution
        experiment.status = ExperimentStatus.RUNNING
        experiment.progress_percentage = 25.0
        experiment.current_step = "Generating test data"
        await session.commit()

        print(f"✅ Experiment running: {experiment.progress_percentage}%")

        # Step 6: Update progress
        experiment.progress_percentage = 50.0
        experiment.current_step = "Executing quicksort"
        await session.commit()

        print(f"✅ Progress updated: {experiment.progress_percentage}%")

        # Step 7: Complete experiment
        experiment.status = ExperimentStatus.COMPLETED
        experiment.progress_percentage = 100.0
        experiment.current_step = "Analysis complete"
        experiment.completed_at = datetime.utcnow()
        experiment.results = {
            "quicksort_time": 0.123,
            "mergesort_time": 0.145,
            "winner": "quicksort",
        }
        await session.commit()

        print(f"✅ Experiment completed")
        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.results is not None

        # Step 8: Mark hypothesis as tested
        hypothesis.tested = True
        hypothesis.test_results = experiment.results
        await session.commit()

        print(f"✅ Hypothesis tested")
        assert hypothesis.tested

        # Step 9: Verify all data
        result = await session.execute(
            select(Experiment).where(Experiment.session_id == research_session.id)
        )
        experiments = result.scalars().all()
        assert len(experiments) == 1

        result = await session.execute(
            select(Idea).where(Idea.session_id == research_session.id)
        )
        ideas = result.scalars().all()
        assert len(ideas) == 1

        result = await session.execute(
            select(Hypothesis).where(Hypothesis.session_id == research_session.id)
        )
        hypotheses = result.scalars().all()
        assert len(hypotheses) == 1

        print(f"\n✅ Complete workflow test passed!")
        print(f"   - Session created")
        print(f"   - Idea generated")
        print(f"   - Hypothesis formulated")
        print(f"   - Experiment executed")
        print(f"   - Results analyzed")


@pytest.mark.asyncio
async def test_multiple_experiments_in_session(test_db):
    """Test running multiple experiments in a single session"""
    print("\n=== Testing Multiple Experiments ===\n")

    async for session in get_session():
        # Create session
        research_session = ResearchSession(
            id="session_multi_1",
            user_id="user_123",
            mode=SessionMode.RESEARCH,
            title="Multi-experiment session",
        )
        session.add(research_session)

        # Create multiple experiments
        experiments = []
        for i in range(3):
            exp = Experiment(
                id=f"exp_multi_{i}",
                session_id=research_session.id,
                experiment_type=ExperimentType.SCIENTIFIC,
                goal=f"Test case {i}",
                status=ExperimentStatus.COMPLETED if i % 2 == 0 else ExperimentStatus.RUNNING,
            )
            session.add(exp)
            experiments.append(exp)

        await session.commit()

        # Query experiments
        result = await session.execute(
            select(Experiment).where(Experiment.session_id == research_session.id)
        )
        all_exps = result.scalars().all()

        assert len(all_exps) == 3
        print(f"✅ Created {len(all_exps)} experiments in session")

        # Query by status
        result = await session.execute(
            select(Experiment).where(
                Experiment.session_id == research_session.id,
                Experiment.status == ExperimentStatus.COMPLETED
            )
        )
        completed = result.scalars().all()

        result = await session.execute(
            select(Experiment).where(
                Experiment.session_id == research_session.id,
                Experiment.status == ExperimentStatus.RUNNING
            )
        )
        running = result.scalars().all()

        print(f"   - Completed: {len(completed)}")
        print(f"   - Running: {len(running)}")

        assert len(completed) == 2  # 0 and 2
        assert len(running) == 1    # 1

        print("✅ Multiple experiments test passed!")


@pytest.mark.asyncio
async def test_experiment_error_handling(test_db):
    """Test experiment error scenarios"""
    print("\n=== Testing Error Handling ===\n")

    async for session in get_session():
        # Create experiment
        experiment = Experiment(
            id="exp_error_1",
            session_id="session_123",
            experiment_type=ExperimentType.SCIENTIFIC,
            goal="Test error handling",
            status=ExperimentStatus.RUNNING,
        )
        session.add(experiment)
        await session.commit()

        # Simulate error
        experiment.status = ExperimentStatus.FAILED
        experiment.error_message = "Simulated error for testing"
        await session.commit()

        assert experiment.status == ExperimentStatus.FAILED
        assert experiment.error_message is not None
        print(f"✅ Error handling: {experiment.error_message}")

        # Verify can query failed experiments
        result = await session.execute(
            select(Experiment).where(Experiment.status == ExperimentStatus.FAILED)
        )
        failed = result.scalars().all()
        assert len(failed) == 1

        print("✅ Error handling test passed!")


@pytest.mark.asyncio
async def test_serialization(test_db):
    """Test model serialization"""
    print("\n=== Testing Serialization ===\n")

    async for session in get_session():
        # Create experiment
        experiment = Experiment(
            id="exp_serial_1",
            session_id="session_123",
            experiment_type=ExperimentType.SCIENTIFIC,
            goal="Test serialization",
            status=ExperimentStatus.COMPLETED,
        )
        experiment.results = {"test": "data", "value": 123}
        session.add(experiment)
        await session.commit()
        await session.refresh(experiment)

        # Test to_dict
        exp_dict = experiment.to_dict()

        assert exp_dict["id"] == "exp_serial_1"
        assert exp_dict["experiment_type"] == "scientific"
        assert exp_dict["status"] == "completed"
        assert exp_dict["results"]["test"] == "data"
        assert exp_dict["results"]["value"] == 123
        assert "created_at" in exp_dict
        assert "progress" in exp_dict

        print("✅ Serialization test passed!")
        print(f"   Serialized fields: {list(exp_dict.keys())}")


if __name__ == "__main__":
    async def run_all_tests():
        # Initialize database
        await init_database("sqlite+aiosqlite:///:memory:")

        print("╔════════════════════════════════════════════════════════════╗")
        print("║  Integration Workflow Tests                                ║")
        print("╚════════════════════════════════════════════════════════════╝")

        try:
            await test_complete_research_workflow(None)
            await test_multiple_experiments_in_session(None)
            await test_experiment_error_handling(None)
            await test_serialization(None)

            print("\n" + "="*60)
            print("✅ All integration tests passed!")
            print("="*60)

        finally:
            await close_database()

    asyncio.run(run_all_tests())
