"""
Basic Usage Examples for UAgent Research Extension

This demonstrates how to use the research extension with OpenHands.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base import init_database, get_session
from models import Experiment, ExperimentType, ExperimentStatus


async def example_1_create_experiment():
    """Example 1: Create an experiment in the database"""
    print("\n=== Example 1: Create Experiment ===\n")

    # Initialize database (in-memory for example)
    await init_database("sqlite+aiosqlite:///:memory:")

    # Create experiment
    async for session in get_session():
        experiment = Experiment(
            id="exp_example_1",
            session_id="session_123",
            experiment_type=ExperimentType.SCIENTIFIC,
            goal="Compare sorting algorithm performance",
            status=ExperimentStatus.PENDING,
        )

        session.add(experiment)
        await session.commit()
        await session.refresh(experiment)

        print(f"Created experiment: {experiment.id}")
        print(f"Status: {experiment.status.value}")
        print(f"Goal: {experiment.goal}")

        # Convert to dict for API response
        exp_dict = experiment.to_dict()
        print(f"\nAs JSON:")
        import json
        print(json.dumps(exp_dict, indent=2))


async def example_2_update_progress():
    """Example 2: Update experiment progress"""
    print("\n=== Example 2: Update Progress ===\n")

    await init_database("sqlite+aiosqlite:///:memory:")

    async for session in get_session():
        # Create experiment
        experiment = Experiment(
            id="exp_example_2",
            session_id="session_123",
            experiment_type=ExperimentType.SCIENTIFIC,
            goal="Test hypothesis about algorithm performance",
            status=ExperimentStatus.PENDING,
        )
        session.add(experiment)
        await session.commit()

        # Simulate progress updates
        steps = [
            ("Generating hypotheses", 20),
            ("Designing experiments", 40),
            ("Executing experiments", 60),
            ("Analyzing results", 80),
            ("Completed", 100),
        ]

        for step, percentage in steps:
            experiment.current_step = step
            experiment.progress_percentage = percentage

            if percentage == 100:
                experiment.status = ExperimentStatus.COMPLETED
                from datetime import datetime
                experiment.completed_at = datetime.utcnow()

            await session.commit()
            print(f"Progress: {percentage}% - {step}")


async def example_3_list_experiments():
    """Example 3: List experiments with filters"""
    print("\n=== Example 3: List Experiments ===\n")

    await init_database("sqlite+aiosqlite:///:memory:")

    async for session in get_session():
        # Create multiple experiments
        experiments = [
            Experiment(
                id=f"exp_{i}",
                session_id="session_123",
                experiment_type=ExperimentType.SCIENTIFIC,
                goal=f"Experiment {i}",
                status=ExperimentStatus.COMPLETED if i % 2 == 0 else ExperimentStatus.RUNNING,
            )
            for i in range(5)
        ]

        for exp in experiments:
            session.add(exp)
        await session.commit()

        # Query experiments
        from sqlalchemy import select

        # Get all experiments
        result = await session.execute(select(Experiment))
        all_exps = result.scalars().all()
        print(f"Total experiments: {len(all_exps)}")

        # Get completed experiments
        result = await session.execute(
            select(Experiment).where(Experiment.status == ExperimentStatus.COMPLETED)
        )
        completed_exps = result.scalars().all()
        print(f"Completed experiments: {len(completed_exps)}")

        # Get running experiments
        result = await session.execute(
            select(Experiment).where(Experiment.status == ExperimentStatus.RUNNING)
        )
        running_exps = result.scalars().all()
        print(f"Running experiments: {len(running_exps)}")


async def example_4_research_session():
    """Example 4: Create research session"""
    print("\n=== Example 4: Research Session ===\n")

    await init_database("sqlite+aiosqlite:///:memory:")

    from models import ResearchSession, SessionMode

    async for session in get_session():
        research_session = ResearchSession(
            id="session_123",
            user_id="user_1",
            mode=SessionMode.RESEARCH,
            title="ML Algorithm Comparison Study",
            description="Comparing various ML algorithms for classification",
            tags=["machine-learning", "algorithms", "comparison"],
        )

        session.add(research_session)
        await session.commit()
        await session.refresh(research_session)

        print(f"Created session: {research_session.id}")
        print(f"Title: {research_session.title}")
        print(f"Mode: {research_session.mode.value}")
        print(f"Tags: {research_session.tags}")


async def example_5_ideas_and_hypotheses():
    """Example 5: Create ideas and hypotheses"""
    print("\n=== Example 5: Ideas and Hypotheses ===\n")

    await init_database("sqlite+aiosqlite:///:memory:")

    from models import Idea, Hypothesis

    async for session in get_session():
        # Create idea
        idea = Idea(
            id="idea_1",
            session_id="session_123",
            title="Use ML to predict optimal compiler flags",
            description="Train an ML model to predict which compiler flags will optimize performance",
            topic="Compiler Optimization",
            novelty_score=0.85,
            feasibility_score=0.70,
            impact_score=0.90,
            tags=["ml", "compilers", "optimization"],
        )
        session.add(idea)

        # Create hypothesis
        hypothesis = Hypothesis(
            id="hyp_1",
            idea_id="idea_1",
            session_id="session_123",
            statement="ML can predict optimal compiler flags with >80% accuracy",
            null_hypothesis="ML predictions are no better than random selection",
            testability_score=0.95,
        )
        session.add(hypothesis)

        await session.commit()

        print(f"Created idea: {idea.title}")
        print(f"  Novelty: {idea.novelty_score}")
        print(f"  Feasibility: {idea.feasibility_score}")
        print(f"  Impact: {idea.impact_score}")

        print(f"\nCreated hypothesis: {hypothesis.statement}")
        print(f"  Testability: {hypothesis.testability_score}")
        print(f"  Tested: {hypothesis.tested}")


async def main():
    """Run all examples"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  UAgent Research Extension - Usage Examples                ║")
    print("╚════════════════════════════════════════════════════════════╝")

    await example_1_create_experiment()
    await example_2_update_progress()
    await example_3_list_experiments()
    await example_4_research_session()
    await example_5_ideas_and_hypotheses()

    print("\n✅ All examples completed successfully!\n")


if __name__ == "__main__":
    asyncio.run(main())
