"""
Tests for data models
"""

import pytest
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from uagent_researchmodels import (
    Base,
    Experiment,
    ExperimentStatus,
    ExperimentType,
    ResearchSession,
    SessionMode,
    Idea,
    Hypothesis,
)


@pytest.fixture
async def async_session():
    """Create async test database session"""
    # Use in-memory SQLite for tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session factory
    async_session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_maker() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_create_experiment(async_session):
    """Test creating experiment"""
    experiment = Experiment(
        id="test_exp_1",
        session_id="test_session_1",
        experiment_type=ExperimentType.SCIENTIFIC,
        goal="Test experiment goal",
        status=ExperimentStatus.PENDING,
    )

    async_session.add(experiment)
    await async_session.commit()

    assert experiment.id == "test_exp_1"
    assert experiment.status == ExperimentStatus.PENDING
    assert experiment.progress_percentage == 0.0


@pytest.mark.asyncio
async def test_experiment_to_dict(async_session):
    """Test experiment serialization"""
    experiment = Experiment(
        id="test_exp_2",
        session_id="test_session_1",
        experiment_type=ExperimentType.CODE,
        goal="Code analysis",
        status=ExperimentStatus.RUNNING,
        progress_percentage=50.0,
        current_step="Analyzing files",
    )

    async_session.add(experiment)
    await async_session.commit()

    exp_dict = experiment.to_dict()

    assert exp_dict['id'] == "test_exp_2"
    assert exp_dict['experiment_type'] == "code"
    assert exp_dict['status'] == "running"
    assert exp_dict['progress']['percentage'] == 50.0
    assert exp_dict['progress']['current_step'] == "Analyzing files"


@pytest.mark.asyncio
async def test_create_research_session(async_session):
    """Test creating research session"""
    session_obj = ResearchSession(
        id="test_session_1",
        user_id="user_1",
        mode=SessionMode.RESEARCH,
        title="Test Research Session",
    )

    async_session.add(session_obj)
    await async_session.commit()

    assert session_obj.id == "test_session_1"
    assert session_obj.mode == SessionMode.RESEARCH


@pytest.mark.asyncio
async def test_create_idea(async_session):
    """Test creating idea"""
    idea = Idea(
        id="idea_1",
        session_id="test_session_1",
        title="Test Idea",
        description="Test idea description",
        topic="Machine Learning",
        novelty_score=0.8,
        feasibility_score=0.7,
        impact_score=0.9,
    )

    async_session.add(idea)
    await async_session.commit()

    assert idea.id == "idea_1"
    assert idea.novelty_score == 0.8


@pytest.mark.asyncio
async def test_create_hypothesis(async_session):
    """Test creating hypothesis"""
    hypothesis = Hypothesis(
        id="hyp_1",
        session_id="test_session_1",
        statement="Test hypothesis statement",
        null_hypothesis="Null hypothesis",
        testability_score=0.85,
        tested=False,
    )

    async_session.add(hypothesis)
    await async_session.commit()

    assert hypothesis.id == "hyp_1"
    assert hypothesis.tested is False
    assert hypothesis.testability_score == 0.85


@pytest.mark.asyncio
async def test_experiment_status_transitions(async_session):
    """Test experiment status transitions"""
    experiment = Experiment(
        id="test_exp_3",
        session_id="test_session_1",
        experiment_type=ExperimentType.SCIENTIFIC,
        goal="Test transitions",
        status=ExperimentStatus.PENDING,
    )

    async_session.add(experiment)
    await async_session.commit()

    # Transition to running
    experiment.status = ExperimentStatus.RUNNING
    experiment.started_at = datetime.utcnow()
    await async_session.commit()

    assert experiment.status == ExperimentStatus.RUNNING
    assert experiment.started_at is not None

    # Transition to completed
    experiment.status = ExperimentStatus.COMPLETED
    experiment.completed_at = datetime.utcnow()
    experiment.progress_percentage = 100.0
    await async_session.commit()

    assert experiment.status == ExperimentStatus.COMPLETED
    assert experiment.completed_at is not None
    assert experiment.progress_percentage == 100.0
