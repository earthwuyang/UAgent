# Implementation Roadmap: Plugin/Extension Model

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 0: Environment Setup](#phase-0-environment-setup)
4. [Phase 1: Core Extension Structure](#phase-1-core-extension-structure)
5. [Phase 2: Backend Integration](#phase-2-backend-integration)
6. [Phase 3: Frontend Integration](#phase-3-frontend-integration)
7. [Phase 4: Testing & Validation](#phase-4-testing--validation)
8. [Phase 5: Deployment](#phase-5-deployment)
9. [Rollback Plan](#rollback-plan)

---

## Overview

This roadmap provides step-by-step implementation instructions for integrating UAgent into OpenHands using the **Plugin/Extension Model** (Option A from Integration Approaches).

**Timeline**: 15-20 weeks
**Team Size**: 2-3 developers
**Risk Level**: Low to Medium

---

## Prerequisites

### Required Knowledge
- [ ] Familiarity with OpenHands architecture
- [ ] Understanding of UAgent research engines
- [ ] Python async/await patterns
- [ ] React/TypeScript for frontend
- [ ] Docker and container orchestration

### Required Access
- [ ] Write access to OpenHands repository (fork)
- [ ] UAgent source code access
- [ ] Development environment setup
- [ ] Test infrastructure access

### Required Tools
```bash
# Development tools
- Python 3.11+
- Node.js 18+
- Docker 24+
- Git 2.40+

# OpenHands dependencies
pip install openhands>=0.9.0

# UAgent dependencies
pip install -r requirements.txt
```

---

## Phase 0: Environment Setup

### Week 1: Development Environment

#### Step 1: Fork OpenHands

```bash
# Clone OpenHands
git clone https://github.com/All-Hands-AI/OpenHands.git
cd OpenHands

# Create feature branch
git checkout -b feature/uagent-extension

# Set up remote for UAgent integration
git remote add uagent https://github.com/yourusername/UAgent.git
```

#### Step 2: Create Extension Directory Structure

```bash
# Create extension directory
mkdir -p extensions/uagent_research

# Initialize Python package
cat > extensions/uagent_research/__init__.py <<EOF
"""
UAgent Research Extension for OpenHands

Provides advanced research capabilities including:
- Scientific research with experiment execution
- Code repository analysis (RepoMaster)
- ROMA tree-based research orchestration
- Idea and hypothesis generation (AI Scientist style)
"""

__version__ = "0.1.0"
__author__ = "UAgent Team"
__license__ = "MIT"
EOF

# Create subdirectories
mkdir -p extensions/uagent_research/{agents,engines,runtime,ui,api,models,utils}

# Create __init__.py for each subdirectory
for dir in agents engines runtime ui api models utils; do
    touch extensions/uagent_research/$dir/__init__.py
done
```

#### Step 3: Set Up Extension Configuration

```bash
# Create setup.py for extension
cat > extensions/uagent_research/setup.py <<EOF
from setuptools import setup, find_packages

setup(
    name="uagent-research-extension",
    version="0.1.0",
    description="Advanced research capabilities for OpenHands",
    packages=find_packages(),
    install_requires=[
        "openhands>=0.9.0",
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "aiohttp>=3.8.0",
        # UAgent-specific dependencies
        "arxiv>=2.0.0",
        "playwright>=1.40.0",
        "beautifulsoup4>=4.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "ruff>=0.1.0",
        ]
    },
    entry_points={
        "openhands.extensions": [
            "uagent_research = uagent_research:UAgentResearchExtension",
        ]
    },
)
EOF

# Create pyproject.toml
cat > extensions/uagent_research/pyproject.toml <<EOF
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "uagent-research-extension"
version = "0.1.0"
description = "Advanced research capabilities for OpenHands"
readme = "README.md"
requires-python = ">=3.11"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.ruff]
line-length = 100
target-version = "py311"
EOF
```

#### Step 4: Set Up Development Database

```bash
# Create SQLite database for development
cat > extensions/uagent_research/init_db.py <<EOF
"""Initialize extension database"""
from sqlalchemy import create_engine
from .models.base import Base

def init_db(database_url: str = "sqlite:///./uagent_research.db"):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    print(f"Database initialized at {database_url}")

if __name__ == "__main__":
    init_db()
EOF

# Run initialization
cd extensions/uagent_research
python -m init_db
```

#### Step 5: Configure Extension in OpenHands

```bash
# Add extension to OpenHands config
cat >> config.toml <<EOF

[extensions]
enabled = ["uagent_research"]

[extensions.uagent_research]
# Extension-specific configuration
database_url = "sqlite:///./uagent_research.db"
workspace_dir = "./workspaces/research"
max_concurrent_experiments = 5
experiment_timeout = 3600  # 1 hour

# Research engine settings
[extensions.uagent_research.scientific]
max_retries = 3
validation_strict = true

[extensions.uagent_research.roma]
max_parallel_branches = 10
branch_timeout = 1800  # 30 minutes

[extensions.uagent_research.code_research]
repomaster_enabled = true
max_repo_size = "10GB"
EOF
```

---

## Phase 1: Core Extension Structure

### Week 2-3: Extension Framework

#### Step 1: Define Extension Interface

```python
# extensions/uagent_research/extension.py

from openhands.core.extension import Extension
from openhands.core.config import Config
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class UAgentResearchExtension(Extension):
    """
    UAgent Research Extension for OpenHands

    Provides advanced research capabilities integrated into OpenHands.
    """

    # Extension metadata
    name = "uagent_research"
    version = "0.1.0"
    description = "Advanced research capabilities for scientific experiments"
    author = "UAgent Team"

    def __init__(self, config: Config):
        super().__init__(config)
        self.research_engines = {}
        self.active_experiments = {}

    async def initialize(self):
        """Initialize extension resources"""
        logger.info(f"Initializing {self.name} extension v{self.version}")

        # Initialize database
        await self._init_database()

        # Initialize research engines
        await self._init_research_engines()

        # Set up workspace
        await self._init_workspace()

        # Register agents
        await self._register_agents()

        logger.info(f"{self.name} extension initialized successfully")

    async def shutdown(self):
        """Cleanup extension resources"""
        logger.info(f"Shutting down {self.name} extension")

        # Stop all active experiments
        for exp_id, experiment in self.active_experiments.items():
            await experiment.stop()

        # Cleanup resources
        await self._cleanup_workspace()

        logger.info(f"{self.name} extension shut down successfully")

    async def _init_database(self):
        """Initialize extension database"""
        from .models.base import init_database
        from .models.experiment import Experiment
        from .models.research_session import ResearchSession

        db_url = self.config.get("database_url", "sqlite:///./uagent_research.db")
        await init_database(db_url)
        logger.info(f"Database initialized: {db_url}")

    async def _init_research_engines(self):
        """Initialize research engines"""
        from .engines.scientific_research import ScientificResearchEngine
        from .engines.code_research import CodeResearchEngine
        from .engines.roma_engine import ROMAEngine

        self.research_engines = {
            "scientific": ScientificResearchEngine(self.config),
            "code": CodeResearchEngine(self.config),
            "roma": ROMAEngine(self.config),
        }
        logger.info(f"Initialized {len(self.research_engines)} research engines")

    async def _init_workspace(self):
        """Set up research workspace directory"""
        import os
        workspace_dir = self.config.get("workspace_dir", "./workspaces/research")
        os.makedirs(workspace_dir, exist_ok=True)
        logger.info(f"Workspace initialized: {workspace_dir}")

    async def _register_agents(self):
        """Register research agents with OpenHands"""
        from .agents.scientific_research_agent import ScientificResearchAgent
        from .agents.code_research_agent import CodeResearchAgent
        from .agents.roma_agent import ROMAOrchestratorAgent

        # Register agents
        self.register_agent("scientific_research", ScientificResearchAgent)
        self.register_agent("code_research", CodeResearchAgent)
        self.register_agent("roma_orchestrator", ROMAOrchestratorAgent)

        logger.info("Research agents registered successfully")

    async def _cleanup_workspace(self):
        """Cleanup temporary workspace files"""
        # Implement workspace cleanup logic
        pass

    # Extension API methods
    def register_routes(self):
        """Register API routes"""
        from .api.research_routes import router
        return [router]

    def register_ui_components(self):
        """Register frontend components"""
        return {
            "routes": [
                {"path": "/research", "component": "ResearchDashboard"},
                {"path": "/research/experiments/:id", "component": "ExperimentDetail"},
                {"path": "/research/tree", "component": "ResearchTreeView"},
            ],
            "navigation": [
                {
                    "id": "research",
                    "label": "Research",
                    "icon": "flask",
                    "path": "/research"
                }
            ],
            "widgets": [
                {
                    "id": "active_experiments",
                    "component": "ActiveExperimentsWidget",
                    "position": "sidebar"
                }
            ]
        }
```

#### Step 2: Define Data Models

```python
# extensions/uagent_research/models/base.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import MetaData

metadata = MetaData()
Base = declarative_base(metadata=metadata)

# Global session factory
async_session_factory = None

async def init_database(database_url: str):
    """Initialize database connection and tables"""
    global async_session_factory

    engine = create_async_engine(database_url, echo=False)
    async_session_factory = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_session() -> AsyncSession:
    """Get database session"""
    async with async_session_factory() as session:
        yield session
```

```python
# extensions/uagent_research/models/experiment.py

from sqlalchemy import Column, String, Text, Integer, Float, DateTime, JSON, Enum
from sqlalchemy.sql import func
from datetime import datetime
import enum
from .base import Base

class ExperimentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class Experiment(Base):
    __tablename__ = "experiments"

    # Primary key
    id = Column(String, primary_key=True)

    # Foreign keys
    session_id = Column(String, nullable=False, index=True)
    parent_experiment_id = Column(String, nullable=True)

    # Metadata
    experiment_type = Column(String, nullable=False)  # scientific, code, roma
    goal = Column(Text, nullable=False)
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.PENDING)

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Progress tracking
    progress_percentage = Column(Float, default=0.0)
    current_step = Column(String, nullable=True)
    total_steps = Column(Integer, nullable=True)

    # Configuration
    config = Column(JSON, nullable=True)
    workspace_path = Column(String, nullable=True)

    # Results
    results = Column(JSON, nullable=True)
    artifacts = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    # Resource usage
    execution_time_seconds = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<Experiment {self.id} ({self.status})>"
```

```python
# extensions/uagent_research/models/research_session.py

from sqlalchemy import Column, String, DateTime, JSON, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from .base import Base

class SessionMode(str, enum.Enum):
    CHAT = "chat"
    RESEARCH = "research"
    HYBRID = "hybrid"

class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    mode = Column(Enum(SessionMode), default=SessionMode.RESEARCH)

    # Metadata
    title = Column(String, nullable=True)
    description = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Session state
    state = Column(JSON, nullable=True)
    research_tree = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<ResearchSession {self.id} ({self.mode})>"
```

#### Step 3: Migrate Core Research Engines

```bash
# Copy UAgent research engines to extension
cp -r /path/to/uagent/backend/app/core/research_engines/* \
      extensions/uagent_research/engines/

# Adapt imports to work with OpenHands
# This requires refactoring UAgent code to remove dependencies
# on UAgent-specific infrastructure and use OpenHands equivalents
```

```python
# extensions/uagent_research/engines/scientific_research.py

from openhands.core.logger import openhands_logger as logger
from openhands.runtime.runtime import Runtime
from openhands.events.stream import EventStream
from typing import Dict, Any, Optional
import asyncio

class ScientificResearchEngine:
    """
    Scientific research engine adapted for OpenHands.

    Migrated from UAgent with minimal changes to core logic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = None  # Will be injected by agent

    async def run_experiment(
        self,
        goal: str,
        runtime: Runtime,
        event_stream: EventStream,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Execute scientific research experiment.

        Args:
            goal: Research objective
            runtime: OpenHands runtime for code execution
            event_stream: Event stream for progress updates
            session_id: Current session ID

        Returns:
            Experiment results dictionary
        """
        logger.info(f"Starting scientific research: {goal}")

        # Create experiment in database
        from ..models.experiment import Experiment, ExperimentStatus
        experiment = Experiment(
            id=f"exp_{session_id}_{int(time.time())}",
            session_id=session_id,
            experiment_type="scientific",
            goal=goal,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.utcnow()
        )

        # Save to database
        async with get_session() as session:
            session.add(experiment)
            await session.commit()

        try:
            # Run research using UAgent's core logic
            # but with OpenHands runtime instead of custom execution
            result = await self._execute_research(
                goal=goal,
                runtime=runtime,
                event_stream=event_stream,
                experiment=experiment
            )

            # Update experiment status
            experiment.status = ExperimentStatus.COMPLETED
            experiment.results = result
            experiment.completed_at = datetime.utcnow()

            return result

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = str(e)
            raise

        finally:
            # Save final state
            async with get_session() as session:
                await session.merge(experiment)
                await session.commit()

    async def _execute_research(
        self,
        goal: str,
        runtime: Runtime,
        event_stream: EventStream,
        experiment: Experiment
    ) -> Dict[str, Any]:
        """Core research execution logic (adapted from UAgent)"""

        # Stream progress update
        await event_stream.add_event({
            "type": "research_progress",
            "experiment_id": experiment.id,
            "step": "initializing",
            "progress": 0.0
        })

        # === COPY CORE LOGIC FROM UAGENT ===
        # This is where we integrate UAgent's research logic
        # Key changes:
        # 1. Use OpenHands runtime instead of OpenHandsClient
        # 2. Use EventStream for progress instead of WebSocketManager
        # 3. Use OpenHands LLM interface instead of custom client

        # Example structure (simplified):
        steps = [
            "literature_review",
            "hypothesis_generation",
            "experiment_design",
            "code_implementation",
            "execution",
            "analysis"
        ]

        results = {}
        for i, step in enumerate(steps):
            logger.info(f"Executing step: {step}")

            # Update progress
            progress = (i + 1) / len(steps)
            await event_stream.add_event({
                "type": "research_progress",
                "experiment_id": experiment.id,
                "step": step,
                "progress": progress * 100
            })

            # Execute step using OpenHands runtime
            step_result = await self._execute_step(
                step=step,
                runtime=runtime,
                context=results
            )

            results[step] = step_result

        return results

    async def _execute_step(
        self,
        step: str,
        runtime: Runtime,
        context: Dict[str, Any]
    ) -> Any:
        """Execute individual research step"""
        # Implement step-specific logic
        # Use OpenHands runtime to execute code, run commands, etc.
        pass
```

---

## Phase 2: Backend Integration

### Week 4-6: Agent Implementation

#### Step 1: Create Scientific Research Agent

```python
# extensions/uagent_research/agents/scientific_research_agent.py

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.schema import AgentState
from openhands.events.action import Action, MessageAction, CmdRunAction
from openhands.events.observation import Observation
from openhands.llm.llm import LLM
from ..engines.scientific_research import ScientificResearchEngine
import logging

logger = logging.getLogger(__name__)

class ScientificResearchAgent(CodeActAgent):
    """
    Agent specialized for scientific research experiments.

    Extends CodeActAgent with research planning and execution capabilities.
    """

    VERSION = "1.0"

    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.research_engine = ScientificResearchEngine(config={})
        self.research_engine.llm = llm  # Inject LLM
        self.current_experiment = None

    async def step(self, state: AgentState) -> Action:
        """
        Execute one agent step.

        Determines if this is a research task and routes appropriately.
        """
        # Check if this is a research task
        if self._is_research_task(state):
            logger.info("Detected research task, using research engine")
            return await self._research_step(state)
        else:
            # Fall back to normal CodeAct behavior
            logger.info("Normal coding task, using CodeAct")
            return await super().step(state)

    def _is_research_task(self, state: AgentState) -> bool:
        """Detect if current task requires research capabilities"""
        research_keywords = [
            "experiment", "hypothesis", "research", "investigate",
            "analyze performance", "compare approaches", "benchmark",
            "scientific method", "ml model", "train model"
        ]

        task_lower = state.task.lower() if state.task else ""
        return any(keyword in task_lower for keyword in research_keywords)

    async def _research_step(self, state: AgentState) -> Action:
        """Execute research-specific step"""

        # First time seeing this task - start experiment
        if not self.current_experiment:
            return await self._start_experiment(state)

        # Continue ongoing experiment
        return await self._continue_experiment(state)

    async def _start_experiment(self, state: AgentState) -> Action:
        """Start new research experiment"""
        logger.info(f"Starting experiment for goal: {state.task}")

        # Create experiment using research engine
        # This is a long-running operation
        result = await self.research_engine.run_experiment(
            goal=state.task,
            runtime=state.runtime,  # OpenHands runtime
            event_stream=state.event_stream,
            session_id=state.session_id
        )

        self.current_experiment = result

        # Return summary action
        return MessageAction(
            content=f"""Experiment completed successfully!

**Goal**: {state.task}

**Results Summary**:
{self._format_results_summary(result)}

**Next Steps**:
Would you like me to:
1. Analyze the results in more detail?
2. Run additional experiments?
3. Generate a research report?
"""
        )

    async def _continue_experiment(self, state: AgentState) -> Action:
        """Continue or modify ongoing experiment"""
        # Handle user feedback and iteration
        last_message = state.history[-1] if state.history else None

        if not last_message:
            return MessageAction("Please provide feedback or next steps.")

        # Parse user intent
        user_input = last_message.content.lower()

        if "analyze" in user_input or "detail" in user_input:
            # Generate detailed analysis
            analysis = await self._generate_analysis(self.current_experiment)
            return MessageAction(analysis)

        elif "report" in user_input:
            # Generate research report
            report = await self._generate_report(self.current_experiment)
            return MessageAction(report)

        elif "new" in user_input or "another" in user_input:
            # Start new experiment
            self.current_experiment = None
            return await self._start_experiment(state)

        else:
            # Default: ask for clarification
            return MessageAction(
                "I'm not sure what you'd like me to do next. "
                "Could you please clarify? (analyze results / generate report / run new experiment)"
            )

    def _format_results_summary(self, results: dict) -> str:
        """Format experiment results for display"""
        summary = []

        if "hypothesis" in results:
            summary.append(f"- **Hypothesis**: {results['hypothesis']}")

        if "experiments_run" in results:
            summary.append(f"- **Experiments Run**: {results['experiments_run']}")

        if "key_findings" in results:
            summary.append("- **Key Findings**:")
            for finding in results["key_findings"]:
                summary.append(f"  - {finding}")

        return "\n".join(summary)

    async def _generate_analysis(self, experiment: dict) -> str:
        """Generate detailed analysis of experiment results"""
        # Use LLM to generate analysis
        prompt = f"""Analyze these experiment results in detail:

{json.dumps(experiment, indent=2)}

Provide:
1. Statistical significance of findings
2. Potential confounding factors
3. Recommendations for follow-up experiments
4. Limitations of current approach
"""

        response = await self.llm.completion(messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    async def _generate_report(self, experiment: dict) -> str:
        """Generate formal research report"""
        # Use LLM to generate formal report
        prompt = f"""Generate a formal research report for these experiment results:

{json.dumps(experiment, indent=2)}

Format as a scientific paper with sections:
- Abstract
- Introduction
- Methods
- Results
- Discussion
- Conclusion
- References (if applicable)
"""

        response = await self.llm.completion(messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
```

#### Step 2: Create Code Research Agent

```python
# extensions/uagent_research/agents/code_research_agent.py

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.schema import AgentState
from openhands.events.action import Action, MessageAction
from openhands.llm.llm import LLM
from ..engines.code_research import CodeResearchEngine
import logging

logger = logging.getLogger(__name__)

class CodeResearchAgent(CodeActAgent):
    """
    Agent specialized for code repository analysis.

    Integrates RepoMaster-style functionality into OpenHands.
    """

    VERSION = "1.0"

    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.code_engine = CodeResearchEngine(config={})
        self.code_engine.llm = llm
        self.current_analysis = None

    async def step(self, state: AgentState) -> Action:
        """Execute one agent step"""

        if self._is_code_analysis_task(state):
            logger.info("Detected code analysis task")
            return await self._analyze_code(state)
        else:
            return await super().step(state)

    def _is_code_analysis_task(self, state: AgentState) -> bool:
        """Detect if task requires code analysis"""
        code_keywords = [
            "analyze code", "understand repository", "find implementation",
            "code structure", "architecture", "how does", "where is",
            "explain code", "trace", "dependency"
        ]

        task_lower = state.task.lower() if state.task else ""
        return any(keyword in task_lower for keyword in code_keywords)

    async def _analyze_code(self, state: AgentState) -> Action:
        """Perform code analysis"""
        logger.info(f"Analyzing code for: {state.task}")

        # Use code research engine
        analysis = await self.code_engine.analyze_repository(
            query=state.task,
            workspace=state.runtime.workdir,
            context=state.history
        )

        self.current_analysis = analysis

        return MessageAction(
            content=f"""Code Analysis Complete

**Query**: {state.task}

**Findings**:
{self._format_analysis(analysis)}

Would you like me to:
1. Dive deeper into specific files?
2. Trace a function call?
3. Generate architecture diagram?
"""
        )

    def _format_analysis(self, analysis: dict) -> str:
        """Format analysis results"""
        output = []

        if "relevant_files" in analysis:
            output.append("**Relevant Files**:")
            for file in analysis["relevant_files"][:5]:
                output.append(f"- `{file}`")

        if "key_components" in analysis:
            output.append("\n**Key Components**:")
            for component in analysis["key_components"]:
                output.append(f"- {component}")

        if "summary" in analysis:
            output.append(f"\n**Summary**:\n{analysis['summary']}")

        return "\n".join(output)
```

#### Step 3: Create API Routes

```python
# extensions/uagent_research/api/research_routes.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.base import get_session
from ..models.experiment import Experiment, ExperimentStatus
from ..models.research_session import ResearchSession
from ..engines.scientific_research import ScientificResearchEngine

router = APIRouter(prefix="/api/research", tags=["research"])

# Request/Response models
class StartResearchRequest(BaseModel):
    goal: str
    session_id: str
    research_type: str = "scientific"
    config: Optional[dict] = None

class ExperimentResponse(BaseModel):
    id: str
    session_id: str
    experiment_type: str
    goal: str
    status: ExperimentStatus
    progress_percentage: float
    current_step: Optional[str]
    results: Optional[dict]
    error_message: Optional[str]

# Endpoints
@router.post("/experiments/start", response_model=ExperimentResponse)
async def start_experiment(
    request: StartResearchRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session)
):
    """Start a new research experiment"""

    # Create experiment record
    experiment = Experiment(
        id=f"exp_{request.session_id}_{int(time.time())}",
        session_id=request.session_id,
        experiment_type=request.research_type,
        goal=request.goal,
        config=request.config,
        status=ExperimentStatus.PENDING
    )

    session.add(experiment)
    await session.commit()

    # Start experiment in background
    # Note: This requires integration with OpenHands runtime
    # For now, we'll return the created experiment

    return ExperimentResponse(
        id=experiment.id,
        session_id=experiment.session_id,
        experiment_type=experiment.experiment_type,
        goal=experiment.goal,
        status=experiment.status,
        progress_percentage=experiment.progress_percentage,
        current_step=experiment.current_step,
        results=experiment.results,
        error_message=experiment.error_message
    )

@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get experiment status and results"""

    result = await session.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(404, f"Experiment {experiment_id} not found")

    return ExperimentResponse(
        id=experiment.id,
        session_id=experiment.session_id,
        experiment_type=experiment.experiment_type,
        goal=experiment.goal,
        status=experiment.status,
        progress_percentage=experiment.progress_percentage,
        current_step=experiment.current_step,
        results=experiment.results,
        error_message=experiment.error_message
    )

@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    session_id: Optional[str] = None,
    status: Optional[ExperimentStatus] = None,
    session: AsyncSession = Depends(get_session)
):
    """List experiments with optional filters"""

    query = select(Experiment)

    if session_id:
        query = query.where(Experiment.session_id == session_id)
    if status:
        query = query.where(Experiment.status == status)

    result = await session.execute(query)
    experiments = result.scalars().all()

    return [
        ExperimentResponse(
            id=exp.id,
            session_id=exp.session_id,
            experiment_type=exp.experiment_type,
            goal=exp.goal,
            status=exp.status,
            progress_percentage=exp.progress_percentage,
            current_step=exp.current_step,
            results=exp.results,
            error_message=exp.error_message
        )
        for exp in experiments
    ]

@router.delete("/experiments/{experiment_id}")
async def cancel_experiment(
    experiment_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Cancel running experiment"""

    result = await session.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(404, f"Experiment {experiment_id} not found")

    if experiment.status not in [ExperimentStatus.PENDING, ExperimentStatus.RUNNING]:
        raise HTTPException(400, f"Cannot cancel experiment in status {experiment.status}")

    # Cancel experiment
    # TODO: Implement actual cancellation logic
    experiment.status = ExperimentStatus.CANCELLED
    await session.commit()

    return {"status": "cancelled", "experiment_id": experiment_id}
```

---

## Phase 3: Frontend Integration

### Week 7-9: UI Components

#### Step 1: Create Research Dashboard

```typescript
// extensions/uagent_research/ui/ResearchDashboard.tsx

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent, CardActions } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { useResearchExtension } from './hooks/useResearchExtension';
import { ExperimentList } from './components/ExperimentList';
import { ResearchTree } from './components/ResearchTree';
import { NewExperimentDialog } from './components/NewExperimentDialog';

export function ResearchDashboard() {
  const {
    experiments,
    activeExperiments,
    completedExperiments,
    startExperiment,
    loading
  } = useResearchExtension();

  const [activeTab, setActiveTab] = useState('active');
  const [showNewDialog, setShowNewDialog] = useState(false);

  return (
    <div className="research-dashboard p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Research Dashboard</h1>
        <Button onClick={() => setShowNewDialog(true)}>
          New Experiment
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="active">
            Active ({activeExperiments.length})
          </TabsTrigger>
          <TabsTrigger value="completed">
            Completed ({completedExperiments.length})
          </TabsTrigger>
          <TabsTrigger value="tree">
            Research Tree
          </TabsTrigger>
        </TabsList>

        <TabsContent value="active">
          <ExperimentList
            experiments={activeExperiments}
            onExperimentClick={(exp) => {
              // Navigate to experiment detail
              window.location.href = `/research/experiments/${exp.id}`;
            }}
          />
        </TabsContent>

        <TabsContent value="completed">
          <ExperimentList
            experiments={completedExperiments}
            onExperimentClick={(exp) => {
              window.location.href = `/research/experiments/${exp.id}`;
            }}
          />
        </TabsContent>

        <TabsContent value="tree">
          <ResearchTree sessionId={currentSessionId} />
        </TabsContent>
      </Tabs>

      <NewExperimentDialog
        open={showNewDialog}
        onClose={() => setShowNewDialog(false)}
        onSubmit={async (config) => {
          await startExperiment(config);
          setShowNewDialog(false);
        }}
      />
    </div>
  );
}
```

```typescript
// extensions/uagent_research/ui/hooks/useResearchExtension.ts

import { useState, useEffect, useCallback } from 'react';
import { useOpenHandsAPI } from '@openhands/hooks';

interface Experiment {
  id: string;
  session_id: string;
  experiment_type: string;
  goal: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress_percentage: number;
  current_step?: string;
  results?: any;
  error_message?: string;
}

export function useResearchExtension() {
  const api = useOpenHandsAPI();
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(false);

  // Load experiments on mount
  useEffect(() => {
    loadExperiments();

    // Set up polling for active experiments
    const interval = setInterval(() => {
      loadExperiments();
    }, 5000);  // Poll every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const loadExperiments = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/research/experiments', {
        params: { session_id: api.sessionId }
      });
      setExperiments(response.data);
    } catch (error) {
      console.error('Failed to load experiments:', error);
    } finally {
      setLoading(false);
    }
  }, [api]);

  const startExperiment = useCallback(async (config: {
    goal: string;
    research_type: string;
    config?: any;
  }) => {
    setLoading(true);
    try {
      const response = await api.post('/api/research/experiments/start', {
        ...config,
        session_id: api.sessionId
      });

      // Reload experiments
      await loadExperiments();

      return response.data;
    } catch (error) {
      console.error('Failed to start experiment:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [api, loadExperiments]);

  const cancelExperiment = useCallback(async (experimentId: string) => {
    try {
      await api.delete(`/api/research/experiments/${experimentId}`);
      await loadExperiments();
    } catch (error) {
      console.error('Failed to cancel experiment:', error);
      throw error;
    }
  }, [api, loadExperiments]);

  // Derived state
  const activeExperiments = experiments.filter(
    exp => exp.status === 'running' || exp.status === 'pending'
  );

  const completedExperiments = experiments.filter(
    exp => exp.status === 'completed' || exp.status === 'failed'
  );

  return {
    experiments,
    activeExperiments,
    completedExperiments,
    loading,
    startExperiment,
    cancelExperiment,
    refresh: loadExperiments
  };
}
```

#### Step 2: Integrate into OpenHands Frontend

```typescript
// openhands/frontend/src/App.tsx (MODIFIED)

import { ResearchDashboard } from '@uagent-research/ui/ResearchDashboard';

function App() {
  return (
    <Router>
      <Routes>
        {/* Existing routes */}
        <Route path="/" element={<ChatInterface />} />
        <Route path="/settings" element={<Settings />} />

        {/* New research routes */}
        <Route path="/research" element={<ResearchDashboard />} />
        <Route path="/research/experiments/:id" element={<ExperimentDetail />} />
        <Route path="/research/tree" element={<ResearchTreeView />} />
      </Routes>
    </Router>
  );
}
```

```typescript
// openhands/frontend/src/components/navigation/Sidebar.tsx (MODIFIED)

import { FlaskConical } from 'lucide-react';

function Sidebar() {
  return (
    <nav className="sidebar">
      {/* Existing nav items */}
      <NavItem icon={<MessageSquare />} label="Chat" to="/" />
      <NavItem icon={<Settings />} label="Settings" to="/settings" />

      {/* New research nav item */}
      <NavItem icon={<FlaskConical />} label="Research" to="/research" />
    </nav>
  );
}
```

---

## Phase 4: Testing & Validation

### Week 10-12: Comprehensive Testing

#### Step 1: Unit Tests

```python
# extensions/uagent_research/tests/test_scientific_research_agent.py

import pytest
from unittest.mock import Mock, AsyncMock
from openhands.core.schema import AgentState
from openhands.llm.llm import LLM
from ..agents.scientific_research_agent import ScientificResearchAgent

@pytest.fixture
def mock_llm():
    llm = Mock(spec=LLM)
    llm.completion = AsyncMock(return_value=Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    ))
    return llm

@pytest.fixture
def agent(mock_llm):
    return ScientificResearchAgent(mock_llm)

@pytest.mark.asyncio
async def test_detect_research_task(agent):
    """Test detection of research tasks"""
    state = AgentState(
        task="Run an experiment to compare algorithm performance",
        session_id="test_session"
    )

    assert agent._is_research_task(state) is True

@pytest.mark.asyncio
async def test_detect_non_research_task(agent):
    """Test detection of non-research tasks"""
    state = AgentState(
        task="Fix the bug in login.py",
        session_id="test_session"
    )

    assert agent._is_research_task(state) is False

@pytest.mark.asyncio
async def test_start_experiment(agent, mock_llm):
    """Test starting new experiment"""
    state = AgentState(
        task="Test hypothesis: Algorithm A is faster than B",
        session_id="test_session",
        runtime=Mock(),
        event_stream=Mock()
    )

    # Mock research engine
    agent.research_engine.run_experiment = AsyncMock(return_value={
        "hypothesis": "Algorithm A is faster",
        "experiments_run": 10,
        "key_findings": ["A is 2x faster on average"]
    })

    action = await agent._start_experiment(state)

    assert "Experiment completed successfully" in action.content
    assert agent.current_experiment is not None
```

#### Step 2: Integration Tests

```python
# extensions/uagent_research/tests/test_api_integration.py

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ..models.base import Base
from ..api.research_routes import router
from fastapi import FastAPI

@pytest.fixture
async def test_app():
    """Create test FastAPI app"""
    app = FastAPI()
    app.include_router(router)
    return app

@pytest.fixture
async def test_db():
    """Create test database"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    yield async_session

    await engine.dispose()

@pytest.mark.asyncio
async def test_create_experiment(test_app, test_db):
    """Test creating new experiment via API"""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        response = await client.post("/api/research/experiments/start", json={
            "goal": "Test experiment",
            "session_id": "test_session",
            "research_type": "scientific"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["goal"] == "Test experiment"
        assert data["status"] == "pending"

@pytest.mark.asyncio
async def test_list_experiments(test_app, test_db):
    """Test listing experiments"""
    # Create some experiments first
    # ...

    async with AsyncClient(app=test_app, base_url="http://test") as client:
        response = await client.get("/api/research/experiments")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
```

#### Step 3: End-to-End Tests

```python
# extensions/uagent_research/tests/test_e2e_research_flow.py

import pytest
from openhands.controller.agent_controller import AgentController
from openhands.runtime.docker.docker_runtime import DockerRuntime
from ..agents.scientific_research_agent import ScientificResearchAgent

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_research_workflow():
    """Test complete research workflow from start to finish"""

    # Create agent
    llm = get_test_llm()
    agent = ScientificResearchAgent(llm)

    # Create runtime
    runtime = DockerRuntime(config=get_test_runtime_config())
    await runtime.initialize()

    # Create controller
    controller = AgentController(
        agent=agent,
        runtime=runtime,
        max_iterations=50
    )

    # Run research task
    result = await controller.run(
        "Run an experiment to compare sorting algorithm performance: quicksort vs mergesort"
    )

    # Verify results
    assert result.status == "completed"
    assert "experiment" in result.output.lower()
    assert result.iterations < 50  # Should complete in reasonable iterations

    # Cleanup
    await runtime.cleanup()
```

---

## Phase 5: Deployment

### Week 13-15: Production Deployment

#### Step 1: Package Extension

```bash
# Build extension package
cd extensions/uagent_research
python -m build

# This creates:
# dist/uagent_research_extension-0.1.0-py3-none-any.whl
# dist/uagent_research_extension-0.1.0.tar.gz
```

#### Step 2: Create Deployment Documentation

```markdown
# UAgent Research Extension - Installation Guide

## Prerequisites
- OpenHands >= 0.9.0
- Python >= 3.11
- Docker >= 24.0

## Installation

### From PyPI (when published)
\`\`\`bash
pip install uagent-research-extension
\`\`\`

### From source
\`\`\`bash
git clone https://github.com/yourusername/uagent-research-extension.git
cd uagent-research-extension
pip install -e .
\`\`\`

## Configuration

Add to your OpenHands config.toml:

\`\`\`toml
[extensions]
enabled = ["uagent_research"]

[extensions.uagent_research]
database_url = "postgresql://user:pass@localhost/uagent_research"
workspace_dir = "./workspaces/research"
max_concurrent_experiments = 5
\`\`\`

## Usage

### Start OpenHands with extension
\`\`\`bash
openhands --config config.toml
\`\`\`

### Access research features
- Navigate to http://localhost:3000/research
- Use research agents in chat: "Run an experiment to test..."

## Troubleshooting

### Extension not loading
- Check logs: `tail -f ~/.openhands/logs/extensions.log`
- Verify installation: `pip show uagent-research-extension`

### Database errors
- Ensure database is accessible
- Run migrations: `python -m uagent_research.init_db`
```

#### Step 3: CI/CD Pipeline

```yaml
# .github/workflows/extension-ci.yml

name: UAgent Research Extension CI

on:
  push:
    paths:
      - 'extensions/uagent_research/**'
  pull_request:
    paths:
      - 'extensions/uagent_research/**'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          cd extensions/uagent_research
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          cd extensions/uagent_research
          pytest tests/ -v --cov=. --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./extensions/uagent_research/coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install linting tools
        run: pip install black mypy ruff

      - name: Run black
        run: black --check extensions/uagent_research

      - name: Run ruff
        run: ruff check extensions/uagent_research

      - name: Run mypy
        run: mypy extensions/uagent_research
```

---

## Rollback Plan

### If Integration Fails

#### Option 1: Disable Extension

```toml
# config.toml
[extensions]
enabled = []  # Remove "uagent_research"
```

#### Option 2: Revert to Standalone UAgent

```bash
# Keep extension code but run UAgent separately
docker-compose up uagent-standalone
```

#### Option 3: Remove Extension Completely

```bash
# Uninstall extension
pip uninstall uagent-research-extension

# Remove extension directory
rm -rf extensions/uagent_research

# Remove from git
git checkout main
git branch -D feature/uagent-extension
```

---

## Success Metrics

### Week-by-Week Checkpoints

- **Week 3**: Extension loads successfully, basic structure in place
- **Week 6**: Scientific research agent can run simple experiments
- **Week 9**: Frontend dashboard shows experiments in real-time
- **Week 12**: All tests passing, >90% coverage
- **Week 15**: Production-ready deployment

### Final Success Criteria

- [ ] All UAgent research features work in OpenHands
- [ ] Extension loads without errors
- [ ] API endpoints respond correctly
- [ ] Frontend components render properly
- [ ] Tests pass with >90% coverage
- [ ] Documentation is complete
- [ ] Performance meets benchmarks
- [ ] No regression in OpenHands core functionality

---

**Next**: See `06_TECHNICAL_SPECIFICATIONS.md` for detailed API specifications and data schemas.
