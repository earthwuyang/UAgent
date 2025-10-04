# Integration Approaches: Detailed Comparison

## Table of Contents
1. [Approach Overview](#approach-overview)
2. [Option A: Plugin/Extension Model](#option-a-pluginextension-model)
3. [Option B: Full Merge](#option-b-full-merge)
4. [Option C: Microservices Architecture](#option-c-microservices-architecture)
5. [Option D: Hybrid Approach](#option-d-hybrid-approach)
6. [Decision Matrix](#decision-matrix)
7. [Recommendation](#recommendation)

---

## Approach Overview

We've identified four potential approaches for integrating UAgent's research capabilities into OpenHands:

| Approach | Coupling | Complexity | Maintenance | Flexibility |
|----------|----------|------------|-------------|-------------|
| **A: Plugin/Extension** | Low | Medium | Easy | High |
| **B: Full Merge** | High | Very High | Hard | Low |
| **C: Microservices** | Very Low | High | Medium | Very High |
| **D: Hybrid** | Medium | High | Medium | High |

---

## Option A: Plugin/Extension Model ⭐ RECOMMENDED

### Architecture

```
OpenHands Core
├── openhands/
│   ├── agenthub/
│   │   ├── codeact_agent/
│   │   ├── delegator_agent/
│   │   └── planner_agent/
│   │
│   ├── controller/
│   ├── runtime/
│   ├── llm/
│   └── server/
│
└── extensions/                          # NEW: Extension system
    └── uagent_research/                 # UAgent as extension
        ├── __init__.py
        ├── agents/                      # Research agents
        │   ├── scientific_research_agent.py
        │   ├── code_research_agent.py
        │   └── roma_orchestrator_agent.py
        │
        ├── engines/                     # Core research logic
        │   ├── scientific_research.py   # From UAgent
        │   ├── code_research.py
        │   └── roma_engine.py
        │
        ├── runtime/                     # Research runtime extensions
        │   └── experiment_runtime.py
        │
        ├── ui/                          # Frontend components
        │   ├── ResearchDashboard.tsx
        │   ├── ResearchTree.tsx
        │   └── ExperimentMonitor.tsx
        │
        └── api/                         # API routes
            └── research_routes.py
```

### Implementation Details

#### 1. Extension Registration

```python
# extensions/uagent_research/__init__.py

from openhands.core.extension import Extension
from .agents import ScientificResearchAgent, CodeResearchAgent, ROMAAgent

class UAgentResearchExtension(Extension):
    """UAgent Research capabilities as OpenHands extension"""

    name = "uagent-research"
    version = "1.0.0"
    description = "Advanced research capabilities for scientific experiments"

    def register_agents(self):
        """Register research agents"""
        return {
            "scientific_research": ScientificResearchAgent,
            "code_research": CodeResearchAgent,
            "roma": ROMAAgent,
        }

    def register_routes(self):
        """Register API routes"""
        from .api import research_routes
        return research_routes.router

    def register_ui_components(self):
        """Register frontend components"""
        return {
            "routes": [
                {"path": "/research", "component": "ResearchDashboard"},
                {"path": "/experiments", "component": "ExperimentMonitor"},
            ],
            "tabs": [
                {"id": "research", "label": "Research", "icon": "flask"},
            ]
        }

    def on_install(self):
        """Called when extension is installed"""
        # Set up database tables for research data
        # Initialize research workspace
        pass

# Register extension
extension = UAgentResearchExtension()
```

#### 2. Agent Integration

```python
# extensions/uagent_research/agents/scientific_research_agent.py

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.schema import AgentState
from openhands.events.action import Action, MessageAction
from openhands.events.observation import Observation

from ..engines.scientific_research import ScientificResearchEngine

class ScientificResearchAgent(CodeActAgent):
    """Agent specialized for scientific research experiments"""

    def __init__(self, llm, *args, **kwargs):
        super().__init__(llm, *args, **kwargs)
        self.research_engine = ScientificResearchEngine(llm)

    async def step(self, state: AgentState) -> Action:
        """
        Execute one research step.

        This agent extends CodeActAgent but adds research-specific
        planning and execution logic.
        """
        # Check if this is a research task
        if self._is_research_task(state):
            return await self._research_step(state)
        else:
            # Fall back to normal CodeAct behavior
            return await super().step(state)

    def _is_research_task(self, state: AgentState) -> bool:
        """Detect if current task requires research capabilities"""
        keywords = ["experiment", "hypothesis", "research", "analyze", "investigate"]
        last_message = state.history[-1] if state.history else None
        if last_message:
            return any(kw in last_message.content.lower() for kw in keywords)
        return False

    async def _research_step(self, state: AgentState) -> Action:
        """Execute a research-specific step"""
        # Use research engine to plan next action
        research_plan = await self.research_engine.plan_next_step(
            goal=state.task,
            history=state.history,
            current_state=state
        )

        # Convert research plan to OpenHands action
        if research_plan.action_type == "run_experiment":
            return MessageAction(
                content=f"Running experiment: {research_plan.experiment_config}"
            )
        elif research_plan.action_type == "analyze_results":
            return MessageAction(
                content=f"Analyzing results from {research_plan.experiment_id}"
            )
        else:
            # Default to code execution
            return await super().step(state)
```

#### 3. Runtime Extension

```python
# extensions/uagent_research/runtime/experiment_runtime.py

from openhands.runtime.runtime import Runtime
from openhands.runtime.docker.docker_runtime import DockerRuntime
from openhands.events.action import Action
from openhands.events.observation import Observation

class ExperimentRuntime(DockerRuntime):
    """Extended runtime for research experiments"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_workspace = None

    async def initialize_experiment_workspace(self, experiment_id: str):
        """Set up isolated workspace for experiment"""
        self.experiment_workspace = f"/experiments/{experiment_id}"

        # Create experiment directory
        await self.run_in_sandbox(
            f"mkdir -p {self.experiment_workspace}"
        )

        # Set up logging
        await self.run_in_sandbox(
            f"mkdir -p {self.experiment_workspace}/logs"
        )

    async def run_experiment_action(
        self,
        action: "ExperimentAction"
    ) -> Observation:
        """Execute experiment-specific action"""
        if not self.experiment_workspace:
            await self.initialize_experiment_workspace(action.experiment_id)

        # Execute experiment in isolated workspace
        result = await self.run_in_sandbox(
            action.command,
            working_dir=self.experiment_workspace
        )

        # Collect experiment artifacts
        artifacts = await self._collect_experiment_artifacts()

        return ExperimentObservation(
            result=result,
            artifacts=artifacts,
            workspace=self.experiment_workspace
        )

    async def _collect_experiment_artifacts(self) -> dict:
        """Collect logs, results, and generated files"""
        artifacts = {}

        # Collect logs
        logs = await self.run_in_sandbox(
            f"cat {self.experiment_workspace}/logs/*.log"
        )
        artifacts["logs"] = logs

        # Collect results
        if await self._file_exists(f"{self.experiment_workspace}/results.json"):
            results = await self.run_in_sandbox(
                f"cat {self.experiment_workspace}/results.json"
            )
            artifacts["results"] = results

        return artifacts
```

#### 4. API Routes

```python
# extensions/uagent_research/api/research_routes.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from ..engines.scientific_research import ScientificResearchEngine
from ..engines.roma_engine import ROMAEngine

router = APIRouter(prefix="/api/research", tags=["research"])

class ResearchRequest(BaseModel):
    goal: str
    session_id: str
    research_type: str = "scientific"  # scientific, code, roma

class ExperimentStatus(BaseModel):
    experiment_id: str
    status: str
    progress: float
    results: Optional[dict] = None

@router.post("/start")
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
):
    """Start a new research session"""

    # Create research engine based on type
    if request.research_type == "scientific":
        engine = ScientificResearchEngine()
    elif request.research_type == "roma":
        engine = ROMAEngine()
    else:
        raise HTTPException(400, f"Unknown research type: {request.research_type}")

    # Start research in background
    background_tasks.add_task(
        engine.run_research,
        goal=request.goal,
        session_id=request.session_id
    )

    return {
        "status": "started",
        "session_id": request.session_id,
        "research_type": request.research_type
    }

@router.get("/experiments/{experiment_id}/status")
async def get_experiment_status(experiment_id: str) -> ExperimentStatus:
    """Get status of running experiment"""
    # Query experiment database
    experiment = await get_experiment_from_db(experiment_id)

    if not experiment:
        raise HTTPException(404, "Experiment not found")

    return ExperimentStatus(
        experiment_id=experiment_id,
        status=experiment.status,
        progress=experiment.progress,
        results=experiment.results if experiment.status == "completed" else None
    )

@router.get("/experiments")
async def list_experiments(
    session_id: Optional[str] = None,
    status: Optional[str] = None
) -> List[ExperimentStatus]:
    """List all experiments, optionally filtered"""
    experiments = await query_experiments(
        session_id=session_id,
        status=status
    )

    return [
        ExperimentStatus(
            experiment_id=exp.id,
            status=exp.status,
            progress=exp.progress,
            results=exp.results
        )
        for exp in experiments
    ]
```

#### 5. Frontend Integration

```typescript
// extensions/uagent_research/ui/ResearchDashboard.tsx

import React, { useState, useEffect } from 'react';
import { useResearchExtension } from './hooks/useResearchExtension';
import { ExperimentList } from './components/ExperimentList';
import { ResearchTree } from './components/ResearchTree';

export function ResearchDashboard() {
  const { experiments, startResearch, loading } = useResearchExtension();
  const [activeView, setActiveView] = useState<'experiments' | 'tree'>('experiments');

  return (
    <div className="research-dashboard">
      <header className="dashboard-header">
        <h1>Research Dashboard</h1>
        <nav>
          <button
            className={activeView === 'experiments' ? 'active' : ''}
            onClick={() => setActiveView('experiments')}
          >
            Experiments
          </button>
          <button
            className={activeView === 'tree' ? 'active' : ''}
            onClick={() => setActiveView('tree')}
          >
            Research Tree
          </button>
        </nav>
      </header>

      <main className="dashboard-content">
        {activeView === 'experiments' ? (
          <ExperimentList
            experiments={experiments}
            onStartNew={startResearch}
            loading={loading}
          />
        ) : (
          <ResearchTree sessionId={currentSessionId} />
        )}
      </main>
    </div>
  );
}

// Register component with OpenHands
export default {
  component: ResearchDashboard,
  route: '/research',
  icon: 'flask',
  title: 'Research'
};
```

```typescript
// extensions/uagent_research/ui/hooks/useResearchExtension.ts

import { useState, useEffect } from 'react';
import { useOpenHandsAPI } from '@openhands/client';

export function useResearchExtension() {
  const api = useOpenHandsAPI();
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/research/experiments');
      setExperiments(response.data);
    } finally {
      setLoading(false);
    }
  };

  const startResearch = async (goal: string, type: string) => {
    setLoading(true);
    try {
      await api.post('/api/research/start', {
        goal,
        research_type: type,
        session_id: api.sessionId
      });
      await loadExperiments();
    } finally {
      setLoading(false);
    }
  };

  return {
    experiments,
    startResearch,
    loading,
    refresh: loadExperiments
  };
}
```

### Pros & Cons

#### Pros ✅

1. **Clean Separation**
   - UAgent code lives in `extensions/` directory
   - No modifications to OpenHands core required
   - Clear boundaries between systems

2. **Easy Maintenance**
   - Can update OpenHands independently
   - Extension has its own versioning
   - Testing can be isolated

3. **Gradual Migration**
   - Can migrate one feature at a time
   - Easy to test each component
   - Can maintain backward compatibility

4. **Open Source Friendly**
   - Extension can be separate repository
   - Can have different license if needed
   - Community can contribute extensions

5. **Flexible Deployment**
   - Users can choose to install extension or not
   - Can be distributed via package manager
   - Easy to enable/disable

#### Cons ⚠️

1. **API Constraints**
   - Limited by OpenHands extension API
   - Some features might require core changes
   - May need to contribute to OpenHands core

2. **Potential Duplication**
   - Extension API might not expose everything needed
   - May need to duplicate some utility code
   - Performance overhead from abstraction layer

3. **Integration Complexity**
   - Need to understand OpenHands extension system deeply
   - May require changes to OpenHands if extension API is insufficient
   - Documentation and examples may be limited

### Development Workflow

```bash
# 1. Clone OpenHands
git clone https://github.com/All-Hands-AI/OpenHands.git
cd OpenHands

# 2. Create extension directory
mkdir -p extensions/uagent_research

# 3. Copy UAgent research engines
cp -r /path/to/uagent/backend/app/core/research_engines/* \
      extensions/uagent_research/engines/

# 4. Install extension
pip install -e extensions/uagent_research

# 5. Enable extension in config
echo "extensions:
  - uagent_research" >> config.toml

# 6. Run OpenHands with extension
openhands --config config.toml
```

---

## Option B: Full Merge

### Architecture

```
OpenHands (Merged)
├── openhands/
│   ├── agenthub/
│   │   ├── codeact_agent/
│   │   ├── scientific_research_agent/    # MERGED from UAgent
│   │   ├── code_research_agent/          # MERGED from UAgent
│   │   └── roma_agent/                   # MERGED from UAgent
│   │
│   ├── controller/
│   │   ├── agent_controller.py
│   │   └── research_controller.py        # MERGED: Research orchestration
│   │
│   ├── runtime/
│   │   ├── docker/
│   │   └── experiment/                   # MERGED: Experiment runtime
│   │
│   ├── research/                          # NEW: Core research logic
│   │   ├── engines/
│   │   │   ├── scientific_research.py
│   │   │   ├── code_research.py
│   │   │   └── roma_engine.py
│   │   │
│   │   ├── validation/
│   │   │   └── experiment_validator.py
│   │   │
│   │   └── workspace/
│   │       └── experiment_workspace.py
│   │
│   ├── server/
│   │   ├── routes/
│   │   │   ├── conversations.py
│   │   │   └── research.py               # MERGED: Research routes
│   │   │
│   │   └── listen.py                     # MODIFIED: Add research endpoints
│   │
│   └── frontend/
│       └── src/
│           ├── components/
│           │   ├── research/              # MERGED: Research UI
│           │   │   ├── ResearchDashboard.tsx
│           │   │   ├── ResearchTree.tsx
│           │   │   └── ExperimentMonitor.tsx
│           │   │
│           │   └── chat/
│           │       └── ChatInterface.tsx  # MODIFIED: Add research tab
│           │
│           └── services/
│               └── researchService.ts     # MERGED: Research API client
```

### Implementation Details

#### 1. Agent Integration (Direct)

```python
# openhands/agenthub/scientific_research_agent/scientific_research_agent.py

from openhands.core.schema import AgentState
from openhands.events.action import Action
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent

# Direct import of UAgent research engine (now part of OpenHands)
from openhands.research.engines.scientific_research import ScientificResearchEngine

class ScientificResearchAgent(CodeActAgent):
    """
    Scientific Research Agent - merged from UAgent.

    This agent specializes in conducting scientific experiments,
    hypothesis testing, and research workflows.
    """

    sandbox_plugins = CodeActAgent.sandbox_plugins + [
        "jupyter",  # For data analysis
        "postgresql",  # For database experiments
    ]

    def __init__(self, llm):
        super().__init__(llm)
        self.research_engine = ScientificResearchEngine(llm)
        self.current_experiment = None

    async def step(self, state: AgentState) -> Action:
        """Execute research step"""
        # Deep integration - can access all OpenHands internals
        return await self.research_engine.plan_next_action(state)
```

#### 2. Server Integration (Direct)

```python
# openhands/server/routes/research.py

from fastapi import APIRouter, WebSocket
from openhands.controller.agent_controller import AgentController
from openhands.agenthub.scientific_research_agent import ScientificResearchAgent
from openhands.research.engines.roma_engine import ROMAEngine

router = APIRouter(prefix="/api/research")

@router.post("/experiments/start")
async def start_experiment(request: ExperimentRequest):
    """Start scientific research experiment"""

    # Create research agent
    agent = ScientificResearchAgent(llm=get_llm_for_research())

    # Create controller with research agent
    controller = AgentController(
        agent=agent,
        workdir=request.workspace,
        max_iterations=request.max_iterations or 100
    )

    # Run research
    result = await controller.run(request.goal)

    return {
        "experiment_id": result.id,
        "status": result.status,
        "results": result.output
    }

@router.websocket("/experiments/{experiment_id}/stream")
async def stream_experiment(websocket: WebSocket, experiment_id: str):
    """Stream experiment progress in real-time"""
    await websocket.accept()

    # Direct access to OpenHands event stream
    async for event in get_experiment_event_stream(experiment_id):
        await websocket.send_json({
            "type": event.type,
            "data": event.data
        })
```

#### 3. Frontend Integration (Direct)

```typescript
// openhands/frontend/src/components/chat/ChatInterface.tsx
// MODIFIED to add research tab

import { ResearchPanel } from '../research/ResearchPanel';

function ChatInterface() {
  const [activeMode, setActiveMode] = useState<'chat' | 'research'>('chat');

  return (
    <div className="interface">
      <Tabs value={activeMode} onChange={setActiveMode}>
        <Tab value="chat">Chat</Tab>
        <Tab value="research">Research</Tab>  {/* NEW */}
      </Tabs>

      {activeMode === 'chat' ? (
        <ChatPanel />
      ) : (
        <ResearchPanel />  {/* Direct integration */}
      )}
    </div>
  );
}
```

### Pros & Cons

#### Pros ✅

1. **Deepest Integration**
   - No API boundaries
   - Can optimize performance across entire stack
   - Single unified codebase

2. **Consistent User Experience**
   - Research features feel native
   - Uniform UI/UX patterns
   - Single authentication/authorization

3. **Simplified Deployment**
   - One application to deploy
   - No inter-service communication
   - Easier dependency management

#### Cons ❌

1. **Massive Refactoring**
   - Need to refactor ALL UAgent code to fit OpenHands patterns
   - High risk of breaking things
   - Months of work

2. **Maintenance Complexity**
   - Hard to track OpenHands updates
   - Merge conflicts on every update
   - Need to maintain fork or contribute to upstream

3. **All-or-Nothing**
   - Can't migrate gradually
   - Hard to test in isolation
   - Difficult to rollback

4. **Upstream Dependency**
   - Need OpenHands maintainers to accept changes
   - May not align with OpenHands roadmap
   - Could be rejected

### Migration Effort

```
Estimated Time: 6-9 months
Estimated LOC Changes: 50,000+
Risk Level: Very High

Week 1-4:   Fork OpenHands, set up development environment
Week 5-12:  Refactor UAgent research engines to OpenHands patterns
Week 13-20: Integrate agents into agenthub
Week 21-24: Merge frontend components
Week 25-28: Database and state management integration
Week 29-32: End-to-end testing
Week 33-36: Performance optimization
Week 37-40: Documentation and migration tools
```

---

## Option C: Microservices Architecture

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway / Load Balancer              │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│  OpenHands   │    │  UAgent Research │    │   Shared     │
│   Service    │    │     Service      │    │   Frontend   │
│              │    │                  │    │              │
│ - Agent Loop │    │ - Research Eng.  │    │ - React App  │
│ - Runtime    │    │ - Experiments    │    │ - OpenHands  │
│ - Code Tasks │    │ - ROMA Engine    │    │ - Research   │
└──────────────┘    └──────────────────┘    └──────────────┘
        │                     │                     │
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│                    Shared Database / Cache                    │
│  - User Sessions   - Experiments   - Agent State   - Results  │
└──────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### 1. Service Communication

```python
# uagent_research_service/main.py

from fastapi import FastAPI
from .research.endpoints import router as research_router
from .integration.openhands_client import OpenHandsClient

app = FastAPI(title="UAgent Research Service")

# Register routers
app.include_router(research_router, prefix="/api/research")

# OpenHands client for delegating code tasks
openhands_client = OpenHandsClient(base_url="http://openhands-service:8000")

@app.post("/api/research/experiments/start")
async def start_experiment(request: ExperimentRequest):
    """Start experiment, delegate code execution to OpenHands"""

    # Create experiment
    experiment = await create_experiment(request)

    # Delegate code execution to OpenHands service
    openhands_task = await openhands_client.create_task(
        task=experiment.code_execution_goal,
        workspace=experiment.workspace,
        context={
            "experiment_id": experiment.id,
            "callback_url": f"http://research-service:8001/api/experiments/{experiment.id}/results"
        }
    )

    return {
        "experiment_id": experiment.id,
        "openhands_task_id": openhands_task.id,
        "status": "running"
    }

@app.post("/api/experiments/{experiment_id}/results")
async def receive_results(experiment_id: str, results: dict):
    """Callback from OpenHands with execution results"""

    # Process results
    experiment = await get_experiment(experiment_id)
    await experiment.process_results(results)

    # Analyze and determine next steps
    next_action = await experiment.analyze_and_plan()

    if next_action:
        # Delegate next code task to OpenHands
        await openhands_client.create_task(
            task=next_action.task,
            workspace=experiment.workspace,
            context={"experiment_id": experiment_id}
        )

    return {"status": "processed"}
```

#### 2. Shared State Management

```python
# shared/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Shared database for both services
DATABASE_URL = "postgresql://user:pass@shared-db:5432/unified_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Shared models
class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    user_id = Column(String)
    mode = Column(String)  # 'chat', 'research', 'hybrid'
    created_at = Column(DateTime)

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    goal = Column(Text)
    status = Column(String)
    results = Column(JSON)
    openhands_tasks = relationship("OpenHandsTask")

class OpenHandsTask(Base):
    __tablename__ = "openhands_tasks"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.id"))
    task = Column(Text)
    status = Column(String)
    output = Column(JSON)
```

#### 3. Frontend Integration

```typescript
// frontend/src/services/api.ts

// Unified API client that routes to appropriate service

class UnifiedAPIClient {
  private openhandsClient: OpenHandsClient;
  private researchClient: ResearchClient;

  constructor() {
    this.openhandsClient = new OpenHandsClient('/openhands');
    this.researchClient = new ResearchClient('/research');
  }

  async startChat(message: string) {
    // Route to OpenHands service
    return this.openhandsClient.sendMessage(message);
  }

  async startResearch(goal: string) {
    // Route to Research service
    return this.researchClient.startExperiment({ goal });
  }

  async getSession(sessionId: string) {
    // Query shared database through either service
    return this.openhandsClient.getSession(sessionId);
  }
}

export const api = new UnifiedAPIClient();
```

#### 4. Docker Compose Deployment

```yaml
# docker-compose.yml

version: '3.8'

services:
  api-gateway:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - openhands-service
      - research-service
      - frontend

  openhands-service:
    build: ./openhands
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/unified_db
      - REDIS_URL=redis://cache:6379
    depends_on:
      - db
      - cache

  research-service:
    build: ./uagent_research
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/unified_db
      - REDIS_URL=redis://cache:6379
      - OPENHANDS_SERVICE_URL=http://openhands-service:8000
    depends_on:
      - db
      - cache
      - openhands-service

  frontend:
    build: ./frontend
    environment:
      - API_GATEWAY_URL=http://api-gateway

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=unified_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass

  cache:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Pros & Cons

#### Pros ✅

1. **Complete Independence**
   - Services can be developed separately
   - Different teams can own different services
   - Easy to version independently

2. **Scalability**
   - Can scale services independently
   - Research-heavy workloads don't affect chat
   - Can use different infrastructure per service

3. **Technology Flexibility**
   - Can use different languages/frameworks per service
   - Easier to upgrade dependencies
   - Can optimize each service independently

4. **Fault Isolation**
   - If research service fails, chat still works
   - Easier to debug issues
   - Can deploy updates independently

#### Cons ❌

1. **Communication Complexity**
   - Need API contracts between services
   - Network latency between services
   - Error handling across service boundaries

2. **Deployment Complexity**
   - Need orchestration (Kubernetes, Docker Compose)
   - More infrastructure to maintain
   - Complex monitoring and logging

3. **Data Consistency**
   - Need distributed transaction handling
   - Potential for data sync issues
   - Complex state management

4. **Development Overhead**
   - Need to run multiple services locally
   - More complex testing
   - Duplicate code across services

### Migration Effort

```
Estimated Time: 4-6 months
Estimated Infrastructure Changes: High
Risk Level: Medium

Week 1-4:   Service boundary definition and API design
Week 5-8:   Extract UAgent into standalone service
Week 9-12:  Implement service communication layer
Week 13-16: Shared database and state management
Week 17-20: Frontend integration
Week 21-24: Deployment and orchestration setup
```

---

## Option D: Hybrid Approach

### Architecture

```
OpenHands Core (Minimal Changes)
├── openhands/
│   ├── agenthub/
│   ├── controller/
│   ├── runtime/
│   └── server/
│       └── listen.py              # MODIFIED: Proxy research requests
│
└── plugins/                        # NEW: Plugin system
    └── research/
        ├── service/                # Microservice for heavy research
        │   ├── engines/
        │   └── api/
        │
        └── integration/            # Light integration into OpenHands
            ├── agents/             # Thin wrapper agents
            │   └── research_delegator_agent.py
            │
            └── routes/             # API routes (proxy to service)
                └── research_proxy.py
```

### Implementation Details

#### 1. Delegator Agent (Lightweight)

```python
# plugins/research/integration/agents/research_delegator_agent.py

from openhands.agenthub.delegator_agent import DelegatorAgent
from openhands.events.action import Action, MessageAction

class ResearchDelegatorAgent(DelegatorAgent):
    """
    Lightweight agent that delegates research tasks to research service.

    This agent runs in OpenHands but delegates heavy research work
    to the separate research microservice.
    """

    async def step(self, state: AgentState) -> Action:
        # Detect if this is a research task
        if self._should_delegate_to_research(state):
            # Delegate to research service
            return await self._delegate_to_research_service(state)
        else:
            # Handle normally
            return await super().step(state)

    async def _delegate_to_research_service(self, state: AgentState) -> Action:
        """Send task to research microservice"""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://research-service:8001/api/research/execute",
                json={
                    "goal": state.task,
                    "context": state.history,
                    "workspace": state.workspace
                }
            )

            result = response.json()

        # Convert research service response to OpenHands action
        return MessageAction(
            content=f"Research completed: {result['summary']}\n\nFull results: {result['url']}"
        )
```

#### 2. API Proxy (Lightweight)

```python
# plugins/research/integration/routes/research_proxy.py

from fastapi import APIRouter, HTTPException
import httpx

router = APIRouter(prefix="/api/research")

RESEARCH_SERVICE_URL = "http://research-service:8001"

@router.post("/experiments/start")
async def start_experiment_proxy(request: dict):
    """Proxy request to research service"""

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{RESEARCH_SERVICE_URL}/api/experiments/start",
                json=request,
                timeout=60.0
            )
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(500, f"Research service unavailable: {e}")

@router.get("/experiments/{experiment_id}/status")
async def get_experiment_status_proxy(experiment_id: str):
    """Proxy request to research service"""

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{RESEARCH_SERVICE_URL}/api/experiments/{experiment_id}/status"
        )
        return response.json()

# WebSocket proxy for real-time updates
@router.websocket("/experiments/{experiment_id}/stream")
async def stream_experiment_proxy(websocket: WebSocket, experiment_id: str):
    """Proxy WebSocket connection to research service"""

    await websocket.accept()

    async with websockets.connect(
        f"ws://research-service:8001/api/experiments/{experiment_id}/stream"
    ) as research_ws:
        # Bidirectional proxy
        async def forward_to_client():
            async for message in research_ws:
                await websocket.send_text(message)

        async def forward_to_service():
            async for message in websocket.iter_text():
                await research_ws.send(message)

        await asyncio.gather(
            forward_to_client(),
            forward_to_service()
        )
```

#### 3. Research Service (Heavy Lifting)

```python
# plugins/research/service/main.py

from fastapi import FastAPI
from .engines.scientific_research import ScientificResearchEngine
from .engines.roma_engine import ROMAEngine

app = FastAPI(title="UAgent Research Service")

@app.post("/api/research/execute")
async def execute_research(request: ResearchRequest):
    """Execute research task (heavy computation here)"""

    # Create appropriate research engine
    if request.research_type == "scientific":
        engine = ScientificResearchEngine()
    elif request.research_type == "roma":
        engine = ROMAEngine()

    # Run research (this is CPU/memory intensive)
    result = await engine.run(
        goal=request.goal,
        context=request.context,
        workspace=request.workspace
    )

    return {
        "experiment_id": result.id,
        "status": result.status,
        "summary": result.summary,
        "url": f"/experiments/{result.id}/results"
    }

@app.post("/api/experiments/start")
async def start_experiment(request: ExperimentRequest):
    """Start long-running experiment"""

    # Create experiment in database
    experiment = await create_experiment(request)

    # Run in background
    background_tasks.add_task(
        run_experiment_background,
        experiment_id=experiment.id,
        goal=request.goal
    )

    return {
        "experiment_id": experiment.id,
        "status": "started"
    }
```

### Pros & Cons

#### Pros ✅

1. **Best of Both Worlds**
   - Light integration into OpenHands (easy to maintain)
   - Heavy research work isolated in service (scalable)
   - Clear separation of concerns

2. **Flexible Deployment**
   - Can deploy research service separately or embedded
   - Can scale research service independently
   - Easy to enable/disable research features

3. **Gradual Migration**
   - Start with service separation
   - Gradually integrate more tightly if needed
   - Low risk migration path

4. **Performance Optimization**
   - Research service can use different hardware (GPUs, more RAM)
   - OpenHands stays lightweight
   - Can optimize each component independently

#### Cons ⚠️

1. **Still Some Duplication**
   - Need both agent and service code
   - Some API/routing code duplicated
   - Two codebases to maintain

2. **Network Dependency**
   - Research features require service to be running
   - Network latency for research requests
   - Need to handle service failures

3. **Complexity**
   - More complex than pure plugin
   - More components to deploy than full merge
   - Need to coordinate versions between components

---

## Decision Matrix

### Evaluation Criteria

| Criteria | Weight | Plugin | Full Merge | Microservices | Hybrid |
|----------|--------|--------|------------|---------------|--------|
| **Ease of Implementation** | 20% | 9/10 | 3/10 | 6/10 | 7/10 |
| **Maintenance Burden** | 20% | 9/10 | 4/10 | 6/10 | 7/10 |
| **Performance** | 15% | 7/10 | 10/10 | 6/10 | 8/10 |
| **Scalability** | 15% | 6/10 | 7/10 | 10/10 | 9/10 |
| **Flexibility** | 10% | 8/10 | 4/10 | 10/10 | 9/10 |
| **Integration Depth** | 10% | 7/10 | 10/10 | 5/10 | 7/10 |
| **Risk Level** | 10% | 9/10 | 3/10 | 7/10 | 8/10 |
| ****Total Score** | 100% | **8.0** | **5.6** | **7.1** | **7.9** |

### Detailed Analysis

#### Option A: Plugin/Extension Model
- **Best for**: Quick integration, minimal risk, easy maintenance
- **Use when**: OpenHands extension API is sufficient
- **Avoid when**: Need very deep integration or performance is critical
- **Score: 8.0/10** ⭐

#### Option B: Full Merge
- **Best for**: Deepest integration, maximum performance
- **Use when**: UAgent becomes core part of OpenHands
- **Avoid when**: Want to maintain independence or move fast
- **Score: 5.6/10**

#### Option C: Microservices
- **Best for**: Independent scaling, fault isolation
- **Use when**: Have DevOps resources and need scalability
- **Avoid when**: Simple deployment is priority
- **Score: 7.1/10**

#### Option D: Hybrid
- **Best for**: Balance between integration and independence
- **Use when**: Want best of plugin + microservices
- **Avoid when**: Want simpler architecture
- **Score: 7.9/10** ⭐

---

## Recommendation

### Primary Recommendation: Option A (Plugin/Extension) ⭐

**Why**:
1. Lowest risk, fastest to implement
2. Easy to maintain alongside OpenHands updates
3. Can migrate to other options later if needed
4. Good enough for 90% of use cases

**When to Choose**:
- Want to get research features working ASAP
- OpenHands extension API covers needs
- Small team (1-3 developers)
- Want to contribute back to OpenHands ecosystem

### Alternative Recommendation: Option D (Hybrid)

**Why**:
1. Better scalability than pure plugin
2. Still relatively easy to implement
3. Flexibility to optimize performance
4. Good for production deployment

**When to Choose**:
- Expect high research workload
- Need independent scaling
- Have DevOps resources
- Want maximum flexibility

### NOT Recommended: Option B (Full Merge)

Unless:
- OpenHands maintainers explicitly want UAgent features in core
- Have 6+ months for integration
- Can dedicate team to tracking OpenHands changes

### Consider Later: Option C (Microservices)

If:
- Research workload grows significantly
- Need independent scaling for research vs chat
- Have Kubernetes/cloud infrastructure
- Can convert from Plugin (Option A) to Microservices later

---

## Migration Path

### Phase 1: Start with Plugin (Option A)
- Week 1-12: Implement as extension
- Get working prototype
- Test with real users

### Phase 2: Evaluate
- Month 4: Analyze performance and usage patterns
- Identify bottlenecks
- Determine if need more scalability

### Phase 3: Evolve if Needed
- **If performance is good**: Stay with Plugin ✅
- **If need scalability**: Migrate to Hybrid (Option D)
- **If OpenHands wants to adopt**: Consider Full Merge (Option B)

---

**Next**: See `05_IMPLEMENTATION_ROADMAP.md` for step-by-step implementation guide for Option A (Plugin).
