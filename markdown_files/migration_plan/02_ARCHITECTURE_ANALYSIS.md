# Architecture Analysis: UAgent vs OpenHands

## Table of Contents
1. [UAgent Architecture](#uagent-architecture)
2. [OpenHands Architecture](#openhands-architecture)
3. [Component Mapping](#component-mapping)
4. [Integration Points](#integration-points)

---

## UAgent Architecture

### Backend Structure

```
backend/
├── app/
│   ├── api/                          # FastAPI routes
│   │   ├── research.py              # Research endpoints
│   │   ├── experiments.py           # Experiment management
│   │   ├── sessions.py              # Session management
│   │   └── websocket.py             # Real-time updates
│   │
│   ├── core/                         # Core business logic
│   │   ├── research_engines/        # ⭐ KEY COMPONENT
│   │   │   ├── scientific_research.py    # Scientific experiments
│   │   │   ├── code_research.py          # Code analysis (RepoMaster)
│   │   │   └── roma_engine.py            # Tree-based research
│   │   │
│   │   ├── exec/                    # Execution engines
│   │   │   ├── ras.py              # Research-as-Service
│   │   │   ├── ras_executor.py     # Experiment executor
│   │   │   └── ras_validator.py    # Result validation
│   │   │
│   │   ├── openhands/              # OpenHands integration
│   │   │   ├── client.py           # OpenHands client
│   │   │   ├── workspace_manager.py
│   │   │   └── streaming_integration.py
│   │   │
│   │   ├── llm/                    # LLM abstraction
│   │   │   ├── streaming_llm_client.py
│   │   │   └── smart_router.py     # Multi-provider routing
│   │   │
│   │   ├── session_manager.py      # Session orchestration
│   │   ├── experiment_manager.py   # Experiment lifecycle
│   │   └── websocket_manager.py    # Real-time communication
│   │
│   └── integrations/                # External integrations
│       ├── openhands_runtime.py
│       ├── openhands_single_container.py
│       └── repomaster_bridge.py
│
└── main.py                          # FastAPI application
```

### Frontend Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── research/               # ⭐ KEY COMPONENT
│   │   │   ├── ResearchTree.tsx   # ROMA tree visualization
│   │   │   ├── ExperimentDashboard.tsx
│   │   │   ├── IdeaGenerator.tsx
│   │   │   └── HypothesisPanel.tsx
│   │   │
│   │   ├── chat/                   # Chat interface
│   │   │   ├── ChatInterface.tsx
│   │   │   └── MessageList.tsx
│   │   │
│   │   └── workspace/              # Workspace management
│   │       └── WorkspaceViewer.tsx
│   │
│   ├── hooks/                      # React hooks
│   │   ├── useResearch.ts
│   │   ├── useExperiments.ts
│   │   └── useWebSocket.ts
│   │
│   ├── services/                   # API clients
│   │   ├── api.ts
│   │   └── websocket.ts
│   │
│   └── store/                      # State management
│       └── researchStore.ts
```

### Key Features

| Feature | Implementation | Complexity |
|---------|---------------|------------|
| **Scientific Research** | `scientific_research.py` with OpenHands integration | High |
| **ROMA Tree** | Tree-based parallel research with visualization | High |
| **Idea Generation** | LLM-based idea generation with scoring | Medium |
| **Hypothesis Testing** | Automated experiment design and execution | High |
| **Code Research** | RepoMaster integration for codebase analysis | Medium |
| **Multi-LLM** | Smart routing across providers | Medium |
| **Real-time Updates** | WebSocket-based streaming | Low |

---

## OpenHands Architecture

### Backend Structure

```
openhands/
├── agenthub/                        # Agent implementations
│   ├── codeact_agent/              # Code-focused agent
│   ├── delegator_agent/            # Task delegation
│   └── planner_agent/              # Planning agent
│
├── controller/                      # Agent orchestration
│   ├── agent_controller.py         # Main loop
│   └── state/                      # State management
│
├── runtime/                         # Execution environments
│   ├── cli/                        # CLI runtime
│   ├── docker/                     # Docker runtime
│   ├── remote/                     # Remote runtime
│   └── action_execution_server.py  # Action executor
│
├── events/                          # Event system
│   ├── action/                     # Action events
│   ├── observation/                # Observation events
│   └── stream.py                   # Event streaming
│
├── llm/                            # LLM integration
│   ├── llm.py                      # LLM abstraction
│   └── providers/                  # Provider implementations
│
├── server/                         # Web server
│   ├── routes/                     # API routes
│   ├── session/                    # Session management
│   └── listen.py                   # Main server
│
└── core/                           # Core components
    ├── config.py
    ├── schema/
    └── logger.py
```

### Frontend Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── chat/                   # Chat interface
│   │   │   ├── ChatInterface.tsx
│   │   │   └── Messages.tsx
│   │   │
│   │   ├── features/               # Feature components
│   │   │   ├── terminal/
│   │   │   ├── browser/
│   │   │   └── jupyter/
│   │   │
│   │   └── modals/                 # Modal dialogs
│   │
│   ├── services/
│   │   ├── chatService.ts
│   │   └── settingsService.ts
│   │
│   ├── state/                      # State management
│   │   └── chatSlice.ts
│   │
│   └── api/
│       └── open-hands.ts           # API client
```

### Key Features

| Feature | Implementation | Extensibility |
|---------|---------------|---------------|
| **Agent Loop** | `agent_controller.py` | High - pluggable agents |
| **Multi-runtime** | `runtime/` with abstract base | High - new runtimes easy |
| **Action System** | Event-based with observations | High - extensible actions |
| **Plugin System** | Runtime plugins | Medium - limited scope |
| **Web UI** | React with WebSocket | Medium - component-based |
| **Session Management** | Server-side sessions | High - isolated contexts |

---

## Component Mapping

### Backend Components

| UAgent Component | OpenHands Equivalent | Integration Strategy |
|------------------|---------------------|---------------------|
| `research_engines/` | New agent types | ✅ Add as specialized agents |
| `exec/ras_executor.py` | `runtime/action_execution_server.py` | ✅ Extend action system |
| `llm/streaming_llm_client.py` | `llm/llm.py` | ⚠️ Merge or adapt |
| `session_manager.py` | `server/session/` | ⚠️ Extend existing |
| `websocket_manager.py` | `server/listen.py` WebSocket | ✅ Use OpenHands' |
| `openhands/client.py` | N/A (internal) | ❌ Remove (no longer needed) |
| `experiment_manager.py` | New module | ✅ Add as extension |
| `smart_router.py` | `llm/llm.py` | ⚠️ Add routing logic |

### Frontend Components

| UAgent Component | OpenHands Equivalent | Integration Strategy |
|------------------|---------------------|---------------------|
| `ResearchTree.tsx` | New component | ✅ Add to features/ |
| `ExperimentDashboard.tsx` | New component | ✅ Add to features/ |
| `IdeaGenerator.tsx` | New component | ✅ Add to features/ |
| `ChatInterface.tsx` | `chat/ChatInterface.tsx` | ⚠️ Extend existing |
| `useResearch.ts` | New hook | ✅ Add custom hook |
| `researchStore.ts` | `chatSlice.ts` | ⚠️ Extend state |
| `websocket.ts` | Existing WebSocket | ✅ Use OpenHands' |

**Legend**:
- ✅ Clean integration - add as new module
- ⚠️ Requires adaptation - merge/extend existing
- ❌ Remove - no longer needed

---

## Integration Points

### 1. Agent System Integration

**UAgent Research Engines → OpenHands Agents**

```python
# Current UAgent
class ScientificResearchEngine:
    async def execute_research(self, goal: str) -> ResearchResult:
        # Complex research logic
        ...

# OpenHands Integration
class ResearchAgent(Agent):
    """OpenHands agent for scientific research"""

    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.research_engine = ScientificResearchEngine(llm)

    async def step(self, state: State) -> Action:
        # Use research engine to generate next action
        research_plan = await self.research_engine.plan_next_step(state)
        return convert_to_openhands_action(research_plan)
```

**Benefits**:
- ✅ Leverages OpenHands' mature agent loop
- ✅ Consistent with OpenHands architecture
- ✅ Easy to test and debug

### 2. Runtime Integration

**UAgent Experiment Execution → OpenHands Runtime**

```python
# Current UAgent
class OpenHandsSingleContainer:
    async def run(self, config):
        # Manages Docker container
        ...

# OpenHands Integration
class ResearchRuntime(Runtime):
    """Extended runtime for research experiments"""

    async def run_experiment(self, experiment: Experiment) -> Result:
        # Use OpenHands runtime infrastructure
        action = ExperimentAction(experiment)
        observation = await self.execute_action(action)
        return parse_experiment_result(observation)
```

**Benefits**:
- ✅ No need to manage Docker ourselves
- ✅ Consistent runtime behavior
- ✅ Better error handling

### 3. LLM Integration

**UAgent Smart Router → OpenHands LLM System**

```python
# UAgent Smart Router
class SmartRouter:
    async def route(self, prompt: str) -> str:
        # Route to best LLM provider
        provider = self.select_provider(prompt)
        return await provider.complete(prompt)

# OpenHands Integration (Option A: Extend)
class ResearchLLM(LLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.router = SmartRouter()

    async def completion(self, messages) -> str:
        # Use smart routing
        return await self.router.route(messages)

# OpenHands Integration (Option B: Middleware)
class LLMRouter:
    """Middleware for LLM routing"""
    def select_llm_for_task(self, task_type: str) -> LLM:
        if task_type == "research":
            return get_research_llm()
        elif task_type == "code":
            return get_code_llm()
        ...
```

### 4. State Management

**UAgent Session State → OpenHands State**

```python
# Current UAgent
class ResearchSession:
    research_tree: ResearchTree
    experiments: List[Experiment]
    current_hypothesis: Hypothesis

# OpenHands Integration
class ResearchState(State):
    """Extended state for research sessions"""
    research_tree: Optional[ResearchTree] = None
    experiments: List[Experiment] = []
    current_hypothesis: Optional[Hypothesis] = None

    # Inherit from OpenHands State
    history: List[Event] = []
    ...
```

### 5. UI Integration

**UAgent Frontend → OpenHands Frontend**

```typescript
// Current UAgent
function ResearchDashboard() {
  const { experiments } = useResearch();
  return <ExperimentList experiments={experiments} />;
}

// OpenHands Integration
// Add new route in OpenHands app
function App() {
  return (
    <Routes>
      <Route path="/" element={<ChatInterface />} />
      <Route path="/research" element={<ResearchDashboard />} /> {/* NEW */}
      <Route path="/experiments" element={<ExperimentView />} /> {/* NEW */}
    </Routes>
  );
}

// Add research tab to main interface
function MainInterface() {
  const [activeTab, setActiveTab] = useState('chat');

  return (
    <Tabs value={activeTab} onChange={setActiveTab}>
      <Tab value="chat">Chat</Tab>
      <Tab value="research">Research</Tab> {/* NEW */}
      <Tab value="experiments">Experiments</Tab> {/* NEW */}
    </Tabs>
  );
}
```

---

## Data Flow Comparison

### UAgent Current Flow

```
User Request
    ↓
FastAPI Route
    ↓
Research Engine
    ↓
OpenHands Client (subprocess)
    ↓
OpenHands Container
    ↓
Results back through chain
    ↓
WebSocket to Frontend
    ↓
React Components Update
```

### OpenHands Integrated Flow

```
User Request
    ↓
OpenHands Server (FastAPI-like)
    ↓
Agent Controller
    ↓
Research Agent (using research engine)
    ↓
Runtime (CLI/Docker/etc)
    ↓
Action Execution
    ↓
Event Stream
    ↓
WebSocket to Frontend
    ↓
React Components Update
```

**Key Differences**:
- ❌ No external OpenHands subprocess
- ✅ Direct integration with agent loop
- ✅ Unified event system
- ✅ Single WebSocket connection

---

## Technical Debt & Cleanup Opportunities

### Can Be Removed After Integration

1. ✂️ **`integrations/openhands_*.py`** - All OpenHands integration code
2. ✂️ **`openhands/client.py`** - Direct OpenHands client
3. ✂️ **`docker_container_manager.py`** - Use OpenHands' runtime
4. ✂️ **Duplicate WebSocket handling** - Use OpenHands' system

### Must Be Preserved/Migrated

1. ✅ **`research_engines/`** - Core research logic
2. ✅ **`exec/ras_*.py`** - Experiment execution logic
3. ✅ **Frontend research components** - UI visualizations
4. ✅ **`smart_router.py`** - LLM routing logic
5. ✅ **Research data models** - Experiment schemas

### Estimated Code Reduction

| Category | Current Lines | After Integration | Reduction |
|----------|--------------|-------------------|-----------|
| Backend Integration Code | ~5,000 | ~500 | **90%** |
| Duplicate Infrastructure | ~3,000 | ~0 | **100%** |
| Frontend Duplication | ~2,000 | ~500 | **75%** |
| **Total** | **~10,000** | **~1,000** | **90%** |

**Net Result**: Simpler, cleaner codebase with same functionality!

---

## Conclusion

**Integration Feasibility**: ✅ **HIGH**

**Key Findings**:
1. OpenHands has excellent extension points (agent system, runtime, UI)
2. Most UAgent features can map cleanly to OpenHands concepts
3. Significant code reduction possible (90% of integration code)
4. Clean separation allows gradual migration

**Recommended Path**: **Plugin/Extension Model** (see Executive Summary)

---

**Next**: See `03_INTEGRATION_APPROACHES.md` for detailed comparison of integration strategies
