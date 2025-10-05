# DeepResearch + RepoMaster Integration Plan
## Unified Research Tree Architecture for OpenHands

**Status**: Design Phase
**Last Updated**: 2025-10-04
**Based on**: Codex AI Architecture Review

---

## Executive Summary

This plan integrates **Tongyi DeepResearch** (web-based research) and **RepoMaster** (code repository exploration) into OpenHands as pluggable adapters under the `uagent_research` extension. The integration preserves OpenHands core while adding powerful research capabilities through:

1. **Unified Research Tree**: Single tree showing Ideas → Hypotheses → Web/Code Search → Experiments → Results
2. **Agent Adapters**: Pluggable adapters for DeepResearch, RepoMaster, and CodeActAgent
3. **Parallel Execution**: AsyncIO-based tree search with branch-level concurrency
4. **Shared Tools**: Standardized tools (Serper, Jina, GitHub API) accessible by all adapters
5. **Episode Recording**: AgentFounder-style trajectory logging for continual pre-training

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ResearchTreeView                                             │  │
│  │  ├─ Root: "Compare ML algorithms"                            │  │
│  │  ├─ Idea 1: "Use neural architecture search" (DeepResearch)  │  │
│  │  │  ├─ WebSearch: "NAS papers" (Serper)                     │  │
│  │  │  ├─ WebBrowse: "arxiv.org/..." (Jina)                    │  │
│  │  │  └─ Summary: "NAS shows promise..."                      │  │
│  │  ├─ Idea 2: "Find existing implementations" (RepoMaster)     │  │
│  │  │  ├─ GitHubSearch: "neural architecture search pytorch"   │  │
│  │  │  ├─ RepoAnalysis: "microsoft/nni"                        │  │
│  │  │  └─ CodeExecution: "python examples/nas/..."            │  │
│  │  └─ Hypothesis 1: "NAS+NNI achieves >90% accuracy"          │  │
│  │     └─ Experiment (CodeActAgent): Run benchmarks            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                            ↕ SSE/WebSocket Events
┌─────────────────────────────────────────────────────────────────────┐
│              TreeSearchOrchestrator (Backend)                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Branch Scheduler (TaskGroup + PriorityQueue)                │  │
│  │  ├─ Branch 1: DeepResearchAdapter → WebWalker → WebResummer │  │
│  │  ├─ Branch 2: RepoMasterAdapter → Scheduler → CodeExplorer  │  │
│  │  └─ Branch 3: CodeActAgentAdapter → Sandbox Execution       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  SkillRouter (Intelligent Dispatch)                          │  │
│  │  ├─ "search web" → DeepResearchAdapter                      │  │
│  │  ├─ "find GitHub repo" → RepoMasterAdapter                  │  │
│  │  └─ "run code" → CodeActAgentAdapter                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Shared Tool Registry                                        │  │
│  │  ├─ SerperSearchTool (web search)                           │  │
│  │  ├─ JinaReadTool (web page reading)                         │  │
│  │  ├─ GitHubSearchTool (repo discovery)                       │  │
│  │  ├─ RepoAnalysisTool (code understanding)                   │  │
│  │  └─ PythonExecutionTool (sandbox)                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────────────┐
│              Database (SQLite + JSON research_tree)                  │
│  research_tree: {                                                    │
│    nodes: {id: {type, title, content, status, parent_id, ...}},     │
│    edges: [{from, to, relation}],                                   │
│    episodes: [{trajectory, rewards, artifacts}]                     │
│  }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Agent Adapters

All adapters implement the same interface:

```python
class AgentAdapter(ABC):
    """Base adapter interface for research agents"""

    @abstractmethod
    async def run(self, task: Task, context: Context) -> AsyncIterator[Event]:
        """
        Execute task and yield events.

        Args:
            task: Task with goal, constraints, budget
            context: Execution context (parent nodes, tools, secrets)

        Yields:
            Event: Plan, Step, ToolCall, Observation, Summary, Complete
        """
        pass

    @abstractmethod
    async def cancel(self):
        """Cancel ongoing execution"""
        pass
```

#### a) DeepResearchAdapter

**Source**: `/home/wuy/AI/UAgent/DeepResearch/`

**Wraps**:
- `inference/run_multi_react.py`: ReAct loop with tools
- `WebWalker`: Multi-hop query planning
- `WebResummer`: Result aggregation
- `WebSailor`: Deep browsing

**Tools Used**:
- `SerperSearchTool`: Web search via Serper API
- `JinaReadTool`: Web page reading via Jina Reader API
- `PythonExecutionTool`: Code execution in sandbox

**Event Flow**:
```
Plan("Search for NAS papers")
  → ToolCall(SerperSearchTool, query="neural architecture search")
  → Observation(results=[...])
  → ToolCall(JinaReadTool, url="https://arxiv.org/...")
  → Observation(content="...")
  → Summary("NAS is a technique for...")
  → Complete(artifacts=[urls, snippets, summary])
```

**File**: `extensions/uagent_research/adapters/deepresearch/adapter.py`

```python
class DeepResearchAdapter(AgentAdapter):
    """Adapter for Tongyi DeepResearch (web research)"""

    def __init__(self, config: DeepResearchConfig):
        self.walker = WebWalker(config)
        self.resummer = WebResummer(config)
        self.tools = [SerperSearchTool(), JinaReadTool(), PythonExecutionTool()]

    async def run(self, task: Task, context: Context) -> AsyncIterator[Event]:
        # 1. Plan multi-hop queries
        plan = await self.walker.plan(task.goal)
        yield PlanEvent(steps=plan.steps)

        # 2. Execute ReAct loop
        for step in plan.steps:
            action = await self._select_action(step, context)
            yield StepEvent(action=action)

            tool_call = await self._execute_tool(action)
            yield ToolCallEvent(tool=tool_call.name, args=tool_call.args)

            observation = await tool_call.invoke()
            yield ObservationEvent(result=observation)

        # 3. Aggregate results
        summary = await self.resummer.summarize(observations)
        yield SummaryEvent(content=summary.content, citations=summary.citations)

        yield CompleteEvent(artifacts=summary.artifacts)
```

#### b) RepoMasterAdapter

**Source**: `/home/wuy/AI/UAgent/RepoMaster/`

**Wraps**:
- `src/core/agent_scheduler.py`: Multi-agent orchestrator
- `src/services/agents/deep_search_agent.py`: GitHub repo search
- `src/core/agent_code_explore.py`: Repository analysis
- Autogen group chat framework

**Tools Used**:
- `GitHubSearchTool`: Search GitHub repos
- `RepoAnalysisTool`: Analyze code structure
- `CodeExecutionTool`: Run code in sandbox

**Event Flow**:
```
Plan("Find NAS implementations on GitHub")
  → ToolCall(GitHubSearchTool, query="neural architecture search pytorch")
  → Observation(repos=[{name: "microsoft/nni", stars: 13k, ...}])
  → ToolCall(RepoAnalysisTool, repo="microsoft/nni")
  → Observation(structure={main_modules: [...], entry_points: [...]})
  → ToolCall(CodeExecutionTool, command="python examples/nas/darts/main.py")
  → Observation(output="Accuracy: 92.3%")
  → Summary("NNI provides DARTS implementation with 92.3% accuracy")
  → Complete(artifacts=[repo_url, code_files, execution_results])
```

**File**: `extensions/uagent_research/adapters/repomaster/adapter.py`

```python
class RepoMasterAdapter(AgentAdapter):
    """Adapter for RepoMaster (code repository research)"""

    def __init__(self, config: RepoMasterConfig):
        self.scheduler = RepoMasterAgent(llm_config=config.llm_config)
        self.autogen_session = AutogenSession()
        self.tools = [GitHubSearchTool(), RepoAnalysisTool(), CodeExecutionTool()]

    async def run(self, task: Task, context: Context) -> AsyncIterator[Event]:
        # 1. Start autogen group chat
        chat_result = await self.autogen_session.start(
            agents=[self.scheduler.scheduler_agent, self.scheduler.user_proxy],
            message=task.goal
        )

        # 2. Stream autogen turns as events
        async for turn in chat_result:
            if turn.role == "scheduler_agent":
                if turn.tool_calls:
                    for tool_call in turn.tool_calls:
                        yield ToolCallEvent(
                            tool=tool_call.name,
                            args=tool_call.arguments
                        )
                else:
                    yield StepEvent(reasoning=turn.content)

            elif turn.role == "user_proxy":
                yield ObservationEvent(result=turn.content)

            if "TERMINATE" in turn.content:
                break

        # 3. Extract final summary
        summary = await self._extract_summary(chat_result)
        yield SummaryEvent(content=summary)

        yield CompleteEvent(artifacts=self._collect_artifacts(chat_result))
```

#### c) CodeActAgentAdapter

**Wraps**: OpenHands existing CodeActAgent

**File**: `extensions/uagent_research/adapters/codeact/adapter.py`

```python
class CodeActAgentAdapter(AgentAdapter):
    """Adapter for OpenHands CodeActAgent"""

    def __init__(self, runtime, llm_config):
        self.agent = CodeActAgent(llm=LLM(llm_config))
        self.runtime = runtime

    async def run(self, task: Task, context: Context) -> AsyncIterator[Event]:
        # Initialize controller
        controller = AgentController(
            agent=self.agent,
            max_iterations=task.budget.max_iterations
        )

        # Stream agent steps
        async for event in controller.run(task.goal):
            if isinstance(event, AgentDelegateEvent):
                yield StepEvent(action=event.action.to_dict())
            elif isinstance(event, ObservationEvent):
                yield ObservationEvent(result=event.observation.to_dict())

        yield CompleteEvent(artifacts=controller.state.outputs)
```

### 2. Shared Tools

All tools implement a common interface:

```python
class Tool(ABC):
    """Base tool interface"""

    name: str
    cost_per_call: float
    rate_limit: RateLimit

    @abstractmethod
    async def invoke(self, **kwargs) -> ToolResult:
        """Execute tool and return result"""
        pass

    @abstractmethod
    def cache_key(self, **kwargs) -> str:
        """Generate cache key for deduplication"""
        pass
```

**Examples**:

**SerperSearchTool** (`extensions/uagent_research/tools/search/serper_tool.py`):
```python
class SerperSearchTool(Tool):
    name = "web_search"
    cost_per_call = 0.001  # $0.001 per query

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient()
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.rate_limiter = TokenBucket(rate=10, capacity=20)  # 10 QPS

    async def invoke(self, query: str, num_results: int = 10) -> ToolResult:
        # Check cache
        cache_key = self.cache_key(query=query, num_results=num_results)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Rate limit
        await self.rate_limiter.acquire()

        # Call API
        response = await self.client.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": num_results},
            headers={"X-API-KEY": self.api_key}
        )
        response.raise_for_status()

        result = ToolResult(
            success=True,
            data=response.json()["organic"],
            cost=self.cost_per_call
        )

        # Cache
        self.cache[cache_key] = result

        return result

    def cache_key(self, **kwargs) -> str:
        return hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()
```

**JinaReadTool** (`extensions/uagent_research/tools/browse/jina_tool.py`):
```python
class JinaReadTool(Tool):
    name = "web_read"
    cost_per_call = 0.0002  # $0.0002 per page

    async def invoke(self, url: str) -> ToolResult:
        # Use Jina Reader API
        reader_url = f"https://r.jina.ai/{url}"

        response = await self.client.get(
            reader_url,
            headers={"X-Return-Format": "markdown"}
        )

        content = self._clean_html(response.text)

        return ToolResult(
            success=True,
            data={
                "url": url,
                "content": content,
                "word_count": len(content.split())
            },
            cost=self.cost_per_call
        )
```

### 3. TreeSearchOrchestrator

**File**: `extensions/uagent_research/orchestrator/tree_orchestrator.py`

```python
class TreeSearchOrchestrator:
    """
    Orchestrates parallel research tree expansion.

    - Manages multiple branches (ideas, hypotheses, experiments)
    - Routes tasks to appropriate adapters
    - Enforces budgets and concurrency limits
    - Streams events to frontend
    """

    def __init__(
        self,
        research_id: str,
        router: SkillRouter,
        adapters: Dict[str, AgentAdapter],
        config: OrchestratorConfig
    ):
        self.research_id = research_id
        self.router = router
        self.adapters = adapters
        self.config = config

        # State
        self.tree = ResearchTree(research_id=research_id)
        self.active_branches: Dict[str, asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(config.max_parallel)
        self.event_bus = EventBus(research_id)

    async def expand_idea(self, idea: ResearchNode) -> List[ResearchNode]:
        """
        Expand an idea into hypotheses using parallel adapters.

        1. Route idea to appropriate adapter (DeepResearch or RepoMaster)
        2. Execute adapter and collect results
        3. Generate hypotheses from results
        4. Return hypothesis nodes
        """
        # 1. Route to adapter
        adapter_name = await self.router.route(idea)
        adapter = self.adapters[adapter_name]

        # 2. Create task
        task = Task(
            goal=idea.content,
            budget=Budget(
                max_iterations=10,
                max_cost=5.0,
                deadline=datetime.utcnow() + timedelta(minutes=10)
            )
        )

        # 3. Execute with concurrency control
        async with self.semaphore:
            results = []
            async for event in adapter.run(task, context=self._build_context(idea)):
                # Stream event to UI
                await self.event_bus.publish(event)

                # Collect artifacts
                if isinstance(event, (SummaryEvent, CompleteEvent)):
                    results.append(event)

        # 4. Generate hypotheses from results
        hypotheses = await self._generate_hypotheses(idea, results)

        # 5. Add to tree
        for hyp in hypotheses:
            self.tree.add_node(hyp, parent=idea)

        return hypotheses

    async def run_experiment(self, hypothesis: ResearchNode) -> ResearchNode:
        """
        Run experiment to test hypothesis.

        Uses CodeActAgentAdapter to execute code and collect results.
        """
        adapter = self.adapters["codeact"]

        task = Task(
            goal=f"Test hypothesis: {hypothesis.content}",
            budget=Budget(max_iterations=50, max_cost=10.0)
        )

        result_node = ResearchNode(
            id=f"result_{uuid.uuid4().hex[:8]}",
            type=NodeType.RESULT,
            title=f"Results for {hypothesis.title}",
            status=NodeStatus.RUNNING,
            parent_id=hypothesis.id
        )

        self.tree.add_node(result_node, parent=hypothesis)

        async with self.semaphore:
            async for event in adapter.run(task, context=self._build_context(hypothesis)):
                await self.event_bus.publish(event)

                if isinstance(event, CompleteEvent):
                    result_node.status = NodeStatus.COMPLETE
                    result_node.artifacts = event.artifacts
                    result_node.content = event.summary

        return result_node
```

### 4. Event Schema

**File**: `extensions/uagent_research/models/events.py`

```python
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel

class EventType(str, Enum):
    PLAN = "plan"
    STEP = "step"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    SUMMARY = "summary"
    CRITIQUE = "critique"
    COMPLETE = "complete"
    ERROR = "error"

class Event(BaseModel):
    """Base event"""
    type: EventType
    timestamp: datetime
    branch_id: str
    node_id: Optional[str] = None

class PlanEvent(Event):
    type: EventType = EventType.PLAN
    steps: List[str]
    reasoning: str

class StepEvent(Event):
    type: EventType = EventType.STEP
    action: str
    reasoning: Optional[str] = None

class ToolCallEvent(Event):
    type: EventType = EventType.TOOL_CALL
    tool: str
    args: Dict[str, Any]
    cost: float = 0.0

class ObservationEvent(Event):
    type: EventType = EventType.OBSERVATION
    result: Dict[str, Any]
    success: bool = True

class SummaryEvent(Event):
    type: EventType = EventType.SUMMARY
    content: str
    citations: List[str] = []
    confidence: float = 0.0

class CompleteEvent(Event):
    type: EventType = EventType.COMPLETE
    artifacts: List[Artifact]
    summary: str

class ErrorEvent(Event):
    type: EventType = EventType.ERROR
    message: str
    traceback: Optional[str] = None
```

---

## File Structure

```
OpenHands/
├── extensions/
│   └── uagent_research/
│       ├── __init__.py
│       ├── config.py                           # Configuration
│       │
│       ├── models/
│       │   ├── events.py                       # Event models
│       │   ├── research_tree.py                # Tree models
│       │   └── artifacts.py                    # Artifact models
│       │
│       ├── adapters/
│       │   ├── base.py                         # AgentAdapter interface
│       │   ├── codeact/
│       │   │   └── adapter.py                  # CodeActAgentAdapter
│       │   ├── deepresearch/
│       │   │   ├── adapter.py                  # DeepResearchAdapter
│       │   │   ├── webwalker.py                # WebWalker wrapper
│       │   │   ├── resummer.py                 # WebResummer wrapper
│       │   │   └── sailor.py                   # WebSailor wrapper
│       │   └── repomaster/
│       │       ├── adapter.py                  # RepoMasterAdapter
│       │       ├── autogen_session.py          # Autogen wrapper
│       │       └── scheduler_wrapper.py        # Scheduler wrapper
│       │
│       ├── tools/
│       │   ├── base.py                         # Tool interface
│       │   ├── search/
│       │   │   └── serper_tool.py              # SerperSearchTool
│       │   ├── browse/
│       │   │   └── jina_tool.py                # JinaReadTool
│       │   ├── code/
│       │   │   ├── github_search_tool.py       # GitHubSearchTool
│       │   │   └── repo_analysis_tool.py       # RepoAnalysisTool
│       │   └── common/
│       │       ├── cache.py                    # Caching utilities
│       │       ├── rate_limit.py               # Rate limiting
│       │       └── html_utils.py               # HTML parsing
│       │
│       ├── orchestrator/
│       │   ├── tree_orchestrator.py            # Main orchestrator
│       │   ├── branch_executor.py              # Per-branch execution
│       │   └── event_bus.py                    # Event streaming
│       │
│       ├── router/
│       │   └── skill_router.py                 # Intelligent routing
│       │
│       ├── telemetry/
│       │   └── episode_recorder.py             # AgentFounder logging
│       │
│       ├── api/
│       │   └── routes.py                       # FastAPI routes
│       │
│       └── services/
│           ├── github.py                       # GitHub utilities
│           └── vector_store.py                 # Optional embeddings
│
├── frontend/src/extensions/uagent_research/
│   ├── components/
│   │   ├── ResearchTreeView.tsx               # Tree visualization
│   │   ├── NodeInspector.tsx                  # Node details
│   │   └── BranchRunner.tsx                   # Controls
│   ├── hooks/
│   │   └── useResearchTree.ts                 # SSE/WebSocket hook
│   └── api/
│       └── research.ts                        # API client
│
└── DeepResearch/                              # Copied from /home/wuy/AI/UAgent/
    ├── inference/                             # Use as library
    └── WebAgent/
│
└── RepoMaster/                                # Copied from /home/wuy/AI/UAgent/
    └── src/                                   # Use as library
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create extension structure
- [ ] Define AgentAdapter interface
- [ ] Implement Event models
- [ ] Create SkillRouter (heuristic-based)
- [ ] Setup EventBus

### Phase 2: Tools (Week 2)
- [ ] Implement SerperSearchTool
- [ ] Implement JinaReadTool
- [ ] Implement GitHubSearchTool
- [ ] Add caching and rate limiting
- [ ] Add tool registry

### Phase 3: Adapters (Week 3)
- [ ] Copy DeepResearch source to `extensions/uagent_research/vendor/deepresearch/`
- [ ] Implement DeepResearchAdapter
- [ ] Copy RepoMaster source to `extensions/uagent_research/vendor/repomaster/`
- [ ] Implement RepoMasterAdapter
- [ ] Implement CodeActAgentAdapter

### Phase 4: Orchestration (Week 4)
- [ ] Implement TreeSearchOrchestrator
- [ ] Add branch executor with TaskGroup
- [ ] Implement PUCT scoring (from previous plan)
- [ ] Add budget tracking

### Phase 5: Frontend (Week 5)
- [ ] Create ResearchTreeView component
- [ ] Add NodeInspector for details
- [ ] Implement SSE event streaming
- [ ] Add branch controls

### Phase 6: Testing & Refinement (Week 6)
- [ ] Integration tests
- [ ] End-to-end workflow tests
- [ ] Performance optimization
- [ ] Documentation

---

## Configuration

**Environment Variables**:
```bash
# API Keys
SERPER_API_KEY=xxx
JINA_API_KEY=xxx
GITHUB_TOKEN=xxx

# Paths (optional - for development)
DEEPRESEARCH_PATH=/home/wuy/AI/UAgent/DeepResearch
REPOMASTER_PATH=/home/wuy/AI/UAgent/RepoMaster

# Limits
RESEARCH_MAX_CONCURRENCY=8
RESEARCH_MAX_COST_PER_BRANCH=10.0
RESEARCH_TIMEOUT_SECONDS=600
```

**Optional Dependencies** (`pyproject.toml`):
```toml
[project.optional-dependencies]
deepresearch = ["httpx", "beautifulsoup4", "trafilatura", "lxml", "tenacity"]
repomaster = ["pyautogen", "tiktoken", "docker"]
research_tools = ["httpx", "backoff", "pydantic>=2.0", "rapidfuzz"]
research_all = [
    "uagent-research[deepresearch,repomaster,research_tools]"
]
```

---

## Next Steps

1. **Review this plan** - Provide feedback on architecture
2. **Start Phase 1** - Create extension structure and interfaces
3. **Iterate with Codex** - Get implementation details for adapters

**Ready to proceed?**
