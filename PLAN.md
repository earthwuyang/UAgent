I want to implement a universal agent (uagent) that can do deep reserach (like chatgpt deep research) and do scientific research (like paper2code,ai scientist, ai researcher, agent laboratory) that can do literature review, idea generation, experimentation (code generation, execution, debug like repomaster that can find opensource github repo and utilize existing repos as much as possible), and report generation.

<!-- I previously fail at backup directory, the previous uagent struggles at understanding instructions, executing commands and code as instructed, so now I want to start from scratch but base on the openhands source code, specifically the cli version (headless) version of openhands that run using uvx without docker, but I also want you claude code to build a frontend web page, to enable openhands to do web deep research, github repo search analysis and deploy like repomaster, do parallel tree search like scientific research like ai scientist and agent laboratory.

You should first read the code in backup and openhands (specifically the without docker cli headless part of openhands), and copy necessary files to this directory, for example , create a backend and frontend directories to contain separate source code files to implement this uagent. -->

# DETAILED IMPLEMENTATION PLAN FOR UNIVERSAL AGENT (UAGENT) SYSTEM

## 1. SYSTEM ARCHITECTURE OVERVIEW

### Core Philosophy
- **Simplicity First**: Start with minimal viable functionality, then expand incrementally
- **OpenHands Foundation**: Leverage OpenHands CLI/headless architecture as the execution engine (copy the source code and reuse)
- **Modular Design**: Clean separation between research orchestration and agent execution
- **Progressive Enhancement**: Add advanced features only after core stability is achieved
- **Smart Routing**: Implement intelligent routing to automatically identify whether user requests are for:
  - **Deep Research**: ChatGPT-style comprehensive research across web, academic, and technical sources
  - **Scientific Research**: AI Scientist/AgentLab-style experimental research with hypothesis testing
  - **Code Research**: RepoMaster-style repository analysis and code understanding

### High-Level Architecture with Smart Routing
```
                              ┌─────────────────┐
                              │   User Input    │
                              │   Processing    │
                              └─────────┬───────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │  Smart Router   │
                              │  (LLM-based)    │
                              └─────┬───┬───┬───┘
                                    │   │   │
        ┌───────────────────────────┘   │   └───────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
│  Deep Research  │            │ Scientific Res  │            │  Code Research  │
│     Engine      │            │     Engine      │            │     Engine      │
└─────────┬───────┘            └─────────┬───────┘            └─────────┬───────┘
          │                              │                              │
          └──────────────┐               │               ┌──────────────┘
                         │               │               │
                         ▼               ▼               ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │◄──►│    Backend      │◄──►│  OpenHands CLI  │
│   (React/TS)    │    │   (FastAPI)     │    │   (Execution)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Web Interface   │    │ Research Tree   │    │ Code Execution  │
│ • Dashboard     │    │ • Task Planning │    │ • Workspace Mgmt│
│ • Progress View │    │ • Multi-Agent   │    │ • Tool Integration│
│ • Results View  │    │ • Report Gen    │    │ • Debug Support │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 2. BACKEND SYSTEM DESIGN

### 2.1 Core Components Structure (This is only a demo, you needn't implement these exact files, you can do what you want)
```
backend/
├── app/
│   ├── main.py                 # FastAPI application entry
│   ├── core/
│   │   ├── __init__.py
│   │   ├── smart_router.py     # Intelligent request routing system
│   │   ├── openhands_client.py # OpenHands CLI integration (copied from OpenHands)
│   │   ├── research_engines/   # Specialized research engines
│   │   │   ├── __init__.py
│   │   │   ├── deep_research.py    # Web/academic comprehensive research
│   │   │   ├── scientific_research.py # Experimental/hypothesis research
│   │   │   └── code_research.py    # Repository/code analysis research
│   │   ├── research_tree.py    # Core research methodology
│   │   ├── task_orchestrator.py# Task coordination
│   │   ├── agent_manager.py    # Agent lifecycle management
│   │   └── workspace_manager.py# Workspace and file management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── research.py         # Research data models
│   │   ├── tasks.py            # Task and workflow models
│   │   └── agents.py           # Agent configuration models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── search_service.py   # Multi-modal search
│   │   ├── github_service.py   # GitHub integration
│   │   ├── report_service.py   # Report generation
│   │   └── memory_service.py   # Context and memory management
│   ├── api/
│   │   ├── __init__.py
│   │   ├── research.py         # Research endpoints
│   │   ├── tasks.py            # Task management endpoints
│   │   ├── agents.py           # Agent control endpoints
│   │   └── websocket.py        # Real-time communication
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       ├── logging.py          # Logging setup
│       └── helpers.py          # Utility functions
├── requirements.txt
└── config/
    ├── development.yaml
    └── production.yaml
```

### 2.2 Key Backend Functionalities

#### 2.2.1 Smart Router (`smart_router.py`)
**Purpose**: Intelligent routing system to automatically classify user requests and route to appropriate research engines

**Core Functions**:
- `classify_request()`: Analyze user input to determine research type
- `route_to_engine()`: Direct requests to appropriate research engine
- `validate_routing()`: Ensure routing decision is correct
- `fallback_handler()`: Handle ambiguous or multi-type requests

**Classification Logic**:
```python
class SmartRouter:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.classification_prompt = """
        Analyze the user request and classify it into one of these categories.
        Note: Scientific research is the most complex and may include elements
        of both deep research and code research.

        1. DEEP_RESEARCH: General information gathering, literature reviews,
           market analysis, comprehensive topic exploration, fact-finding

        2. SCIENTIFIC_RESEARCH: **MOST COMPLEX** - Hypothesis testing, experimental
           design, data analysis, research methodology. Often includes:
           - Literature review (deep research component)
           - Code implementation and testing (code research component)
           - Iterative experimentation with feedback loops
           - Statistical analysis and validation
           - Paper writing and peer review

        3. CODE_RESEARCH: Repository analysis, code understanding,
           implementation patterns, technical documentation, best practices

        Classification Priority:
        - If request involves hypothesis, experiments, or scientific methodology → SCIENTIFIC_RESEARCH
        - If request is purely about finding/understanding existing code → CODE_RESEARCH
        - If request is general information gathering without experimentation → DEEP_RESEARCH

        Examples:
        - "Research the latest trends in AI" → DEEP_RESEARCH
        - "Design and run experiments to test if attention mechanisms improve transformer performance" → SCIENTIFIC_RESEARCH
        - "Find transformer implementations and test a new attention mechanism hypothesis" → SCIENTIFIC_RESEARCH
        - "Study how different transformer architectures work" → SCIENTIFIC_RESEARCH (involves experimentation)
        - "Find and analyze open source implementations of transformers" → CODE_RESEARCH
        - "What are the best practices for implementing transformers?" → CODE_RESEARCH
        """

    async def classify_and_route(self, user_request: str) -> Tuple[str, dict]:
        # LLM-based classification with complexity analysis
        classification_result = await self.llm_client.classify(user_request, self.classification_prompt)

        primary_engine = classification_result.engine
        complexity_score = classification_result.complexity_score
        sub_components = classification_result.sub_components

        # Route to appropriate engine with context
        if primary_engine == "SCIENTIFIC_RESEARCH":
            return await self.route_to_scientific_research(
                user_request,
                complexity_score=complexity_score,
                requires_deep_research=sub_components.get("deep_research", False),
                requires_code_research=sub_components.get("code_research", False),
                requires_experimentation=sub_components.get("experimentation", True),
                requires_iteration=sub_components.get("iteration", True)
            )
        elif primary_engine == "DEEP_RESEARCH":
            return await self.route_to_deep_research(user_request)
        elif primary_engine == "CODE_RESEARCH":
            return await self.route_to_code_research(user_request)
        else:
            return await self.handle_mixed_request(user_request, classification_result)

    async def route_to_scientific_research(self, request: str, **kwargs) -> Tuple[str, dict]:
        """
        Scientific research routing with multi-engine coordination
        """
        workflow_plan = {
            "primary_engine": "scientific_research",
            "complexity_level": kwargs.get("complexity_score", 0.8),
            "sub_workflows": [],
            "iteration_enabled": kwargs.get("requires_iteration", True),
            "feedback_loops": True
        }

        # Add sub-workflows based on requirements
        if kwargs.get("requires_deep_research"):
            workflow_plan["sub_workflows"].append({
                "engine": "deep_research",
                "phase": "literature_review",
                "priority": "high"
            })

        if kwargs.get("requires_code_research"):
            workflow_plan["sub_workflows"].append({
                "engine": "code_research",
                "phase": "implementation_analysis",
                "priority": "high"
            })

        # Always include experimentation for scientific research
        workflow_plan["sub_workflows"].append({
            "engine": "scientific_research",
            "phase": "hypothesis_generation",
            "priority": "critical"
        })

        workflow_plan["sub_workflows"].append({
            "engine": "scientific_research",
            "phase": "experimental_design",
            "priority": "critical"
        })

        workflow_plan["sub_workflows"].append({
            "engine": "scientific_research",
            "phase": "code_generation_and_execution",
            "priority": "critical",
            "includes_openhands": True
        })

        workflow_plan["sub_workflows"].append({
            "engine": "scientific_research",
            "phase": "analysis_and_validation",
            "priority": "critical"
        })

        return "SCIENTIFIC_RESEARCH", workflow_plan
```

#### 2.2.2 Research Engines (`research_engines/`)

##### Deep Research Engine (`deep_research.py`)
**Purpose**: ChatGPT-style comprehensive research across multiple sources
**Capabilities**:
- Multi-source web search and synthesis
- Academic paper analysis and citation
- Market research and trend analysis
- Fact-checking and verification
- Comprehensive report generation

##### Scientific Research Engine (`scientific_research.py`)
**Purpose**: AI Scientist/AgentLab-style experimental research - **MOST COMPLEX ENGINE**

**Core Characteristics**:
- **Multi-Engine Orchestrator**: Coordinates and integrates all other research engines
- **Iterative Feedback Loops**: Continuous refinement based on experimental results
- **Code Generation & Execution**: Real-time code creation, testing, and debugging via OpenHands
- **Full Research Lifecycle**: From literature review to peer-reviewed conclusions

**Comprehensive Capabilities**:

1. **Literature Review Phase** (Deep Research Integration):
   - Comprehensive academic paper analysis
   - State-of-the-art research identification
   - Gap analysis and research positioning
   - Citation network analysis

2. **Code Discovery & Analysis** (Code Research Integration):
   - Repository mining for relevant implementations
   - Code pattern analysis and best practices extraction
   - Existing solution evaluation and adaptation
   - Open source library integration assessment

3. **Hypothesis Generation & Experimental Design**:
   - Research question formulation
   - Testable hypothesis generation
   - Experimental methodology design
   - Statistical power analysis and sample size determination
   - Control group and variable identification

4. **Implementation Phase** (Heavy OpenHands Integration):
   - Experimental code generation and setup
   - Data collection pipeline implementation
   - Real-time debugging and error resolution
   - Performance optimization and scaling
   - Version control and experiment tracking

5. **Execution & Monitoring**:
   - Automated experiment execution
   - Real-time progress monitoring
   - Error detection and recovery
   - Resource usage optimization
   - Intermediate result analysis

6. **Analysis & Validation**:
   - Statistical analysis and significance testing
   - Result interpretation and visualization
   - Reproducibility verification
   - Peer review simulation
   - Conclusion formulation and confidence assessment

7. **Iteration & Refinement**:
   - Hypothesis refinement based on results
   - Experimental design improvements
   - Code optimization and bug fixes
   - Additional experiment generation
   - Continuous learning and adaptation

**Workflow Complexity Management**:
```python
class ScientificResearchEngine:
    def __init__(self, deep_research_engine, code_research_engine, openhands_client):
        self.deep_research = deep_research_engine
        self.code_research = code_research_engine
        self.openhands = openhands_client
        self.iteration_count = 0
        self.max_iterations = 5

    async def execute_research_workflow(self, request: str, workflow_plan: dict):
        results = {}

        # Phase 1: Background Research (if required)
        if workflow_plan.get("requires_deep_research"):
            literature_review = await self.deep_research.comprehensive_search(
                query=f"literature review: {request}",
                sources=["academic", "patents", "preprints"]
            )
            results["literature_review"] = literature_review

        # Phase 2: Code Analysis (if required)
        if workflow_plan.get("requires_code_research"):
            code_analysis = await self.code_research.analyze_repositories(
                query=f"implementations: {request}",
                include_patterns=True,
                include_benchmarks=True
            )
            results["code_analysis"] = code_analysis

        # Phase 3: Iterative Experimentation
        for iteration in range(self.max_iterations):
            # Generate/refine hypothesis
            hypothesis = await self.generate_hypothesis(request, results, iteration)

            # Design experiments
            experiment_design = await self.design_experiments(hypothesis, results)

            # Generate and execute code via OpenHands
            execution_result = await self.openhands.execute_scientific_experiment(
                design=experiment_design,
                context=results,
                iteration=iteration
            )

            # Analyze results
            analysis = await self.analyze_results(execution_result, hypothesis)
            results[f"iteration_{iteration}"] = {
                "hypothesis": hypothesis,
                "experiment": experiment_design,
                "execution": execution_result,
                "analysis": analysis
            }

            # Check if we have conclusive results
            if analysis.confidence > 0.9 or analysis.conclusive:
                break

            # Refine for next iteration
            request = await self.refine_research_question(request, analysis)

        # Phase 4: Final synthesis and validation
        final_report = await self.synthesize_research(results, request)

        return final_report
```

##### Code Research Engine (`code_research.py`)
**Purpose**: RepoMaster-style repository analysis and code understanding
**Capabilities**:
- GitHub repository discovery and analysis
- Code pattern recognition and documentation
- Implementation example generation
- Best practices extraction
- Integration testing and validation

#### 2.2.3 OpenHands Integration (`openhands_client.py`)
**Purpose**: Bridge between uagent and OpenHands CLI execution engine

**Core Functions**:
- `initialize_openhands_session()`: Start new OpenHands CLI session
- `execute_task()`: Execute tasks via OpenHands agent
- `get_session_status()`: Monitor execution progress
- `stream_output()`: Real-time output streaming
- `manage_workspace()`: Workspace creation and management

**Implementation Details**:
```python
class OpenHandsClient:
    def __init__(self, config_path: str = None):
        # Initialize OpenHands CLI configuration
        # Set up session management
        # Configure agent selection (CodeAct, etc.)

    async def execute_research_task(self, task: ResearchTask) -> TaskResult:
        # Create isolated workspace
        # Configure agent with task context
        # Execute via CLI with streaming output
        # Return structured results

    async def execute_code_task(self, code_request: CodeRequest) -> CodeResult:
        # Set up development environment
        # Execute code generation/debugging
        # Run tests and validation
        # Return execution results
```

#### 2.2.2 Research Tree System (`research_tree.py`)
**Purpose**: Hierarchical research methodology with intelligent task decomposition

**Core Functions**:
- `create_research_goal()`: Initialize new research objective
- `decompose_task()`: Break down complex tasks into subtasks
- `execute_node()`: Execute individual research nodes
- `synthesize_results()`: Combine results from multiple nodes
- `generate_report()`: Create comprehensive research reports

**Node Types**:
1. **Root Node**: Top-level research objective
2. **Literature Review Node**: Academic and technical literature analysis
3. **Code Analysis Node**: Repository and code base analysis
4. **Experimentation Node**: Hypothesis testing and validation
5. **Synthesis Node**: Result combination and analysis
6. **Validation Node**: Result verification and testing

**Implementation Details**:
```python
class ResearchTree:
    def __init__(self, goal: str, max_depth: int = 5):
        # Initialize tree structure
        # Set expansion limits
        # Configure node types

    async def expand_node(self, node_id: str) -> List[ResearchNode]:
        # LLM-guided task decomposition
        # Create child nodes based on research methodology
        # Assign appropriate node types
        # Return expansion plan

    async def execute_node(self, node_id: str) -> NodeResult:
        # Route to appropriate execution engine
        # OpenHands for code tasks
        # Search service for literature
        # Return structured results
```

#### 2.2.3 Task Orchestrator (`task_orchestrator.py`)
**Purpose**: Coordinate multiple research tasks and manage execution flow

**Core Functions**:
- `orchestrate_research()`: Main research workflow coordination
- `manage_parallel_tasks()`: Handle concurrent task execution
- `handle_dependencies()`: Manage task dependencies
- `monitor_progress()`: Track overall research progress
- `handle_failures()`: Error recovery and retries

**Implementation Details**:
```python
class TaskOrchestrator:
    def __init__(self, openhands_client: OpenHandsClient):
        # Initialize with OpenHands client
        # Set up task queues
        # Configure parallel execution limits

    async def execute_research_plan(self, plan: ResearchPlan) -> ResearchResults:
        # Decompose into executable tasks
        # Schedule tasks based on dependencies
        # Execute via OpenHands
        # Aggregate and synthesize results
```

#### 2.2.4 Agent Manager (`agent_manager.py`)
**Purpose**: Manage different types of specialized agents for various research tasks

**Agent Types**:
1. **Literature Review Agent**: Academic paper analysis
2. **Code Analysis Agent**: Repository and code examination
3. **Web Research Agent**: General web search and analysis
4. **Experimentation Agent**: Hypothesis testing and validation
5. **Synthesis Agent**: Result combination and report generation
6. **GitHub Agent**: Repository search and analysis

**Implementation Details**:
```python
class AgentManager:
    def __init__(self):
        # Initialize agent configurations
        # Set up agent-specific prompts
        # Configure tool access for each agent

    async def deploy_agent(self, agent_type: str, task: Task) -> Agent:
        # Select appropriate OpenHands agent
        # Configure with specialized prompts
        # Set up task-specific context
        # Return configured agent instance
```

#### 2.2.5 Multi-Modal Search Service (`search_service.py`)
**Purpose**: Comprehensive search across multiple sources

**Search Types**:
1. **Academic Search**: Papers, patents, technical documents
2. **GitHub Search**: Repository analysis and code search
3. **Web Search**: General web information gathering
4. **Documentation Search**: API docs, technical documentation

**Implementation Details**:
```python
class SearchService:
    def __init__(self):
        # Initialize search engines
        # Set up API connections
        # Configure result synthesis

    async def comprehensive_search(self, query: str, sources: List[str]) -> SearchResults:
        # Execute parallel searches
        # Synthesize results using LLM
        # Rank by relevance and quality
        # Return structured results
```

#### 2.2.6 Report Generation Service (`report_service.py`)
**Purpose**: Generate comprehensive research reports

**Report Types**:
1. **Executive Summary**: High-level findings and recommendations
2. **Technical Report**: Detailed analysis and methodology
3. **Code Documentation**: Generated code explanation and usage
4. **Literature Review**: Academic and technical source analysis

### 2.3 API Endpoints Structure

#### 2.3.1 Smart Routing API (`/api/router/`)
- `POST /classify` - Classify user request and suggest research type
- `POST /route` - Route request to appropriate research engine
- `GET /engines` - List available research engines and capabilities

#### 2.3.2 Research Management API (`/api/research/`)
- `POST /goals` - Create new research goal (with automatic routing)
- `GET /goals/{id}` - Get research goal status
- `POST /goals/{id}/execute` - Start research execution
- `GET /goals/{id}/progress` - Get execution progress
- `GET /goals/{id}/results` - Get research results
- `GET /goals/{id}/report` - Generate/download report
- `DELETE /goals/{id}` - Cancel research goal
- `GET /goals/{id}/engine` - Get research engine used for this goal

#### 2.3.3 Engine-Specific APIs
##### Deep Research API (`/api/deep-research/`)
- `POST /search` - Multi-source comprehensive search
- `POST /synthesize` - Synthesize results from multiple sources
- `GET /sources` - List available search sources

##### Scientific Research API (`/api/scientific-research/`)
- `POST /hypothesis` - Generate research hypothesis
- `POST /experiment` - Design experimental methodology
- `POST /analyze` - Analyze experimental data

##### Code Research API (`/api/code-research/`)
- `POST /repositories` - Search and analyze repositories
- `POST /patterns` - Extract code patterns and best practices
- `POST /integrate` - Test code integration

#### 2.3.2 Task Management API (`/api/tasks/`)
- `GET /tasks` - List all tasks
- `GET /tasks/{id}` - Get specific task details
- `POST /tasks/{id}/retry` - Retry failed task
- `DELETE /tasks/{id}` - Cancel task

#### 2.3.3 Agent Management API (`/api/agents/`)
- `GET /agents` - List available agent types
- `GET /agents/active` - List currently active agents
- `POST /agents/{type}/deploy` - Deploy specific agent
- `DELETE /agents/{id}` - Stop agent

#### 2.3.4 WebSocket Endpoints (`/ws/`)
- `/ws/research/{goal_id}` - Real-time research progress with multi-engine coordination
- `/ws/tasks/{task_id}` - Real-time task execution with OpenHands output streaming
- `/ws/agents/{agent_id}` - Real-time agent output and status updates
- `/ws/engines/status` - Live status updates for all research engines
- `/ws/openhands/{session_id}` - Live OpenHands code execution output
- `/ws/research/{goal_id}/journal` - Real-time research journal updates
- `/ws/metrics/live` - Live performance and progress metrics streaming

## 3. FRONTEND SYSTEM DESIGN

### 3.1 Technology Stack
- **Framework**: React 18 with TypeScript
- **State Management**: Zustand for simple state management
- **UI Library**: Tailwind CSS + Shadcn/ui components
- **Real-time**: WebSocket integration for live updates
- **Visualization**: D3.js for research tree visualization
- **Build Tool**: Vite for fast development and building

### 3.2 Component Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── common/
│   │   │   ├── Layout.tsx           # Main layout wrapper
│   │   │   ├── Navigation.tsx       # Navigation component
│   │   │   ├── StatusIndicator.tsx  # Status display
│   │   │   └── LoadingSpinner.tsx   # Loading states
│   │   ├── research/
│   │   │   ├── SmartRequestInput.tsx # Intelligent request input with routing
│   │   │   ├── GoalCreator.tsx      # Research goal creation
│   │   │   ├── EngineSelector.tsx   # Research engine selection/override
│   │   │   ├── TreeVisualizer.tsx   # Research tree display
│   │   │   ├── ProgressTracker.tsx  # Progress monitoring
│   │   │   ├── ResultsViewer.tsx    # Results display
│   │   │   ├── ResearchProgressStream.tsx  # Live progress streaming
│   │   │   ├── EngineStatusIndicators.tsx  # Real-time engine status
│   │   │   ├── LiveResearchJournal.tsx     # Streaming research log
│   │   │   ├── CodeExecutionTerminal.tsx   # Live OpenHands output
│   │   │   ├── InteractiveResearchTree.tsx # Real-time tree updates
│   │   │   └── MetricsDashboard.tsx        # Live performance metrics
│   │   ├── tasks/
│   │   │   ├── TaskList.tsx         # Task management
│   │   │   ├── TaskDetails.tsx      # Task detail view
│   │   │   └── ExecutionLogs.tsx    # Real-time logs
│   │   └── agents/
│   │       ├── AgentDashboard.tsx   # Agent overview
│   │       ├── AgentMonitor.tsx     # Agent monitoring
│   │       └── AgentControls.tsx    # Agent management
│   ├── pages/
│   │   ├── Dashboard.tsx            # Main dashboard
│   │   ├── Research.tsx             # Research interface
│   │   ├── Tasks.tsx                # Task management
│   │   └── Reports.tsx              # Report viewing
│   ├── services/
│   │   ├── api.ts                   # API client
│   │   ├── websocket.ts             # WebSocket client
│   │   └── types.ts                 # TypeScript types
│   ├── stores/
│   │   ├── researchStore.ts         # Research state
│   │   ├── taskStore.ts             # Task state
│   │   └── agentStore.ts            # Agent state
│   └── utils/
│       ├── formatting.ts            # Data formatting
│       └── constants.ts             # App constants
├── public/
└── package.json
```

### 3.3 Key Frontend Features

#### 3.3.1 Smart Request Input (`SmartRequestInput.tsx`)
**Purpose**: Intelligent interface for research request input with automatic routing suggestions

**Features**:
- **Natural Language Input**: Large text area for natural language research requests
- **Auto-Classification**: Real-time classification suggestions as user types
- **Engine Preview**: Show which research engine will be used and why
- **Manual Override**: Allow users to manually select different research engine
- **Request Examples**: Show example requests for each research type
- **Confidence Indicators**: Visual confidence levels for automatic classification

**Implementation**:
```typescript
interface SmartRequestInputProps {
  onSubmit: (request: string, engineType: EngineType) => void;
  isLoading: boolean;
}

const SmartRequestInput: React.FC<SmartRequestInputProps> = ({
  onSubmit,
  isLoading
}) => {
  const [request, setRequest] = useState('');
  const [suggestedEngine, setSuggestedEngine] = useState<EngineType | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [manualOverride, setManualOverride] = useState<EngineType | null>(null);

  // Real-time classification as user types
  useEffect(() => {
    const debounced = debounce(async () => {
      if (request.length > 20) {
        const classification = await api.classifyRequest(request);
        setSuggestedEngine(classification.engine);
        setConfidence(classification.confidence);
      }
    }, 500);

    debounced();
  }, [request]);

  return (
    <div className="smart-request-input">
      <textarea
        value={request}
        onChange={(e) => setRequest(e.target.value)}
        placeholder="Describe your research goal in natural language..."
        className="w-full h-32 p-4 border rounded-lg"
      />

      {suggestedEngine && (
        <EngineClassificationDisplay
          engine={suggestedEngine}
          confidence={confidence}
          onOverride={setManualOverride}
        />
      )}

      <RequestExamples onExampleClick={setRequest} />

      <button
        onClick={() => onSubmit(request, manualOverride || suggestedEngine)}
        disabled={!request || isLoading}
        className="submit-btn"
      >
        Start Research
      </button>
    </div>
  );
};
```

#### 3.3.2 Main Dashboard
**Purpose**: Central hub for all uagent activities

**Features**:
- **Quick Start**: Easy research goal creation
- **Active Research**: Overview of ongoing research
- **Recent Results**: Quick access to recent findings
- **System Status**: Overall system health and performance
- **Quick Actions**: Common tasks and shortcuts

**Components**:
```typescript
interface DashboardProps {
  activeResearch: ResearchGoal[];
  recentResults: ResearchResult[];
  systemStatus: SystemStatus;
}

const Dashboard: React.FC<DashboardProps> = ({
  activeResearch,
  recentResults,
  systemStatus
}) => {
  // Render dashboard with real-time updates
  // WebSocket connections for live data
  // Quick action buttons
  // Status indicators
};
```

#### 3.3.2 Research Tree Visualizer
**Purpose**: Interactive visualization of research progress

**Features**:
- **Hierarchical Tree View**: D3.js-based tree visualization
- **Node Status Indicators**: Visual status for each research node
- **Interactive Navigation**: Click to drill down into nodes
- **Real-time Updates**: Live progress updates via WebSocket
- **Export Options**: Save tree as image or data

**Implementation**:
```typescript
interface TreeNode {
  id: string;
  type: NodeType;
  status: NodeStatus;
  title: string;
  children: TreeNode[];
  results?: NodeResult;
}

const TreeVisualizer: React.FC<{tree: TreeNode}> = ({tree}) => {
  // D3.js tree rendering
  // Interactive node selection
  // Real-time status updates
  // Zoom and pan functionality
};
```

#### 3.3.3 Real-time Progress Tracker
**Purpose**: Live monitoring of research execution

**Features**:
- **Live Output Streaming**: Real-time command output
- **Progress Indicators**: Visual progress bars and percentages
- **Error Handling**: Clear error display and recovery options
- **Execution Logs**: Detailed log viewing with filtering
- **Performance Metrics**: Execution time and resource usage

#### 3.3.4 Results and Report Viewer
**Purpose**: Display and manage research results

**Features**:
- **Structured Results**: Organized display of findings
- **Interactive Reports**: Expandable sections and navigation
- **Export Options**: Multiple format support (PDF, Markdown, JSON)
- **Search and Filter**: Find specific results and insights
- **Collaboration**: Share results and add annotations

#### 3.3.5 Agent Management Interface
**Purpose**: Monitor and control research agents

**Features**:
- **Agent Overview**: Status and current tasks for all agents
- **Resource Monitoring**: CPU, memory, and execution time tracking
- **Agent Configuration**: Adjust agent parameters and behavior
- **Task Assignment**: Manual task assignment to specific agents
- **Performance Analytics**: Agent efficiency and success metrics

## 4. INTEGRATION AND EXECUTION FLOW

### 4.1 Research Workflow
1. **Goal Creation**: User defines research objective via frontend
2. **Task Decomposition**: Research tree system breaks down goal
3. **Agent Assignment**: Appropriate agents assigned to tasks
4. **OpenHands Execution**: Tasks executed via OpenHands CLI
5. **Result Synthesis**: Results combined and analyzed
6. **Report Generation**: Comprehensive report created
7. **User Review**: Results presented in frontend interface

### 4.2 Code Research Workflow
1. **Repository Analysis**: GitHub service finds relevant repositories
2. **Code Understanding**: OpenHands analyzes code structure
3. **Documentation Review**: Automated documentation analysis
4. **Usage Examples**: Generate working code examples
5. **Integration Testing**: Test code in isolated environment
6. **Best Practices**: Extract patterns and recommendations

### 4.3 Scientific Research Workflow (Most Complex - Multi-Engine Integration)
**Phase 1: Background Research & Context Building**
1. **Literature Review** (Deep Research Engine): Comprehensive academic paper analysis, SOTA identification
2. **Code Discovery** (Code Research Engine): Find and analyze existing implementations
3. **Gap Analysis**: Identify research opportunities and positioning

**Phase 2: Research Planning**
4. **Hypothesis Generation**: Formulate testable research questions
5. **Experimental Design**: Create rigorous experimental methodology
6. **Resource Planning**: Determine computational requirements and datasets

**Phase 3: Implementation & Execution (Heavy OpenHands Integration)**
7. **Environment Setup**: Create isolated research workspace
8. **Code Generation**: Generate experimental code, data pipelines, and testing infrastructure
9. **Iterative Development**: Real-time debugging, optimization, and refinement
10. **Experiment Execution**: Run experiments with monitoring and error recovery

**Phase 4: Analysis & Validation**
11. **Statistical Analysis**: Automated statistical testing and significance evaluation
12. **Result Interpretation**: Extract insights and evaluate hypothesis support
13. **Reproducibility Testing**: Verify results across multiple runs
14. **Peer Review Simulation**: Automated methodology and result validation

**Phase 5: Iteration & Refinement**
15. **Hypothesis Refinement**: Refine research questions based on findings
16. **Experimental Improvements**: Optimize methodology and implementation
17. **Additional Experiments**: Generate follow-up experiments
18. **Convergence Assessment**: Determine when research is conclusive

**Phase 6: Documentation & Dissemination**
19. **Comprehensive Reporting**: Generate publication-ready research reports
20. **Code Documentation**: Create reusable code packages
21. **Visualization**: Generate figures, plots, and interactive demonstrations
22. **Knowledge Integration**: Update research knowledge base

**Feedback Loops**:
- Continuous iteration between hypothesis, implementation, and results
- Real-time error detection and experimental adjustment
- Progressive refinement of research questions
- Integration of findings from literature and code analysis

### 4.4 Web Research Workflow
1. **Multi-source Search**: Search across web, GitHub, academic sources
2. **Content Analysis**: Extract and synthesize key information
3. **Fact Verification**: Cross-reference information across sources
4. **Trend Analysis**: Identify patterns and emerging trends
5. **Recommendation Generation**: Provide actionable insights

## 5. TECHNICAL IMPLEMENTATION DETAILS

### 5.1 Development Environment Setup
- **Backend**: Python 3.11+, FastAPI, SQLAlchemy, Alembic
- **Frontend**: Node.js 18+, React 18, TypeScript, Vite
- **Database**: SQLite for development, PostgreSQL for production
- **OpenHands**: Integrated as Git submodule or package dependency
- **Development Tools**: Docker Compose for local development

### 5.2 Security and Safety
- **Workspace Isolation**: All code execution in isolated environments
- **Input Sanitization**: Comprehensive input validation
- **Resource Limits**: CPU, memory, and time limits for all operations
- **Access Control**: Role-based access to sensitive operations
- **Audit Logging**: Comprehensive logging of all actions

### 5.3 Performance and Scalability
- **Async Operations**: Non-blocking operations throughout
- **Connection Pooling**: Efficient database and API connections
- **Caching**: Intelligent caching of search results and computations
- **Resource Management**: Efficient memory and CPU usage
- **Horizontal Scaling**: Design for multi-instance deployment

### 5.4 Error Handling and Recovery
- **Graceful Degradation**: System continues functioning with component failures
- **Automatic Retries**: Intelligent retry logic for transient failures
- **Error Reporting**: Comprehensive error tracking and reporting
- **Recovery Mechanisms**: Automatic recovery from common failure scenarios
- **User Feedback**: Clear error messages and recovery suggestions

## 6. DEPLOYMENT AND OPERATIONS

### 6.1 Development Deployment
- **Local Development**: Docker Compose with hot reloading
- **Database**: SQLite with automatic migrations
- **Frontend**: Vite dev server with proxy to backend
- **OpenHands**: Local CLI installation

### 6.2 Production Deployment
- **Containerization**: Docker containers for all components
- **Orchestration**: Docker Compose or Kubernetes
- **Database**: PostgreSQL with backup and replication
- **Load Balancing**: Nginx or cloud load balancer
- **Monitoring**: Health checks and performance monitoring

### 6.3 Configuration Management
- **Environment Variables**: Secure configuration via environment
- **Config Files**: YAML-based configuration for complex settings
- **Secrets Management**: Secure handling of API keys and credentials
- **Feature Flags**: Enable/disable features without deployment

## 7. TESTING STRATEGY

### 7.1 Backend Testing
- **Unit Tests**: Comprehensive unit test coverage (>90%)
- **Integration Tests**: API endpoint and service integration tests
- **End-to-End Tests**: Complete workflow testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability and penetration testing

### 7.2 Frontend Testing
- **Component Tests**: React component testing with Jest
- **Integration Tests**: User flow testing with Playwright
- **Visual Tests**: Screenshot comparison testing
- **Accessibility Tests**: WCAG compliance testing
- **Cross-browser Tests**: Compatibility across browsers

### 7.3 OpenHands Integration Testing
- **CLI Integration**: Test OpenHands CLI integration
- **Workspace Management**: Test workspace creation and cleanup
- **Task Execution**: Test various task types and scenarios
- **Error Handling**: Test failure scenarios and recovery
- **Performance**: Test execution time and resource usage

## 8. MONITORING AND OBSERVABILITY

### 8.1 Application Monitoring
- **Health Checks**: Endpoint health monitoring
- **Performance Metrics**: Response time and throughput monitoring
- **Error Tracking**: Comprehensive error logging and alerting
- **User Analytics**: Usage patterns and feature adoption
- **Resource Monitoring**: CPU, memory, and disk usage

### 8.2 Research Monitoring
- **Research Progress**: Track research goal completion rates
- **Agent Performance**: Monitor agent efficiency and success rates
- **Task Execution**: Track task completion times and failure rates
- **Result Quality**: Monitor result accuracy and usefulness
- **User Satisfaction**: Track user feedback and ratings

## 9. IMPLEMENTATION PHASES

### Phase 1: Foundation (Weeks 1-2)
- Set up project structure and development environment
- Copy and integrate OpenHands CLI source code
- Implement smart routing system with LLM-based classification
- Build basic frontend with smart request input interface
- Create simple research tree structure

### Phase 2: Core Research Engines (Weeks 3-5)
- Implement deep research engine (web + academic search)
- Implement code research engine (GitHub + repository analysis)
- **Scientific research engine foundation** (basic workflow without iteration)
- Create research tree visualization with engine-specific workflows
- Add real-time progress tracking and engine status monitoring

### Phase 3: Scientific Research Engine Complexity (Weeks 6-8)
- **Multi-engine integration**: Scientific research coordinating other engines
- **Iterative experimentation loops**: Hypothesis refinement and continuous improvement
- **Advanced OpenHands integration**: Complex code generation, execution, and debugging
- **Statistical analysis pipeline**: Automated result analysis and validation
- **Feedback loop implementation**: Real-time experimental adjustment

### Phase 4: Advanced Features & Integration (Weeks 9-10)
- **Cross-engine communication**: Seamless data flow between all engines
- **Advanced reporting**: Publication-ready research documentation
- **Performance optimization**: Handle complex, long-running scientific workflows
- **Error recovery systems**: Robust handling of experimental failures

### Phase 5: Polish and Production (Weeks 11-12)
- **System-wide performance optimization**: Optimize for complex scientific workflows
- **Security hardening and audit**: Secure handling of research data and code execution
- **Production deployment setup**: Scalable infrastructure for scientific computing
- **Comprehensive documentation**: User guides for scientific research workflows
- **Integration testing**: End-to-end testing of complex multi-engine workflows

## 10. SUCCESS METRICS

### 10.1 Technical Metrics
- **System Uptime**: >99.9% availability
- **Response Time**: <2s for API responses
- **Task Success Rate**: >95% successful task completion
- **Error Rate**: <1% unhandled errors
- **Resource Efficiency**: Optimal CPU and memory usage

### 10.2 Research Quality Metrics
- **Result Accuracy**: High-quality, relevant research results
- **Completeness**: Comprehensive coverage of research topics
- **Timeliness**: Fast research completion times
- **Reproducibility**: Consistent results across runs
- **User Satisfaction**: Positive user feedback and adoption

This comprehensive plan provides a solid foundation for building a reliable, scalable, and user-friendly universal agent system that leverages the proven strengths of OpenHands while learning from the lessons of the previous implementation. 