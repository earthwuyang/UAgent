# Technical Specifications: UAgent Research Extension

## Table of Contents
1. [API Specifications](#api-specifications)
2. [Data Models](#data-models)
3. [Agent Interfaces](#agent-interfaces)
4. [Event System](#event-system)
5. [WebSocket Protocol](#websocket-protocol)
6. [Extension Configuration](#extension-configuration)
7. [Performance Requirements](#performance-requirements)

---

## API Specifications

### Base URL
```
/api/research
```

### Authentication
All endpoints use OpenHands session-based authentication. Include session token in requests:

```http
Authorization: Bearer <session_token>
```

---

### Endpoints

#### 1. Start Research Experiment

**POST** `/api/research/experiments/start`

Initiates a new research experiment.

**Request Body:**
```typescript
{
  goal: string;                    // Research objective
  session_id: string;              // OpenHands session ID
  research_type: "scientific" | "code" | "roma";  // Type of research
  config?: {                       // Optional configuration
    max_iterations?: number;       // Max agent iterations (default: 100)
    timeout_seconds?: number;      // Timeout (default: 3600)
    llm_model?: string;            // Specific LLM model
    validation_strict?: boolean;   // Strict validation (default: true)
    workspace_path?: string;       // Custom workspace path
  };
  context?: {                      // Optional context
    previous_results?: any[];      // Results from related experiments
    constraints?: string[];        // Constraints to apply
    parameters?: Record<string, any>;  // Custom parameters
  };
}
```

**Response (200 OK):**
```typescript
{
  experiment_id: string;           // Unique experiment ID
  session_id: string;              // Session ID
  status: "pending" | "running";   // Initial status
  created_at: string;              // ISO 8601 timestamp
  estimated_duration_seconds?: number;  // Estimated completion time
  websocket_url: string;           // WebSocket for real-time updates
}
```

**Error Responses:**
```typescript
// 400 Bad Request
{
  error: "validation_error";
  message: string;
  details?: Record<string, string[]>;
}

// 409 Conflict
{
  error: "max_concurrent_experiments_reached";
  message: string;
  current_count: number;
  max_allowed: number;
}

// 500 Internal Server Error
{
  error: "internal_error";
  message: string;
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/research/experiments/start \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "goal": "Compare performance of quicksort vs mergesort on random data",
    "session_id": "session_123",
    "research_type": "scientific",
    "config": {
      "max_iterations": 50,
      "timeout_seconds": 1800
    }
  }'
```

---

#### 2. Get Experiment Status

**GET** `/api/research/experiments/{experiment_id}`

Retrieves current status and results of an experiment.

**Path Parameters:**
- `experiment_id` (string, required): Experiment identifier

**Query Parameters:**
- `include_logs` (boolean, optional): Include execution logs (default: false)
- `include_artifacts` (boolean, optional): Include generated artifacts (default: false)

**Response (200 OK):**
```typescript
{
  id: string;
  session_id: string;
  experiment_type: "scientific" | "code" | "roma";
  goal: string;
  status: "pending" | "running" | "completed" | "failed" | "timeout" | "cancelled";

  // Progress tracking
  progress: {
    percentage: number;            // 0-100
    current_step: string;          // Human-readable current step
    steps_completed: number;
    total_steps: number;
    estimated_time_remaining_seconds?: number;
  };

  // Timestamps
  created_at: string;              // ISO 8601
  started_at?: string;
  completed_at?: string;

  // Results (only if status is "completed")
  results?: {
    summary: string;
    hypothesis?: string;
    key_findings: string[];
    data: Record<string, any>;
    visualizations?: Array<{
      type: "chart" | "table" | "image";
      data: any;
      caption?: string;
    }>;
    recommendations?: string[];
  };

  // Error information (only if status is "failed")
  error?: {
    message: string;
    type: string;
    traceback?: string;
    retry_suggestion?: string;
  };

  // Resource usage
  metrics: {
    execution_time_seconds: number;
    memory_usage_mb?: number;
    tokens_used?: number;
    api_calls?: number;
  };

  // Optional fields
  logs?: string[];                 // If include_logs=true
  artifacts?: Array<{              // If include_artifacts=true
    name: string;
    type: string;
    url: string;
    size_bytes: number;
  }>;
}
```

**Example:**
```bash
curl http://localhost:3000/api/research/experiments/exp_123?include_logs=true \
  -H "Authorization: Bearer <token>"
```

---

#### 3. List Experiments

**GET** `/api/research/experiments`

Lists experiments with optional filtering.

**Query Parameters:**
- `session_id` (string, optional): Filter by session
- `status` (string, optional): Filter by status
- `research_type` (string, optional): Filter by type
- `limit` (number, optional): Max results (default: 50, max: 200)
- `offset` (number, optional): Pagination offset (default: 0)
- `sort_by` (string, optional): Sort field (default: "created_at")
- `sort_order` (string, optional): "asc" or "desc" (default: "desc")

**Response (200 OK):**
```typescript
{
  experiments: Array<ExperimentSummary>;  // See experiment schema
  total: number;                          // Total matching experiments
  limit: number;
  offset: number;
  has_more: boolean;
}

interface ExperimentSummary {
  id: string;
  goal: string;
  status: string;
  progress_percentage: number;
  created_at: string;
  execution_time_seconds?: number;
}
```

**Example:**
```bash
curl "http://localhost:3000/api/research/experiments?status=running&limit=10" \
  -H "Authorization: Bearer <token>"
```

---

#### 4. Cancel Experiment

**DELETE** `/api/research/experiments/{experiment_id}`

Cancels a running or pending experiment.

**Path Parameters:**
- `experiment_id` (string, required): Experiment to cancel

**Query Parameters:**
- `force` (boolean, optional): Force cancellation even if unsafe (default: false)

**Response (200 OK):**
```typescript
{
  experiment_id: string;
  status: "cancelled";
  cancelled_at: string;            // ISO 8601
  partial_results?: any;           // Results collected before cancellation
}
```

**Error Responses:**
```typescript
// 400 Bad Request - Cannot cancel
{
  error: "cannot_cancel";
  message: "Cannot cancel experiment in status 'completed'";
  current_status: string;
}
```

**Example:**
```bash
curl -X DELETE http://localhost:3000/api/research/experiments/exp_123 \
  -H "Authorization: Bearer <token>"
```

---

#### 5. Get Research Tree (ROMA)

**GET** `/api/research/sessions/{session_id}/tree`

Retrieves ROMA research tree structure.

**Path Parameters:**
- `session_id` (string, required): Research session ID

**Response (200 OK):**
```typescript
{
  session_id: string;
  tree: {
    root: ResearchNode;
    total_nodes: number;
    active_branches: number;
  };
  updated_at: string;
}

interface ResearchNode {
  id: string;
  parent_id?: string;
  type: "hypothesis" | "experiment" | "analysis";
  content: string;
  status: "pending" | "running" | "completed" | "failed";
  children: ResearchNode[];

  // Node-specific data
  metadata: {
    created_at: string;
    experiment_id?: string;
    confidence_score?: number;
    tags?: string[];
  };

  // Results if completed
  results?: any;
}
```

**Example:**
```bash
curl http://localhost:3000/api/research/sessions/session_123/tree \
  -H "Authorization: Bearer <token>"
```

---

#### 6. Generate Ideas (AI Scientist)

**POST** `/api/research/ideas/generate`

Generates research ideas based on a topic.

**Request Body:**
```typescript
{
  topic: string;                   // Research topic
  context?: string;                // Additional context
  num_ideas?: number;              // Number of ideas (default: 5, max: 20)
  creativity?: number;             // 0.0-1.0 (default: 0.7)
  constraints?: string[];          // Constraints to apply
}
```

**Response (200 OK):**
```typescript
{
  topic: string;
  ideas: Array<{
    id: string;
    title: string;
    description: string;
    novelty_score: number;         // 0-1
    feasibility_score: number;     // 0-1
    impact_score: number;          // 0-1
    tags: string[];
    related_work?: string[];
    potential_challenges?: string[];
  }>;
  generated_at: string;
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/research/ideas/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "topic": "Machine learning for code optimization",
    "num_ideas": 3,
    "creativity": 0.8
  }'
```

---

#### 7. Generate Hypotheses

**POST** `/api/research/hypotheses/generate`

Generates testable hypotheses from an idea.

**Request Body:**
```typescript
{
  idea: string;                    // Research idea
  background?: string;             // Background information
  num_hypotheses?: number;         // Number to generate (default: 3, max: 10)
}
```

**Response (200 OK):**
```typescript
{
  idea: string;
  hypotheses: Array<{
    id: string;
    statement: string;
    null_hypothesis: string;
    testability_score: number;     // 0-1
    expected_outcome: string;
    experimental_design: {
      approach: string;
      metrics: string[];
      controls: string[];
      estimated_duration: string;
    };
  }>;
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/research/hypotheses/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "idea": "ML can predict optimal compiler flags",
    "num_hypotheses": 3
  }'
```

---

#### 8. Code Repository Analysis

**POST** `/api/research/code/analyze`

Analyzes a code repository (RepoMaster integration).

**Request Body:**
```typescript
{
  repository: {
    type: "git" | "local";
    url?: string;                  // Git URL (if type=git)
    path?: string;                 // Local path (if type=local)
    branch?: string;               // Branch to analyze (default: main)
  };
  query: string;                   // Analysis query
  scope?: {
    include_paths?: string[];      // Paths to include
    exclude_paths?: string[];      // Paths to exclude
    file_extensions?: string[];    // File types to analyze
  };
  depth?: "quick" | "deep";        // Analysis depth (default: quick)
}
```

**Response (200 OK):**
```typescript
{
  repository: {
    name: string;
    url?: string;
    commit?: string;
    analyzed_at: string;
  };

  analysis: {
    summary: string;
    architecture: {
      components: Array<{
        name: string;
        type: string;
        description: string;
        files: string[];
        dependencies: string[];
      }>;
      layers: string[];
      patterns: string[];
    };

    relevant_files: Array<{
      path: string;
      relevance_score: number;      // 0-1
      summary: string;
      key_functions?: string[];
    }>;

    code_map: {
      entry_points: string[];
      critical_paths: string[][];
      complexity_hotspots: Array<{
        file: string;
        function: string;
        complexity: number;
      }>;
    };

    insights: string[];
    recommendations: string[];
  };

  metrics: {
    files_analyzed: number;
    lines_of_code: number;
    analysis_time_seconds: number;
  };
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/research/code/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "repository": {
      "type": "git",
      "url": "https://github.com/example/repo.git"
    },
    "query": "How does authentication work?",
    "depth": "deep"
  }'
```

---

## Data Models

### Experiment Model

```python
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, JSON, Enum
from sqlalchemy.sql import func
import enum

class ExperimentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class ExperimentType(str, enum.Enum):
    SCIENTIFIC = "scientific"
    CODE = "code"
    ROMA = "roma"

class Experiment(Base):
    __tablename__ = "experiments"

    # Primary key
    id = Column(String(64), primary_key=True)

    # Foreign keys
    session_id = Column(String(64), nullable=False, index=True)
    parent_experiment_id = Column(String(64), nullable=True, index=True)

    # Metadata
    experiment_type = Column(Enum(ExperimentType), nullable=False)
    goal = Column(Text, nullable=False)
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.PENDING, index=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Progress
    progress_percentage = Column(Float, default=0.0)
    current_step = Column(String(256), nullable=True)
    total_steps = Column(Integer, nullable=True)
    steps_completed = Column(Integer, default=0)

    # Configuration
    config = Column(JSON, nullable=True)
    workspace_path = Column(String(512), nullable=True)

    # Results
    results = Column(JSON, nullable=True)
    artifacts = Column(JSON, nullable=True)
    logs = Column(JSON, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_type = Column(String(128), nullable=True)
    error_traceback = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Resource usage
    execution_time_seconds = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    api_calls = Column(Integer, default=0)

    # Indexes
    __table_args__ = (
        Index('idx_session_status', 'session_id', 'status'),
        Index('idx_created_at', 'created_at'),
    )
```

### Research Session Model

```python
class SessionMode(str, enum.Enum):
    CHAT = "chat"
    RESEARCH = "research"
    HYBRID = "hybrid"

class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), nullable=True, index=True)
    mode = Column(Enum(SessionMode), default=SessionMode.RESEARCH)

    # Metadata
    title = Column(String(256), nullable=True)
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # Array of strings

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_activity_at = Column(DateTime, default=func.now())

    # Session state
    state = Column(JSON, nullable=True)
    research_tree = Column(JSON, nullable=True)  # ROMA tree structure
    context = Column(JSON, nullable=True)

    # Settings
    settings = Column(JSON, nullable=True)

    # Relationships (if using SQLAlchemy relationships)
    # experiments = relationship("Experiment", back_populates="session")
```

### Idea Model

```python
class Idea(Base):
    __tablename__ = "ideas"

    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), nullable=False, index=True)

    # Content
    title = Column(String(512), nullable=False)
    description = Column(Text, nullable=False)
    topic = Column(String(256), nullable=False)

    # Scores
    novelty_score = Column(Float, nullable=True)
    feasibility_score = Column(Float, nullable=True)
    impact_score = Column(Float, nullable=True)

    # Metadata
    tags = Column(JSON, nullable=True)
    related_work = Column(JSON, nullable=True)
    potential_challenges = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now())

    # Status
    status = Column(String(32), default="generated")  # generated, selected, tested
```

### Hypothesis Model

```python
class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id = Column(String(64), primary_key=True)
    idea_id = Column(String(64), nullable=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)

    # Content
    statement = Column(Text, nullable=False)
    null_hypothesis = Column(Text, nullable=True)

    # Experimental design
    testability_score = Column(Float, nullable=True)
    expected_outcome = Column(Text, nullable=True)
    experimental_design = Column(JSON, nullable=True)

    # Testing
    tested = Column(Boolean, default=False)
    experiment_id = Column(String(64), nullable=True)

    # Results
    result = Column(String(32), nullable=True)  # supported, rejected, inconclusive
    confidence = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    tested_at = Column(DateTime, nullable=True)
```

---

## Agent Interfaces

### Base Research Agent

```python
from abc import ABC, abstractmethod
from openhands.core.schema import AgentState
from openhands.events.action import Action
from openhands.llm.llm import LLM

class ResearchAgent(ABC):
    """Base class for all research agents"""

    VERSION: str = "1.0"

    def __init__(self, llm: LLM):
        self.llm = llm
        self.current_research = None

    @abstractmethod
    async def step(self, state: AgentState) -> Action:
        """Execute one agent step"""
        pass

    @abstractmethod
    def is_research_task(self, state: AgentState) -> bool:
        """Determine if task requires research capabilities"""
        pass

    @abstractmethod
    async def start_research(self, goal: str, context: dict) -> dict:
        """Start new research"""
        pass

    @abstractmethod
    async def continue_research(self, feedback: str) -> dict:
        """Continue ongoing research with user feedback"""
        pass

    async def summarize_results(self, results: dict) -> str:
        """Generate human-readable summary of results"""
        pass
```

### Research Engine Interface

```python
from typing import Dict, Any, Optional, AsyncIterator
from openhands.runtime.runtime import Runtime
from openhands.events.stream import EventStream

class ResearchEngine(ABC):
    """Base class for research engines"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = None  # Injected by agent

    @abstractmethod
    async def run_experiment(
        self,
        goal: str,
        runtime: Runtime,
        event_stream: EventStream,
        session_id: str
    ) -> Dict[str, Any]:
        """Run experiment and return results"""
        pass

    @abstractmethod
    async def stream_progress(
        self,
        experiment_id: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream progress updates"""
        pass

    @abstractmethod
    async def validate_results(
        self,
        results: Dict[str, Any]
    ) -> bool:
        """Validate experiment results"""
        pass
```

---

## Event System

### Event Types

```typescript
// Research lifecycle events
type ResearchEvent =
  | ExperimentStartedEvent
  | ExperimentProgressEvent
  | ExperimentCompletedEvent
  | ExperimentFailedEvent
  | ExperimentCancelledEvent
  | StepStartedEvent
  | StepCompletedEvent
  | LLMInteractionEvent
  | ArtifactGeneratedEvent;

interface BaseEvent {
  type: string;
  timestamp: string;              // ISO 8601
  experiment_id: string;
  session_id: string;
}

interface ExperimentStartedEvent extends BaseEvent {
  type: "experiment_started";
  goal: string;
  experiment_type: string;
  config: any;
}

interface ExperimentProgressEvent extends BaseEvent {
  type: "experiment_progress";
  progress: {
    percentage: number;
    current_step: string;
    steps_completed: number;
    total_steps: number;
    estimated_time_remaining_seconds?: number;
  };
}

interface ExperimentCompletedEvent extends BaseEvent {
  type: "experiment_completed";
  results: any;
  execution_time_seconds: number;
}

interface ExperimentFailedEvent extends BaseEvent {
  type: "experiment_failed";
  error: {
    message: string;
    type: string;
    traceback?: string;
  };
}

interface StepStartedEvent extends BaseEvent {
  type: "step_started";
  step_name: string;
  step_description: string;
}

interface StepCompletedEvent extends BaseEvent {
  type: "step_completed";
  step_name: string;
  result: any;
  duration_seconds: number;
}

interface LLMInteractionEvent extends BaseEvent {
  type: "llm_interaction";
  interaction: {
    prompt: string;
    response: string;
    model: string;
    tokens_used: number;
  };
}

interface ArtifactGeneratedEvent extends BaseEvent {
  type: "artifact_generated";
  artifact: {
    name: string;
    type: string;
    url: string;
    size_bytes: number;
  };
}
```

### Event Emission

```python
async def emit_event(
    event_stream: EventStream,
    event_type: str,
    experiment_id: str,
    session_id: str,
    data: Dict[str, Any]
):
    """Emit research event"""
    event = {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_id": experiment_id,
        "session_id": session_id,
        **data
    }

    await event_stream.add_event(event)
```

---

## WebSocket Protocol

### Connection

```typescript
// Connect to experiment stream
const ws = new WebSocket(
  `ws://localhost:3000/api/research/experiments/${experimentId}/stream`,
  {
    headers: {
      Authorization: `Bearer ${token}`
    }
  }
);

ws.onopen = () => {
  console.log("Connected to experiment stream");
};
```

### Message Format

**Server → Client:**
```typescript
{
  type: "event";
  event: ResearchEvent;
}
```

**Client → Server:**
```typescript
// Subscribe to specific event types
{
  type: "subscribe";
  event_types: string[];
}

// Pause/Resume experiment
{
  type: "control";
  action: "pause" | "resume";
}

// Send feedback
{
  type: "feedback";
  message: string;
}
```

### Example Client Implementation

```typescript
class ExperimentStream {
  private ws: WebSocket;
  private listeners: Map<string, Function[]> = new Map();

  constructor(experimentId: string, token: string) {
    this.ws = new WebSocket(
      `ws://localhost:3000/api/research/experiments/${experimentId}/stream`,
      { headers: { Authorization: `Bearer ${token}` } }
    );

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };
  }

  on(eventType: string, callback: Function) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, []);
    }
    this.listeners.get(eventType)!.push(callback);

    // Subscribe to this event type
    this.ws.send(JSON.stringify({
      type: "subscribe",
      event_types: [eventType]
    }));
  }

  private handleMessage(message: any) {
    if (message.type === "event") {
      const eventType = message.event.type;
      const callbacks = this.listeners.get(eventType) || [];
      callbacks.forEach(cb => cb(message.event));
    }
  }

  pause() {
    this.ws.send(JSON.stringify({ type: "control", action: "pause" }));
  }

  resume() {
    this.ws.send(JSON.stringify({ type: "control", action: "resume" }));
  }

  sendFeedback(message: string) {
    this.ws.send(JSON.stringify({ type: "feedback", message }));
  }

  close() {
    this.ws.close();
  }
}

// Usage
const stream = new ExperimentStream("exp_123", token);

stream.on("experiment_progress", (event) => {
  console.log(`Progress: ${event.progress.percentage}%`);
});

stream.on("step_completed", (event) => {
  console.log(`Step completed: ${event.step_name}`);
});

stream.on("experiment_completed", (event) => {
  console.log("Experiment complete!", event.results);
  stream.close();
});
```

---

## Extension Configuration

### Configuration Schema

```toml
[extensions.uagent_research]
# Database
database_url = "postgresql://user:pass@localhost:5432/uagent_research"
database_pool_size = 10
database_max_overflow = 20

# Workspace
workspace_dir = "./workspaces/research"
workspace_cleanup_policy = "on_session_end"  # or "manual", "daily"
workspace_max_size_gb = 100

# Concurrency
max_concurrent_experiments = 5
max_concurrent_llm_calls = 10
background_task_workers = 4

# Timeouts
default_experiment_timeout_seconds = 3600
max_experiment_timeout_seconds = 7200
llm_call_timeout_seconds = 120

# Scientific Research
[extensions.uagent_research.scientific]
max_retries = 3
validation_strict = true
simulation_detection = true
min_experiment_iterations = 1
max_experiment_iterations = 10

# ROMA
[extensions.uagent_research.roma]
max_parallel_branches = 10
branch_timeout_seconds = 1800
tree_max_depth = 5
pruning_strategy = "low_confidence"  # or "balanced", "aggressive"

# Code Research (RepoMaster)
[extensions.uagent_research.code_research]
repomaster_enabled = true
max_repo_size_gb = 10
analysis_depth = "quick"  # or "deep"
cache_analyses = true
cache_ttl_hours = 24

# LLM Configuration
[extensions.uagent_research.llm]
default_model = "gpt-4"
streaming_enabled = true
temperature = 0.7
max_tokens = 4096

# Logging
[extensions.uagent_research.logging]
level = "INFO"
format = "json"
log_llm_interactions = true
log_experiment_steps = true
```

### Environment Variables

```bash
# Override config with environment variables
UAGENT_RESEARCH_DATABASE_URL="postgresql://..."
UAGENT_RESEARCH_MAX_CONCURRENT_EXPERIMENTS=10
UAGENT_RESEARCH_WORKSPACE_DIR="/data/research"

# OpenHands integration
OPENHANDS_SESSION_TOKEN="..."
OPENHANDS_API_URL="http://localhost:3000"
```

---

## Performance Requirements

### Response Time SLAs

| Endpoint | Target (p95) | Maximum |
|----------|-------------|---------|
| Start Experiment | < 500ms | 2s |
| Get Status | < 100ms | 500ms |
| List Experiments | < 200ms | 1s |
| Cancel Experiment | < 300ms | 1s |
| Generate Ideas | < 3s | 10s |
| Generate Hypotheses | < 2s | 8s |
| Code Analysis (quick) | < 10s | 30s |
| Code Analysis (deep) | < 60s | 300s |

### Throughput Requirements

| Operation | Target | Notes |
|-----------|--------|-------|
| Concurrent experiments per instance | 5 | Configurable |
| WebSocket connections per instance | 100 | |
| API requests per second | 100 | Per instance |
| Experiment starts per minute | 20 | |

### Resource Limits

| Resource | Limit | Notes |
|----------|-------|-------|
| Max experiment duration | 2 hours | Configurable |
| Max workspace size | 100 GB | Per experiment |
| Max database size | 1 TB | Total |
| Max LLM tokens per experiment | 1M | Configurable |
| Max memory per experiment | 16 GB | |

### Scalability Targets

| Metric | Target | Horizon |
|--------|--------|---------|
| Active users | 100 | 6 months |
| Experiments per day | 1,000 | 6 months |
| Total experiments stored | 100,000 | 12 months |
| Database size | 500 GB | 12 months |

---

## Security Specifications

### Authentication

- All API endpoints require valid OpenHands session token
- WebSocket connections require authentication during handshake
- API keys must be rotated every 90 days

### Authorization

- Users can only access their own experiments and sessions
- Admin users can view all experiments (read-only)
- Workspace isolation enforced by OpenHands runtime

### Data Protection

- Experiment results containing sensitive data must be encrypted at rest
- Workspace files cleaned up after session end (configurable)
- Database backups encrypted

### Rate Limiting

```typescript
interface RateLimits {
  experiments_per_hour: 20;
  ideas_per_hour: 100;
  api_calls_per_minute: 100;
  websocket_messages_per_second: 10;
}
```

---

**Next**: See `07_FRONTEND_INTEGRATION.md` for detailed UI/UX specifications and component architecture.
