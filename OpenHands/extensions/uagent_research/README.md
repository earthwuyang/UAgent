# UAgent Research Extension for OpenHands

Advanced research capabilities integrated into OpenHands, providing scientific research, code analysis, and AI-powered idea generation.

## Features

### ✅ Implemented (Production-Ready)

- **Data Models**: Complete SQLAlchemy models for experiments, sessions, ideas, and hypotheses
- **Scientific Research Engine**: Hypothesis generation, experiment design, execution, and analysis
- **Code Research Engine**: Repository analysis, code comprehension, and architecture understanding
- **Research Agents**: OpenHands agents that extend CodeActAgent with research capabilities
- **API Routes**: FastAPI routes for experiment management and research operations
- **Database Support**: Async PostgreSQL and SQLite support
- **Testing**: Comprehensive test suite with 100% model coverage

### 🚧 In Progress

- Frontend React components
- WebSocket streaming for real-time updates
- ROMA tree visualization

## Installation

### From Source

```bash
cd /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Initialize Database

```python
from uagent_research.models.base import init_database

# For SQLite (development)
await init_database("sqlite+aiosqlite:///./research.db")

# For PostgreSQL (production)
await init_database("postgresql+asyncpg://user:pass@localhost/research_db")
```

### 2. Use Research Engines

#### Scientific Research

```python
from uagent_research.engines import ScientificResearchEngine
from openhands.llm.llm import LLM
from openhands.runtime.runtime import Runtime
from openhands.events.stream import EventStream

# Initialize
llm = LLM(config)
engine = ScientificResearchEngine(llm)

# Run experiment
results = await engine.run_experiment(
    goal="Compare performance of quicksort vs mergesort",
    runtime=runtime,
    event_stream=event_stream,
    session_id="session_123"
)
```

#### Code Research

```python
from uagent_research.engines import CodeResearchEngine

engine = CodeResearchEngine(llm)

analysis = await engine.analyze_repository(
    query="How does authentication work in this codebase?",
    workspace="/path/to/repo",
    runtime=runtime
)
```

### 3. Use Research Agents

```python
from uagent_research.agents import ScientificResearchAgent
from openhands.core.config import AgentConfig
from openhands.llm.llm_registry import LLMRegistry

# Create agent
config = AgentConfig(...)
llm_registry = LLMRegistry()
agent = ScientificResearchAgent(config, llm_registry)

# Use in OpenHands controller
from openhands.controller.agent_controller import AgentController

controller = AgentController(agent=agent, ...)
result = await controller.run("Run experiment to test hypothesis X")
```

### 4. Use API Routes

```python
from fastapi import FastAPI
from uagent_research.api import router
from uagent_research.models.base import init_database

app = FastAPI()

# Initialize database
@app.on_event("startup")
async def startup():
    await init_database("sqlite+aiosqlite:///./research.db")

# Include research routes
app.include_router(router)

# Run server
# uvicorn main:app --reload
```

#### API Endpoints

- `POST /api/research/experiments/start` - Start experiment
- `GET /api/research/experiments/{id}` - Get experiment status
- `GET /api/research/experiments` - List experiments
- `DELETE /api/research/experiments/{id}` - Cancel experiment
- `GET /api/research/sessions/{id}/tree` - Get ROMA tree
- `POST /api/research/ideas/generate` - Generate ideas
- `POST /api/research/hypotheses/generate` - Generate hypotheses
- `GET /api/research/health` - Health check

## Architecture

### Directory Structure

```
uagent_research/
├── agents/                       # OpenHands agents
│   ├── scientific_research_agent.py
│   └── code_research_agent.py
│
├── engines/                      # Research engines
│   ├── scientific_research.py
│   └── code_research.py
│
├── models/                       # Data models
│   ├── base.py
│   ├── experiment.py
│   ├── research_session.py
│   ├── idea.py
│   └── hypothesis.py
│
├── api/                         # API routes
│   └── research_routes.py
│
├── tests/                       # Test suite
│   └── test_models.py
│
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
├── pytest.ini                   # Test configuration
└── README.md                    # This file
```

### Integration with OpenHands

The extension integrates with OpenHands at multiple levels:

1. **Agent Level**: Research agents extend `CodeActAgent`
2. **Runtime Level**: Use OpenHands runtime for code execution
3. **Event Level**: Emit events to OpenHands event stream
4. **API Level**: FastAPI routes integrate with OpenHands server
5. **LLM Level**: Use OpenHands LLM infrastructure

## Agent Coordination Mode

Research agents now operate as conversation coordinators instead of blocking the main chat while experiments run. When a research goal is detected, the agent:

- Launches the background experiment via `TreeSearchOrchestrator`
- Polls `ResearchSessionManager` every few seconds for status updates
- Streams meaningful progress reports back to the user
- Remains responsive to ad-hoc questions and control commands

### Real-time Interaction

While coordination mode is active, users can:

- Ask “how’s progress?” to receive an immediate status snapshot
- Issue control commands such as “pause research”, “resume research”, or “cancel research”
- Target specific nodes, e.g. “cancel node idea-2”
- Steer execution with guidance like “focus on the DARTS approach”

### Coordination Flow

```
User Message → Research Coordinator Agent
              ↓
              ├→ start_research(goal, session)
              ├→ poll ResearchSessionManager (default 10s cadence)
              ├→ relay progress updates as chat messages
              └→ forward control commands through ControlBus
```

### Configuration

Fine-tune coordination behaviour via environment variables (see `.env.example`):

- `ENABLE_AGENT_COORDINATION` – toggle coordination mode (default: `true`)
- `RESEARCH_POLL_INTERVAL` – seconds between automatic progress polls (default: `10`)
- `PROGRESS_CACHE_TTL` – seconds to cache status responses for fast repeated queries (default: `2`)

### Troubleshooting

- **No updates while research runs:** ensure `ENABLE_AGENT_COORDINATION=true` and the middleware is loaded
- **Updates too frequent or sparse:** adjust `RESEARCH_POLL_INTERVAL`
- **Duplicate progress notifications:** verify that legacy session-level progress reporters are disabled or running at a higher interval

## Testing

### Run All Tests

```bash
cd /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research
pytest
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html
```

### Test Results

```
======================== 6 passed, 2 warnings in 2.70s =========================
```

Current test coverage: **100% for models**, **7% overall** (agents and engines need integration tests)

## Development

### Running Tests During Development

```bash
# Watch mode
pytest --watch

# Specific test file
pytest tests/test_models.py

# Specific test
pytest tests/test_models.py::test_create_experiment -v
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy .
```

## Configuration

### Environment Variables

```bash
# Database
RESEARCH_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/research_db

# LLM
LITELLM_MODEL=gpt-4
LITELLM_API_KEY=your_key

# Research settings
RESEARCH_MAX_RETRIES=3
RESEARCH_VALIDATION_STRICT=true
RESEARCH_MAX_ITERATIONS=10
```

### Python Configuration

```python
config = {
    'max_retries': 3,
    'validation_strict': True,
    'simulation_detection': True,
    'max_iterations': 10,
}

engine = ScientificResearchEngine(llm, config)
```

## API Examples

### Start Experiment

```bash
curl -X POST http://localhost:8000/api/research/experiments/start \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Compare sorting algorithm performance",
    "session_id": "session_123",
    "research_type": "scientific"
  }'
```

Response:
```json
{
  "id": "exp_session_123_1696429200_abc123",
  "session_id": "session_123",
  "experiment_type": "scientific",
  "goal": "Compare sorting algorithm performance",
  "status": "pending",
  "progress_percentage": 0.0,
  "created_at": "2025-10-04T12:00:00"
}
```

### Get Experiment Status

```bash
curl http://localhost:8000/api/research/experiments/exp_session_123_1696429200_abc123
```

### List Experiments

```bash
curl "http://localhost:8000/api/research/experiments?session_id=session_123&status=completed"
```

## Data Models

### Experiment

```python
experiment = Experiment(
    id="exp_123",
    session_id="session_123",
    experiment_type=ExperimentType.SCIENTIFIC,
    goal="Research goal",
    status=ExperimentStatus.RUNNING,
    progress_percentage=50.0,
    current_step="Analyzing results",
    results={...},
    config={...}
)
```

### Research Session

```python
session = ResearchSession(
    id="session_123",
    user_id="user_1",
    mode=SessionMode.RESEARCH,
    title="My Research Project",
    research_tree={...}
)
```

### Idea

```python
idea = Idea(
    id="idea_1",
    session_id="session_123",
    title="Novel ML Approach",
    description="...",
    topic="Machine Learning",
    novelty_score=0.9,
    feasibility_score=0.7,
    impact_score=0.8
)
```

### Hypothesis

```python
hypothesis = Hypothesis(
    id="hyp_1",
    session_id="session_123",
    statement="Algorithm A is faster than B",
    null_hypothesis="No difference in performance",
    testability_score=0.9,
    tested=False
)
```

## Troubleshooting

### Database Connection Error

```python
# Ensure database is initialized
await init_database("sqlite+aiosqlite:///./research.db")
```

### Import Errors

```bash
# Install in editable mode
pip install -e .
```

### Test Failures

```bash
# Clean and reinstall
pip uninstall uagent-research-extension
pip install -e .
pytest
```

## Roadmap

### Phase 1: Core Implementation ✅ (Complete)
- [x] Data models
- [x] Scientific research engine
- [x] Code research engine
- [x] Research agents
- [x] API routes
- [x] Unit tests

### Phase 2: Integration (In Progress)
- [ ] Integrate with OpenHands server
- [ ] WebSocket streaming
- [ ] Real-time progress updates
- [ ] Integration tests

### Phase 3: Frontend
- [ ] React components
- [ ] Research dashboard
- [ ] Experiment viewer
- [ ] ROMA tree visualizer

### Phase 4: Advanced Features
- [ ] ROMA parallel research
- [ ] Idea generation with LLM
- [ ] Hypothesis scoring
- [ ] Experiment templates

## Contributing

### Development Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`
4. Make changes
5. Run tests again
6. Submit PR

### Code Standards

- **Type hints** for all functions
- **Docstrings** for all classes and public methods
- **Tests** for all new functionality
- **No mocking** in production code
- **Production-ready** code only

## License

MIT

## Contact

- GitHub Issues: [Report bugs or request features]
- Documentation: See `markdown_files/migration_plan/` for detailed design docs

## Acknowledgments

- Built on [OpenHands](https://github.com/All-Hands-AI/OpenHands)
- Integrates UAgent research capabilities
- Inspired by AI Scientist and RepoMaster

---

## Real-Time Updates and Heartbeat Mechanism

### Overview

The UAgent Research Extension provides real-time progress updates through an EventBus system with built-in heartbeat functionality. This ensures clients stay informed even when long-running operations are in progress.

### Heartbeat Mechanism

**What are Heartbeats?**

Heartbeats are synthetic events automatically emitted when research branches are idle for longer than the configured interval. They serve multiple purposes:

- **Keep WebSocket connections alive** - Prevent timeouts during long-running operations
- **Show progress** - Indicate that the system is still working even when no actual events occur
- **Monitor health** - Allow clients to detect if a branch has stalled or crashed

**How Heartbeats Work:**

1. When a branch publishes its first event, a heartbeat supervisor is automatically started
2. The supervisor monitors idle time (time since last event)
3. If idle time exceeds the heartbeat interval, a synthetic `StepEvent` with `action="heartbeat"` is emitted
4. Heartbeats continue until the branch completes or is explicitly stopped
5. When a branch finishes (success or failure), its heartbeat supervisor is automatically cleaned up

**Default Behavior:**

- Heartbeat interval: **5 seconds** (configurable)
- Enabled by default
- Automatic cleanup on branch completion
- Independent supervisors for each branch (parallel branches get independent heartbeats)

### Configuration

#### Environment Variables

Add these to your `.env` file or export them in your shell:

```bash
# Enable heartbeats (default: true)
RESEARCH_HEARTBEAT_ENABLED=true

# Heartbeat interval in seconds (default: 5, range: 2-10, 0 to disable)
RESEARCH_HEARTBEAT_INTERVAL=5
```

#### Disabling Heartbeats

For testing or debugging, you may want to disable heartbeats:

```bash
# Option 1: Set interval to 0
export RESEARCH_HEARTBEAT_INTERVAL=0

# Option 2: Disable explicitly (if implemented)
export RESEARCH_HEARTBEAT_ENABLED=false
```

#### Adjusting Heartbeat Frequency

**Faster heartbeats** (more frequent updates, higher overhead):
```bash
export RESEARCH_HEARTBEAT_INTERVAL=2  # Heartbeat every 2 seconds
```

**Slower heartbeats** (less frequent updates, lower overhead):
```bash
export RESEARCH_HEARTBEAT_INTERVAL=10  # Heartbeat every 10 seconds
```

**Note:** The system automatically caps the interval at 10 seconds and logs a warning if you exceed this limit.

### WebSocket Message Format

#### Heartbeat Event Structure

When a heartbeat is emitted, clients receive a message like:

```json
{
  "type": "event_log",
  "event_type": "StepEvent",
  "branch_id": "idea-0",
  "action": "heartbeat",
  "reasoning": "Synthetic heartbeat - no progress since last event",
  "timestamp": "2025-10-08T01:30:00Z",
  "metadata": {
    "is_heartbeat": true,
    "idle_time_seconds": 5.2
  }
}
```

#### Regular Event Structure

Regular progress events have the same structure but with actual action/reasoning:

```json
{
  "type": "event_log",
  "event_type": "StepEvent",
  "branch_id": "idea-0",
  "action": "Analyzing repository structure",
  "reasoning": "Scanning repository layout and files",
  "timestamp": "2025-10-08T01:30:05Z"
}
```

### Frontend Integration

#### Filtering Heartbeats

You may want to display heartbeats differently from regular events:

```javascript
function handleWebSocketMessage(message) {
  if (message.action === 'heartbeat') {
    // Show subtle loading indicator
    updateBranchSpinner(message.branch_id, true);
  } else {
    // Show actual progress update
    addProgressLog(message.branch_id, message.action);
  }
}
```

#### Detecting Stalls

Use heartbeats to detect if a branch has stalled:

```javascript
const branchTimestamps = {};

function handleWebSocketMessage(message) {
  branchTimestamps[message.branch_id] = Date.now();
}

// Check for stalled branches
setInterval(() => {
  const now = Date.now();
  for (const [branchId, lastUpdate] of Object.entries(branchTimestamps)) {
    const timeSinceUpdate = (now - lastUpdate) / 1000;
    
    // No heartbeat for 15 seconds? Branch may be stalled
    if (timeSinceUpdate > 15) {
      showWarning(`Branch ${branchId} appears stalled`);
    }
  }
}, 5000);
```

### Troubleshooting

#### Heartbeats Too Frequent

**Problem:** Receiving too many heartbeat events, cluttering logs

**Solutions:**
```bash
# Increase interval (slower heartbeats)
export RESEARCH_HEARTBEAT_INTERVAL=10

# Or disable heartbeats entirely
export RESEARCH_HEARTBEAT_INTERVAL=0
```

#### Heartbeats Continue After Completion

**Problem:** Heartbeats continue even after branch finishes

**Possible Causes:**
1. `stop_branch_heartbeat()` not called in orchestrator
2. Branch completed with exception, cleanup didn't run
3. Branch ID mismatch

**Debug Steps:**
```python
# Check active heartbeats
stats = event_bus.get_stats()
print(f"Active heartbeats: {stats['active_heartbeats']}")
print(f"Active branches: {stats['heartbeat_branches']}")

# Manually stop heartbeat
event_bus.stop_branch_heartbeat("problematic-branch-id")
```

#### No Heartbeats Appearing

**Problem:** Not receiving any heartbeat events

**Checklist:**
1. ✅ Verify `RESEARCH_HEARTBEAT_INTERVAL > 0`
2. ✅ Ensure EventBus is properly initialized
3. ✅ Check that WebSocketPublisher is started
4. ✅ Verify WebSocket client is connected
5. ✅ Wait long enough (idle time must exceed interval)

**Debug Commands:**
```bash
# Check environment variable
echo $RESEARCH_HEARTBEAT_INTERVAL

# Check EventBus initialization
python -c "
from extensions.uagent_research.orchestrator.event_bus import EventBus
bus = EventBus()
print(f'Heartbeat interval: {bus.heartbeat_interval}s')
"

# Check logs for heartbeat activity
tail -f logs/research.log | grep -i heartbeat
```

#### High Memory Usage

**Problem:** Memory grows over time

**Possible Cause:** Heartbeat supervisors not being cleaned up

**Solution:** Ensure `stop_branch_heartbeat()` is called:
```python
# In tree_orchestrator.py
async def _execute_node(self, node):
    try:
        # ... node execution ...
    finally:
        # Cleanup heartbeat
        self.event_bus.stop_branch_heartbeat(node.id)
```

### Performance Considerations

#### Overhead

Heartbeats have minimal overhead:
- Memory: ~1KB per active branch supervisor
- CPU: Negligible (asyncio sleep loop)
- Network: One WebSocket message every N seconds per branch

#### Scaling

For systems with many parallel branches:
```bash
# Reduce heartbeat frequency for better scalability
export RESEARCH_HEARTBEAT_INTERVAL=10

# Or disable for high-throughput scenarios
export RESEARCH_HEARTBEAT_INTERVAL=0
```

### Implementation Details

#### EventBus Heartbeat Supervisor

The heartbeat supervisor runs as an async task per branch:

```python
async def _heartbeat_supervisor(self, branch_id: str):
    """Emit periodic heartbeats when branch is idle"""
    while True:
        await asyncio.sleep(self.heartbeat_interval)
        
        idle_time = time.time() - self._last_event_time.get(branch_id, 0)
        
        if idle_time >= self.heartbeat_interval:
            # Emit synthetic StepEvent
            heartbeat_event = StepEvent(
                branch_id=branch_id,
                action="heartbeat",
                reasoning=f"Synthetic heartbeat - idle for {idle_time:.1f}s"
            )
            await self.publish(heartbeat_event)
```

#### Automatic Cleanup

Heartbeats are automatically stopped when:

1. **Branch completes successfully:**
   ```python
   # In tree_orchestrator.py:_execute_node()
   finally:
       self.event_bus.stop_branch_heartbeat(node.id)
   ```

2. **Branch is cancelled:**
   ```python
   # In tree_orchestrator.py:_control_loop()
   elif action["action"] == "cancel_node":
       self.event_bus.stop_branch_heartbeat(node_id)
   ```

3. **Orchestrator is cancelled:**
   ```python
   # In tree_orchestrator.py:cancel()
   for node in self.tree.nodes.values():
       self.event_bus.stop_branch_heartbeat(node.id)
   ```

### Testing

Run heartbeat tests:

```bash
# Unit tests
pytest tests/test_websocket.py::test_heartbeat_delivery
pytest tests/test_websocket.py::test_heartbeat_stops_on_completion

# Integration tests
pytest tests/test_heartbeat_integration.py

# All heartbeat tests
pytest tests/ -k heartbeat -v
```

### Monitoring

#### Statistics

Get heartbeat statistics at runtime:

```python
stats = event_bus.get_stats()

print(f"Total heartbeats sent: {stats['heartbeats_sent']}")
print(f"Active heartbeat supervisors: {stats['active_heartbeats']}")
print(f"Branches with heartbeats: {stats['heartbeat_branches']}")
```

#### Logging

Enable debug logging to monitor heartbeat activity:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Logs will show:
# DEBUG: Heartbeat supervisor started for branch idea-0 (interval=5s)
# DEBUG: Emitting heartbeat for branch idea-0 (idle=5.2s)
# DEBUG: Heartbeat delivered to 3 subscribers
# INFO: Stopped heartbeat for branch idea-0 (active for 45.3s, sent 127 total)
```

---

## Additional Resources

- **EventBus Implementation:** `orchestrator/event_bus.py`
- **WebSocket Publisher:** `orchestrator/ws_publisher.py`
- **Heartbeat Tests:** `tests/test_heartbeat_integration.py`
- **Configuration:** `.env.example`

For questions or issues, check the troubleshooting section above or review the test files for usage examples.


---

## Research Control API

The UAgent Research Extension provides a comprehensive REST API for controlling and monitoring research experiments.

### Get Experiment Status

Get detailed real-time status of a running experiment.

**Endpoint:** `GET /api/research/experiments/{experiment_id}/status`

**Response:**
```json
{
  "experiment_id": "exp_...",
  "status": "running",
  "stats": {
    "total_nodes": 10,
    "completed": 5,
    "failed": 1,
    "running": 4,
    "pending": 0,
    "total_cost": 0.25,
    "total_tokens": 5000
  },
  "adapters": {
    "deepresearch": {
      "status": "running",
      "current_step": "Browsing documentation",
      "last_event": "2025-01-06T10:30:00",
      "cost": 0.05,
      "tokens": 1200
    }
  },
  "active_branches": [
    {
      "branch_id": "idea-0-hyp-0",
      "title": "Test using pgvector",
      "adapter": "codeact",
      "status": "running",
      "progress": "Running tests...",
      "cost": 0.10
    }
  ],
  "created_at": "2025-01-06T10:00:00",
  "last_update": "2025-01-06T10:30:05"
}
```

### Get Experiment Events

Retrieve event history for an experiment (fallback to WebSocket).

**Endpoint:** `GET /api/research/experiments/{experiment_id}/events?since_version=0&limit=100`

**Parameters:**
- `since_version` (optional): Get events after this version number (0-based, default: 0)
- `limit` (optional): Maximum events to return (1-1000, default: 100)

**Response:**
```json
{
  "experiment_id": "exp_...",
  "since_version": 0,
  "current_version": 25,
  "earliest_available_version": 1,
  "events": [
    {
      "version": 1,
      "timestamp": "2025-01-06T10:25:00",
      "type": "PLAN",
      "branch_id": "idea-0",
      "data": {...}
    }
  ],
  "has_more": false,
  "has_gap": false
}
```

**Response Fields:**
- `earliest_available_version`: Oldest version still in buffer (may be > 1 if events dropped due to 1000-event limit)
- `has_gap`: `true` if `since_version` < `earliest_available_version` (indicates data loss - client should resync from `earliest_available_version`)
- `has_more`: `true` if more events available beyond the requested `limit`

**Note on Data Loss:** Events are stored in a circular buffer (max 1000 per experiment). If an experiment generates more than 1000 events, older events are dropped. Clients should check `has_gap` to detect missed events and resync if necessary.

### Control Experiment

Send control commands to running experiments.

**Endpoint:** `PATCH /api/research/experiments/{experiment_id}`

#### Supported Actions

##### Pause Research
```json
{"action": "pause"}
```

##### Resume Research
```json
{"action": "resume"}
```

##### Cancel Experiment
```json
{"action": "cancel"}
```

##### Cancel Specific Node
```json
{
  "action": "cancel_node",
  "target": {"node_id": "idea-2"}
}
```

##### Reprioritize Nodes
```json
{
  "action": "reprioritize",
  "target": {"adapter": "codeact"},
  "payload": {"delta": 0.2}
}
```

##### Steer Adapter
```json
{
  "action": "steer",
  "target": {"adapter": "codeact"},
  "payload": {"text": "Focus on DARTS algorithm, ignore ENAS"}
}
```

##### Add Research Direction
```json
{
  "action": "add_node",
  "payload": {
    "parent_id": "root",
    "node": {
      "type": "IDEA",
      "title": "Try DuckDB vector extension",
      "content": "Explore DuckDB's native vector support",
      "prior": 0.8
    }
  }
}
```

**Success Response:**
```json
{
  "status": "acknowledged",
  "experiment_id": "exp_...",
  "action": "pause",
  "message": "Control command 'pause' sent successfully",
  "timestamp": "2025-01-06T10:30:00"
}
```

**Error Responses:**
- `400`: Invalid action or missing required fields
- `404`: Experiment not found
- `409`: Experiment not in valid state for action
- `500`: Internal server error

### Usage Examples

#### Python

```python
import requests

# Get status
response = requests.get("http://localhost:3000/api/research/experiments/exp_123/status")
status = response.json()
print(f"Running nodes: {status['stats']['running']}")

# Pause experiment
response = requests.patch(
    "http://localhost:3000/api/research/experiments/exp_123",
    json={"action": "pause"}
)
print(response.json())

# Get events
response = requests.get(
    "http://localhost:3000/api/research/experiments/exp_123/events",
    params={"since_version": 10, "limit": 50}
)
events = response.json()
print(f"Retrieved {len(events['events'])} events")
```

#### cURL

```bash
# Get status
curl http://localhost:3000/api/research/experiments/exp_123/status

# Steer adapter
curl -X PATCH http://localhost:3000/api/research/experiments/exp_123 \
  -H "Content-Type: application/json" \
  -d '{"action": "steer", "target": {"adapter": "codeact"}, "payload": {"text": "Focus on performance"}}'
```

### API Notes

- **Status Endpoint**: Returns real-time status from `ResearchSessionManager` when experiment is actively tracked. Falls back to database for completed experiments.
- **Events Endpoint**: Stores last 1000 events per experiment in a circular buffer. Older events are dropped when the buffer is full. Clients should:
  - Poll with `since_version` to get incremental updates
  - Check `has_gap` to detect data loss and resync from `earliest_available_version`
  - Use `limit` between 1-1000 to control batch size
  - Use WebSocket for real-time streaming in production
- **Control Actions**: All actions are asynchronous—success response indicates command was sent, not completed.
- **Validation**: Request payloads are validated using Pydantic discriminated unions. Invalid requests return 400 with detailed error messages.

---

