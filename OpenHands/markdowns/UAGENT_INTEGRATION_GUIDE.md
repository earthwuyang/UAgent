# UAgent Research Extension - Integration Guide

## Overview

The **UAgent Research Extension** adds advanced research capabilities to OpenHands, including:

- **Scientific Research**: Automated experiment design, execution, and analysis
- **Code Research (RepoMaster)**: Deep code repository analysis and understanding
- **ROMA**: Tree-based research orchestration for complex multi-stage research
- **AI Scientist**: Automated idea generation and hypothesis formulation

## Architecture

### Integration Points

1. **Server Integration** (`openhands/server/app.py`)
   - Extension loaded at startup (lines 37-64)
   - Database initialized in lifespan manager
   - API routes registered automatically

2. **Extension Location**
   ```
   OpenHands/extensions/uagent_research/
   ├── uagent_research/
   │   ├── models/          # Database models
   │   ├── engines/         # Research engines
   │   ├── agents/          # Research agents
   │   ├── api/             # API routes
   │   └── examples/        # Usage examples
   └── tests/              # Integration tests
   ```

3. **Database**
   - Location: `openhands_research.db` (SQLite)
   - Configurable via `RESEARCH_DATABASE_URL` environment variable
   - Stores: Experiments, Sessions, Ideas, Hypotheses

### Core Components

#### 1. Models

- **Experiment**: Research experiments with status tracking
- **ResearchSession**: User research sessions
- **Idea**: Generated research ideas with scoring
- **Hypothesis**: Testable hypotheses derived from ideas

#### 2. Research Types

- `scientific`: Scientific experiments (AI Scientist style)
- `code`: Code repository analysis (RepoMaster)
- `roma`: Multi-stage tree-based research

#### 3. Experiment Status Flow

```
PENDING → RUNNING → COMPLETED
              ↓
            FAILED
              ↓
          CANCELLED
```

## API Endpoints

Base URL: `http://localhost:3000/api/research`

### Health Check
```bash
GET /api/research/health
```

**Response:**
```json
{
  "status": "healthy",
  "extension": "uagent_research",
  "version": "0.1.0",
  "timestamp": "2025-10-04T11:18:59.168216"
}
```

### Start Experiment
```bash
POST /api/research/experiments/start
```

**Request Body:**
```json
{
  "goal": "Compare sorting algorithm performance",
  "session_id": "session_123",
  "research_type": "scientific",
  "config": {
    "max_iterations": 10,
    "timeout": 300
  }
}
```

**Response:**
```json
{
  "id": "exp_session_123_1759576805_e5152ed4",
  "session_id": "session_123",
  "experiment_type": "scientific",
  "goal": "Compare sorting algorithm performance",
  "status": "pending",
  "progress_percentage": 0.0,
  "current_step": null,
  "results": null,
  "error_message": null,
  "created_at": "2025-10-04T11:30:00.000000",
  "started_at": null,
  "completed_at": null
}
```

### Get Experiment Status
```bash
GET /api/research/experiments/{experiment_id}
```

### List Experiments
```bash
GET /api/research/experiments?session_id=xxx&status=completed&limit=50
```

**Query Parameters:**
- `session_id` (optional): Filter by session
- `status` (optional): Filter by status (pending, running, completed, failed, cancelled)
- `limit` (optional): Max results (default: 50)
- `offset` (optional): Pagination offset

### Cancel Experiment
```bash
DELETE /api/research/experiments/{experiment_id}
```

### Generate Ideas
```bash
POST /api/research/ideas/generate
```

**Request Body:**
```json
{
  "topic": "Machine Learning Optimization",
  "context": "Exploring novel approaches",
  "num_ideas": 5,
  "creativity": 0.7
}
```

### Generate Hypotheses
```bash
POST /api/research/hypotheses/generate
```

**Request Body:**
```json
{
  "idea": "Use ML to predict optimal compiler flags",
  "background": "Recent advances in AutoML",
  "num_hypotheses": 3
}
```

### Get Research Tree (ROMA)
```bash
GET /api/research/sessions/{session_id}/tree
```

## Testing UAgent Integration

### Quick Test

1. **Health Check:**
   ```bash
   curl http://120.46.207.248:3000/api/research/health
   ```

2. **Start an Experiment:**
   ```bash
   curl -X POST http://120.46.207.248:3000/api/research/experiments/start \
     -H "Content-Type: application/json" \
     -d '{
       "goal": "Test experiment",
       "session_id": "test_session_1",
       "research_type": "scientific"
     }'
   ```

3. **List Experiments:**
   ```bash
   curl http://120.46.207.248:3000/api/research/experiments
   ```

### Automated Test Suite

Run the integration test suite:

```bash
cd /home/wuy/AI/UAgent/OpenHands
python test_uagent_integration.py
```

This tests:
- ✅ Health check
- ✅ Creating scientific experiments
- ✅ Getting experiment status
- ✅ Listing experiments
- ✅ Generating ideas
- ✅ Generating hypotheses
- ✅ Code research experiments

### Unit Tests

Run model and database tests:

```bash
cd extensions/uagent_research
python -m pytest tests/
```

Or run specific test files:
```bash
python tests/test_integration_workflow.py
python tests/test_models.py
```

## Usage Examples

### Example 1: Scientific Research Workflow

```python
import requests

BASE_URL = "http://localhost:3000/api/research"

# 1. Start experiment
response = requests.post(f"{BASE_URL}/experiments/start", json={
    "goal": "Compare quicksort vs mergesort performance on random data",
    "session_id": "my_session_1",
    "research_type": "scientific",
    "config": {
        "max_iterations": 10,
        "data_size": 10000
    }
})
experiment_id = response.json()["id"]

# 2. Monitor progress
import time
while True:
    response = requests.get(f"{BASE_URL}/experiments/{experiment_id}")
    data = response.json()

    print(f"Status: {data['status']}, Progress: {data['progress_percentage']}%")

    if data['status'] in ['completed', 'failed', 'cancelled']:
        break

    time.sleep(5)

# 3. Get results
if data['status'] == 'completed':
    print("Results:", data['results'])
```

### Example 2: Code Research (RepoMaster)

```python
response = requests.post(f"{BASE_URL}/experiments/start", json={
    "goal": "Analyze codebase architecture and identify design patterns",
    "session_id": "code_session_1",
    "research_type": "code",
    "config": {
        "repository_path": "/workspace",
        "analysis_depth": "deep"
    }
})
```

### Example 3: Idea Generation + Hypothesis Testing

```python
# 1. Generate ideas
response = requests.post(f"{BASE_URL}/ideas/generate", json={
    "topic": "Neural Network Optimization",
    "num_ideas": 5,
    "creativity": 0.8
})
ideas = response.json()["ideas"]

# 2. Generate hypotheses for best idea
best_idea = max(ideas, key=lambda x: x['impact_score'])

response = requests.post(f"{BASE_URL}/hypotheses/generate", json={
    "idea": best_idea['title'],
    "background": best_idea['description'],
    "num_hypotheses": 3
})
hypotheses = response.json()["hypotheses"]

# 3. Test hypothesis with experiment
response = requests.post(f"{BASE_URL}/experiments/start", json={
    "goal": f"Test hypothesis: {hypotheses[0]['statement']}",
    "session_id": "hypothesis_test_1",
    "research_type": "scientific"
})
```

## Environment Configuration

### Required Environment Variables

Set in `start_openhands_research.sh`:

```bash
# Research database
export RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./openhands_research.db"

# LLM configuration (for AI features)
export LLM_MODEL="dashscope/qwen3-coder-plus"
export LLM_API_KEY="${DASHSCOPE_API_KEY}"
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### Database Configuration

The extension uses SQLAlchemy with async support:

- **Development**: SQLite (`sqlite+aiosqlite:///./openhands_research.db`)
- **Production**: Can use PostgreSQL (`postgresql+asyncpg://...`)

## Frontend Integration (Future)

The extension includes WebSocket support for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:3000/api/research/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Experiment update:', data);
  // Update UI with progress
};
```

## Troubleshooting

### Extension Not Loading

Check server logs:
```bash
tmux attach -t uagent-backend
```

Look for:
- ✅ "UAgent Research Extension loaded from source"
- ✅ "Research database initialized"
- ✅ "UAgent Research Extension routes registered"

### Database Issues

Reset database:
```bash
rm openhands_research.db
# Restart server - database will be recreated
```

### API Errors

Check database connection:
```bash
curl http://120.46.207.248:3000/api/research/health
```

Expected response:
```json
{"status": "healthy", "extension": "uagent_research", "version": "0.1.0"}
```

## Current Status

### ✅ Working
- Extension loading and initialization
- Database models and migrations
- API endpoints (health, experiments, ideas, hypotheses)
- Experiment creation and status tracking
- WebSocket support for real-time updates

### 🚧 In Development
- Actual experiment execution engines
- LLM integration for idea/hypothesis generation
- Code analysis engine (RepoMaster)
- ROMA tree-based orchestration
- Frontend UI components

### 📋 Next Steps

1. **Implement Scientific Research Engine**
   - Connect to OpenHands agent for code execution
   - Automated experiment design
   - Results analysis and reporting

2. **Implement Code Research Engine**
   - Repository analysis
   - Dependency graph generation
   - Pattern detection

3. **Frontend UI**
   - Experiment dashboard
   - Real-time progress tracking
   - Results visualization

4. **LLM Integration**
   - Idea generation with configured LLM
   - Hypothesis formulation
   - Experiment design assistance

## References

- **Extension Code**: `/home/wuy/AI/UAgent/OpenHands/extensions/uagent_research/`
- **Server Integration**: `/home/wuy/AI/UAgent/OpenHands/openhands/server/app.py`
- **Database**: `/home/wuy/AI/UAgent/OpenHands/openhands_research.db`
- **Test Script**: `/home/wuy/AI/UAgent/OpenHands/test_uagent_integration.py`
