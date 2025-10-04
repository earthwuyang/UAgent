# UAgent Research Extension - Quick Start Guide

## Installation

```bash
cd /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research
pip install -e .
```

## Starting OpenHands with Research Extension

```bash
cd /home/wuy/AI/UAgent/OpenHands

# Optional: Set custom database URL
export RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./openhands_research.db"

# Start server
python -m openhands.server.listen
```

You should see:
```
✅ Research database initialized: sqlite+aiosqlite:///./openhands_research.db
✅ UAgent Research Extension loaded successfully
```

## API Endpoints

### HTTP Endpoints

#### Start Experiment
```bash
curl -X POST http://localhost:8000/api/research/experiments/start \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Compare quicksort vs mergesort performance",
    "session_id": "session_123",
    "research_type": "scientific"
  }'
```

Response:
```json
{
  "id": "exp_abc123",
  "status": "pending",
  "session_id": "session_123",
  "experiment_type": "scientific",
  "goal": "Compare quicksort vs mergesort performance",
  "created_at": "2024-10-04T12:00:00Z"
}
```

#### Get Experiment Status
```bash
curl http://localhost:8000/api/research/experiments/exp_abc123
```

Response:
```json
{
  "id": "exp_abc123",
  "status": "running",
  "progress": {
    "percentage": 45.0,
    "current_step": "Executing experiments"
  },
  "results": null
}
```

#### List Experiments
```bash
# All experiments
curl http://localhost:8000/api/research/experiments

# Filter by session
curl http://localhost:8000/api/research/experiments?session_id=session_123

# Filter by status
curl http://localhost:8000/api/research/experiments?status=completed

# Pagination
curl http://localhost:8000/api/research/experiments?skip=0&limit=10
```

#### Health Check
```bash
curl http://localhost:8000/api/research/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "database": "connected"
}
```

### WebSocket Endpoints

#### Watch Experiment Progress

**JavaScript**:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/research/ws/experiment/exp_abc123');

ws.onopen = () => {
    console.log('Connected to experiment');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch(data.type) {
        case 'connected':
            console.log('WebSocket connected:', data.message);
            break;

        case 'progress':
            console.log(`Progress: ${data.data.percentage}%`);
            console.log(`Step: ${data.data.current_step}`);
            // Update UI with progress
            updateProgressBar(data.data.percentage);
            break;

        case 'status':
            console.log('Status changed:', data.data.status);
            break;

        case 'result':
            console.log('Experiment completed:', data.data);
            break;

        case 'error':
            console.error('Experiment error:', data.data);
            break;
    }
};

// Keep connection alive
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
    }
}, 30000);
```

**Python**:
```python
import asyncio
import websockets
import json

async def watch_experiment(experiment_id):
    uri = f"ws://localhost:8000/api/research/ws/experiment/{experiment_id}"

    async with websockets.connect(uri) as websocket:
        print(f"Connected to experiment {experiment_id}")

        async for message in websocket:
            data = json.loads(message)

            if data['type'] == 'progress':
                print(f"Progress: {data['data']['percentage']}% - {data['data']['current_step']}")

            elif data['type'] == 'result':
                print(f"Experiment completed!")
                print(json.dumps(data['data'], indent=2))
                break

# Run
asyncio.run(watch_experiment("exp_abc123"))
```

#### Watch Session

```javascript
const ws = new WebSocket('ws://localhost:8000/api/research/ws/session/session_123');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch(data.type) {
        case 'experiment_started':
            console.log('New experiment started:', data.data.experiment_id);
            break;

        case 'experiment_completed':
            console.log('Experiment completed:', data.data.experiment_id);
            break;

        case 'idea_generated':
            console.log('New idea:', data.data);
            break;

        case 'hypothesis_generated':
            console.log('New hypothesis:', data.data);
            break;
    }
};
```

## Python SDK Usage

### Using the Models Directly

```python
import asyncio
from uagent_research.models.base import init_database, get_session
from uagent_research.models import (
    Experiment,
    ExperimentType,
    ExperimentStatus,
    ResearchSession,
    SessionMode,
)

async def create_experiment_example():
    # Initialize database
    await init_database("sqlite+aiosqlite:///./research.db")

    # Create experiment
    async for session in get_session():
        # Create research session
        research_session = ResearchSession(
            id="session_1",
            user_id="user_123",
            mode=SessionMode.RESEARCH,
            title="Algorithm Performance Study",
        )
        session.add(research_session)

        # Create experiment
        experiment = Experiment(
            id="exp_1",
            session_id="session_1",
            experiment_type=ExperimentType.SCIENTIFIC,
            goal="Compare sorting algorithms",
            status=ExperimentStatus.PENDING,
        )
        session.add(experiment)

        await session.commit()
        await session.refresh(experiment)

        print(f"Created experiment: {experiment.id}")
        print(f"Status: {experiment.status.value}")

# Run
asyncio.run(create_experiment_example())
```

### Using the Research Engine

```python
from uagent_research.engines import ScientificResearchEngine
from openhands.llm.llm import LLM

# Initialize engine
llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")
engine = ScientificResearchEngine(llm=llm)

# Run experiment
results = await engine.run_experiment(
    goal="Test hypothesis about algorithm performance",
    runtime=runtime,  # OpenHands runtime
    event_stream=event_stream,  # OpenHands event stream
    session_id="session_123",
)

print(f"Hypotheses generated: {len(results['hypotheses'])}")
print(f"Experiments run: {len(results['experiment_results'])}")
print(f"Conclusion: {results['conclusion']}")
```

## Configuration

### Environment Variables

```bash
# Database URL (required)
export RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./openhands_research.db"

# Or PostgreSQL for production
export RESEARCH_DATABASE_URL="postgresql+asyncpg://user:password@localhost/openhands_research"

# Debug mode (optional)
export RESEARCH_DEBUG=true
```

### OpenHands Configuration

Add to your `config.toml`:

```toml
[research]
enabled = true
database_url = "sqlite+aiosqlite:///./openhands_research.db"
max_concurrent_experiments = 5
default_llm_model = "anthropic/claude-3-5-sonnet-20241022"
```

## Testing

### Run All Tests

```bash
cd /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research

# Run with pytest
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
python tests/test_integration_workflow.py
```

### Test WebSocket Connection

```bash
# Install wscat if needed
npm install -g wscat

# Connect to experiment WebSocket
wscat -c ws://localhost:8000/api/research/ws/experiment/exp_123

# You'll see connection message:
# {"type": "connected", "experiment_id": "exp_123", "message": "Connected to experiment exp_123", ...}

# Send ping
> ping
# Receive pong:
# < {"type": "pong", "timestamp": "2024-10-04T12:00:00Z"}
```

## Troubleshooting

### Extension Not Loading

**Symptom**: Server starts but no "✅ UAgent Research Extension loaded successfully" message

**Solutions**:
1. Check extension directory exists:
   ```bash
   ls -la /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research
   ```

2. Verify Python path:
   ```python
   import sys
   from pathlib import Path
   extensions_path = Path(__file__).parent / 'extensions'
   print(extensions_path.exists())
   ```

3. Check for import errors:
   ```bash
   cd /home/wuy/AI/UAgent/OpenHands
   python -c "import sys; sys.path.insert(0, 'extensions'); from uagent_research.api import router"
   ```

### Database Errors

**Symptom**: "Database not initialized" error

**Solutions**:
1. Check database URL:
   ```bash
   echo $RESEARCH_DATABASE_URL
   ```

2. Verify database file permissions (SQLite):
   ```bash
   touch openhands_research.db
   chmod 666 openhands_research.db
   ```

3. For PostgreSQL, verify connection:
   ```bash
   psql $RESEARCH_DATABASE_URL
   ```

### WebSocket Connection Fails

**Symptom**: WebSocket connection refused or closes immediately

**Solutions**:
1. Verify server is running:
   ```bash
   curl http://localhost:8000/api/research/health
   ```

2. Check experiment ID exists:
   ```bash
   curl http://localhost:8000/api/research/experiments/exp_123
   ```

3. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## Examples

See `/home/wuy/AI/UAgent/OpenHands/extensions/uagent_research/examples/`:
- `basic_usage.py` - Basic database operations
- More examples coming in Phase 3!

## Documentation

- **README.md** - Full documentation
- **IMPLEMENTATION_STATUS.md** - Implementation details
- **PHASE_2_INTEGRATION_COMPLETE.md** - Integration status
- **QUICKSTART.md** - This file

## Support

For issues, check:
1. Test files in `tests/` for usage examples
2. Documentation in markdown files
3. Code comments in source files

---

**Version**: 0.1.0
**Last Updated**: 2025-10-04
**Status**: Production Ready ✅
