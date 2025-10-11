# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**UAgent** is an AI software engineer system that extends OpenHands with advanced research capabilities. The project is located at `/Users/wuy/Desktop/code/UAgent` and contains the OpenHands fork as a subdirectory (`OpenHands/`) plus top-level orchestration scripts.

It combines autonomous software engineering with intelligent research through a PUCT-based tree search system.

### Core Architecture

The system has two operational modes:

1. **Direct Execution Mode**: Standard OpenHands agent flow
   - User prompt → Agent → LLM → Action → Runtime → Observation → State update
   - Event-driven architecture via EventStream
   - Runtime executes actions in Docker sandbox

2. **Research Mode**: Advanced tree-based exploration (UAgent extension)
   - TreeSearchOrchestrator manages PUCT-based tree search (`Q + c * P * sqrt(N) / (1 + n)`)
   - Research adapters: DeepResearchAdapter (web), RepoMasterAdapter (GitHub), CodeActAdapter (code)
   - EventBus coalesces events for WebSocket streaming to frontend
   - Real-time visualization via React Flow frontend

**Key architectural insight**: The EventStream serves as the backbone for all communication. Research events flow through EventBus → WebSocket → Frontend in ROMA-compatible format with incremental delta updates.

### Directory Structure

```
/Users/wuy/Desktop/code/UAgent/        # Project root
├── OpenHands/                          # OpenHands fork subdirectory
│   ├── openhands/                     # Core OpenHands system
│   │   ├── agenthub/                  # Agent implementations
│   │   ├── controller/                # AgentController orchestration
│   │   ├── core/                      # State, LLM, and event primitives
│   │   ├── events/                    # Action and Observation types
│   │   ├── runtime/                   # Docker sandbox and action execution
│   │   ├── server/                    # FastAPI server, WebSocket, session management
│   │   ├── llm/                       # LiteLLM integration
│   │   ├── memory/                    # Long-term memory systems
│   │   └── integrations/              # External service integrations
│   ├── extensions/uagent_research/    # Research extension
│   │   ├── orchestrator/              # TreeSearchOrchestrator
│   │   ├── adapters/                  # DeepResearchAdapter, RepoMasterAdapter, CodeActAdapter
│   │   ├── middleware/                # ResearchMiddleware entry point
│   │   ├── event_bus/                 # Event coalescing and WebSocket publishing
│   │   └── tests/                     # Unit and integration tests
│   ├── frontend/                      # React/TypeScript frontend
│   │   ├── src/
│   │   │   ├── components/features/   # Domain components (browser, chat, code editor, terminal)
│   │   │   ├── routes/                # File-based routing (Remix SPA)
│   │   │   ├── state/                 # Redux store
│   │   │   └── api/                   # Backend API calls
│   ├── Makefile                       # OpenHands build/test commands
│   ├── pyproject.toml                 # Python dependencies
│   └── execute_ml_routing_research.py # Example research script
├── start_openhands_research.sh        # Main startup script (project root)
├── requirements.txt                   # Top-level Python dependencies
├── README.md                          # Research session notes
├── .env*                              # Environment configuration files
└── workspace/                         # Generated workspace directories
```

## Development Commands

### Initial Setup

**Important**: The project root is `/Users/wuy/Desktop/code/UAgent`, while build commands are in `OpenHands/` subdirectory.

```bash
# Navigate to project root
cd /Users/wuy/Desktop/code/UAgent

# Install dependencies via OpenHands Makefile (requires Python 3.12, Node.js 22+, Poetry 1.8+, Docker)
cd OpenHands && make build && cd ..

# Alternative: Setup with specific Poetry groups
cd OpenHands && POETRY_GROUP=dev,test,runtime make install-python-dependencies && cd ..

# Setup configuration file (sets LLM API key, model, workspace)
cd OpenHands && make setup-config && cd ..

# Note: Python virtual environment is at /Users/wuy/Desktop/code/UAgent/.venv
# Always activate it before running Python commands:
source .venv/bin/activate
```

### Running the Application

```bash
# RECOMMENDED: Start backend with research extension in tmux
# CRITICAL: Must activate virtual environment BEFORE starting!
cd /Users/wuy/Desktop/code/UAgent  # Parent directory
tmux new-session -d -s uagent-backend 'bash -c "source .venv/bin/activate && ./start_openhands_research.sh"'
# Access the tmux session: tmux attach -t uagent-backend
# Backend will be available at: http://localhost:2999

# Check if backend is running:
curl http://localhost:2999/api/options/health

# Alternative: Run full stack (backend + frontend)
make run

# Run backend only (uvicorn on port 3000) - NOT recommended for research mode
make start-backend

# Run frontend only (Vite on port 3001)
make start-frontend

# Run in Docker
make docker-run

# Run in development container
make docker-dev

# Stop backend tmux session
tmux kill-session -t uagent-backend
```

### Testing

```bash
# Backend tests (from OpenHands directory)
cd OpenHands
poetry run pytest tests/unit/              # Unit tests
poetry run pytest tests/integration/        # Integration tests
poetry run pytest tests/e2e/                # End-to-end tests
poetry run pytest -k "test_name"           # Run specific test
cd ..

# Frontend tests (from OpenHands directory)
cd OpenHands/frontend
npm run test                               # Run all tests
npm run test:coverage                      # With coverage
npm run test:e2e                           # Playwright E2E tests
cd ../..

# Research extension tests (from OpenHands directory)
cd OpenHands/extensions/uagent_research
pytest tests/unit/
pytest tests/integration/
python tests/test_tools_integration.py     # Integration test
cd ../../..
```

### Linting and Formatting

```bash
# All linters (from OpenHands directory)
cd OpenHands && make lint && cd ..

# Backend only (ruff, mypy, pre-commit)
cd OpenHands && make lint-backend && cd ..

# Frontend only (eslint, prettier, type-check)
cd OpenHands && make lint-frontend && cd ..

# Fix frontend issues
cd OpenHands/frontend && npm run lint:fix && cd ../..
```

### Building

```bash
# Build frontend production bundle (from OpenHands directory)
cd OpenHands && make build-frontend && cd ..

# Full project build (from OpenHands directory)
cd OpenHands && make build && cd ..
```

## Research Mode Execution

### Auto-Trigger Research Mode

Research mode automatically activates when you send a complex task. The system uses an LLM-based classifier (`TaskClassifier`) to detect:
- Research & investigation tasks (comparing, benchmarking, evaluating)
- Complex multi-stage work (3+ distinct phases)
- Experimental systems (ML models, data collection, experiments)
- Source code modification (database/kernel-level changes)
- Systematic comparisons (multiple baselines, threshold methods)

**Configuration** (in `.env`):
```bash
ENABLE_AUTO_RESEARCH_TRIGGER=true
RESEARCH_CONFIDENCE_THRESHOLD=0.5  # 0.0-1.0, lower = more sensitive
```

**Important**: The TaskClassifier uses OpenHands' LLM configuration (from `.env`):
- `LLM_MODEL`: e.g., `openai/qwen3-coder-plus`
- `LLM_API_KEY`: Your API key (DashScope, OpenAI, etc.)
- `LLM_BASE_URL`: API endpoint URL

If LLM classification fails, it falls back to heuristic keyword matching.

### Manual Research Mode

You can also start research mode programmatically:

```python
# Example: Execute research task
from extensions.uagent_research.middleware.research_middleware import ResearchMiddleware

middleware = ResearchMiddleware()
experiment_id = await middleware.start_research(
    goal="Research and implement ML-based query routing",
    session_id="unique_session_id",
    research_type='code',  # 'web', 'code', or 'hybrid'
    config={
        'max_iterations': 50,
        'max_parallel': 3,
        'workspace_dir': '/path/to/workspace',
    }
)
```

**Research task examples** (see `execute_ml_routing_research.py`):
- Complex multi-step implementations requiring exploration
- Literature review and synthesis
- GitHub repository analysis and comparison
- Hypothesis generation and validation

## Critical Development Notes

### Python Environment

- **Python version**: 3.12 (3.13 not supported)
- **Package manager**: Poetry 1.8+
- **Virtual environment**: Managed by Poetry (`poetry env use python3.12`)

### Frontend Environment

- **Node.js**: 22.x or later required
- **Package manager**: npm 10.5.0
- **Framework**: Remix SPA Mode (React + Vite + React Router)
- **State**: Redux + TanStack Query
- **Testing**: Vitest + React Testing Library + MSW (Mock Service Worker)

### Event System

The EventStream is central to all operations:
- **Actions**: Commands to execute (edit file, run command, etc.)
- **Observations**: Results from actions (file contents, command output)
- **Event flow**: Agent → EventStream → Runtime → EventStream → AgentController
- **Research extension**: EventBus coalesces events and publishes to WebSocket

### Runtime and Sandbox

- **Default runtime**: Docker-based sandboxing
- **Runtime location**: `openhands/runtime/`
- **Sandbox config**: Configurable via environment variables
- **Security**: Isolated execution environment for code actions

### Configuration

- **Config file**: `config.toml` (create with `make setup-config`)
- **Environment proxy**: Set `HTTP_PROXY` and `HTTPS_PROXY` if needed (see research scripts)
- **Workspace**: Default `./workspace`, configurable in config.toml

## Common Workflows

### Adding a New Agent

1. Create agent class in `openhands/agenthub/`
2. Extend base `Agent` class
3. Implement `step()` method (prompt generation and response parsing)
4. Register in `openhands/agenthub/__init__.py`

### Adding Research Adapter

1. Create adapter in `extensions/uagent_research/adapters/`
2. Extend `AgentAdapter` base class
3. Implement `run()` and `get_skill_types()` methods
4. Register in adapter_registry

### Modifying Frontend Components

1. Components by domain in `frontend/src/components/features/`
2. Use `renderWithProviders()` for testing Redux-connected components
3. Mock API calls with MSW in `frontend/src/mocks/`
4. Update types in `frontend/src/types/`

### Working with EventStream

```python
# Subscribe to events
event_stream.subscribe(EventStreamSubscriber.RUNTIME, callback)

# Publish action
event_stream.add_event(RunCommandAction(command="ls"), source="agent")

# Add event listener
def on_event(event: Event):
    if isinstance(event, Observation):
        # Process observation
        pass

event_stream.add_event_listener(on_event)
```

## Environment Variables

### Backend
- `BACKEND_HOST`: Backend host (default: 127.0.0.1)
- `BACKEND_PORT`: Backend port (default: 3000)
- `DEFAULT_MODEL`: LLM model (default: gpt-4o)
- `LLM_API_KEY`: API key for LLM provider
- `WORKSPACE_BASE`: Workspace directory

### Frontend
- `VITE_BACKEND_HOST`: Backend host:port (default: 127.0.0.1:3000)
- `VITE_FRONTEND_PORT`: Frontend port (default: 3001)
- `VITE_MOCK_API`: Enable MSW mocking (default: false)
- `VITE_USE_TLS`: Use HTTPS/WSS (default: false)

## Troubleshooting

### Poetry issues
```bash
# Reset Poetry environment
poetry env remove python3.12
poetry env use python3.12
poetry install
```

### Playwright not installed (macOS with Manjaro detection issue)
```bash
poetry run playwright install chromium
```

### Frontend type errors
```bash
cd frontend
npm run typecheck
# Fix errors, then run
npm run lint:fix
```

### Docker sandbox issues
```bash
# Check Docker is running
docker ps

# Rebuild Docker images
make docker-dev
```

### Research mode not starting

**Symptom**: Research Tree tab shows "Idle" and "Disconnected", no experiment ID

**Cause**: TaskClassifier failed to trigger research mode (usually due to missing LLM API key)

**Solution**:
1. Check backend logs for "LLM classification failed" errors:
   ```bash
   tmux attach -t uagent-backend
   # Look for: "TaskClassifier initialized with..."
   ```

2. Verify LLM configuration in `.env`:
   ```bash
   # In /Users/wuy/Desktop/code/UAgent/.env
   LLM_API_KEY=${DASHSCOPE_API_KEY}  # Must be set
   LLM_BASE_URL=${DASHSCOPE_BASE_URL}
   LLM_MODEL=openai/qwen3-coder-plus
   ```

3. Ensure environment variables are exported:
   ```bash
   # In parent directory or in start_openhands_research.sh
   export DASHSCOPE_API_KEY="your-key-here"
   export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
   ```

4. Restart backend:
   ```bash
   tmux kill-session -t uagent-backend
   cd /Users/wuy/Desktop/code/UAgent
   tmux new-session -d -s uagent-backend 'bash start_openhands_research.sh'
   ```

**WebSocket Connection Issues**:
- The Research WebSocket connects to: `ws://localhost:2999/api/research/ws/experiment/{experiment_id}`
- If research mode doesn't trigger, no experiment ID exists, so WebSocket fails with 404
- Frontend falls back to polling if WebSocket fails
- Check "Research WS" logs in browser console for connection status

**Fallback**: If LLM classification consistently fails, the system uses heuristic keyword matching

## UAgent Functionality Testing Procedure

### Standard Testing Protocol

When asked to test UAgent-OpenHands functionality, follow this exact procedure:

1. **Environment Setup**:
   ```bash
   cd /Users/wuy/Desktop/code/UAgent
   source .venv/bin/activate
   ```

2. **Start Backend in tmux**:
   ```bash
   tmux new-session -d -s uagent-backend 'source .venv/bin/activate && bash start_openhands_research.sh'
   # Access the session: tmux attach -t uagent-backend
   ```

3. **Use Playwright MCP to navigate to localhost:2999**

4. **Start New Conversation and Send Research Message**:
   Send this exact message:
   ```
   research goal: modify postgres and pg_duckdb source code （ to download source code you can utilize the proxy on port localhost:7890, do not use the system-wide postgresql）, first extract pre-opt features from postgres kernel and log to files, then collect dual-execution data (pre-optimization query features that can be found in kernel structures and execution times on dual engine) and train a machine learning model to predict whether postgres engine or duckdb engine executes a query fast and embed the machine learning model into database source code (using the language of the database for example c language) to online route each query to the faster engine, and execute end-to-end experiments to test the ml-based system's performance. A baseline method called threshold-based method should also be implemented, which routes query based on threshold, for example threshold can be 10000 or 50000 or any other value, if postgres estimates the cost of a query is above threshold, then send to duckdb, otherwise send to postgres, and compare the postgres-only, duckdb-only, different threshold-based methods and lightgbm-based method. please record every successful  necessary commands in README.md so that later people can reproduce your results. also record your python packages dependencies in requirements.txt.
   ```

5. **Monitor and Debug**:
   - Monitor the parallel research progress in the Research Tree tab
   - Debug any issues that appear during execution
   - Check backend logs: `tmux attach -t uagent-backend`
   - Verify WebSocket connection status in browser console

### Expected Behavior

- Research mode should auto-trigger due to the complex, multi-stage nature of the task
- TaskClassifier should detect ML model training, database kernel modification, and experimental comparison
- Research Tree should show active experiment ID and connected WebSocket status
- Multiple research adapters should be utilized for different aspects of the task

### Troubleshooting During Testing

If research mode doesn't trigger:
- Check LLM API key configuration in `.env`
- Verify `ENABLE_AUTO_RESEARCH_TRIGGER=true` in environment
- Monitor tmux session for TaskClassifier errors
- Ensure proxy settings are properly configured for source code downloads

## Key Files

### Project Root Files
- `start_openhands_research.sh` - Main startup script with environment setup
- `requirements.txt` - Top-level Python dependencies
- `README.md` - Research session notes
- `.env*` - Environment configuration files (e.g., `.env.qwen3-coder-plus`, `.env.dashscope_kimi`)
- `WARP.md` - This file (development guide)

### OpenHands Core Files
- `OpenHands/openhands/core/agent.py` - Base Agent class
- `OpenHands/openhands/controller/agent_controller.py` - Main orchestration loop
- `OpenHands/openhands/server/listen.py` - FastAPI server entry point (actual: `openhands/server/__main__.py`)
- `OpenHands/openhands/server/app.py` - FastAPI app initialization
- `OpenHands/extensions/uagent_research/orchestrator/tree_orchestrator.py` - Research tree search
- `OpenHands/extensions/uagent_research/middleware/research_middleware.py` - Research mode entry
- `OpenHands/frontend/src/routes/_oh.app/route.tsx` - Main app route
- `OpenHands/pyproject.toml` - Python dependencies
- `OpenHands/frontend/package.json` - Frontend dependencies
- `OpenHands/Makefile` - Build, test, and lint commands
- `OpenHands/execute_ml_routing_research.py` - Example research execution script
