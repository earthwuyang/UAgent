# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**OpenHands-UAgent** is an AI software engineer system that extends OpenHands with advanced research capabilities. It combines autonomous software engineering with intelligent research through a PUCT-based tree search system.

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
openhands/          # Core OpenHands system
├── agenthub/       # Agent implementations
├── controller/     # AgentController orchestration
├── core/           # State, LLM, and event primitives
├── events/         # Action and Observation types
├── runtime/        # Docker sandbox and action execution
├── server/         # FastAPI server, WebSocket, session management
├── llm/            # LiteLLM integration
├── memory/         # Long-term memory systems
└── integrations/   # External service integrations

extensions/uagent_research/  # Research extension
├── orchestrator/            # TreeSearchOrchestrator
├── adapters/                # DeepResearchAdapter, RepoMasterAdapter, CodeActAdapter
├── middleware/              # ResearchMiddleware entry point
├── event_bus/               # Event coalescing and WebSocket publishing
└── tests/                   # Unit and integration tests

frontend/           # React/TypeScript frontend
├── src/
│   ├── components/features/  # Domain components (browser, chat, code editor, terminal)
│   ├── routes/              # File-based routing (Remix SPA)
│   ├── state/               # Redux store
│   └── api/                 # Backend API calls
```

## Development Commands

### Initial Setup

```bash
# Install dependencies (requires Python 3.12, Node.js 22+, Poetry 1.8+, Docker)
make build

# Alternative: Setup with specific Poetry groups
POETRY_GROUP=dev,test,runtime make install-python-dependencies

# Setup configuration file (sets LLM API key, model, workspace)
make setup-config
```

### Running the Application

```bash
# Run full stack (backend + frontend)
make run

# Run backend only (uvicorn on port 3000)
make start-backend

# Run frontend only (Vite on port 3001)
make start-frontend

# Run in Docker
make docker-run

# Run in development container
make docker-dev
```

### Testing

```bash
# Backend tests
poetry run pytest tests/unit/              # Unit tests
poetry run pytest tests/integration/        # Integration tests
poetry run pytest tests/e2e/                # End-to-end tests
poetry run pytest -k "test_name"           # Run specific test

# Frontend tests
cd frontend
npm run test                               # Run all tests
npm run test:coverage                      # With coverage
npm run test:e2e                           # Playwright E2E tests

# Research extension tests
cd extensions/uagent_research
pytest tests/unit/
pytest tests/integration/
python tests/test_tools_integration.py     # Integration test
```

### Linting and Formatting

```bash
# All linters
make lint

# Backend only (ruff, mypy, pre-commit)
make lint-backend

# Frontend only (eslint, prettier, type-check)
make lint-frontend

# Fix frontend issues
cd frontend && npm run lint:fix
```

### Building

```bash
# Build frontend production bundle
make build-frontend

# Full project build
make build
```

## Research Mode Execution

The UAgent extension provides advanced research capabilities:

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
- Verify workspace directory exists and is writable
- Check proxy settings if behind firewall
- Ensure all adapters are registered
- Review logs in event bus output

## Key Files

- `openhands/core/agent.py` - Base Agent class
- `openhands/controller/agent_controller.py` - Main orchestration loop
- `openhands/server/listen.py` - FastAPI server entry point
- `extensions/uagent_research/orchestrator/tree_orchestrator.py` - Research tree search
- `extensions/uagent_research/middleware/research_middleware.py` - Research mode entry
- `frontend/src/routes/_oh.app/route.tsx` - Main app route
- `pyproject.toml` - Python dependencies
- `frontend/package.json` - Frontend dependencies
