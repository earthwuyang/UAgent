# UAgent Project

## Project Overview

UAgent is a research-enhanced fork of OpenHands, an AI-powered software development platform. This fork adds autonomous parallel research capabilities through the UAgent Research Extension, enabling the system to automatically trigger tree-based research processes for complex queries. The core of the research extension is a Tree Search Orchestrator that uses the PUCT (Predictor + Upper Confidence bounds applied to Trees) algorithm, the same algorithm used by AlphaZero.

The project is built with Python and uses FastAPI for the backend server. The frontend is a React application.

## Building and Running

### Local Development

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -e OpenHands/extensions/uagent_research
    ```

2.  **Start the server:**
    ```bash
    ./start_openhands_research.sh
    ```

    Alternatively, to run the backend in a tmux session:
    ```bash
    tmux send-keys -t uagent-backend 'source /Users/wuy/Desktop/code/UAgent/.venv/bin/activate && cd /Users/wuy/Desktop/code/UAgent && ./start_openhands_research.sh' C-m
    ```

### Production

```bash
# Set environment variables
export ENABLE_AUTO_RESEARCH_TRIGGER=true
export RESEARCH_CONFIDENCE_THRESHOLD=0.7

# Start with Docker
docker-compose up -d
```

### Testing

```bash
cd /Users/wuy/Desktop/code/UAgent/OpenHands/extensions/uagent_research
pytest -v
```

## Development Conventions

The project follows a modular architecture with a clear separation of concerns. The core OpenHands platform provides the foundation, and the UAgent Research Extension is a self-contained module.

### Code Style

The codebase is primarily Python. While no specific linter is mentioned, the code appears to follow standard Python conventions.

### Testing

The research extension has its own test suite, which can be run with pytest. The tests cover extension loading, integration workflows, data models, server integration, and WebSockets.

### Contribution Guidelines

To add a new adapter to the research extension:

1.  Create an adapter directory in `OpenHands/extensions/uagent_research/adapters/`.
2.  Implement the `BaseAdapter` interface.
3.  Register the adapter in `OpenHands/extensions/uagent_research/adapters/__init__.py`.
4.  Update the routing logic in `OpenHands/extensions/uagent_research/router/skill_router.py`.
5.  Add tests for the new adapter.
