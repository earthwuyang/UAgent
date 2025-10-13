# UAgent Project - Comprehensive Documentation

## Project Overview

UAgent is a research-enhanced fork of OpenHands (formerly OpenDevin), an AI-powered software development platform. This project adds autonomous parallel research capabilities through the UAgent Research Extension, enabling the system to automatically trigger tree-based research processes for complex queries using PUCT (Predictor + Upper Confidence bounds applied to Trees) algorithm.

### Key Innovation
- Automatic tree-based research using PUCT algorithm (same algorithm as AlphaZero)
- Parallel exploration of multiple research paths
- Budget-controlled execution with cost monitoring
- Real-time progress tracking

## Architecture

### High-Level Structure
```
UAgent/
├── OpenHands/                          # Modified OpenHands fork (main codebase)
│   ├── openhands/                      # Core OpenHands platform
│   │   ├── server/                     # FastAPI backend server
│   │   ├── agenthub/                   # Agent implementations
│   │   ├── controller/                 # Agent control logic
│   │   ├── runtime/                    # Execution environments (Docker, etc.)
│   │   ├── llm/                        # LLM integration layer
│   │   ├── events/                     # Event system
│   │   └── storage/                    # Data persistence
│   │
│   ├── extensions/                     # Extension system
│   │   └── uagent_research/            # UAgent Research Extension
│   │       ├── classifier/             # Task classification (triggers research)
│   │       ├── middleware/             # Request interception
│   │       ├── orchestrator/           # Tree search orchestrator (PUCT algorithm)
│   │       ├── adapters/               # Execution adapters (DeepResearch, RepoMaster, CodeAct)
│   │       ├── router/                 # Skill routing
│   │       ├── tools/                  # Research tools (search, browse, code analysis)
│   │       ├── api/                    # Research API endpoints
│   │       └── models/                 # Data models
│   │
│   ├── frontend/                       # React frontend
│   └── workspace/                      # User workspaces
├── original_openhands/                 # Original OpenHands (reference)
└── requirements.txt                    # Python dependencies
```

### Core Components

1. **Task Classifier**: Determines if a user query should trigger research mode based on keywords and complexity indicators
2. **Research Middleware**: Intercepts user messages and triggers research when appropriate
3. **Tree Orchestrator**: Core research engine using PUCT algorithm for systematic exploration
4. **Adapters**: Execute different types of research tasks (DeepResearch, RepoMaster, CodeAct)
5. **Research API**: Exposes research tree state and control endpoints

## Building and Running

### Prerequisites
- Python 3.10+
- Docker and Docker Compose
- API keys for LLM providers (Anthropic, OpenAI, or DashScope)

### Setup
1. Clone the repository and navigate to the directory
2. Set up Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Set your LLM API keys and other configuration options

### Running the Application
```bash
./start_openhands_research.sh
```

The server will start and be accessible at:
- Main UI: `http://localhost:3000`
- Research API: `http://localhost:3000/api/research`

### Configuration
Key environment variables:
- `ENABLE_AUTO_RESEARCH_TRIGGER`: Enable/disable research auto-trigger
- `RESEARCH_CONFIDENCE_THRESHOLD`: Trigger threshold (0.0 to 1.0)
- `RESEARCH_MAX_ITERATIONS`: Max research iterations
- `RESEARCH_MAX_COST`: Max cost in dollars
- `RESEARCH_MAX_PARALLEL`: Max concurrent branches

## Development Conventions

### Code Structure
- Follow existing code style and patterns found in the OpenHands base
- Add comprehensive docstrings for new functions and classes
- Include type hints where possible
- Write tests for new functionality

### Adding New Adapters
1. Create adapter directory in `extensions/uagent_research/adapters/my_adapter/`
2. Implement `BaseAdapter` interface
3. Register in `adapters/__init__.py`
4. Update routing logic in `router/skill_router.py`
5. Add tests

### Testing
Run tests using:
```bash
cd OpenHands/extensions/uagent_research
pytest -v
```

## Key Files Reference

### Core OpenHands
- `openhands/server/app.py` - Main FastAPI application
- `openhands/server/session/session.py` - Agent session management
- `openhands/agenthub/codeact_agent/codeact_agent.py` - Primary coding agent

### UAgent Research Extension
- `extensions/uagent_research/classifier/task_classifier.py` - Task classification
- `extensions/uagent_research/middleware/research_middleware.py` - Request interception
- `extensions/uagent_research/orchestrator/tree_orchestrator.py` - PUCT tree search
- `extensions/uagent_research/router/skill_router.py` - Task routing
- `extensions/uagent_research/api/research_routes.py` - Research API

## Research Workflow Example

For a query like "Modify postgres and pg_duckdb to support vector search":

1. **Classification**: Task classifier identifies this as a complex research task with high confidence
2. **Middleware Intercepts**: Triggers research mode, creates experiment ID
3. **Tree Initialization**: Creates ROOT node with user query
4. **Expansion**: Generates multiple ideas (web research, code research, etc.)
5. **Execution**: Runs ideas in parallel using appropriate adapters
6. **PUCT Algorithm**: Selects most promising paths based on success metrics
7. **Iteration**: Continues until budget is exhausted or goal is achieved

The system maintains a tree structure with nodes of different types (ROOT, IDEA, HYPOTHESIS, EXPERIMENT) and tracks metrics like visits, success rate, and costs.

## Troubleshooting

### Research Not Triggering
- Check `ENABLE_AUTO_RESEARCH_TRIGGER=true`
- Verify query confidence >= threshold
- Ensure server is restarted after config changes

### High Costs
- Lower `RESEARCH_MAX_COST`
- Reduce `RESEARCH_MAX_ITERATIONS`
- Increase `RESEARCH_CONFIDENCE_THRESHOLD`
- Use cheaper LLM models

### Frontend Issues
- Check server logs for errors
- Verify research API is responding
- Clear browser cache if needed