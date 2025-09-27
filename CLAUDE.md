# UNIVERSAL AGENT (UAGENT) - KEY DEVELOPMENT INSTRUCTIONS

## Recent Updates (2025-09-26)

### OpenHands V3 Integration (Default Enabled)
- **OpenHands V3 headless mode is now ENABLED by default** for scientific research experiments
- Uses deterministic subprocess execution with artifact-based communication
- To disable and use legacy V2 mode: `export UAGENT_OPENHANDS_V3=0`
- See `OPENHANDS_V3_MIGRATION.md` for full migration details

# UNIVERSAL AGENT (UAGENT) - KEY DEVELOPMENT INSTRUCTIONS

## Core Principles

1. **Test-Driven Development**: After implementing any new function, write comprehensive tests in `test/*test.py` before proceeding
2. **Specification-First**: Use github/spec_kit repository patterns to write detailed specifications before coding
3. **OpenHands Foundation**: Copy and integrate OpenHands CLI source code as the execution engine
4. **Progressive Implementation**: Start simple, test thoroughly, then add complexity
5. **DO NOT MOCK OR SIMULATE**: Always use real LLM integration (DashScope Qwen), real APIs, and real services - no mocking or simulation in production code

## Smart Routing System

The system must intelligently route user requests to appropriate research engines:
- **Deep Research**: Web/academic comprehensive research (ChatGPT-style)
- **Scientific Research**: Experimental research with hypothesis testing (AI Scientist/AgentLab-style) - **MOST COMPLEX**
- **Code Research**: Repository analysis and code understanding (RepoMaster-style)

**Critical**: Scientific research is the most complex engine and orchestrates the other two engines.

## Testing Requirements

### Required Test Structure
```
test/
├── unit/
│   ├── test_smart_router.py       # Router classification accuracy
│   ├── test_deep_research.py      # Web search and synthesis
│   ├── test_code_research.py      # Repository analysis
│   ├── test_scientific_research.py # Experimental workflows
│   ├── test_openhands_client.py   # OpenHands integration
│   ├── test_websocket_manager.py  # WebSocket connection management
│   ├── test_streaming_llm_client.py # LLM interaction streaming
│   └── test_progress_tracker.py   # Research progress event tracking
├── integration/
│   ├── test_multi_engine.py       # Cross-engine communication
│   ├── test_workflows.py          # End-to-end research workflows
│   ├── test_api_endpoints.py      # API functionality
│   ├── test_websocket_streaming.py # Real-time WebSocket communication
│   └── test_completion_events.py  # Research completion event flow
└── e2e/
    ├── test_deep_research_flow.py
    ├── test_code_research_flow.py
    ├── test_scientific_research_flow.py
    └── test_realtime_visualization.py # End-to-end streaming and visualization
```

### Test Requirements Before Each Commit
1. **Unit Tests**: >90% coverage for new functions
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete user workflows
4. **Performance Tests**: Verify response times and resource usage

## Specification Requirements

Before implementing any major component, create specifications using spec_kit patterns:

### Required Specifications
```
specs/
├── smart_router_spec.md          # Router behavior and classification rules
├── research_engines_spec.md      # Engine capabilities and interfaces
├── openhands_integration_spec.md # Integration patterns and workflows
├── api_endpoints_spec.md         # REST API specifications
└── frontend_components_spec.md   # UI component specifications
```

### Spec Format (following spec_kit)
```markdown
# Component Specification

## Overview
Brief description and purpose

## Requirements
- Functional requirements
- Non-functional requirements
- Performance requirements

## Interface
- Input parameters
- Output format
- Error conditions

## Behavior
- Normal operation flow
- Edge cases
- Error handling

## Testing
- Test scenarios
- Success criteria
- Performance benchmarks
```

## Implementation Order

1. **Phase 1**: OpenHands integration + Smart router + Basic tests
2. **Phase 2**: Deep research engine + Code research engine + Integration tests
3. **Phase 3**: Scientific research engine foundation + Multi-engine coordination
4. **Phase 4**: Scientific research complexity (iteration, feedback loops)
5. **Phase 5**: Frontend + End-to-end tests + Production deployment

## Code Quality Standards

- **Type hints**: All Python functions must have type annotations
- **Documentation**: Docstrings for all classes and public methods
- **Error handling**: Comprehensive exception handling with logging
- **Async/await**: Use async patterns throughout for performance
- **Configuration**: External configuration files, no hardcoded values
- **Real Integration Only**: Use DashScope Qwen for all LLM operations, no mock clients in production

## OpenHands Integration Requirements

1. **Copy source code**: Copy relevant OpenHands CLI modules to our codebase
2. **Workspace isolation**: Each research task gets isolated workspace
3. **Session management**: Persistent sessions with state restoration
4. **Stream output**: Real-time output streaming to frontend
5. **Error recovery**: Robust error handling and retry mechanisms

## Real-Time Research Process Visualization Requirements

### Core Visualization Features ✅ **IMPLEMENTED**
1. **Live Progress Streaming**: Real-time display of research process execution
2. **Multi-Engine Status**: Visual indicators for all active research engines
3. **Interactive Process Tree**: Live updates to research tree with expanding nodes
4. **Research Journal**: Real-time research log with timestamps and engine sources
5. **Code Execution Monitor**: Live streaming of OpenHands code execution output
6. **Research Metrics Dashboard**: Live performance and progress metrics
7. **LLM Interaction Streaming**: Real-time display of LLM conversations during research

### Frontend Real-Time Components ✅ **IMPLEMENTED**
1. **ResearchProgressStream**: WebSocket-based live progress updates
2. **EngineStatusIndicators**: Real-time status for Deep/Code/Scientific engines
3. **LiveResearchJournal**: Streaming research log with filtering and search
4. **CodeExecutionTerminal**: Live OpenHands execution output with syntax highlighting
5. **InteractiveResearchTree**: Real-time tree updates with progress indicators
6. **MetricsDashboard**: Live charts for execution time, success rates, token usage
7. **LLMConversationLogs**: Real-time LLM interaction display with streaming tokens

### Backend Streaming Requirements ✅ **IMPLEMENTED**
1. **WebSocket Integration**: Real-time communication for all research engines
2. **Progress Event System**: Structured events for all research phases
3. **Multi-Engine Coordination Events**: Cross-engine communication visibility
4. **OpenHands Output Streaming**: Live code execution and debugging output
5. **Research State Synchronization**: Consistent state across all components
6. **Research Completion Events**: Proper broadcasting of completion status with 100% progress
7. **LLM Streaming Integration**: Automatic broadcasting of LLM interactions via StreamingLLMClient

### Recent Implementation Fixes (2025-09-19)
#### Issue 1: Tree Nodes Stuck at "Running" Status ✅ **FIXED**
- **Root Cause**: Research completion events were never being broadcast
- **Solution**: Added `log_research_completed` method to progress tracker
- **Files Modified**:
  - `backend/app/core/websocket_manager.py` (lines 315-345)
  - `backend/app/routers/smart_router.py` (completion logging for all engines)

#### Issue 2: LLM Chat Not Showing Interactions ✅ **FIXED**
- **Root Cause**: Research engines used LLM but interactions weren't broadcast to WebSocket
- **Solution**: Created `StreamingLLMClient` wrapper for automatic LLM interaction broadcasting
- **Files Added**: `backend/app/core/streaming_llm_client.py`
- **Files Modified**: `backend/app/routers/smart_router.py` (temporary LLM client replacement during research)

## Critical Success Metrics

- **Router Accuracy**: >95% correct classification of user requests
- **Test Coverage**: >90% code coverage with meaningful tests
- **Response Time**: <2s for API responses, <30s for simple research tasks
- **Reliability**: >99% uptime, graceful failure handling
- **User Experience**: Clear progress indicators, helpful error messages

## Development Workflow

1. **Write Specification**: Create detailed spec using spec_kit patterns
2. **Write Tests**: Create comprehensive test suite for the component
3. **Implement**: Write the actual implementation
4. **Test**: Run all tests and ensure they pass
5. **Integration**: Test with other components
6. **Review**: Code review for quality and performance
7. **Deploy**: Deploy to staging/production environments

## Key Commands

- **Run Tests**: `python -m pytest test/ -v --cov=app`
- **Type Check**: `mypy app/`
- **Lint Code**: `ruff check app/`
- **Format Code**: `black app/`
- **Start Backend**: `uvicorn app.main:app --reload --port 8000`
- **Start Frontend**: `cd frontend && npm run dev`
- **Headless Research Monitor**: `python -m backend.scripts.cli_research --query "<prompt>"`
- **Attach to Existing Session**: `python -m backend.scripts.cli_research --session <session_id>`
- **Install Playwright Chromium (RepoMaster search)**: `playwright install --with-deps chromium`

## Emergency Protocols

- **Test Failures**: Do not proceed with implementation if tests fail
- **Performance Issues**: Profile and optimize before adding new features
- **Security Concerns**: Audit all code execution and input validation
- **Data Loss**: Implement backup and recovery for all research data

For detailed implementation plan, see PLAN.md
