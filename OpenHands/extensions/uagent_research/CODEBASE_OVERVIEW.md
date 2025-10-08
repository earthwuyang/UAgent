# UAgent Research Codebase Overview

This document provides a comprehensive overview of the UAgent Research extension codebase, focusing on the intelligent node expansion feature.

## Directory Structure

```
extensions/uagent_research/
├── services/
│   ├── __init__.py
│   ├── idea_generation_service.py      # NEW: LLM-based node generation
│   └── research_session_manager.py
├── orchestrator/
│   ├── __init__.py
│   ├── tree_orchestrator.py            # MODIFIED: Intelligent expansion
│   └── event_bus.py
├── middleware/
│   ├── __init__.py
│   └── research_middleware.py          # MODIFIED: LLM integration
├── uagent_research/
│   ├── engines/
│   │   ├── scientific_research_original.py
│   │   ├── scientific_research.py
│   │   └── code_research.py
│   ├── models/
│   │   ├── research_tree.py
│   │   ├── experiment.py
│   │   └── ...
│   └── ...
├── tests/
│   ├── test_idea_generation_service.py  # NEW: Service tests
│   ├── test_orchestrator_intelligent_expansion.py  # NEW: Integration tests
│   └── ...
├── config.py                            # MODIFIED: New configuration
├── README.md                            # MODIFIED: Documentation
├── IMPLEMENTATION_STATUS.md             # MODIFIED: Phase 4 added
├── ARCHITECTURE_DIAGRAM.md              # NEW: Architecture docs
└── CODEBASE_OVERVIEW.md                 # NEW: This file
```

## Core Components

### 1. IdeaGenerationService

**Location**: `services/idea_generation_service.py`

**Purpose**: Bridges the tree orchestrator and research engines to provide LLM-based node generation.

**Key Methods**:
- `generate_ideas(goal, context, max_ideas)` → List[ResearchNode]
  - Generates research ideas from a goal
  - Calls ScientificResearchEngine
  - Transforms engine output to ResearchNode objects

- `generate_hypotheses(idea_content, parent_node, max_hypotheses)` → List[ResearchNode]
  - Generates testable hypotheses for an idea
  - Uses LLM to create specific, measurable hypotheses
  - Returns nodes with parent_id set

- `generate_experiments(hypothesis_content, parent_node)` → List[ResearchNode]
  - Generates concrete experiments for a hypothesis
  - Includes methodology and expected outcomes
  - Returns executable experiment nodes

**Error Handling**:
- Retry logic for transient LLM failures (default: 2 retries)
- Returns empty list on failure (graceful degradation)
- Logs all errors with full stack traces

**Configuration**:
```python
config = {
    'max_ideas': 3,
    'max_hypotheses': 2,
    'max_experiments': 1,
    'retry_count': 2,
}
service = IdeaGenerationService(llm=llm, config=config)
```

### 2. TreeSearchOrchestrator (Modified)

**Location**: `orchestrator/tree_orchestrator.py`

**Purpose**: Manages research tree exploration using PUCT algorithm. Now supports intelligent node expansion.

**Key Modifications**:

1. **Constructor** (lines ~56-100):
   ```python
   def __init__(
       self,
       max_parallel: int = 3,
       budget: Optional[Budget] = None,
       router: Optional[SkillRouter] = None,
       event_bus: Optional[EventBus] = None,
       control_bus: Optional[ControlBus] = None,
       llm = None,  # NEW
       idea_service: Optional[IdeaGenerationService] = None,  # NEW
   )
   ```
   - Accepts LLM instance
   - Accepts optional IdeaGenerationService
   - Creates service automatically if LLM provided
   - Sets `use_intelligent_expansion` flag

2. **_expand_node() Method** (lines ~320-450):
   - **Before**: Hardcoded placeholder node generation
   - **After**: Intelligent LLM-based generation with fallback
   
   Flow:
   ```python
   if self.use_intelligent_expansion:
       try:
           if node.type == NodeType.ROOT:
               children = await self.idea_service.generate_ideas(goal, context)
           elif node.type == NodeType.IDEA:
               children = await self.idea_service.generate_hypotheses(...)
           elif node.type == NodeType.HYPOTHESIS:
               children = await self.idea_service.generate_experiments(...)
       except Exception as e:
           logger.error(...)
           children = []  # Will fall through to fallback
   
   if not children:
       # Fallback to hardcoded nodes
       children = [...]
   ```

3. **Backward Compatibility**:
   - Existing tests pass without modification
   - No LLM → Falls back to placeholders
   - No breaking changes to API

### 3. ResearchMiddleware (Modified)

**Location**: `middleware/research_middleware.py`

**Purpose**: Intercepts user messages and triggers research mode. Now passes LLM to orchestrator.

**Key Modifications** (lines ~650-680):

```python
# Try to get LLM instance for intelligent node expansion
llm = None
try:
    # Try to get LLM from session manager
    if session_mgr and hasattr(session_mgr, 'llm'):
        llm = session_mgr.llm
        logger.info("[RESEARCH_MIDDLEWARE] Acquired LLM from session manager")
    else:
        logger.warning("[RESEARCH_MIDDLEWARE] LLM not available from session manager")
except Exception as e:
    logger.warning(f"[RESEARCH_MIDDLEWARE] Failed to acquire LLM: {e}")

logger.info(
    f"[RESEARCH_MIDDLEWARE] Creating TreeSearchOrchestrator with llm={'available' if llm else 'unavailable'}"
)
orchestrator = TreeSearchOrchestrator(
    max_parallel=max_parallel,
    budget=budget,
    event_bus=event_bus,
    control_bus=control_bus,
    llm=llm,  # NEW: Pass LLM
)
```

**Best-Effort Approach**:
- Attempts to get LLM from session manager
- Logs availability status
- Never blocks research startup if LLM unavailable
- Orchestrator handles missing LLM gracefully

### 4. Configuration (Modified)

**Location**: `config.py`

**New Configuration Options**:
```python
# Enable LLM-based node generation (vs hardcoded placeholders)
ENABLE_INTELLIGENT_EXPANSION = os.getenv('RESEARCH_ENABLE_INTELLIGENT_EXPANSION', 'true').lower() == 'true'

# Maximum number of ideas to generate from root
MAX_RESEARCH_IDEAS = int(os.getenv('RESEARCH_MAX_IDEAS', '3'))

# Maximum hypotheses per idea
MAX_HYPOTHESES_PER_IDEA = int(os.getenv('RESEARCH_MAX_HYPOTHESES', '2'))

# Maximum experiments per hypothesis
MAX_EXPERIMENTS_PER_HYPOTHESIS = int(os.getenv('RESEARCH_MAX_EXPERIMENTS', '1'))

# Number of retries for failed LLM calls
IDEA_GENERATION_RETRY_COUNT = int(os.getenv('RESEARCH_IDEA_RETRY_COUNT', '2'))
```

**Environment Variables**:
- `RESEARCH_ENABLE_INTELLIGENT_EXPANSION` - Feature toggle (default: true)
- `RESEARCH_MAX_IDEAS` - Max ideas per root (default: 3)
- `RESEARCH_MAX_HYPOTHESES` - Max hypotheses per idea (default: 2)
- `RESEARCH_MAX_EXPERIMENTS` - Max experiments per hypothesis (default: 1)
- `RESEARCH_IDEA_RETRY_COUNT` - LLM retry attempts (default: 2)

## Data Models

### ResearchNode

**Location**: `uagent_research/models/research_tree.py`

**Structure**:
```python
@dataclass
class ResearchNode:
    id: str                    # Unique identifier
    type: NodeType             # ROOT, IDEA, HYPOTHESIS, EXPERIMENT
    title: str                 # Human-readable title
    content: str               # Detailed content
    status: NodeStatus         # PENDING, RUNNING, COMPLETE, FAILED
    prior: float               # Prior probability (0.0-1.0)
    parent_id: Optional[str]   # Parent node ID
    children: List[str]        # Child node IDs
    visits: int                # PUCT visit count
    value: float               # PUCT value
```

**Node Types**:
- `ROOT` - Starting point of research tree
- `IDEA` - High-level research direction
- `HYPOTHESIS` - Testable hypothesis
- `EXPERIMENT` - Concrete experiment to run

### Research Tree

**Location**: `uagent_research/models/research_tree.py`

**Purpose**: Stores the complete research tree with nodes and edges.

**Key Methods**:
- `add_node(node, parent_id)` - Add node to tree
- `get_node(node_id)` - Retrieve node by ID
- `get_children(node_id)` - Get child nodes
- `to_dict()` - Serialize tree for API/frontend

## Integration Points

### 1. Orchestrator ↔ IdeaGenerationService

**Interface**:
```python
# Orchestrator calls service
children = await self.idea_service.generate_ideas(goal, context)

# Service returns ResearchNode objects
return [
    ResearchNode(id="idea-1", type=NodeType.IDEA, ...),
    ResearchNode(id="idea-2", type=NodeType.IDEA, ...),
]
```

**Contract**:
- Service MUST return List[ResearchNode] or empty list
- Nodes MUST have unique IDs
- Nodes MUST have correct type (IDEA/HYPOTHESIS/EXPERIMENT)
- Service MUST NOT raise exceptions (handles internally)

### 2. IdeaGenerationService ↔ ScientificResearchEngine

**Interface**:
```python
# Service calls engine
ideas = await self.engine._generate_research_ideas(goal, context, max_ideas)

# Engine returns idea objects
return [
    ResearchIdea(title="...", summary="...", confidence=0.8),
    ...
]
```

**Transformation**:
```python
# Service transforms to ResearchNode
node = ResearchNode(
    id=f"idea-{uuid.uuid4().hex[:8]}",
    type=NodeType.IDEA,
    title=idea.title,
    content=idea.summary,
    prior=idea.confidence,
    ...
)
```

### 3. Middleware ↔ Orchestrator

**Interface**:
```python
# Middleware creates orchestrator with LLM
orchestrator = TreeSearchOrchestrator(
    max_parallel=3,
    budget=Budget(...),
    llm=llm,  # OpenHands LLM instance
)

# Orchestrator handles LLM gracefully
if llm:
    self.idea_service = IdeaGenerationService(llm)
    self.use_intelligent_expansion = True
else:
    self.use_intelligent_expansion = False
```

## Testing Strategy

### 1. Unit Tests for IdeaGenerationService

**File**: `tests/test_idea_generation_service.py`

**Coverage**:
- Service initialization with/without config
- Idea generation with mock LLM
- Hypothesis generation
- Experiment generation
- Error handling and retries
- Empty response handling
- Node structure validation

**Mocking Strategy**:
```python
class MockLLM:
    def __init__(self, responses):
        self.responses = responses
    
    async def completion(self, messages, temperature=0.7):
        response_text = self.responses[self.call_count]
        self.call_count += 1
        return mock_response_object(response_text)
```

### 2. Integration Tests for Orchestrator

**File**: `tests/test_orchestrator_intelligent_expansion.py`

**Coverage**:
- Orchestrator with intelligent service enabled
- Orchestrator without service (fallback)
- ROOT node expansion
- IDEA node expansion
- HYPOTHESIS node expansion
- Service failure handling
- Tree stats updates
- Node parent-child relationships

**Testing Approach**:
```python
# Create orchestrator with mock service
mock_service = MockIdeaGenerationService()
orchestrator = TreeSearchOrchestrator(
    max_parallel=1,
    idea_service=mock_service
)

# Test expansion
children = await orchestrator._expand_node(root, goal="Test")

# Verify service was called
assert mock_service.generate_ideas_called is True

# Verify intelligent nodes generated
assert children[0].title.startswith("LLM-Generated")
```

## Code Flow Example

### Complete Flow: User Goal → Intelligent Tree

```
1. User: "Implement vector search in PostgreSQL"
   ↓
2. ResearchMiddleware.intercept_message()
   - Detects research task
   - Calls start_research()
   ↓
3. ResearchMiddleware.start_research()
   - Gets LLM from session_mgr.llm
   - Creates TreeSearchOrchestrator(llm=llm)
   ↓
4. TreeSearchOrchestrator.__init__()
   - Receives llm
   - Creates IdeaGenerationService(llm)
   - Sets use_intelligent_expansion = True
   ↓
5. TreeSearchOrchestrator.run()
   - Starts PUCT loop
   - Iteration 1: Select ROOT
   ↓
6. TreeSearchOrchestrator._expand_node(ROOT)
   - if use_intelligent_expansion:
       - Call idea_service.generate_ideas(goal)
   ↓
7. IdeaGenerationService.generate_ideas()
   - Calls engine._generate_research_ideas()
   ↓
8. ScientificResearchEngine._generate_research_ideas()
   - Creates LLM prompt
   - Calls LLM.completion()
   ↓
9. LLM.completion()
   - Returns JSON response with ideas
   ↓
10. ScientificResearchEngine
    - Parses JSON
    - Returns ResearchIdea objects
    ↓
11. IdeaGenerationService
    - Transforms to ResearchNode objects
    - Returns list
    ↓
12. TreeSearchOrchestrator
    - Receives intelligent nodes
    - Adds to tree
    - Updates stats
    - Publishes event
    ↓
13. Frontend
    - Receives event
    - Displays intelligent ideas:
      • "Use pgvector extension with HNSW"
      • "Implement custom GiST index"
      • "Leverage tsvector with embeddings"
```

## Key Files Reference

| File | Purpose | Lines | Key Changes |
|------|---------|-------|-------------|
| `services/idea_generation_service.py` | LLM-based node generation | ~350 | NEW |
| `services/__init__.py` | Module exports | ~6 | MODIFIED |
| `orchestrator/tree_orchestrator.py` | Tree search with intelligent expansion | ~900 | MODIFIED (~150 lines) |
| `middleware/research_middleware.py` | LLM integration | ~800 | MODIFIED (~30 lines) |
| `config.py` | Configuration options | ~50 | MODIFIED (~15 lines) |
| `tests/test_idea_generation_service.py` | Service unit tests | ~350 | NEW |
| `tests/test_orchestrator_intelligent_expansion.py` | Integration tests | ~250 | NEW |

## Development Guide

### Adding a New Node Type

1. **Update NodeType enum** in `uagent_research/models/research_tree.py`:
   ```python
   class NodeType(Enum):
       ROOT = "root"
       IDEA = "idea"
       HYPOTHESIS = "hypothesis"
       EXPERIMENT = "experiment"
       ANALYSIS = "analysis"  # NEW
   ```

2. **Add generation method** in `IdeaGenerationService`:
   ```python
   async def generate_analyses(
       self,
       experiment_content: str,
       parent_node: ResearchNode
   ) -> List[ResearchNode]:
       # Implementation
   ```

3. **Update _expand_node()** in `TreeSearchOrchestrator`:
   ```python
   elif node.type == NodeType.EXPERIMENT:
       children = await self.idea_service.generate_analyses(...)
   ```

### Adding a New Research Engine

1. **Create engine class** in `uagent_research/engines/`:
   ```python
   class CodeResearchEngine:
       async def _generate_research_ideas(self, goal, context, max_ideas):
           # Code-specific research logic
   ```

2. **Modify IdeaGenerationService** to support multiple engines:
   ```python
   def __init__(self, llm, config, engine_type='scientific'):
       if engine_type == 'scientific':
           self.engine = ScientificResearchEngine(llm)
       elif engine_type == 'code':
           self.engine = CodeResearchEngine(llm)
   ```

3. **Update orchestrator** to pass engine type:
   ```python
   self.idea_service = IdeaGenerationService(
       llm=llm,
       config=config,
       engine_type='code'
   )
   ```

## Performance Considerations

1. **LLM Call Latency**:
   - Each node expansion requires 1 LLM call
   - Average latency: 1-3 seconds per call
   - Consider caching for repeated goals

2. **Parallel Execution**:
   - Orchestrator executes children in parallel
   - Bounded by `max_parallel` setting
   - Node generation is sequential (one level at a time)

3. **Memory Usage**:
   - Tree stored in memory
   - Each node: ~500 bytes
   - For 100 nodes: ~50 KB
   - Acceptable for research trees (typically < 100 nodes)

4. **Database Impact**:
   - Tree periodically saved to database
   - Not on every expansion (performance optimization)
   - Can be configured via `save_interval`

## Future Enhancements

1. **Caching Layer**:
   - Cache idea generation by goal hash
   - Reduce LLM costs for repeated goals
   - TTL-based cache invalidation

2. **Multi-Engine Routing**:
   - Route based on node type and context
   - Scientific engine for hypothesis generation
   - Code engine for implementation ideas
   - User-provided custom engines

3. **Parallel Node Generation**:
   - Generate multiple hypothesis levels concurrently
   - Faster tree construction
   - Requires careful dependency management

4. **User Feedback Loop**:
   - Allow users to rate generated ideas
   - Use ratings to improve future generation
   - Reinforcement learning for LLM prompts

