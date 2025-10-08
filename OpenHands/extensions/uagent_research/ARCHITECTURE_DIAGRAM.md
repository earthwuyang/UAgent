# UAgent Research Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenHands Application                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Research Middleware                          │  │
│  │  - Intercepts user messages                               │  │
│  │  - Detects research tasks                                 │  │
│  │  - Creates & manages orchestrators                        │  │
│  │  - Provides LLM to orchestrator                           │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                            │
│                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         TreeSearchOrchestrator                            │  │
│  │  - PUCT-based tree search                                 │  │
│  │  - Manages research tree                                  │  │
│  │  - Coordinates parallel execution                         │  │
│  │  - Uses IdeaGenerationService for node expansion          │  │
│  └──────────┬──────────────────────────┬────────────────────┘  │
│             │                           │                        │
│             ▼                           ▼                        │
│  ┌─────────────────────┐    ┌─────────────────────────────┐   │
│  │ IdeaGenerationService│    │    ResearchTree             │   │
│  │ - Generates ideas    │    │    - Stores nodes/edges     │   │
│  │ - Generates hypotheses│   │    - Tracks state           │   │
│  │ - Generates experiments│  │    - PUCT statistics        │   │
│  └──────────┬───────────┘    └─────────────────────────────┘   │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │       ScientificResearchEngine                            │  │
│  │  - Research idea generation                               │  │
│  │  - Hypothesis formulation                                 │  │
│  │  - Experiment design                                      │  │
│  │  - Result analysis                                        │  │
│  └──────────┬───────────────────────────────────────────────┘  │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    LLM (OpenHands)                        │  │
│  │  - GPT-4, Claude, etc.                                    │  │
│  │  - Generates intelligent responses                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Intelligent Node Expansion Flow

### ROOT → IDEAS

```
User Goal: "Implement vector search in PostgreSQL"
                    ↓
        TreeSearchOrchestrator._expand_node(ROOT)
                    ↓
        IdeaGenerationService.generate_ideas(goal)
                    ↓
        ScientificResearchEngine._generate_research_ideas()
                    ↓
                  LLM.completion(prompt)
                    ↓
        Parse JSON: [
          {"title": "Use pgvector extension", ...},
          {"title": "Implement GiST index", ...},
          {"title": "Leverage tsvector", ...}
        ]
                    ↓
        Transform to ResearchNode[]
                    ↓
        Add nodes to tree
                    ↓
        Tree now has 3 IDEA children
```

### IDEA → HYPOTHESES

```
IDEA Node: "Use pgvector extension with HNSW indexing"
                    ↓
        TreeSearchOrchestrator._expand_node(IDEA)
                    ↓
        IdeaGenerationService.generate_hypotheses(idea_content, parent)
                    ↓
                LLM.completion(prompt)
                    ↓
        Parse JSON: [
          {"title": "HNSW provides O(log n) search", ...},
          {"title": "95%+ recall achievable", ...}
        ]
                    ↓
        Transform to ResearchNode[]
                    ↓
        Add nodes to tree with parent_id
                    ↓
        IDEA now has 2 HYPOTHESIS children
```

### HYPOTHESIS → EXPERIMENTS

```
HYPOTHESIS Node: "HNSW provides O(log n) search"
                    ↓
        TreeSearchOrchestrator._expand_node(HYPOTHESIS)
                    ↓
        IdeaGenerationService.generate_experiments(hypothesis, parent)
                    ↓
                LLM.completion(prompt)
                    ↓
        Parse JSON: [
          {
            "title": "Benchmark HNSW vs IVFFlat",
            "methodology": "...",
            "expected_outcome": "..."
          }
        ]
                    ↓
        Transform to ResearchNode[]
                    ↓
        Add nodes to tree
                    ↓
        HYPOTHESIS now has 1 EXPERIMENT child
```

## Component Interaction Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                      Research Session Lifecycle                     │
└────────────────────────────────────────────────────────────────────┘

1. User Message → ResearchMiddleware
   - Classifies message as research task
   - Extracts research goal

2. ResearchMiddleware → TreeSearchOrchestrator
   - Creates orchestrator with LLM
   - Starts background task

3. TreeSearchOrchestrator Initialization
   - Creates IdeaGenerationService(llm)
   - Initializes ResearchTree
   - Starts PUCT loop

4. PUCT Loop - Iteration 1
   a) Select node: ROOT (only node)
   b) Expand node:
      - Call IdeaGenerationService.generate_ideas(goal)
      - Service calls ScientificResearchEngine
      - Engine calls LLM
      - LLM returns 3 ideas
      - Service transforms to ResearchNode objects
      - Orchestrator adds to tree
   c) Simulate/execute children (parallel)
   d) Backpropagate results

5. PUCT Loop - Iteration 2
   a) Select node: Best IDEA (highest PUCT score)
   b) Expand node:
      - Call IdeaGenerationService.generate_hypotheses()
      - LLM returns hypotheses
      - Add to tree
   c) Execute hypotheses
   d) Backpropagate

6. PUCT Loop - Iteration N
   - Continue until budget exhausted or goal achieved
   - Each iteration refines the research tree
   - Best paths explored first (PUCT selection)

7. Result
   - Complete research tree with meaningful nodes
   - Execution results for experiments
   - Analysis and recommendations
```

## Data Flow

```
User Input
    ↓
[Research Goal]
    ↓
TreeSearchOrchestrator
    ↓
IdeaGenerationService.generate_ideas()
    ↓
ScientificResearchEngine._generate_research_ideas()
    ↓
LLM Prompt:
  "Generate 3 research ideas for: {goal}
   Return JSON with title, summary, confidence"
    ↓
LLM Response (JSON)
    ↓
Parse & Validate
    ↓
Create ResearchNode objects:
  - id: "idea-{uuid}"
  - type: NodeType.IDEA
  - title: "LLM-generated title"
  - content: "LLM-generated summary"
  - prior: confidence score
    ↓
Return List[ResearchNode]
    ↓
Orchestrator adds to tree
    ↓
[Tree Updated]
    ↓
Event published to frontend
    ↓
User sees meaningful ideas
```

## Fallback Mechanism

```
Orchestrator._expand_node(node)
    ↓
if use_intelligent_expansion:
    ↓
    try:
        ↓
        service.generate_ideas(goal)
        ↓
        if children:
            ↓
            return children  ✅ Intelligent nodes
        else:
            ↓
            (fall through to fallback)
    ↓
    except Exception:
        ↓
        log error
        ↓
        (fall through to fallback)
    ↓
else:
    ↓
(FALLBACK - Hardcoded nodes)
    ↓
    return [
        ResearchNode("Idea 1: Web Research"),
        ResearchNode("Idea 2: Code Research"),
        ...
    ]  ✅ Placeholder nodes
```

## Configuration Flow

```
Environment Variables
    ↓
RESEARCH_ENABLE_INTELLIGENT_EXPANSION=true
RESEARCH_MAX_IDEAS=3
RESEARCH_MAX_HYPOTHESES=2
    ↓
Loaded by config.py
    ↓
Used by IdeaGenerationService.__init__(config)
    ↓
Service respects limits during generation
    ↓
max_ideas=3 → LLM generates ≤3 ideas
```

## Key Design Decisions

1. **Separation of Concerns**
   - IdeaGenerationService: Node generation logic
   - TreeSearchOrchestrator: Tree management & PUCT
   - ScientificResearchEngine: LLM interaction & prompting

2. **Graceful Degradation**
   - LLM unavailable → Placeholder nodes
   - Service error → Fallback to placeholders
   - Ensures research tree always has structure

3. **Stateless Service**
   - IdeaGenerationService has no state between calls
   - Each generation request is independent
   - Simplifies testing and debugging

4. **Async Throughout**
   - All LLM calls are async
   - Non-blocking orchestrator operation
   - Supports parallel node execution

5. **Type Safety**
   - Strong typing with ResearchNode, NodeType, NodeStatus
   - Clear contracts between components
   - IDE autocomplete and error detection

## Future Architecture Enhancements

1. **Multi-Engine Routing**
   ```
   IdeaGenerationService
       ↓
   Router (based on node type/context)
       ├→ ScientificResearchEngine (for scientific research)
       ├→ CodeResearchEngine (for code analysis)
       └→ CustomEngine (user-provided)
   ```

2. **Caching Layer**
   ```
   IdeaGenerationService
       ↓
   Check cache(goal hash)
       ├→ Cache hit: Return cached ideas
       └→ Cache miss: Generate → Store → Return
   ```

3. **Parallel Generation**
   ```
   Generate multiple levels concurrently:
   ROOT → [IDEA1, IDEA2, IDEA3] (parallel)
       ↓
   Immediately generate hypotheses for all ideas
       ↓
   Faster tree construction
   ```

