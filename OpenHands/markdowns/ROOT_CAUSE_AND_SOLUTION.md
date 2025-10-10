# Root Cause Analysis: Research Tree Not Auto-Starting

## Problem
The parallel research tree never starts when typing a research prompt in chat.

## Root Cause Identified

The issue is a **complex package structure problem** with nested directories:

```
extensions/uagent_research/          ← Outer package (added to sys.path)
├── middleware/                      ← At outer level
│   └── research_middleware.py
├── orchestrator/                    ← At outer level  
│   └── tree_orchestrator.py
├── classifier/                      ← At outer level
├── adapters/                        ← At outer level
└── uagent_research/                 ← Inner nested package!
    ├── api/
    ├── models/
    │   └── research_tree.py         ← Budget class is here
    └── ...
```

When `app.py` adds `extensions/uagent_research` to `sys.path`:
- Modules at outer level become top-level imports: `middleware.`, `orchestrator.`
- Inner `uagent_research/` becomes: `uagent_research.models.research_tree`

But all the outer-level modules use **relative imports** like:
- `from ..classifier` (tries to go up from outer level = error!)
- `from ..uagent_research.models` (works but brittle)

This causes `ImportError: attempted relative import beyond top-level package`

## Why This Breaks Auto-Trigger

1. `session.py` tries to import research middleware
2. Import fails due to relative import issues
3. `RESEARCH_MIDDLEWARE_AVAILABLE = False`
4. `dispatch()` skips the research middleware path entirely
5. Message goes to normal agent loop instead

## Proper Solution

The codebase needs **architectural reorganization**:

### Option 1: Flatten Structure (Recommended)

Move everything from outer level INTO the inner `uagent_research/` package:

```
extensions/uagent_research/
└── uagent_research/                 ← Single package root
    ├── middleware/
    ├── orchestrator/
    ├── classifier/
    ├── adapters/
    ├── api/
    ├── models/
    └── ...
```

Then all imports become clean relative imports within one package.

### Option 2: Fix All Relative Imports

Convert ALL relative imports in ALL modules to absolute imports based on the actual structure. This requires fixing:
- `middleware/research_middleware.py` ✅ (done)
- `orchestrator/tree_orchestrator.py` ❌ (needs fixing)
- `classifier/task_classifier.py` ❌ (likely needs fixing)
- `adapters/` modules ❌ (likely need fixing)
- `services/` modules ❌ (likely need fixing)
- etc...

This is error-prone and creates maintenance burden.

## Immediate Workaround

Until the structure is reorganized, use the **REST API** to start research instead of auto-trigger:

```bash
# Start OpenHands server
poetry run python openhands/server/listen.py

# Use API to start research
curl -X POST http://localhost:3000/api/research/start \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Your research goal here",
    "session_id": "unique_session_id",
    "config": {
      "max_iterations": 50,
      "max_cost": 20.0
    }
  }'
```

The API routes work because they're in the inner `uagent_research/` package which has proper structure.

## What We Fixed

1. ✅ `session.py` - Updated to try multiple import paths
2. ✅ `middleware/research_middleware.py` - Converted relative imports to work with outer level
3. ✅ `openhands/runtime/utils/bash.py` - Fixed libtmux compatibility
4. ✅ Added `extensions/__init__.py`

## What Still Needs Fixing

The auto-trigger won't work until ALL outer-level modules have their imports fixed or the structure is reorganized.

##Status

- **Import errors**: ✅ Partially fixed (middleware level)
- **Auto-trigger**: ❌ Blocked by orchestrator/classifier/etc imports  
- **Manual API start**: ✅ Works
- **WebSocket updates**: ✅ Works when started via API

## Recommendation

**Reorganize the package structure** (Option 1) rather than playing whack-a-mole with relative imports. This is the clean, maintainable solution.

