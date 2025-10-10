# Quick Start Guide - ML Routing Research

## ✅ Status: Import Errors Fixed

All import errors have been resolved. The system is ready to execute the ML-based query routing research.

## How to Run

### Option 1: Direct Execution (Recommended)
```bash
cd /home/wuy/AI/UAgent/OpenHands
python execute_ml_routing_research.py
```

### Option 2: As a Python Module
```bash
cd /home/wuy/AI/UAgent/OpenHands
python -m execute_ml_routing_research
```

## What Was Fixed

1. **Circular Import**: Fixed lazy loading of OpenHands events in `tree_orchestrator.py`
2. **Module Path**: Corrected import path in `research_middleware.py` 
3. **Import Strategy**: Simplified `execute_ml_routing_research.py` to use standard imports

## Files Modified

- `extensions/uagent_research/orchestrator/tree_orchestrator.py`
- `extensions/uagent_research/middleware/research_middleware.py`
- `execute_ml_routing_research.py`

## Verification

Run this to verify all imports work:
```bash
cd /home/wuy/AI/UAgent/OpenHands
python3 -c "
from extensions.uagent_research.middleware.research_middleware import ResearchMiddleware
print('✅ All imports working!')
"
```

## Next Steps

The script will:
1. Create workspace directory
2. Initialize research middleware
3. Start parallel research execution
4. Monitor progress and display updates

See `IMPORT_FIXES_SUMMARY.md` for detailed technical information.
