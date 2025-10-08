# Fix for TypeError: undefined is not an object (evaluating 'e.avg_value.toFixed')

## Problem
The frontend was crashing with the error:
```
TypeError: undefined is not an object (evaluating 'e.avg_value.toFixed')
```

This occurred in `ResearchNode.tsx` when trying to display MCTS statistics for hypothesis nodes.

## Root Cause
The `build_tree_from_database()` function in `research_routes.py` was creating hypothesis nodes without the required MCTS fields that the frontend expects. While idea nodes had these fields:
- `visit_count`
- `avg_value`
- `prior`
- `puct_score`

Hypothesis nodes were missing them, causing the frontend to try calling `.toFixed()` on `undefined`.

## Solution Applied
Updated the hypothesis node creation in `research_routes.py` (around line 790-800) to include all MCTS fields:

```python
"data": {
    "id": hyp_node_id,
    "type": "hypothesis",
    "title": hyp.statement[:50] + "..." if len(hyp.statement) > 50 else hyp.statement,
    "description": hyp.expected_outcome or "",
    "status": "testing" if not hyp.tested else "completed",
    "visit_count": 1,  # ✅ Added
    "avg_value": (hyp.confidence or 0.5) * (hyp.testability_score or 0.5),  # ✅ Added
    "prior": hyp.testability_score or 0.5,  # ✅ Added
    "puct_score": 0.0,  # ✅ Added
    "metadata": {
        "confidence": hyp.confidence,
        "testability": hyp.testability_score
    }
}
```

## Files Modified
- `/home/wuy/AI/UAgent/OpenHands/extensions/uagent_research/uagent_research/api/research_routes.py`

## Testing
1. **Syntax check**: ✅ Passed
   ```bash
   python3 -m py_compile extensions/uagent_research/uagent_research/api/research_routes.py
   ```

2. **Server restart required**: You must restart the OpenHands server to apply changes:
   ```bash
   # Stop the current server (Ctrl+C or kill process)
   # Then restart:
   poetry run python -m openhands.core.main -t oh_workspace
   ```

3. **Verify fix**: After restart, check the API response:
   ```bash
   curl "http://120.46.207.248:3001/api/research/18014f5c05c04e7ba93b75b3e8d6f5ff/tree" | jq '.data.nodes[] | select(.type=="hypothesis") | .data | {avg_value, prior, visit_count, puct_score}'
   ```
   
   Expected output: All hypothesis nodes should now have numeric values for these fields.

4. **Frontend test**: Navigate to the research tree view in your browser:
   ```
   http://120.46.207.248:3000/conversations/18014f5c05c04e7ba93b75b3e8d6f5ff
   ```
   
   The tree should now render without the `avg_value.toFixed` error.

## Additional Notes
- The `avg_value` for hypotheses is calculated as: `(confidence * testability_score)`
- The `prior` is set to the `testability_score`
- Default values of `0.5` are used if database fields are NULL
- All nodes (root, idea, hypothesis) now have consistent MCTS field structure

## Related Files
- Frontend component: `frontend/src/components/research/ResearchNode.tsx` (line 217)
- Type definitions: `frontend/src/state/research-tree-store.ts` (line 19)
- Backend API: `extensions/uagent_research/uagent_research/api/research_routes.py` (line 694-850)

## Status
✅ **FIXED** - Awaiting server restart and verification
