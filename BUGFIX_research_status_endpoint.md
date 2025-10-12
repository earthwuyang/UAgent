# Research Tree Status Endpoint Fix

## Problem
The Research Tree panel in the OpenHands UI showed "Disconnected" status and the "Start" button was disabled because the frontend could not fetch experiment status.

## Root Cause
**API Endpoint Mismatch:**
- **Frontend expected:** `GET /api/research/experiments/{experiment_id}/status`
- **Backend provided:** Only `/tree`, `/events`, and control (PATCH) endpoints
- **Result:** 404 error → status check fails → UI disabled

## Solution
Added the missing `/status` endpoint to the backend API.

## Changes Made

### File: `OpenHands/extensions/uagent_research/api/research_routes.py`

#### 1. Added Response Model (lines 36-42)
```python
class ExperimentStatusResponse(BaseModel):
    """Response model for experiment status"""
    experiment_id: str
    status: str  # idle, running, paused, complete, failed, cancelled
    stats: dict
    adapters: dict
    active_branches: list
```

#### 2. Added Status Endpoint (lines 101-203)
```python
@router.get("/experiments/{experiment_id}/status")
async def get_experiment_status(experiment_id: str) -> ExperimentStatusResponse:
    """
    Get the current status of a research experiment.
    
    Returns experiment execution state, statistics, adapter states, and active branches.
    """
```

## How It Works

1. **Idle State**: If experiment not in `_active_trees`, returns "idle" status with zero stats
2. **Active State**: Analyzes tree nodes to determine status:
   - `running`: Has nodes with status='running'
   - `complete`: All nodes completed
   - `failed`: Has failed nodes
   - `idle`: No running/completed/failed nodes

3. **Additional Data**:
   - **Stats**: Node counts, costs, tokens from tree data
   - **Adapters**: Active adapter types and their current actions
   - **Active Branches**: Running nodes with branch info

## Testing

### Test the endpoint directly:
```bash
# Test with non-existent experiment (should return idle)
curl http://localhost:2999/api/research/experiments/test-id/status

# Check diagnostics
curl http://localhost:2999/api/research/diagnostics
```

### Expected Response for Idle Experiment:
```json
{
  "experiment_id": "aa7e12ea213443ea94b4a3492cc1c9db",
  "status": "idle",
  "stats": {
    "total_nodes": 0,
    "total_edges": 0,
    "total_cost": 0.0,
    "total_tokens": 0,
    "completed_nodes": 0,
    "failed_nodes": 0
  },
  "adapters": {},
  "active_branches": []
}
```

## Frontend Integration

The frontend already has the correct call structure in:
- `frontend/src/api/research-api.ts` (line 77-87)
- `frontend/src/routes/research-tab.tsx` (uses getExperimentStatus)

No frontend changes needed - the fix is backend-only.

## Next Steps

1. **Restart the OpenHands backend** to load the new endpoint
2. **Refresh the browser** at http://localhost:2999/conversations/{conversation_id}
3. **Verify**: The Start button should now be enabled
4. **Test**: Click Start to begin a research experiment

## Verification

After restarting, check:
1. ✅ No more 404 errors in browser console for `/status`
2. ✅ Start button is enabled in Research Tree panel
3. ✅ Status updates correctly (idle → running → complete)
4. ✅ Connection status shows "Connected" when active

## Related Files
- Backend: `OpenHands/extensions/uagent_research/api/research_routes.py`
- Frontend API: `OpenHands/frontend/src/api/research-api.ts`
- Frontend UI: `OpenHands/frontend/src/routes/research-tab.tsx`
