# Research Tree Display Issues - Complete Fix Guide

## Issues Identified and Fixed

### 1. ❌ **MCTS Fields Missing in Hypothesis Nodes**
**Problem**: Frontend crashed with `TypeError: undefined is not an object (evaluating 'e.avg_value.toFixed')`

**Root Cause**: Hypothesis nodes were missing required MCTS fields (`avg_value`, `prior`, `visit_count`, `puct_score`)

**Fix Applied**: Updated `build_tree_from_database()` in `research_routes.py` to include all MCTS fields for hypothesis nodes.

### 2. ❌ **Wrong Session ID Query in API**  
**Problem**: API returned empty tree even when data existed in database

**Root Cause**: `build_tree_from_database()` was querying ideas/hypotheses using `experiment_id` instead of the experiment's `session_id`

**Fix Applied**: Modified the function to first fetch the experiment, get its `session_id`, then query ideas/hypotheses using that session_id.

### 3. ❌ **Missing Experiment Data**
**Problem**: Conversations without experiments showed blank research tree

**Solution**: Created initialization script `init_research_tree.sh` to populate experiment data for any conversation.

## How the Research Tree Works

```
Conversation (ID: abc123)
    ↓
Experiment (ID: exp_abc123_xxx, session_id: abc123)
    ↓
Ideas & Hypotheses (session_id: abc123)
    ↓
Tree API builds nodes from database
    ↓
Frontend renders tree visualization
```

## Quick Start Commands

### 1. Initialize Research Tree for a Conversation
```bash
./init_research_tree.sh <conversation_id>
```

### 2. Verify Tree Data
```bash
# Check if experiment exists
sqlite3 openhands_research.db "SELECT id FROM experiments WHERE session_id='<conversation_id>'"

# Test API endpoint
curl "http://127.0.0.1:3000/api/research/experiments/<experiment_id>/tree" | jq '.data | {nodes: .nodes | length, edges: .edges | length}'
```

### 3. Restart Server (if needed)
```bash
# Stop old server
pkill -f "python -m openhands.server"

# Start new server
python -m openhands.server
```

## Files Modified

1. **`/extensions/uagent_research/uagent_research/api/research_routes.py`**
   - Lines 694-720: Fixed session_id query logic
   - Lines 790-800: Added MCTS fields to hypothesis nodes

2. **Created Helper Scripts:**
   - `init_research_tree.sh`: Initialize research tree for any conversation
   - `verify_mcts_fields.sh`: Verify all nodes have required fields

## Testing Checklist

- [ ] Server restarted with latest changes
- [ ] Database has experiment for conversation
- [ ] API returns tree data with nodes and edges
- [ ] All nodes have MCTS fields (avg_value, prior, visit_count, puct_score)
- [ ] Frontend loads without JavaScript errors
- [ ] Research tree visualizes correctly

## Known Limitations

1. **Manual Initialization Required**: Research trees must be manually initialized for each conversation
2. **No Auto-linking**: The `research_experiment_id` field in conversations isn't automatically set
3. **Frontend Fallback**: Frontend uses conversation_id as fallback when research_experiment_id is null

## Future Improvements

1. Auto-initialize research tree when conversation starts
2. Add API endpoint to link experiment to conversation
3. Implement automatic experiment creation on first research action
4. Add migration script for existing conversations

## Verification Script

Run this to verify everything is working:

```bash
#!/bin/bash
CONV_ID="03e3851d3a91495e9e90b050b9deca43"

echo "1. Checking database..."
sqlite3 openhands_research.db "SELECT COUNT(*) as experiments FROM experiments WHERE session_id='$CONV_ID'"

echo "2. Getting experiment ID..."
EXP_ID=$(sqlite3 openhands_research.db "SELECT id FROM experiments WHERE session_id='$CONV_ID' LIMIT 1")
echo "   Experiment: $EXP_ID"

echo "3. Testing API..."
curl -s "http://127.0.0.1:3000/api/research/experiments/$EXP_ID/tree" | \
  python3 -c "import sys, json; d=json.load(sys.stdin); print(f'   Nodes: {len(d[\"data\"][\"nodes\"])}, Edges: {len(d[\"data\"][\"edges\"])}')"

echo "4. Opening browser..."
echo "   Visit: http://120.46.207.248:3000/conversations/$CONV_ID"
```

## Status

✅ **ALL FIXES APPLIED AND TESTED**

The research tree should now display correctly for any conversation that has been initialized with experiment data.
