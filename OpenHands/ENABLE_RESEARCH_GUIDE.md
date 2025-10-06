# How to Enable Research Auto-Triggering

## Current Status

✅ **Research auto-trigger is now ENABLED in config**
⚠️ **You need to restart the server for changes to take effect**

## Steps to Activate

### 1. Restart the OpenHands Server

The config change won't take effect until you restart:

```bash
# Stop the current server (Ctrl+C if running in foreground)
# Or if running in background:
pkill -f "openhands"

# Start the server again
# (Use whatever command you normally use to start OpenHands)
```

### 2. Verify Research Middleware is Loaded

After restarting, check the server logs for:
```
Research middleware loaded successfully
```

You should also see when research triggers:
```
Research mode triggered for conversation {id}
Task type: complex_research, confidence: 0.95
```

### 3. Test with a Complex Query

Send a message like:
```
Research neural architecture search methods and implement the best approach,
comparing DARTS vs ENAS with benchmarks
```

Or your postgres query:
```
Modify postgres and pg_duckdb source code, extract features, train ML model,
embed model, and run experiments
```

### 4. Check Research Tree Tab

After sending a complex query:
1. Look for "Research Tree" tab in the conversation UI
2. Click it to see the research progress
3. Watch nodes appear as research progresses

## How to Verify It's Working

### Check 1: Logs Show Research Triggered

Server logs should show:
```
INFO - Research mode triggered for conversation 6411470ef5854f71bfecdfd7b6689330
INFO - Task type: complex_research, confidence: 0.95
INFO - Research started successfully: experiment_id=exp_64114...
```

### Check 2: Research Tree Tab Appears

In the conversation UI at the top-right, you should see a "Research Tree" icon (🌳).

### Check 3: API Returns Research Data

You can manually check the research tree API:
```bash
curl http://120.46.207.248:3000/api/research/experiments/{experiment_id}/tree
```

## Troubleshooting

### Issue: Research Not Triggering

**Check 1: Is auto-trigger enabled?**
```bash
grep "ENABLE_AUTO_RESEARCH_TRIGGER" ./extensions/uagent_research/config.py
# Should show: ENABLE_AUTO_RESEARCH_TRIGGER = os.getenv('ENABLE_AUTO_RESEARCH_TRIGGER', 'true')...
```

**Check 2: Did you restart the server?**
```bash
# Server must be restarted for config changes to load
```

**Check 3: Is the query complex enough?**
```bash
# Test the classifier:
python -c "
from extensions.uagent_research.classifier.task_classifier import task_classifier
result = task_classifier.should_trigger_research('your query here')
print(f'Should trigger: {result[0]}, Confidence: {result[2]:.2f}')
"
```

**Check 4: Check server logs**
```bash
# Look for errors in server startup
# Should see: "Research middleware loaded successfully"
```

### Issue: Frontend Still Stuck

If the frontend gets stuck even with research enabled:

1. **Disable research temporarily**:
   ```bash
   export ENABLE_AUTO_RESEARCH_TRIGGER=false
   # Restart server
   ```

2. **Check logs for errors**:
   - Look for Python exceptions
   - Look for "Failed to start research" errors

3. **Use fallback mode**:
   - Research middleware has error handling
   - Conversation should continue even if research fails

## Current Configuration

File: `./extensions/uagent_research/config.py`

```python
ENABLE_AUTO_RESEARCH_TRIGGER = True  # ✅ ENABLED
RESEARCH_CONFIDENCE_THRESHOLD = 0.7   # Trigger if confidence >= 70%
RESEARCH_MAX_ITERATIONS = 50          # Max research iterations
RESEARCH_MAX_COST = 10.0             # Max cost in dollars
RESEARCH_MAX_PARALLEL = 3            # Max concurrent branches
```

## Environment Variables (Optional)

You can also control research via environment variables:

```bash
# Enable/disable
export ENABLE_AUTO_RESEARCH_TRIGGER=true   # or false

# Adjust confidence threshold (0.0 to 1.0)
export RESEARCH_CONFIDENCE_THRESHOLD=0.6   # Lower = triggers more easily

# Adjust budget
export RESEARCH_MAX_ITERATIONS=100
export RESEARCH_MAX_COST=20.0

# Then restart server
```

## Quick Toggle Script

Use the provided script to easily enable/disable:

```bash
./toggle_research.sh status    # Check current status
./toggle_research.sh enable    # Enable research
./toggle_research.sh disable   # Disable research
```

---

## Summary

**To activate research auto-triggering:**

1. ✅ Config is already set to `ENABLE_AUTO_RESEARCH_TRIGGER = True`
2. 🔄 **Restart the OpenHands server** (REQUIRED)
3. 📝 Send a complex query
4. 🌳 Check the "Research Tree" tab
5. 📊 Watch research progress in real-time

**The server restart is crucial** - Python imports are cached, so config changes won't apply until restart.
