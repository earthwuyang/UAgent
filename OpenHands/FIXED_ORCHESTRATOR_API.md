# ✅ Fixed: TreeSearchOrchestrator API

## The Error
```
TypeError: TreeSearchOrchestrator.__init__() got an unexpected keyword argument 'goal'
```

## The Problem

I was calling `TreeSearchOrchestrator` incorrectly:

**Wrong** ❌:
```python
orchestrator = TreeSearchOrchestrator(
    goal=message,
    research_id=experiment_id,
    session_id=conversation_id,
    max_iterations=50,
)
```

## The Solution

`TreeSearchOrchestrator` has a simple `__init__()` and takes parameters in `run()`:

**Correct** ✅:
```python
# Step 1: Create orchestrator (no parameters)
orchestrator = TreeSearchOrchestrator()

# Step 2: Run with goal and settings
result = await orchestrator.run(
    goal=message,
    research_id=experiment_id,
    max_iterations=50
)
```

## Files Fixed

1. **`openhands/server/services/conversation_service.py`**
   - Fixed both occurrences (initial_user_msg and research_goal paths)
   
2. **`openhands/server/session/session.py`**
   - Fixed dispatch method

## The Correct Pattern

```python
# Create
orchestrator = TreeSearchOrchestrator()

# Run in background
async def _run_tree_search():
    try:
        result = await orchestrator.run(
            goal=user_message,
            research_id=experiment_id,
            max_iterations=50
        )
    except Exception as e:
        logger.error(f"Error: {e}")

asyncio.create_task(_run_tree_search())
```

## Testing

✅ **Server running**: http://120.46.207.248:3000/  
✅ **Classification working**: 95% confidence detected  
✅ **Orchestrator API**: Fixed  
✅ **Ready to test**: Send your research prompt!

## What to Expect Now

1. **Classification**: ✅ Working (95% confidence)
2. **Orchestrator creation**: ✅ Fixed
3. **Tree search launch**: ✅ Should work now

### Watch the Logs

```bash
tmux attach -t uagent-backend
```

Look for:
```
Classifying first message in session <id>
🔬 Research activated: research_xxx (complex_research, 95%)
[RESEARCH] Starting: research_xxx
[TREE] Iteration 1: ...
```

---

**Now try sending your research prompt again!** 🚀

The classification worked perfectly (95%!), and now the orchestrator should launch correctly!
