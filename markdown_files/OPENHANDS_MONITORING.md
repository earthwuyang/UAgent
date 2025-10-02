# OpenHands Live Monitoring Guide

## Overview

The enhanced OpenHands single container integration now provides **real-time monitoring** of everything OpenHands is doing inside the container. You can watch commands, prompts, responses, and debug issues as they happen!

## Workspace Location

The OpenHands runtime container workspace is mounted to:
```
${UAGENT_WORKSPACE_DIR}/uagent_workspaces/experiment_session_*/
```

Where `UAGENT_WORKSPACE_DIR` is typically `/home/wuy/AI/uagent-workspace`.

## Live Monitoring Files

When OpenHands runs, you'll find these monitoring files in:
```
${WORKSPACE}/logs/openhands_live/
```

### Core Monitoring Files

1. **📄 live_combined.log**
   - All OpenHands output with timestamps
   - Best file to watch for general monitoring

2. **📋 live_stdout.log**
   - Standard output (commands, results)
   - Shows normal execution flow

3. **⚠️ live_stderr.log**
   - Error output and warnings
   - First place to check when things go wrong

4. **📊 container_status.json**
   - Current container status and metadata
   - Shows exit codes, duration, errors

5. **📖 README_MONITORING.md**
   - Human-readable monitoring guide
   - Updated with final results when complete

## Live Monitoring Commands

### Watch OpenHands Activity in Real-Time
```bash
# Watch all OpenHands activity
tail -f ${WORKSPACE}/logs/openhands_live/live_combined.log

# Watch just errors and warnings
tail -f ${WORKSPACE}/logs/openhands_live/live_stderr.log

# Check current status
cat ${WORKSPACE}/logs/openhands_live/container_status.json
```

### Monitor Container Directly
```bash
# Find running OpenHands containers
docker ps | grep openhands

# Watch container logs directly
docker logs <container_id> -f

# Execute commands inside the container
docker exec -it <container_id> bash
```

## What You'll See

### Normal Execution
- ✅ OpenHands initialization (runtime setup, LLM connection)
- 🤖 Agent reasoning and planning
- 💻 Bash commands being executed
- 📝 File creation and editing
- 🔍 Code analysis and debugging
- 📊 Result compilation and final.json creation

### Error Scenarios
- ❌ Import errors or missing dependencies
- 🔌 LLM connection issues
- 💥 Runtime crashes or timeouts
- 📂 File permission or path issues
- 🚫 Docker container problems

## Debugging Workflow

1. **Start Monitoring**: Begin watching logs before starting experiment
   ```bash
   tail -f /path/to/workspace/logs/openhands_live/live_combined.log
   ```

2. **Launch Experiment**: Start your UAgent research session

3. **Monitor Progress**: Watch real-time logs to see:
   - What commands OpenHands is running
   - What files it's creating/editing
   - Any errors or warnings
   - LLM interactions and responses

4. **Debug Issues**: If problems occur:
   - Check `live_stderr.log` for errors
   - Check `container_status.json` for exit codes
   - Examine workspace files for partial results
   - Use `docker logs <container_id>` for low-level details

## Example Monitoring Session

```bash
# Terminal 1: Watch live activity
tail -f ~/AI/uagent-workspace/uagent_workspaces/experiment_session_123/logs/openhands_live/live_combined.log

# Terminal 2: Monitor errors
tail -f ~/AI/uagent-workspace/uagent_workspaces/experiment_session_123/logs/openhands_live/live_stderr.log

# Terminal 3: Check status periodically
watch -n 5 'cat ~/AI/uagent-workspace/uagent_workspaces/experiment_session_123/logs/openhands_live/container_status.json'
```

## Troubleshooting

### No Log Files Appearing
- Check if container started successfully
- Verify workspace permissions
- Look for container startup errors in UAgent logs

### Container Exits Immediately
- Check `container_status.json` for exit code
- Review `live_stderr.log` for startup errors
- Verify Docker image and environment variables

### Missing final.json
- Check if OpenHands completed its task
- Look for file creation commands in logs
- Verify experiment goal was clear enough
- Check workspace permissions

## Benefits

✅ **Real-time visibility** into OpenHands execution
✅ **Better debugging** with detailed error logs
✅ **Progress monitoring** to see current status
✅ **Command transparency** to understand what's being executed
✅ **LLM interaction logs** to debug prompt/response issues
✅ **File system monitoring** to track created/modified files

This enhanced monitoring makes OpenHands completely transparent, allowing you to understand exactly what's happening during scientific experiments!