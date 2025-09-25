# Fix for Absolute Path Issues in Command Wrapping

## Problem
Error message: `/workspace/logs/commands/20250925_165140_export_PIP_NO_INPUT=1_: No such file or directory`

The command wrapping was using absolute paths starting with `/workspace/` which only exist inside the OpenHands container, not on the host filesystem.

## Root Cause
The original implementation used hardcoded `/workspace/` paths for log files:
```python
log_realtime = f"/workspace/logs/commands/{timestamp}_{safe_cmd}.realtime"
```

This path doesn't exist when the command runs, causing the "No such file or directory" error.

## Solution

### Changed to Relative Paths
Use relative paths that work from the container's working directory:

```python
# Before (incorrect)
log_realtime = f"/workspace/logs/commands/{timestamp}_{safe_cmd}.realtime"

# After (correct)
log_dir = "logs/commands"
log_filename = f"{timestamp}_{safe_cmd}.realtime"
log_realtime = f"{log_dir}/{log_filename}"
```

### Implementation Details

1. **Non-blocking commands** (lines 208-222):
   - Uses relative path: `logs/commands/`
   - Creates directory with: `mkdir -p logs/commands`
   - Logs output to: `logs/commands/{timestamp}_{cmd}.realtime`

2. **Blocking commands** (lines 223-238):
   - Same relative path structure
   - Works with `script` command

3. **Monitoring paths**:
   - Shows actual filesystem path for monitoring
   - Uses `self._workspace_path / log_realtime` for absolute path

## Files Modified
- `/backend/app/integrations/openhands_runtime.py` (lines 208-238)

## Impact
✅ Commands can now create log files correctly
✅ No more "No such file or directory" errors
✅ Logs are created relative to the workspace
✅ Monitoring paths show correct absolute locations

## Log File Locations

### Inside Container
- Working directory: `/workspace/`
- Logs created at: `/workspace/logs/commands/*.realtime`

### On Host Filesystem
- Actual location: `/home/wuy/AI/uagent-workspace/*/logs/commands/*.realtime`
- Can monitor with: `tail -f /home/wuy/AI/uagent-workspace/*/logs/commands/*.realtime`

## Testing
```bash
# Verify logs are created
ls -la /home/wuy/AI/uagent-workspace/*/logs/commands/

# Monitor latest log
tail -f /home/wuy/AI/uagent-workspace/*/logs/commands/*.realtime
```