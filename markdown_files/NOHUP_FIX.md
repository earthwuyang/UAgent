# Fix for Double Nohup Wrapping and Command Timeout Issues

## Problem
Commands with `nohup` were timing out even though they should run in background:
```
ERROR - [CodeAct] Command timeout: nohup mkdir -p /workspace/logs/commands && nohup export PIP_NO_INPUT=1 && ...
```

## Root Causes

### 1. Double Nohup Wrapping
When a command already contained `nohup`, the code was wrapping it again:
- Original command: `nohup some_command`
- After wrapping: `nohup nohup some_command` ❌
- This creates invalid syntax and causes timeouts

### 2. Malformed Commands
Commands were being constructed incorrectly:
- `nohup mkdir` - nohup on a fast command is unnecessary
- `nohup export` - export is a shell builtin, can't be run with nohup

### 3. Quote Escaping Issues
Commands with single quotes weren't properly escaped when wrapped in `bash -c '...'`

## Solutions Implemented

### 1. Check for Existing Nohup/Background
```python
# Check if command already has nohup or is already backgrounded
if "nohup" in command_text or command_text.rstrip().endswith("&"):
    logger.info(f"[CodeAct] Command already has nohup or background operator, not wrapping")
    args["blocking"] = False
    args["thought"] = "Command already configured for background execution"
```

### 2. Proper Command Wrapping
```python
# Use bash -c to handle complex commands properly, escape single quotes
escaped_cmd = command_text.replace("'", "'\\''")
wrapped_command = f"mkdir -p {log_dir} && nohup bash -c '{escaped_cmd}' > {log_realtime} 2>&1 & ..."
```

### 3. Skip Wrapping for Special Commands
```python
# Don't wrap if already has script or nohup
if "script " in command_text or "nohup" in command_text:
    logger.info(f"[CodeAct] Command already has script/nohup, not wrapping")
```

## Implementation Details

### Non-Blocking Commands (lines 201-230)
1. Checks if command already has `nohup` or `&`
2. If not, wraps with `nohup bash -c`
3. Escapes single quotes properly
4. Creates log files with relative paths

### Blocking Commands (lines 231-252)
1. Checks if command already has `script` or `nohup`
2. If not, wraps with `script -q -c`
3. Escapes single quotes properly
4. Streams output to log files

## Benefits

✅ No more double `nohup` wrapping
✅ Commands with existing `nohup` work correctly
✅ Complex commands with quotes are handled properly
✅ Background commands actually run in background (no timeout)
✅ Cleaner command construction

## Testing

### Before Fix
```bash
# Would timeout after 120 seconds
nohup pip install pandas
# Result: nohup nohup pip install pandas (invalid)
```

### After Fix
```bash
# Runs in background immediately
nohup pip install pandas
# Result: Command not wrapped, runs as-is
```

### New Wrapping
```bash
# Command without nohup
pip install pandas
# Result: nohup bash -c 'pip install pandas' > logs/commands/xxx.realtime 2>&1 &
```

## Monitoring
```bash
# Check if background processes are running
ps aux | grep pip

# Monitor log files
tail -f /home/wuy/AI/uagent-workspace/*/logs/commands/*.realtime
```

## Files Modified
- `/backend/app/integrations/openhands_runtime.py` (lines 201-252)