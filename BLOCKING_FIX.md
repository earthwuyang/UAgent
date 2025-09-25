# Fix for OpenHands Blocking Issues

## Problem
Read operations and other simple commands were timing out when a long-running command (like `pip install`) was executing, suggesting the action server was processing commands sequentially.

## Root Cause
The OpenHands action server appears to process commands in sequence. When a long-running blocking command is executing, it prevents other operations from being processed, causing timeouts.

## Implementation Status: ✅ COMPLETE

## Solution

### 1. Automatic Non-Blocking for Long Commands
Long-running commands are now automatically set to non-blocking mode:

```python
long_running_patterns = [
    "pip install", "pip3 install", "npm install", "yarn install",
    "apt-get", "apt install", "yum install", "brew install",
    "docker build", "docker pull", "make", "cmake",
    "wget", "curl -O", "git clone", "sleep"
]
```

When detected, these commands are automatically:
- Set to `blocking: false` to return immediately
- Wrapped with `nohup` for background execution
- Output streamed to `.realtime` log files in `/workspace/logs/commands/`
- Process ID saved to `.pid` file for monitoring

For blocking long-running commands:
- Wrapped with `script -q -c` to capture all output including progress bars
- Output streamed to `.realtime` files for real-time monitoring
- Full terminal output preserved including ANSI colors

### 2. Optimized Timeouts by Operation Type

| Operation | HTTP Timeout | Reason |
|-----------|-------------|---------|
| `read` | 10 seconds | Should be instant |
| `write/edit` | 30 seconds | File ops are fast |
| `run` (non-blocking) | 15 seconds | Returns immediately |
| `run` (blocking) | up to 300 seconds | May take time |

### 3. Background Execution with Logging
Non-blocking commands are wrapped as:
```bash
nohup {command} > {log}.realtime 2>&1 &
echo $! > {log}.pid
```

This ensures:
- Command runs in background
- Output streams to log file
- Process ID is saved for monitoring
- Other operations aren't blocked

## How It Works

### Before (Blocking)
```
1. pip install pandas  [blocking: true]  → Runs for 5 minutes
2. read file.txt                         → Timeout after 30s (blocked)
3. write result.txt                      → Timeout (blocked)
```

### After (Non-blocking)
```
1. pip install pandas  [blocking: false] → Returns immediately (PID: 12345)
2. read file.txt                         → Executes immediately
3. write result.txt                      → Executes immediately
   Monitor: tail -f logs/commands/*.realtime
```

## Monitoring Background Commands

### Check running background processes:
```bash
# See all background process PIDs
ls /home/wuy/AI/uagent-workspace/*/logs/commands/*.pid

# Check if process is still running
ps -p $(cat /path/to/log.pid)

# Monitor output
tail -f /path/to/log.realtime
```

### List recent logs:
```bash
./list_openhands_logs.sh
```

## Benefits
1. **No more blocking** - Long commands don't block short operations
2. **Better parallelism** - Multiple operations can run concurrently
3. **Real-time monitoring** - All output streams to log files
4. **Automatic detection** - No manual configuration needed
5. **Faster operations** - Read/write operations complete quickly

## Files Modified
- `/backend/app/integrations/openhands_runtime.py`
  - Auto-detect long-running commands (lines 190-197)
  - Set non-blocking mode for long commands (lines 201-206)
  - Wrap commands with nohup for background execution (lines 208-217)
  - Wrap blocking long commands with script for streaming (lines 219-228)
  - Optimize HTTP timeouts by operation type (lines 801-823)
  - Backend fallback execution with streaming (lines 630-726)

## Testing
Run a long pip install and verify other operations work:
```python
# This will run in background
action: run
command: pip install tensorflow

# These will execute immediately
action: read
path: /some/file.txt

action: write
path: /output.txt
content: "test"
```