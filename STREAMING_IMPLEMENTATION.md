# OpenHands Real-Time Streaming Implementation

## Overview
All OpenHands commands now automatically stream their output to log files in real-time, allowing you to monitor command execution progress as it happens.

## Key Features

### 1. Automatic Log Creation
- Every OpenHands command creates a log file in `workspace/logs/commands/`
- Log files are named with timestamp and command preview for easy identification
- Example: `20250925_143022_run_pip_install_pandas.log`

### 2. Real-Time Streaming
For long-running commands (pip install, npm install, docker, etc.), output is automatically streamed to `.realtime` files:
- Uses `script` command for full output capture including progress bars
- Preserves ANSI colors and formatting
- Captures both stdout and stderr

### 3. Monitoring Commands
```bash
# List all recent logs
./list_openhands_logs.sh

# Monitor latest pip install
tail -f /home/wuy/AI/uagent-workspace/*/logs/commands/*pip*.realtime

# Watch specific log with colors
tail -f <log_file> | grep --color=always -E "ERROR|SUCCESS|installing|"
```

### 4. Log File Structure
Each log contains:
- Header with timestamp, action, command
- CONTEXT section with full arguments
- OUTPUT/STDOUT/STDERR sections with actual output
- STATUS section with exit codes
- FULL_RESPONSE with complete JSON response

## Implementation Details

### Automatic Wrapping
Commands are automatically wrapped based on type:
- **Long-running commands** (pip, npm, docker): Use `script -q -c 'command' logfile.realtime`
- **Simple commands**: Use `tee` to duplicate output to log
- **Complex commands** (with pipes/redirects): Log without wrapping

### Log Locations
```
workspace/
└── logs/
    └── commands/
        ├── 20250925_143022_run_pip_install.log          # Main log
        ├── 20250925_143022_run_pip_install.log.realtime # Real-time stream
        └── 20250925_143030_run_python_test.log
```

### Environment Variables
- `OPENHANDS_ACTION_TIMEOUT=120` - Default timeout
- `OPENHANDS_MAX_ACTION_TIMEOUT=900` - Maximum timeout with retries
- `OPENHANDS_RUN_ADAPTIVE_MULTIPLIER=1.75` - Timeout increase factor
- `OPENHANDS_RUN_MAX_ATTEMPTS=3` - Retry attempts
- `OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT=600` - Minimum for package managers

## Usage Examples

### Monitor pip install in real-time
```bash
# Start the UAgent backend
cd /home/wuy/AI/UAgent
python -m backend.app.main

# In another terminal, monitor logs
watch -n 1 'ls -lt /home/wuy/AI/uagent-workspace/*/logs/commands/*.realtime | head -5'

# When pip install starts, tail the log
tail -f /home/wuy/AI/uagent-workspace/*/logs/commands/*pip*.realtime
```

### Check for errors
```bash
grep -r "ERROR\|FAIL\|timeout" /home/wuy/AI/uagent-workspace/*/logs/commands/
```

### View latest commands
```bash
./list_openhands_logs.sh
```

## Benefits
1. **Real-time visibility** - See exactly what's happening during long operations
2. **Progress tracking** - Monitor download/installation progress
3. **Error detection** - Quickly identify where commands fail
4. **Performance analysis** - Review timing and resource usage
5. **Debugging** - Complete command history with inputs/outputs

## Files Modified
- `/backend/app/integrations/openhands_runtime.py` - Core streaming implementation
- `/backend/app/integrations/stream_monitor.py` - Streaming utilities
- `/list_openhands_logs.sh` - Helper script to list logs
- `.env` - Configuration variables

## No Additional Monitoring Needed
The streaming happens automatically - no separate monitor process required. Just run your OpenHands commands normally and check the logs at:
- Main logs: `workspace/logs/commands/*.log`
- Real-time streams: `workspace/logs/commands/*.realtime`