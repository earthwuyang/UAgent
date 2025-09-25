# Unified Command Logging Implementation

## Overview
All OpenHands commands now log to a single unified file instead of creating separate log files for each command.

## Benefits
✅ **Single source of truth** - All command output in one place
✅ **Easy monitoring** - Just tail one file
✅ **Chronological order** - See command execution sequence
✅ **Less file clutter** - No more hundreds of small log files
✅ **Better for debugging** - See full context of what happened

## Implementation

### Log Files Created
1. **`logs/commands.log`** - All command output with timestamps
2. **`logs/background_pids.txt`** - PIDs of background processes

### Command Wrapping

#### Non-blocking (Background) Commands
```bash
# Before each command, adds timestamp header
echo '[2025-09-25 17:00:00] Starting: pip install pandas' >> logs/commands.log

# Runs command in background, appending output
nohup bash -c 'pip install pandas' >> logs/commands.log 2>&1 &

# Saves PID for tracking
echo $! >> logs/background_pids.txt
```

#### Blocking Commands
```bash
# Adds timestamp header
echo '[2025-09-25 17:00:00] Starting (blocking): python train.py' >> logs/commands.log

# Uses script to append output with terminal control sequences preserved
script -q -a logs/commands.log -c 'python train.py'
```

## Log Format

```
[2025-09-25 16:55:00] Starting: pip install pandas
Collecting pandas
  Downloading pandas-2.1.4-cp39-cp39-linux_x86_64.whl (12.3 MB)
     |████████████████████████████████| 12.3 MB 5.1 MB/s
Successfully installed pandas-2.1.4

[2025-09-25 16:55:30] Starting (blocking): python train.py
Training model...
Epoch 1/10: loss=0.523
Epoch 2/10: loss=0.412
...

[2025-09-25 16:56:00] Starting: npm install express
added 57 packages in 3.421s
```

## Monitoring Commands

### View the unified log
```bash
# Real-time monitoring
tail -f /home/wuy/AI/uagent-workspace/*/logs/commands.log

# View last 50 lines
tail -n 50 /home/wuy/AI/uagent-workspace/*/logs/commands.log

# Use the helper script
./list_openhands_logs.sh
```

### Search for specific commands
```bash
# Find all pip installs
grep "pip install" logs/commands.log

# Find errors
grep -E "ERROR|FAIL|error|failed" logs/commands.log

# Find specific timestamp
grep "2025-09-25 16:55" logs/commands.log
```

### Check background processes
```bash
# View all PIDs
cat logs/background_pids.txt

# Check if processes are still running
while read pid; do
  ps -p $pid > /dev/null && echo "$pid is running" || echo "$pid finished"
done < logs/background_pids.txt
```

## Files Modified

### `/backend/app/integrations/openhands_runtime.py`

#### Lines 222-237: Export command handling
```python
# Use unified log file
log_dir = "logs"
log_file = f"{log_dir}/commands.log"
pid_file = f"{log_dir}/background_pids.txt"

# Append to unified log with timestamp and command info
log_header = f"echo '[{timestamp}] Starting: {actual_cmd[:100]}' >> {log_file}"
wrapped_command = f"mkdir -p {log_dir} && {log_header} && {export_part} && nohup bash -c '{escaped_cmd}' >> {log_file} 2>&1 & echo $! >> {pid_file}"
```

#### Lines 247-261: Non-blocking commands
```python
# Same unified log approach
log_header = f"echo '[{timestamp}] Starting: {command_text[:100]}' >> {log_file}"
wrapped_command = f"mkdir -p {log_dir} && {log_header} && nohup bash -c '{escaped_cmd}' >> {log_file} 2>&1 & echo $! >> {pid_file}"
```

#### Lines 268-283: Blocking commands
```python
# Use script -a to append to log
log_header = f"echo '[{timestamp}] Starting (blocking): {command_text[:100]}' >> {log_file}"
wrapped_command = f"mkdir -p {log_dir} && {log_header} && script -q -a {log_file} -c '{escaped_cmd}'"
```

### `/list_openhands_logs.sh`
Updated to show unified log instead of listing individual files.

## Advantages Over Previous Implementation

### Before (Separate Files)
```
logs/commands/
├── 20250925_165133_pip_install.realtime
├── 20250925_165133_pip_install.pid
├── 20250925_165135_python_train.realtime
├── 20250925_165135_python_train.pid
├── 20250925_165138_npm_install.realtime
├── 20250925_165138_npm_install.pid
... (hundreds of files)
```

### After (Unified)
```
logs/
├── commands.log         # All output here
└── background_pids.txt  # All PIDs here
```

## Usage Example

```bash
# Start the backend
python -m backend.app.main

# In another terminal, monitor the unified log
tail -f /home/wuy/AI/uagent-workspace/*/logs/commands.log

# Run commands through UAgent
# All output appears in the single log file with timestamps

# Check for errors
grep ERROR /home/wuy/AI/uagent-workspace/*/logs/commands.log

# See command sequence
grep "Starting:" /home/wuy/AI/uagent-workspace/*/logs/commands.log
```

## Notes

1. **Timestamps**: Each command is prefixed with `[YYYY-MM-DD HH:MM:SS]` for easy tracking
2. **Background vs Blocking**: Blocking commands show "(blocking)" in their header
3. **Append mode**: All commands append to the same file, preserving history
4. **Script command**: Used for blocking commands to preserve terminal output (colors, progress bars)
5. **PID tracking**: All background process PIDs saved to `background_pids.txt`