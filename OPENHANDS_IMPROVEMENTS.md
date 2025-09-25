# OpenHands Runtime Improvements

## Overview
Enhanced the OpenHands runtime to handle long-running commands (especially pip install) with adaptive retry, increasing timeouts, and backend fallback execution.

## Key Improvements

### 1. Adaptive Timeout Retry
- **Problem**: Commands like `pip install` timeout after 2-3 minutes
- **Solution**: Implemented adaptive retry with exponentially increasing timeouts
- **Configuration**:
  - `OPENHANDS_ACTION_TIMEOUT=120` - Initial timeout (2 minutes)
  - `OPENHANDS_MAX_ACTION_TIMEOUT=900` - Maximum timeout (15 minutes)
  - `OPENHANDS_RUN_ADAPTIVE_MULTIPLIER=1.75` - Timeout multiplier per retry
  - `OPENHANDS_RUN_MAX_ATTEMPTS=3` - Maximum retry attempts
  - `OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT=600` - Minimum timeout for package managers (10 minutes)

### 2. Backend Fallback Execution
- **Problem**: OpenHands action server sometimes doesn't execute commands
- **Solution**: When action server times out, execute command directly via subprocess
- **Features**:
  - Logs output to `workspace/logs/backend_fallback_*.log`
  - Periodic progress reporting every 5 seconds
  - Full stdout/stderr capture

### 3. Non-Interactive Flags
- **Problem**: Package managers wait for user input causing timeouts
- **Solution**: Automatically add non-interactive flags on first timeout
- **Supported**:
  - `pip install --no-input`
  - `apt-get install -y`
  - `npm install --yes`
  - `conda install -y`
  - And more...

### 4. Background Execution (Planned)
- **Problem**: Very long-running commands block the agent
- **Solution**: Run commands with `nohup` and monitor logs
- **Usage**: Agent can suggest:
  ```bash
  nohup pip install large-package > install.log 2>&1 &
  tail -f install.log  # Monitor progress
  ```

## How It Works

### Execution Flow
1. **Initial Attempt**: Run command with base timeout (2-10 minutes depending on command type)
2. **First Timeout**: Add non-interactive flags and retry
3. **Second Timeout**: Increase timeout by 1.75x and retry
4. **Third Timeout**: Fall back to backend subprocess execution
5. **Final Failure**: Return detailed error with suggestions

### Example Scenario: pip install
```
Attempt 1: pip install pandas numpy scikit-learn (timeout: 600s)
  ↓ Timeout
Attempt 2: pip install --no-input pandas numpy scikit-learn (timeout: 600s)
  ↓ Timeout
Attempt 3: Same command (timeout: 900s - max limit)
  ↓ Timeout
Backend Fallback: Execute via subprocess with logging
```

## Usage

### Environment Variables
Add to `.env`:
```env
OPENHANDS_ACTION_TIMEOUT=120
OPENHANDS_MAX_ACTION_TIMEOUT=900
OPENHANDS_RUN_ADAPTIVE_MULTIPLIER=1.75
OPENHANDS_RUN_MAX_ATTEMPTS=3
OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT=600
```

### Monitoring
- Check logs in `workspace/logs/` directory
- Backend fallback logs: `backend_fallback_*.log`
- Progress is reported in real-time to the agent

### Testing
Run the test suite:
```bash
python test_adaptive_retry.py
```

## Benefits
1. **Robustness**: Commands that previously failed now succeed with retries
2. **Visibility**: Detailed logging and progress reporting
3. **Flexibility**: Configurable timeouts via environment variables
4. **Fallback**: Backend execution ensures commands run even if action server fails
5. **Intelligence**: Automatic detection and handling of package manager commands

## Files Modified
- `/backend/app/integrations/openhands_runtime.py` - Main runtime improvements
- `.env` - Added configuration variables
- New test files for validation

## Future Enhancements
1. Implement full background execution with job management
2. Add command-specific timeout profiles
3. Implement intelligent command splitting for very long operations
4. Add health checks for action server reliability