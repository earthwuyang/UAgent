# OpenHands Integration Complex Problem Statement for GPT-5-Pro

## Executive Summary
We are integrating OpenHands (an autonomous coding agent) into UAgent's scientific research engine. While basic integration works, we face critical issues with command execution, process management, and state tracking that prevent reliable operation.

## Current Architecture
```
UAgent Backend (FastAPI)
    ↓
OpenHands Runtime (Session Manager)
    ↓
CodeAct Agent (LLM-powered)
    ↓
Action Server (Docker Container)
    ↓
Workspace File System
```

## Core Problems

### Problem 1: Command Re-execution Without Success Detection
**Symptom**: pip install commands execute 12+ times despite packages being already installed
**Root Cause**: Background execution returns immediately with "Started background process" message, but CodeAct doesn't know if the command actually succeeded
**Impact**: Wastes tokens, time, and creates confusion in execution flow

### Problem 2: Malformed Command Wrapping
**Symptom**: Commands containing export statements or already using nohup get double-wrapped
**Example**:
```bash
# Original command from CodeAct:
export PIP_NO_INPUT=1 && pip install numpy

# Gets wrapped as:
nohup bash -c 'export PIP_NO_INPUT=1 && pip install numpy' > log 2>&1 &

# Sometimes double-wrapped:
nohup bash -c 'nohup pip install numpy &' > log 2>&1 &
```
**Impact**: Commands fail or behave unexpectedly

### Problem 3: Blocking vs Non-blocking Execution Confusion
**Symptom**: Long-running commands (pip install, docker operations) block other operations
**Current Approach**: Detect "long-running" commands and wrap with nohup
**Problems**:
- Detection logic is fragile (string matching)
- No reliable way to check if background process completed
- CodeAct doesn't understand async execution model
- Read/write operations timeout waiting for long commands

### Problem 4: State Synchronization Between Components
**Symptom**: CodeAct doesn't know the actual state of executed commands
**Details**:
- Background processes return immediately
- No feedback loop to inform CodeAct of completion
- Log files exist but CodeAct doesn't automatically check them
- Process PIDs are captured but not utilized

## Current Implementation Details

### Key Files:
1. **openhands_runtime.py**:
   - Handles command wrapping and execution
   - Detects long-running commands
   - Manages session and workspace

2. **codeact_runner.py**:
   - Runs the CodeAct agent loop
   - Processes LLM responses
   - Executes actions via runtime

3. **openhands_bridge.py**:
   - Bridge between UAgent and OpenHands
   - Session management
   - WebSocket communication

### Current Command Execution Flow:
```python
def execute_bash(self, command: str):
    # Detect if long-running
    if self._is_long_running_command(command):
        # Wrap with nohup
        wrapped = f"nohup bash -c '{command}' > log 2>&1 &"
        # Execute and return immediately
        return "Started background process"
    else:
        # Execute normally and wait
        return subprocess.run(command, capture_output=True)
```

## Attempted Solutions That Failed

1. **Timeout with Retry**: Added exponential backoff but doesn't solve root cause
2. **Unified Logging**: Created single commands.log but CodeAct doesn't read it
3. **Environment Variables**: Added OPENHANDS_ACTION_TIMEOUT but doesn't help with background processes
4. **Token Limit Increase**: Raised to 20000 but doesn't fix execution issues

## Requirements for Solution

### Must Have:
1. **Reliable Command Execution**: Each command executes exactly once with clear success/failure status
2. **Non-blocking for Long Operations**: pip install, docker, etc. shouldn't block other operations
3. **State Awareness**: CodeAct must know when background operations complete
4. **Clean Command Construction**: No double-wrapping or malformed commands

### Nice to Have:
1. **Real-time Progress Monitoring**: Stream output to logs as commands execute
2. **Automatic Retry on Failure**: With exponential backoff
3. **Process Management**: Track and manage background processes

## Specific Questions for GPT-5-Pro

### Q1: Execution Model Design
How should we design the execution model to handle both blocking and non-blocking commands while maintaining state consistency? Should we:
- Use a job queue with status tracking?
- Implement a callback mechanism?
- Use subprocess.Popen with proper polling?
- Something else?

### Q2: CodeAct Agent Adaptation
How can we make the CodeAct agent (which expects synchronous execution) work reliably with asynchronous operations? Options:
- Modify CodeAct's prompts to understand async operations?
- Create a synchronization layer that makes async look sync?
- Implement a state machine for command execution?

### Q3: Command Wrapping Strategy
What's the best approach for command wrapping that handles:
- Export statements
- Compound commands (&&, ||, ;)
- Already-wrapped commands (nohup, &)
- Shell built-ins vs external commands

### Q4: Success Detection for Background Processes
How should we reliably detect when a background process completes successfully? Consider:
- PID tracking and polling
- Output file monitoring
- Exit code capture for background processes
- Integration with CodeAct's execution loop

### Q5: State Management Architecture
What's the optimal architecture for managing state across:
- OpenHands runtime (Python)
- CodeAct agent (LLM)
- Action server (Docker)
- Workspace files

Should we use:
- Event-driven architecture with webhooks?
- Polling-based status checks?
- Shared state store (Redis/SQLite)?
- File-based state tracking?

## Example Problematic Sequence

Current behavior when CodeAct tries to install packages:
```
1. CodeAct: "pip install numpy pandas"
2. Runtime: Detects as long-running, wraps with nohup
3. Runtime: Returns "Started background process with PID 12345"
4. CodeAct: Sees success message, continues
5. CodeAct: "import numpy"
6. Runtime: Fails - numpy not installed yet
7. CodeAct: "pip install numpy" (tries again)
8. Runtime: Wraps again, returns immediately
9. [Repeats 12+ times]
```

Desired behavior:
```
1. CodeAct: "pip install numpy pandas"
2. Runtime: Starts background process, tracks it
3. Runtime: Returns "Installing packages (job-123)..."
4. CodeAct: "check_job job-123" (or automatic)
5. Runtime: "Still running... (30s elapsed)"
6. CodeAct: Waits or does other work
7. CodeAct: "check_job job-123"
8. Runtime: "Completed successfully"
9. CodeAct: "import numpy" - works!
```

## Constraints

1. **Cannot modify OpenHands core**: We can only modify our integration layer
2. **Must maintain CodeAct compatibility**: The agent expects certain action formats
3. **Docker environment**: Commands run inside Docker container
4. **Token limits**: LLM has context limits, can't send huge logs
5. **Real-time requirement**: User expects to see progress in real-time

## Success Criteria

A successful solution will:
1. Execute each command exactly once
2. Provide accurate status for all operations
3. Allow long operations without blocking
4. Give CodeAct visibility into background operations
5. Handle all command types correctly (simple, compound, wrapped)
6. Maintain clear logs for debugging
7. Work reliably across 100+ command sequences

## Additional Context

- Using DashScope Qwen for LLM
- Docker container for execution isolation
- WebSocket for real-time communication
- Python 3.11+ environment
- Ubuntu Linux host system

Please provide a detailed implementation plan that addresses these issues systematically, with specific code examples and architecture decisions.