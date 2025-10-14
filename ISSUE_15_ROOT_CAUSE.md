# GitHub Issue #15 Root Cause Analysis

## Issue Summary
Research conversations showing "Stopped" status when runtime container exits.

## Root Cause
The containers are NOT crashing or failing - they are being **deliberately killed** by OpenHands' conversation limit system.

### Key Findings

1. **Max Concurrent Conversations Limit**
   - Default limit: 3 conversations per user
   - Location: `OpenHands/openhands/core/config/openhands_config.py:112-114`
   ```python
   max_concurrent_conversations: int = Field(
       default=3
   )  # Maximum number of concurrent agent loops allowed per user
   ```

2. **Automatic Cleanup Mechanism**
   - Location: `OpenHands/openhands/server/conversation_manager/standalone_conversation_manager.py:307-338`
   - When a new conversation starts and the limit is exceeded, the system:
     - Identifies the oldest conversations
     - Sends a "TOO_MANY_CONVERSATIONS" error message to the client
     - Calls `close_session()` to kill the container

3. **Exit Code Evidence**
   - All exited containers show exit code **137** (SIGKILL)
   - This confirms they were killed by the system, not crashed
   - Command to verify:
     ```bash
     docker ps -a --filter "name=openhands-runtime-" --format "{{.Names}}\t{{.Status}}"
     ```

4. **Timing Correlation**
   - Backend logs show:
     ```
     21:09:46 - Runtime initialization timed out or failed, but continuing anyway
     21:09:46 - Initial environment setup failed: [Errno 54] Connection reset by peer
     ```
   - However, the container actually exited at 21:15:22 (about 6 minutes later)
   - The "Stopped" status appears when the container is killed, not when initialization fails

## Solution

The fix for issue #15 should focus on **one or more** of the following approaches:

### Option 1: Increase the Default Limit
Modify `openhands/core/config/openhands_config.py`:
```python
max_concurrent_conversations: int = Field(
    default=10  # or higher
)
```

### Option 2: Make Configuration More Visible
- Add the setting to `config.template.toml` with a clear comment
- Document the setting in user-facing documentation
- Show a warning in the UI when approaching the limit

### Option 3: Improve User Communication
- Currently, the "TOO_MANY_CONVERSATIONS" error is sent to the **oldest** conversation being killed
- The **new** conversation that triggered the limit doesn't get any feedback
- Consider showing a warning to the user starting the new conversation

### Option 4: Smarter Cleanup Strategy
- Instead of immediately killing conversations, consider:
  - Warning the user first
  - Allowing graceful shutdown
  - Preserving conversation state before killing

## Testing the Fix

To verify the fix works:

1. Set `max_concurrent_conversations` to a higher value (e.g., 10)
2. Start multiple research conversations
3. Verify that containers are not killed when starting new ones (up to the new limit)
4. Monitor Docker containers: `docker ps -a --filter "name=openhands-runtime-"`

## Related Code Locations

- Container limit definition: `OpenHands/openhands/core/config/openhands_config.py:112-114`
- Cleanup logic: `OpenHands/openhands/server/conversation_manager/standalone_conversation_manager.py:307-338`
- Similar logic in Docker nested manager: `OpenHands/openhands/server/conversation_manager/docker_nested_conversation_manager.py:500-533`

## Conclusion

The issue description in #15 mentioned "automatic restart" and "readiness polling" as implemented features. However, the actual problem is that containers are being killed due to conversation limits, not failing to start. The solution is to adjust the conversation limit or improve how the system handles this limit.
