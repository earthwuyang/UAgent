# Research Auto-Trigger Issue

## Problem

The research middleware currently only checks messages at **conversation initialization**, not for messages sent after the conversation has started.

## Current Behavior

1. User creates new conversation (clicks "New Conversation")
2. `start_conversation()` is called with `initial_user_msg=None`
3. Middleware check is skipped (no initial message)
4. User sends research message
5. Message goes directly to the regular agent, bypassing middleware
6. Regular agent processes it as a normal task (creates task list)
7. Research tree remains disconnected

## Root Cause

In `/Users/wuy/Desktop/code/UAgent/OpenHands/openhands/server/services/conversation_service.py`:

```python
# Lines 169-220: Middleware only runs for initial_user_msg at conversation start
if RESEARCH_MIDDLEWARE_AVAILABLE and initial_user_msg:
    result = await research_middleware.process_message(
        user_message=initial_user_msg,
        session_id=conversation_id,
        ...
    )
```

The middleware is NOT integrated into the ongoing message flow after conversation initialization.

## Workaround 1: Start Conversation with Research Goal

Instead of:
1. Click "New Conversation"
2. Send research message

Do:
1. Click "New Conversation"  
2. **Immediately** paste your research goal before the page fully loads
3. The `initial_user_msg` will be set and trigger research

## Workaround 2: Manual API Trigger

You can manually start research via the API:

```bash
CONVERSATION_ID="<your-conversation-id>"
RESEARCH_GOAL="your research goal here"

curl --noproxy "*" -X POST http://localhost:2999/api/research/experiments/start \
  -H "Content-Type: application/json" \
  -d "{
    \"goal\": \"${RESEARCH_GOAL}\",
    \"session_id\": \"${CONVERSATION_ID}\",
    \"experiment_type\": \"code\",
    \"config\": {
      \"max_iterations\": 50,
      \"max_parallel\": 3
    }
  }"
```

## Proper Fix (TODO)

To properly fix this, we need to integrate the research middleware into the **ongoing** message handling, not just conversation initialization.

### Option 1: Intercept in Session Message Handler

Modify `/Users/wuy/Desktop/code/UAgent/OpenHands/openhands/server/session/session.py` to call the middleware before processing each `MessageAction`:

```python
async def on_event(self, event):
    if isinstance(event, MessageAction) and event.source == EventSource.USER:
        # Check if research should be triggered
        if RESEARCH_MIDDLEWARE_AVAILABLE:
            result = await research_middleware.process_message(
                user_message=event.content,
                session_id=self.session_id,
                conversation_metadata=...
            )
            
            if result.get('should_trigger_research'):
                # Start research without blocking the message
                # The regular agent can continue OR we can skip agent processing
                pass
```

### Option 2: WebSocket Message Interceptor

Add middleware at the WebSocket level in `/Users/wuy/Desktop/code/UAgent/OpenHands/openhands/server/listen_socket.py` to intercept all incoming user messages before they reach the agent.

### Option 3: Event Bus Integration

Integrate the middleware into the event bus so it automatically processes all `MessageAction` events with `source=USER`.

## Current Status

- ✅ Middleware exists and is functional
- ✅ Auto-trigger is enabled by default (`ENABLE_AUTO_RESEARCH_TRIGGER=true`)
- ✅ Classifier works correctly (detects "research goal:" prefix)
- ❌ Integration only at conversation start, not ongoing messages
- ✅ Manual API triggering works fine

## Testing

The current setup was successfully tested:
1. Server starts without database errors
2. Conversations can be created
3. Manual API trigger works and starts research
4. Research tree connects properly when experiment is running

The only missing piece is automatic detection of research goals in ongoing conversation messages.
