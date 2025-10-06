# UAgent Research Integration Flow - Complete Documentation

## 🎯 Integration Points

The UAgent research middleware is hooked at **TWO** points in OpenHands:

### 1. Initial Conversation Start (First Message)
**File**: `openhands/server/services/conversation_service.py`
**Function**: `start_conversation()`
**Line**: 165-206

### 2. Subsequent Messages (All Messages)
**File**: `openhands/server/session/session.py`
**Function**: `WebSession.dispatch()`
**Line**: 364-414

---

## 📊 Message Flow Diagram

### Flow for First Message (New Conversation)

```
User sends first message in new conversation
  ↓
Frontend → POST /api/conversations
  ↓
openhands/server/routes/manage_conversations.py
  ↓
conversation_service.create_new_conversation()
  ↓
conversation_service.start_conversation()
  ├─ Line 158-163: Create initial_message_action from user message
  ├─ Line 165-206: ✅ RESEARCH MIDDLEWARE HOOK #1
  │  ├─ Check if RESEARCH_MIDDLEWARE_AVAILABLE
  │  ├─ Call research_middleware.process_message()
  │  ├─ Classify message (complex_research vs simple)
  │  ├─ If should_trigger_research:
  │  │  ├─ Start TreeSearchOrchestrator in background
  │  │  ├─ Log experiment_id
  │  │  └─ Append research context to message
  │  └─ Continue even if research fails (non-blocking)
  └─ Line 208-214: Start agent loop with (possibly modified) message
      ↓
conversation_manager.maybe_start_agent_loop()
  ↓
Agent begins processing (normal conversation flow)
```

### Flow for Subsequent Messages (Existing Conversation)

```
User sends message in existing conversation
  ↓
Frontend WebSocket → oh_action event
  ↓
openhands/server/listen_socket.py:149 → oh_action()
  ↓
conversation_manager.send_to_event_stream(connection_id, data)
  ↓
openhands/server/conversation_manager/standalone_conversation_manager.py:369
  ↓
session.dispatch(data)
  ↓
openhands/server/session/session.py:364 → WebSession.dispatch()
  ├─ Line 365: Parse event from dict
  ├─ Line 367-398: ✅ RESEARCH MIDDLEWARE HOOK #2
  │  ├─ Check if RESEARCH_MIDDLEWARE_AVAILABLE
  │  ├─ Check if event is MessageAction with content
  │  ├─ Call research_middleware.process_message()
  │  ├─ Classify message (complex_research vs simple)
  │  ├─ If should_trigger_research:
  │  │  ├─ Start TreeSearchOrchestrator in background
  │  │  ├─ Log experiment_id
  │  │  └─ Append research context to message content
  │  └─ Continue even if research fails (non-blocking)
  ├─ Line 400-413: Image support validation
  └─ Line 414: Add event to agent_session.event_stream
      ↓
Agent processes message (normal conversation flow)
```

---

## 🔍 Detailed Code Walkthrough

### Hook #1: Initial Message

**Location**: `openhands/server/services/conversation_service.py:165-206`

```python
# Check if research mode should be triggered (non-blocking)
research_triggered = False
if RESEARCH_MIDDLEWARE_AVAILABLE and initial_user_msg:
    try:
        # Process research in a non-blocking way
        result = await research_middleware.process_message(
            user_message=initial_user_msg,
            session_id=conversation_id,
            conversation_metadata={
                'user_id': user_id,
                'repository': conversation_metadata.selected_repository,
                'branch': conversation_metadata.selected_branch,
            }
        )

        if result.get('should_trigger_research'):
            research_triggered = True
            experiment_id = result.get('experiment_id', 'N/A')
            logger.info(
                f"Research mode triggered for conversation {conversation_id}",
                extra={
                    'task_type': result.get('task_type'),
                    'confidence': result.get('confidence'),
                    'experiment_id': experiment_id,
                }
            )

            # Add research info to initial message for context
            if result.get('status') == 'research_started' and initial_message_action:
                research_context = (
                    f"\n\n[System: Research mode activated - "
                    f"Experiment ID: {experiment_id}, "
                    f"Task Type: {result.get('task_type')}, "
                    f"Confidence: {result.get('confidence', 0):.2f}. "
                    f"Check the Research Tree tab for progress.]"
                )
                initial_message_action.content += research_context
    except Exception as e:
        # Don't let research middleware errors block conversation startup
        logger.error(f"Failed to process research middleware: {str(e)}", exc_info=True)
        logger.info("Continuing with normal conversation despite research middleware error")
```

**Triggers on**: First message when creating a new conversation

### Hook #2: Subsequent Messages

**Location**: `openhands/server/session/session.py:364-414`

```python
async def dispatch(self, data: dict) -> None:
    event = event_from_dict(data.copy())

    # Check if research should be triggered for this message
    if RESEARCH_MIDDLEWARE_AVAILABLE and isinstance(event, MessageAction) and event.content:
        try:
            result = await research_middleware.process_message(
                user_message=event.content,
                session_id=self.sid,
                conversation_metadata={'source': 'subsequent_message'}
            )

            if result.get('should_trigger_research'):
                self.logger.info(
                    f"Research mode triggered for message in conversation {self.sid}",
                    extra={
                        'task_type': result.get('task_type'),
                        'confidence': result.get('confidence'),
                        'experiment_id': result.get('experiment_id'),
                    }
                )

                # Optionally append research context to the message
                if result.get('status') == 'research_started':
                    experiment_id = result.get('experiment_id', 'N/A')
                    research_info = (
                        f"\n\n[System: Research mode activated - "
                        f"Experiment ID: {experiment_id}, "
                        f"Confidence: {result.get('confidence', 0):.2f}. "
                        f"Check the Research Tree tab for progress.]"
                    )
                    event.content += research_info
        except Exception as e:
            self.logger.error(f"Failed to process research middleware: {str(e)}", exc_info=True)
            # Continue with normal message processing

    # ... rest of dispatch logic (image validation, event streaming)
    self.agent_session.event_stream.add_event(event, EventSource.USER)
```

**Triggers on**: Every message sent in an existing conversation

---

## 🔄 Research Middleware Processing

### `research_middleware.process_message()` Flow

**File**: `extensions/uagent_research/middleware/research_middleware.py:65-113`

```python
async def process_message(user_message, session_id, conversation_metadata):
    1. Check if auto_trigger is enabled
       ↓
    2. Call task_classifier.should_trigger_research()
       ↓
    3. task_classifier analyzes message:
       - Count research keywords (search, compare, benchmark, etc.)
       - Count complexity keywords (train, model, experiment, etc.)
       - Detect multi-stage tasks (first...then, multiple ands, etc.)
       - Calculate confidence score (0-1)
       ↓
    4. If should_trigger (confidence >= threshold):
       ↓
    5. Call start_research():
       - Create experiment_id
       - Initialize TreeSearchOrchestrator
       - Start research in background (async task)
       - Store orchestrator reference
       ↓
    6. Return result dict:
       {
           'mode': 'research',
           'should_trigger_research': True,
           'task_type': 'complex_research',
           'confidence': 0.95,
           'experiment_id': 'exp_...',
           'status': 'research_started'
       }
```

### Background Research Execution

```python
async def _run_research(experiment_id):
    1. Get orchestrator from active_orchestrators
       ↓
    2. Call orchestrator.run(goal, max_iterations)
       ↓
    3. TreeSearchOrchestrator executes PUCT search:
       - Generate initial ideas
       - Select best nodes to explore (PUCT scoring)
       - Execute nodes via adapters:
         * DeepResearch: Web search and browsing
         * RepoMaster: GitHub repository analysis
         * CodeAct: Code execution and experiments
       - Update tree with results
       - Publish events to WebSocket
       ↓
    4. Research completes or cancelled
       ↓
    5. Cleanup orchestrator from active_orchestrators
```

---

## 📁 Key Files Modified

### 1. `openhands/server/services/conversation_service.py`
- **Lines 33-43**: Import research_middleware
- **Lines 165-206**: Hook for initial messages

### 2. `openhands/server/session/session.py`
- **Lines 39-46**: Import research_middleware
- **Lines 367-398**: Hook for subsequent messages

### 3. `extensions/uagent_research/middleware/research_middleware.py`
- Middleware implementation
- process_message(), start_research(), _run_research()

### 4. `extensions/uagent_research/classifier/task_classifier.py`
- Task classification logic
- Pattern matching for research indicators

### 5. `extensions/uagent_research/config.py`
- Configuration (reads from .env)

---

## 🎛️ Configuration

### Environment Variables (in `/home/wuy/AI/UAgent/.env`)

```bash
ENABLE_AUTO_RESEARCH_TRIGGER=true       # Enable/disable
RESEARCH_CONFIDENCE_THRESHOLD=0.5       # Min confidence (0-1)
RESEARCH_MAX_ITERATIONS=9999999999      # Max iterations
RESEARCH_MAX_COST=9999999999            # Max LLM cost
RESEARCH_MAX_PARALLEL=3                 # Concurrent branches
```

---

## ✅ When Research Triggers

### Triggers on:
1. ✅ **First message** in a new conversation
2. ✅ **Any message** in an existing conversation
3. ✅ Only if `ENABLE_AUTO_RESEARCH_TRIGGER=true`
4. ✅ Only if confidence >= threshold (0.5 = 50%)

### Example Triggers:

| Message | Confidence | Triggers? |
|---------|-----------|-----------|
| "Research NAS and implement" | 0.90 | ✅ Yes |
| "Compare sorting algorithms" | 0.85 | ✅ Yes |
| "Modify postgres, extract features, train model..." | 0.90 | ✅ Yes |
| "First collect data, then train, finally deploy" | 0.80 | ✅ Yes |
| "Fix login bug" | 0.25 | ❌ No |
| "Add sum function" | 0.20 | ❌ No |

---

## 🔍 How to Verify Integration

### 1. Check Middleware Loads

```bash
# Restart server and look for:
Research middleware loaded successfully
```

### 2. Send Complex Query

Send: "Research neural architecture search and implement the best approach"

### 3. Check Server Logs

Should see:
```
INFO - Research mode triggered for conversation {id}
INFO - Task type: complex_research, confidence: 0.90
INFO - Research started successfully: experiment_id=exp_...
```

### 4. Check Research Tree Tab

- Open "Research Tree" tab in UI (top-right)
- See experiment_id and nodes
- Watch real-time updates

---

## 🎯 Summary

**Integration Status**: ✅ **COMPLETE**

- ✅ Hook #1: Initial messages (conversation startup)
- ✅ Hook #2: Subsequent messages (all messages)
- ✅ Non-blocking (doesn't prevent normal conversation)
- ✅ Error-tolerant (continues even if research fails)
- ✅ Configurable (via .env)
- ✅ Works for both new and existing conversations

**Research triggers on EVERY message (first and subsequent) when enabled!** 🚀

---

## 📝 Developer Notes

### Why Two Hooks?

1. **Hook #1** (conversation_service.py): Catches the initial message when creating a new conversation via the REST API.

2. **Hook #2** (session.py): Catches all subsequent messages sent via WebSocket in an active conversation.

Different code paths require different integration points!

### Non-Blocking Design

Both hooks use `try-except` blocks and don't raise exceptions. If research fails, the conversation continues normally. Research runs in background via `asyncio.create_task()`.

### Message Modification

When research triggers, a system message is appended to the user's message:
```
[System: Research mode activated - Experiment ID: exp_..., Confidence: 0.95. Check the Research Tree tab for progress.]
```

This informs both the agent and user that research is running.
