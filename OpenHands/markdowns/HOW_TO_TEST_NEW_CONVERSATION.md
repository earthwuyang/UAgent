# ⚠️ IMPORTANT: How to Start a NEW Conversation

## The Issue

You are opening an **OLD conversation** (`0dc2d3335d9e482fb39365d0c0285083`). This is why:
1. The agent appears "stopped" (it was stopped in the old conversation)
2. Runtime connection errors (old runtime is dead)
3. Auto-trigger doesn't work (it only triggers on the FIRST message of a NEW conversation)

## Solution: Start a FRESH Conversation

### Option 1: Use the Homepage (Recommended)

1. **Close all existing conversation tabs**
2. Go to: **http://120.46.207.248:3000/**
3. Look for a **"New Conversation"** or **"+"** button
4. Click it to create a completely new conversation
5. Send your research prompt as the FIRST message

### Option 2: Clear Browser Data

If you can't find the "New Conversation" button:

1. **Clear browser cookies** for `120.46.207.248`
2. **Close all tabs** with OpenHands
3. Open a fresh tab: **http://120.46.207.248:3000/**
4. This should give you a new conversation

### Option 3: Use Incognito/Private Mode

1. Open an **incognito/private browser window**
2. Go to: **http://120.46.207.248:3000/**
3. Send your research prompt

### Option 4: Delete Old Conversations via API

```bash
# List all conversations
curl http://localhost:3000/api/conversations

# Delete the old conversation
curl -X DELETE http://localhost:3000/api/conversations/0dc2d3335d9e482fb39365d0c0285083
```

Then refresh the browser and you should get a new conversation.

## How to Identify a NEW Conversation

A NEW conversation will have:
- ✅ A different conversation ID (not `0dc2d3335d9e482fb39365d0c0285083`)
- ✅ Empty chat history
- ✅ Agent status: "Waiting" or "Ready"
- ✅ No previous messages

## What Happens in a NEW Conversation

When you send your research prompt as the FIRST message:

1. **Server receives message**
2. **LLM classifier analyzes it** (~1-2 seconds)
   ```
   LLM classified task as complex_research (confidence: 0.95)
   ```
3. **Research mode triggers**
   ```
   🔬 Research mode triggered
   [COORDINATOR] Middleware returned experiment_id: xxx
   ```
4. **System message appears**
   ```
   [System: Research mode activated - Experiment ID: xxx]
   ```
5. **Research Tree UI appears**
6. **Agent starts working on research**

## Monitor Logs for New Conversation

```bash
tmux attach -t uagent-backend
```

Watch for these messages when you send the FIRST message:
- `TaskClassifier initialized with model: dashscope/qwen3-coder-plus`
- `LLM classified task as complex_research (confidence: 0.95)`
- `🔬 Research mode triggered`
- `[COORDINATOR] Middleware returned experiment_id:`

## Why Old Conversations Don't Work

- ❌ Middleware only checks the **FIRST message**
- ❌ Old conversations already have message history
- ❌ Runtime containers from old sessions are dead
- ❌ Auto-trigger logic is bypassed

## Test Script

If the UI is confusing, you can create a new conversation via API:

```bash
# Create new conversation with research prompt
curl -X POST http://localhost:3000/api/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "CodeActAgent",
    "initial_user_message": "research goal: modify postgres and pg_duckdb source code..."
  }'

# This will return a new conversation_id
# Then open: http://120.46.207.248:3000/conversations/<new_conversation_id>
```

---

**KEY TAKEAWAY**: Stop reusing `0dc2d3335d9e482fb39365d0c0285083`! 

Start a **FRESH NEW CONVERSATION** to test the research auto-trigger! 🚀
