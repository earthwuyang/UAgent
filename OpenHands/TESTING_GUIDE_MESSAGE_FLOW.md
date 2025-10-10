# Testing Guide - Message Flow and Research Auto-Start

## ✅ All Fixes Have Been Successfully Implemented

This guide will help you verify that all the message flow fixes and research auto-start mechanism are working correctly.

## 🎯 What Was Fixed

1. **Frontend Message Send Logging** - Track messages from browser console
2. **Backend Message Receipt Logging** - Track messages in backend logs
3. **Error Handling** - Show errors when messages fail to send
4. **Sending State Management** - Prevent duplicate sends and provide feedback
5. **Visual Feedback** - Disable input while message is sending

## 📝 Test Procedure

### Step 1: Prepare for Testing

Open two terminal windows side by side:
1. **Terminal 1**: Browser console (open DevTools in browser: F12 → Console tab)
2. **Terminal 2**: Backend logs
   ```bash
   tmux attach -t uagent-backend
   ```

### Step 2: Navigate to OpenHands

1. Open browser and go to: http://120.46.207.248:3000
2. Click "New Conversation" button
3. Wait for the conversation page to load (runtime will initialize)
4. Wait until the status shows "Waiting for task" (input will be enabled)

### Step 3: Send Research Goal Message

Type or paste this research goal message:

```
I want to design and implement a highly parallel, ML-based query routing system for PostgreSQL and DuckDB. The system should: 1) Parse incoming SQL queries to extract structural features (table names, join types, aggregations, subqueries, etc.) 2) Use a trained ML model (e.g., Random Forest, XGBoost, or a neural network) to predict whether each query is better suited for Postgres (OLTP) or DuckDB (OLAP) based on historical query performance 3) Route queries dynamically to the appropriate database engine 4) Log routing decisions and actual execution times to continuously improve the model 5) Provide a simple HTTP API for submitting queries and retrieving results 6) Include monitoring dashboards showing query volume, routing accuracy, and performance improvements Please break this down into 3-5 major research branches that can be explored in parallel by sub-agents, and coordinate their findings to produce a final implementation plan.
```

Press Enter or click Send button.

### Step 4: Verify Frontend Logs (Browser Console)

You should see these logs in the browser console:

```javascript
[MESSAGE_SEND] Starting to send message: {
  contentLength: 1095,
  imageCount: 0,
  fileCount: 0,
  timestamp: "2025-10-10T..."
}

[WS_SEND] Sending event via WebSocket { action: "message" }

[MESSAGE_SEND] Sending message via WebSocket: {
  action: "message",
  contentPreview: "I want to design and implement a highly parallel, ML-based query routing system for PostgreSQL...",
  imageUrlCount: 0,
  uploadedFileCount: 0
}

[WS_SEND] Event sent successfully

[MESSAGE_SEND] Message sent successfully
```

✅ **Pass Criteria**: All 5 log messages appear in correct order without errors.

### Step 5: Verify Backend Logs (tmux terminal)

Switch to the tmux terminal and look for these logs:

```python
[MESSAGE_RECEIVED] Dispatch called for conversation fd9028f2db4945a0aa65fb060ff4944a
[MESSAGE_RECEIVED] Event type: MessageAction
[MESSAGE_RECEIVED] Action: message
[MESSAGE_RECEIVED] Content preview: I want to design and implement a highly parallel, ML-based query routing system for PostgreSQL...

[RESEARCH_CHECK] Checking if research should trigger for fd9028f2db4945a0aa65fb060ff4944a
[RESEARCH_CHECK] Event type: MessageAction, already_checked: False
[RESEARCH_CHECK] MessageAction detected with content length: 1095

🔬 Research activated: scientific (85%)

[MIDDLEWARE] start_research called: session_id=fd9028f2db4945a0aa65fb060ff4944a
[MIDDLEWARE] Config: max_iterations=50, max_cost=10.0, max_parallel=3
✅ TreeSearchOrchestrator created for exp_fd9028f2db4945a0aa65fb060ff4944a_...
[RESEARCH_MIDDLEWARE] Background task created, returning experiment_id

[RESEARCH_TRIGGER] ✅ Research auto-started successfully
[RESEARCH_TRIGGER] Experiment ID: exp_fd9028f2db4945a0aa65fb060ff4944a_...
[RESEARCH_TRIGGER] Goal preview: I want to design and implement a highly parallel...
```

✅ **Pass Criteria**: Message received logs appear, followed by research trigger logs with experiment ID.

### Step 6: Verify Research Tree Panel

In the browser, look at the right panel labeled "Research Tree":

**Expected Changes:**
1. Connection status should change from "Disconnected" to "Connected" (green dot)
2. Nodes and Edges counts should start incrementing
3. The tree visualization area should populate with research nodes
4. Current Steps should show research activity (not "Idle")
5. Duration, Cost, and Tokens should start updating

✅ **Pass Criteria**: Research tree connects and shows active research with multiple nodes.

### Step 7: Monitor Research Progress

Watch the research tree panel for:
- **Root node**: Your main research goal
- **Idea nodes**: 3-5 parallel research branches (e.g., "SQL Query Parsing", "ML Model Selection", etc.)
- **Sub-nodes**: Hypotheses and experiments under each idea
- **Node expansion**: New nodes appearing as research progresses

✅ **Pass Criteria**: Multiple parallel research branches appear and expand over time.

## 🐛 Troubleshooting

### Issue: No logs in browser console

**Fix**: Make sure DevTools console is open (F12) and "All levels" is selected (not just "Errors").

### Issue: No logs in backend

**Fix**: Run `tmux capture-pane -t uagent-backend -p | grep MESSAGE_RECEIVED` to search for logs.

### Issue: Message not sent (input clears but no logs)

**Symptoms:**
- Input field clears
- No [MESSAGE_SEND] logs in console
- No [MESSAGE_RECEIVED] logs in backend
- Send button was disabled

**Fix**: This was the original bug we fixed! If this happens:
1. Check for error toast: "Message failed to send..."
2. Check console for [WS_SEND] error logs
3. Verify WebSocket connection status in DevTools → Network → WS tab

### Issue: Research doesn't auto-start

**Possible causes:**
1. **Message not complex enough**: Try a longer, more detailed research query
2. **Not first message**: Research only auto-triggers on the first message in a conversation
3. **Already classified**: Start a fresh conversation

**Fix**: Start a new conversation and try again with the full research goal text above.

### Issue: Research tree shows "Disconnected"

**Fix:**
1. Wait 30 seconds - connection can take time to establish
2. Check backend logs for WebSocket connection errors
3. Verify experiment ID appears in backend logs
4. Refresh the page if needed

## 📊 Success Indicators

Your test is successful when:

1. ✅ All frontend [MESSAGE_SEND] logs appear
2. ✅ All backend [MESSAGE_RECEIVED] logs appear
3. ✅ Research auto-starts without manual intervention
4. ✅ Research tree connects and shows green "Connected" status
5. ✅ Multiple parallel research branches appear as nodes
6. ✅ Research progresses with new nodes appearing over time

## 🎉 What This Proves

A successful test demonstrates:

- **Message Flow Works**: Messages travel from frontend to backend reliably
- **Logging Works**: Complete visibility into message flow for debugging
- **Error Handling Works**: Failures are caught and reported to user
- **Research Auto-Start Works**: Complex queries trigger parallel research automatically
- **No Manual Start Button Needed**: Research begins as soon as message is received

## 📸 Expected Visual Result

After sending the research goal, within 1-2 minutes you should see:

```
Research Tree Panel:
┌─────────────────────────────────────┐
│ 🟢 Connected | Duration: 00:45      │
│ Cost: $0.42 | Tokens: 15,234       │
├─────────────────────────────────────┤
│                                     │
│     [Root: ML Query Routing]        │
│            /    |    \              │
│          /      |      \            │
│    [Idea-1] [Idea-2] [Idea-3]     │
│    SQL      ML       System        │
│    Parsing  Models   Arch          │
│                                     │
│ Nodes: 15 | Edges: 14              │
│ Progress: 60% | 9/15 nodes         │
└─────────────────────────────────────┘
```

## 📄 Related Documents

- `MESSAGE_FLOW_FIXES_SUMMARY.md` - Complete implementation details
- `OpenHands/frontend/src/components/features/chat/chat-interface.tsx` - Frontend logging
- `OpenHands/openhands/server/session/session.py` - Backend logging

---

**Last Updated**: 2025-10-10  
**Status**: All fixes implemented and ready for testing  
**Next Step**: Run this test to verify everything works!
