# Linear Issue Template - Ready to Create

Use this template to create an issue in Linear manually or via Linear MCP with proper authentication.

---

## Issue Details

**Title:** Critical: Message Send Handler Not Executing - Research Auto-Start Blocked

**Priority:** Urgent (P0)

**Labels:** bug, critical, frontend, backend, websocket, research

**Team:** Engineering

**Description:**

### Problem Summary
Research goal messages typed by users are NOT reaching the backend, preventing the research auto-start feature from working. Comprehensive Puppeteer testing reveals that the send handler is completely bypassed.

### Impact
- 🔴 **Severity**: CRITICAL (P0)
- Research auto-start feature completely non-functional
- Users cannot trigger research workflows
- Blocks entire parallel research capability
- Silent failure - no error message to user

### Current Status
**BLOCKED** - Messages typed in chat interface never reach backend

**Evidence:**
- ✗ NO `[MESSAGE_SEND]` logs in browser console
- ✗ NO `[MESSAGE_RECEIVED]` logs in backend
- ✗ NO WebSocket traffic detected
- ✓ Input field clears (some code runs)
- ✗ Research tree never initializes

### Root Cause Analysis
**Hypothesis**: Form submission bypassing React event handlers

The send handler is being bypassed, likely because:
1. Button has `type="submit"` triggering native form submission
2. Native submission bypasses React onClick handlers
3. This clears input but doesn't execute custom WebSocket send code

**Evidence supporting this**:
- Input clearing suggests default behavior executes
- Zero logging indicates custom handler never runs
- Button found with `type="submit"` in tests

### Technical Details

**Files Modified:**

Frontend:
- `frontend/src/components/features/chat/chat-interface.tsx` (lines 51, 122-184)
  - Added `[MESSAGE_SEND]` logging
  - Added `isSendingMessage` state management
  - Added error handling for WebSocket failures

- `frontend/src/components/features/chat/interactive-chat-box.tsx`
  - Added sending state UI feedback
  - Fixed syntax errors

Backend:
- `openhands/server/listen_socket.py` (lines 145-146)
  - Added `[MESSAGE_RECEIVED]` and `[WS_RECEIVE]` logging

Infrastructure:
- `/home/wuy/AI/UAgent/start_openhands_research.sh`
  - Updated to use OpenHands from source via PYTHONPATH

**Test Results:**
```
Puppeteer Automated Test @ http://120.46.207.248:3000

✓ Page loaded successfully
✓ WebSocket connection established
✓ New Conversation button clicked
✓ Chat input found and enabled
✓ Message typed: "Research the latest developments in quantum computing"
✓ Send button found (type=submit)
✓ Button click executed
✓ Input field cleared

✗ NO frontend logs appeared
✗ NO backend logs appeared
✗ NO WebSocket message sent
✗ Research tree never initialized
```

**Backend Logs:**
```
18:02:48 - Successfully joined conversation with connection_id
18:03:11 - sio:disconnect (connection closed)
```
No message logs between connection and disconnect.

### Artifacts Created
1. `/home/wuy/AI/UAgent/ISSUE_REPORT_MESSAGE_FLOW_BLOCKED.md` - Detailed technical report
2. `/home/wuy/AI/UAgent/PROGRESS_SUMMARY.md` - Progress tracking
3. `/tmp/PUPPETEER_TEST_RESULTS.md` - Full test analysis
4. `/tmp/test_message_flow.js` - Reusable test script
5. `/tmp/puppeteer_test_result.png` - UI screenshot

### Next Steps Required

**Priority 1: Debug Message Send**
1. Manual browser testing with DevTools
2. Verify correct build is being served
3. Inspect actual component rendering
4. Check form submission flow
5. Add logging to ALL possible send paths

**Priority 2: Fix Root Cause**
- Ensure `preventDefault()` called on form submit
- Verify React handlers properly attached
- Check component hierarchy
- Clear all caches (browser, service worker)

**Priority 3: Verify Fix**
- Re-run Puppeteer tests
- Confirm `[MESSAGE_SEND]` logs appear
- Confirm backend receives messages
- Confirm research auto-start triggers

### Investigation Questions
- [ ] Is the rebuilt frontend actually being served?
- [ ] Is InteractiveChatBox the component rendering in production?
- [ ] Are error boundaries catching exceptions?
- [ ] Is WebSocket client properly initialized?
- [ ] Does Enter key vs button click behave differently?
- [ ] Is preventDefault() being called?
- [ ] Is form element wrapping the button?

### Environment
- **Server**: http://120.46.207.248:3000
- **Backend Session**: tmux uagent-backend-363
- **Frontend Build**: /home/wuy/AI/UAgent/OpenHands/frontend/build
- **Python**: 3.12 with Poetry virtualenv
- **Node.js**: v22.11.0

### Reproduction Commands
```bash
# Run automated test
cd /tmp && node test_message_flow.js

# Check backend logs
tmux attach -t uagent-backend-363

# Rebuild frontend
cd /home/wuy/AI/UAgent/OpenHands/frontend && npm run build

# Restart backend
cd /home/wuy/AI/UAgent && ./start_openhands_research.sh
```

### Related Work
- Previous debugging resolved WebSocket connection issues
- Frontend syntax errors fixed
- Missing React state declarations added
- Backend Google Cloud dependencies installed
- Start script updated to use source code

### Affected Features
- Research tree visualization
- Multi-agent coordination
- Automated research pipeline
- Message classification for research triggers
- WebSocket message flow for user actions

### User Impact
- Users can type messages
- Messages appear to send (input clears)
- But NO backend processing occurs
- Silent failure with no error feedback
- Confusing UX - looks functional but isn't

---

**Recommendation**: Manual browser testing with DevTools is essential to understand the actual component structure and event flow in the running application.

**Estimated Fix Time**: 2-4 hours (once root cause confirmed)

**Blocking**: All research functionality

---

## Quick Reference

**System**: OpenHands Research Auto-Start
**Component**: Frontend Chat Interface → Backend WebSocket Handler
**Error Type**: Silent failure - handler not executing
**Priority**: P0 CRITICAL
**Status**: BLOCKED

**Key Finding**: The send handler code exists and frontend was rebuilt, but logging never appears, indicating the handler is never called. This suggests a fundamental issue with how form/button submission is wired in the React component tree.

