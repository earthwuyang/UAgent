# Fixes for Issues #23 and #24

## Issue #23: Parallel research tree nodes remain in PENDING state

### Root Cause Analysis
After deep code analysis, the issue appears to be a combination of factors:

1. **Adapter Initialization**: Adapters may be registered but not properly configured with required dependencies (LLM, config, etc.)
2. **Silent Failures**: If adapter.run() fails or hangs, nodes stay in RUNNING/PENDING state without clear error messages
3. **Event Completion**: Nodes only transition to COMPLETE when they receive EventType.COMPLETE from adapters

### Fix Strategy

1. **Add timeout protection** for node execution
2. **Enhance error logging** in adapter execution path
3. **Add fallback behavior** when adapters fail
4. **Ensure adapters emit COMPLETE events**

## Issue #24: Research tree node click doesn't switch left chat UI to node context

### Root Cause Analysis

The frontend code at `ResearchTreeView.tsx` line 248 calls:
```typescript
const conversationId = extractConversationId(metadata);
```

However, child nodes (IDEA, HYPOTHESIS, etc.) don't have their own `conversationId` in metadata because they're not associated with separate conversations. Only the root node has a conversation ID.

### Fix Strategy

Option A (Simple): Each node shares the same conversation but displays its own execution events
Option B (Complex): Create separate conversations for each node that gets executed

We'll implement Option A first as it's simpler and more practical.

## Implementation

See the following files for detailed fixes:
- orchestrator_fixes.py - Fixes for issue #23
- frontend_fixes.tsx - Fixes for issue #24
