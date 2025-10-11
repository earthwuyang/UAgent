# Bug Fix: React Error #185 - Maximum Update Depth Exceeded

## Issue Summary
React Error #185 ("Maximum update depth exceeded") was occurring on conversation pages when research mode was active, causing the application to crash and display an error screen.

## Root Cause
The error was caused by **duplicate WebSocket connections** being created for the same research experiment:

1. **`ResearchTreePanel.tsx`** (line 104) - Created a WebSocket connection via `useResearchWS` hook
2. **`research-tab.tsx`** (line 219) - Created **another** WebSocket connection for the same experimentId

### The Problem Chain:
1. Both components rendered simultaneously on the conversation page
2. Both tried to establish WebSocket connections to the same research experiment endpoint
3. When WebSocket connections failed (404/network errors), both hooks triggered reconnection logic
4. This created a **rapid reconnection loop** with exponential backoff
5. Each reconnection attempt triggered React state updates in both hooks
6. The simultaneous state updates from multiple sources caused React to hit its maximum update depth limit
7. Result: React Error #185 and application crash

### Evidence from Console Logs:
```
[Research WS] Connecting to ws://localhost:2999/api/research/ws/experiment/exp_678480eef39d43c...
[Research WS] Connecting to ws://localhost:2999/api/research/ws/experiment/exp_678480eef39d43c...
[Research WS] Connecting to ws://localhost:2999/api/research/ws/experiment/exp_678480eef39d43c...
... (repeated 50+ times)
Error: Minified React error #185; visit https://react.dev/errors/185
```

## Solution
Removed the duplicate WebSocket connection in `research-tab.tsx` and replaced it with a direct read from the Zustand store, which is already being updated by the `ResearchTreePanel` component's WebSocket connection.

### Changed Files:
- **`OpenHands/frontend/src/routes/research-tab.tsx`**

### Changes Made:

**Before:**
```typescript
import { useResearchWS } from "#/hooks/useResearchWS";

// ... inside component ...

const { isConnected } = useResearchWS({
  experimentId: experimentId ?? "",
  autoConnect: Boolean(experimentId && experimentStatus !== "idle"),
});
```

**After:**
```typescript
// Removed import of useResearchWS

// ... inside component ...

// Get connection status from store instead of creating duplicate WebSocket
const isConnected = useResearchTreeStore((state) => state.isConnected);
```

## Why This Fix Works

1. **Single Source of Truth**: Only `ResearchTreePanel` creates the WebSocket connection
2. **Shared State**: The connection status is shared via Zustand store (`useResearchTreeStore`)
3. **No Duplicate Connections**: Eliminates the reconnection loop that caused the infinite state updates
4. **Same Functionality**: The `research-tab.tsx` component still gets real-time connection status updates, but without creating its own connection

## Architecture Note
The `useResearchWS` hook already implements a **shared connection pattern** (see `sharedConnection` object in `useResearchWS.ts`), but having multiple React components instantiate the hook still caused issues because:

1. Each hook instance registers its own event handlers
2. Each hook instance can trigger reconnection attempts
3. Multiple reconnection timers can be scheduled simultaneously
4. The shared connection management wasn't preventing the duplicate reconnection loops

By ensuring only ONE component uses the hook, we avoid these issues entirely.

## Testing
After applying the fix:
1. Frontend build succeeds: ✅
2. No TypeScript errors introduced: ✅
3. WebSocket connection is properly managed by ResearchTreePanel: ✅
4. Connection status is correctly displayed in research-tab: ✅

## Related Files
- `OpenHands/frontend/src/hooks/useResearchWS.ts` - WebSocket hook implementation
- `OpenHands/frontend/src/components/research/ResearchTreePanel.tsx` - Primary WebSocket consumer
- `OpenHands/frontend/src/routes/research-tab.tsx` - Fixed to use store instead of duplicate hook
- `OpenHands/frontend/src/state/research-tree-store.ts` - Zustand store for shared state

## Future Improvements
Consider implementing a singleton pattern at the module level for WebSocket connections to completely prevent multiple instances, rather than relying on proper hook usage:

```typescript
// Example approach:
class ResearchWebSocketManager {
  private static instance: ResearchWebSocketManager;
  private constructor() { /* ... */ }
  static getInstance() { /* ... */ }
}
```

This would provide an additional safeguard against similar issues in the future.
