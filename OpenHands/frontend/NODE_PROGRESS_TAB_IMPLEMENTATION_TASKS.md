# Node Progress Tab Implementation - Detailed Task Breakdown

**Feature**: Refactor node context switching from in-place UI switch to new tab approach

**Goal**: Add "View Progress" button that opens dedicated node progress page in new browser tab with independent WebSocket connection. Main conversation remains unchanged.

---

## Architecture Overview

```
User Flow:
1. User clicks node in research tree → Detail panel opens (right sidebar)
2. User clicks "View Progress" button → New browser tab opens
3. New tab shows dedicated node progress page with:
   - NodeProgressHeader (node info, status, metrics, back button)
   - NodeEventTimeline (chronological event stream with expand/collapse)
4. Each tab has independent WebSocket connection to node events
5. Main conversation page remains unchanged (no context switching)

URL Structure:
/conversations/{conversationId}/nodes/{nodeId}?experiment_id={experimentId}
```

---

## Task Dependencies

```
Task 1 (Cleanup) ──┐
                   ├──> Task 2 (Refactor Store)
                   │
Task 3 (Route) ────┼──> Task 4 (Button)
                   │
Task 5 (Header) ───┤
                   │
Task 6 (Timeline) ─┤
                   │
                   └──> Task 7 (Integration) ──> Task 8 (Testing)
```

---

## Task 1: Clean Up Context Switching Implementation

**ID**: `35c1bc6e-ace1-4cd1-9b9f-6822138a7149`

**Priority**: High

**Description**: Remove existing context switching code from ResearchTreeView, conversation.tsx, and chat-interface.tsx. Simplify onNodeDoubleClick handler to only open detail panel without context switching or navigation.

### Implementation Steps

1. **In ResearchTreeView.tsx** (lines ~239-267):
   - Remove import: `useNodeEventStore`
   - Modify `onNodeDoubleClick` handler:
     - **Keep**: `event.preventDefault()`, `selectNode()`, `expandNode()`, `setHasRightPanelToggled()`, `setIsRightPanelShown()`, `setSelectedTab('research')`
     - **Remove**: `extractConversationId()`, `switchToNodeContext()`, navigation logic with `navigate()`

2. **In conversation.tsx** (lines ~97-110):
   - Remove imports: `NodeEventWebSocket`, `useNodeEventStore`
   - Remove entire `useEffect` that creates `NodeEventWebSocket` and cleanup

3. **In chat-interface.tsx**:
   - Remove import: `NodeContextIndicator`, `useNodeEventStore`
   - Remove `<NodeContextIndicator />` component rendering
   - Remove `isNodeContext` state variable if unused elsewhere

### Files Modified

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/ResearchTreeView.tsx`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/routes/conversation.tsx`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/features/chat/chat-interface.tsx`

### Verification Criteria

- [ ] TypeScript compilation passes with no errors (`npx tsc --noEmit`)
- [ ] Double-click node in research tree opens detail panel (no context switch)
- [ ] No console errors in browser
- [ ] Main chat UI remains showing root conversation context
- [ ] Research tree visualization still works correctly

### Dependencies

None

### Notes

This task removes all in-place context switching behavior. Double-clicking nodes will still open the detail panel but will no longer change the chat UI context.

---

## Task 2: Refactor NodeEventStore - Remove Context State

**ID**: `d649cdd8-9e82-49b2-b503-9ea04ce85f2b`

**Priority**: High

**Description**: Simplify NodeEventStore by removing all context switching state and actions. Keep only event storage, subscription management, and pagination functionality.

### Implementation Steps

1. **In `/src/state/node-event-store.ts`**:

   **Remove from `NodeEventState` interface**:
   - `contextMode: ContextMode`
   - `activeNodeId: string | null`
   - `activeExperimentId: string | null`

   **Remove from actions**:
   - `switchToNodeContext()`
   - `switchToRootContext()`
   - `setActiveContext()`

   **Keep all event management**:
   - `nodeEvents: Map<string, NodeEvent[]>`
   - `loadNodeEvents()`
   - `appendNodeEvent()`
   - `clearNodeEvents()`
   - `clearAllNodeEvents()`
   - `subscribeToNode()`
   - `unsubscribeFromNode()`
   - Pagination helpers (`loadMoreEvents`, `resetPagination`)
   - Error handling (`setNodeError`, `clearNodeError`)

   **Remove types**:
   - `ContextMode` type if no longer used

2. **Update store implementation**:
   - Remove context state initialization
   - Remove context-related logic from actions

3. **Update tests** in `/src/__tests__/state/node-event-store.test.ts`:
   - Remove context switching test cases
   - Keep event management tests

### Files Modified

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/state/node-event-store.ts`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/__tests__/state/node-event-store.test.ts`

### Verification Criteria

- [ ] TypeScript compilation passes (`npx tsc --noEmit`)
- [ ] Unit tests pass (`npm test node-event-store.test`)
- [ ] Store exports only event management functions
- [ ] No references to `contextMode`, `activeNodeId`, `activeExperimentId` in codebase
- [ ] Zustand devtools still work

### Dependencies

- **Requires**: Task 1 (Clean Up Context Switching Implementation)

### Notes

This simplifies the store to pure event storage without UI context concerns. The store becomes a simple event cache and subscription tracker.

---

## Task 3: Add Route for Node Progress Page

**ID**: `b8d6854c-1b7e-4d7e-be17-e40c606787e0`

**Priority**: High

**Description**: Add new route definition for node progress pages at `/conversations/:conversationId/nodes/:nodeId` and create skeleton route component with basic navigation.

### Implementation Steps

1. **In `/src/routes.ts`**:
   - Add route definition after conversation route:
     ```typescript
     route("conversations/:conversationId/nodes/:nodeId", "routes/node-progress.tsx")
     ```

2. **Create `/src/routes/node-progress.tsx`**:
   ```typescript
   import React from 'react';
   import { useParams, useNavigate } from 'react-router';
   import { ArrowLeft } from 'lucide-react';

   export default function NodeProgressPage() {
     const { conversationId, nodeId } = useParams();
     const navigate = useNavigate();

     const handleBack = () => {
       navigate(`/conversations/${conversationId}`);
     };

     return (
       <div className="flex flex-col h-screen p-6">
         <button
           onClick={handleBack}
           className="flex items-center gap-2 text-blue-500 hover:text-blue-600 mb-4"
         >
           <ArrowLeft size={20} />
           Back to Conversation
         </button>
         <h1 className="text-2xl font-bold">Node Progress Page</h1>
         <p>Conversation ID: {conversationId}</p>
         <p>Node ID: {nodeId}</p>
       </div>
     );
   }
   ```

### Files Created

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/routes/node-progress.tsx`

### Files Modified

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/routes.ts`

### Verification Criteria

- [ ] TypeScript compilation passes
- [ ] Navigate manually to `/conversations/test-conv/nodes/test-node`
- [ ] Page loads without errors
- [ ] Page displays placeholder content with `conversationId` and `nodeId`
- [ ] Back button navigates to `/conversations/test-conv`
- [ ] No console errors

### Dependencies

None

### Notes

This creates the routing infrastructure. The skeleton page allows testing navigation before implementing full functionality.

---

## Task 4: Add View Progress Button to Detail Panel

**ID**: `c40a954e-a35b-4af1-bd69-c3aeb98c0504`

**Priority**: High

**Description**: Add "View Progress" button to ResearchNodeDetailPanel that opens node progress page in new browser tab using `window.open()`. Button should be disabled if experimentId is missing.

### Implementation Steps

1. **In `/src/components/research/ResearchNodeDetailPanel.tsx`**:

   **Add imports**:
   ```typescript
   import { ExternalLink } from 'lucide-react';
   import { useParams } from 'react-router';
   ```

   **Extract conversationId**:
   ```typescript
   const { conversationId } = useParams();
   ```

   **Extract experimentId**:
   ```typescript
   const experimentId = selectedNode?.metadata?.experiment_id || selectedNode?.metadata?.experimentId;
   ```

   **Add button handler**:
   ```typescript
   const handleViewProgress = () => {
     if (!experimentId || !conversationId) return;
     const url = `/conversations/${conversationId}/nodes/${selectedNode.id}?experiment_id=${experimentId}`;
     window.open(url, '_blank', 'noopener,noreferrer');
   };
   ```

   **Add button in JSX** (after Tokens section, around line 150):
   ```typescript
   <section className="research-detail-section">
     <button
       type="button"
       onClick={handleViewProgress}
       disabled={!experimentId}
       className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors"
       title={!experimentId ? 'Experiment ID not available' : 'Open node progress in new tab'}
     >
       <ExternalLink size={16} />
       View Node Progress
     </button>
   </section>
   ```

### Files Modified

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/ResearchNodeDetailPanel.tsx`

### Verification Criteria

- [ ] TypeScript compilation passes
- [ ] Button appears in detail panel after Tokens section
- [ ] Button is enabled when experimentId exists
- [ ] Button is disabled (grayed out) when experimentId missing
- [ ] Clicking button opens new tab with correct URL format
- [ ] URL is shareable (can be copied and pasted)
- [ ] New tab loads node-progress page
- [ ] Multiple clicks open multiple independent tabs

### Dependencies

- **Requires**: Task 3 (Add Route for Node Progress Page)

### Notes

Button uses `window.open()` pattern consistent with existing code (vscode-tab.tsx, served-tab.tsx). Disabled state provides feedback when experimentId is missing.

---

## Task 5: Create NodeProgressHeader Component

**ID**: `5235b9e5-6c2a-4b78-a7df-d75de4584950`

**Priority**: High

**Description**: Create NodeProgressHeader component by refactoring NodeContextIndicator. Display node title, type, status badge, metrics grid, and back button. Static header without animations.

### Implementation Steps

1. **Create directory**:
   ```bash
   mkdir -p /Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/node-progress
   ```

2. **Create `/src/components/research/node-progress/NodeProgressHeader.tsx`**:

See full implementation in the task output above (component code is ~130 lines).

**Key features**:
- Status color mapping: running=blue, complete=green, failed=red, cancelled=gray, pending=yellow
- Type label formatting: `web_search` → `Web Search`
- Metrics grid: 4 columns (Visits, Q Value, Prior, Cost)
- Back button using Link component
- Gradient header background matching NodeContextIndicator style
- No animations (static header)

### Files Created

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/node-progress/NodeProgressHeader.tsx`

### Reference Files

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/NodeContextIndicator.tsx` (for status colors, type labels, styling patterns)

### Verification Criteria

- [ ] TypeScript compilation passes
- [ ] Component renders with mock node data
- [ ] Status colors match existing design (blue/green/red/yellow)
- [ ] Type label formatting works (`web_search` → `Web Search`)
- [ ] Back button navigates to conversation
- [ ] Metrics grid displays 4 columns correctly
- [ ] Dark mode compatible (uses theme colors)
- [ ] No console errors or warnings
- [ ] Gradient background displays correctly

### Dependencies

None (can be developed in parallel)

### Notes

Repurposes NodeContextIndicator design but removes animations. Uses Link instead of button for back navigation. Metrics displayed in 4-column grid matching existing detail panel layout.

---

## Task 6: Create NodeEventTimeline Component

**ID**: `cc92349b-cb7f-4552-a990-9f20155aea89`

**Priority**: High

**Description**: Create NodeEventTimeline component to display node events chronologically. Include event type badges, timestamps, expandable details, auto-scroll, and loading/empty states.

### Implementation Steps

1. **Create `/src/components/research/node-progress/NodeEventTimeline.tsx`**:

See full implementation in the task output above (component code is ~120 lines).

**Key features**:
- Event type configuration with icons and colors:
  - `task_start`: Play icon, blue
  - `task_complete`: CheckCircle icon, green
  - `observation`: Eye icon, purple
  - `action`: Zap icon, yellow
  - `error`: XCircle icon, red
  - `warning`: XCircle icon, orange
  - `message`: MessageSquare icon, gray
- Expandable event details (JSON content)
- Auto-scroll to latest event using ref
- Loading state with spinner
- Empty state with placeholder
- Responsive design

### Files Created

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/node-progress/NodeEventTimeline.tsx`

### Reference Files

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/state/node-event-store.ts` (for NodeEvent type definition)

### Verification Criteria

- [ ] TypeScript compilation passes
- [ ] Component renders with mock events array
- [ ] Event type icons display correctly for all types
- [ ] Event type colors match configuration
- [ ] Expand/collapse details works
- [ ] Timestamps format correctly (locale string)
- [ ] Auto-scroll to bottom works when new events added
- [ ] Loading state shows spinner and message
- [ ] Empty state shows placeholder message
- [ ] No console errors
- [ ] JSON content displays in readable format

### Dependencies

None (can be developed in parallel)

### Notes

Component handles event display with expandable details. Auto-scrolls to latest event. Includes loading skeleton and empty state. Event type icons and colors provide visual hierarchy.

---

## Task 7: Implement Full NodeProgressPage with WebSocket

**ID**: `96c3dfbe-9bb6-4fa5-bc71-28ab1aeb885a`

**Priority**: High

**Description**: Implement complete NodeProgressPage route component with WebSocket connection, event subscription, data fetching, and rendering of NodeProgressHeader and NodeEventTimeline. Handle loading, error, and cleanup states.

### Implementation Steps

1. **Update `/src/routes/node-progress.tsx`** with full implementation:

See full implementation in the task output above (component code is ~110 lines).

**Key features**:
- URL parameter extraction: `conversationId`, `nodeId` from params, `experimentId` from query
- Node data fetching from ResearchTreeStore
- WebSocket initialization and connection
- Event subscription to specific node
- Historical events loading
- Loading state management
- Error handling with user-friendly messages
- Cleanup on unmount (disconnect, unsubscribe, clear events)
- Error boundary fallback UI
- Node not found handling

### Files Modified

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/routes/node-progress.tsx` (replace skeleton with full implementation)

### Dependencies Used

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/node-progress/NodeProgressHeader.tsx`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/node-progress/NodeEventTimeline.tsx`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/services/node-event-websocket.ts`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/state/node-event-store.ts`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/state/research-tree-store.ts`

### Verification Criteria

- [ ] TypeScript compilation passes
- [ ] Page loads without errors when navigating from button
- [ ] WebSocket connects successfully (check browser Network tab)
- [ ] Node data displays correctly in header
- [ ] Events load and display in timeline
- [ ] New events stream in real-time via WebSocket
- [ ] Multiple tabs can open simultaneously and work independently
- [ ] Closing tab disconnects WebSocket (verify in Network tab)
- [ ] Error state shows when parameters missing
- [ ] Loading state shows while connecting
- [ ] Empty state shows when no events yet
- [ ] "Node Not Found" shows when node doesn't exist
- [ ] No console errors or warnings
- [ ] Memory leak test: open/close multiple tabs, check memory usage

### Dependencies

- **Requires**: Task 5 (Create NodeProgressHeader Component)
- **Requires**: Task 6 (Create NodeEventTimeline Component)

### Notes

Complete integration connecting all pieces. WebSocket connects on mount, subscribes to node events, loads historical events. Cleanup on unmount prevents memory leaks. Error and loading states provide good UX.

---

## Task 8: Testing and Documentation

**ID**: `f31b482f-2c79-4042-985f-905e56c1f9bc`

**Priority**: Medium

**Description**: Update unit tests, create E2E tests, update documentation, and polish UI/UX. Verify dark mode, add error boundaries, test mobile responsiveness.

### Implementation Steps

#### 1. Update Unit Tests

**Update `/src/__tests__/state/node-event-store.test.ts`**:
- Remove context switching tests:
  - Remove tests for `switchToNodeContext()`
  - Remove tests for `switchToRootContext()`
  - Remove tests for `contextMode` state
  - Remove tests for `activeNodeId` and `activeExperimentId`
- Keep event management tests:
  - Event storage and retrieval
  - Subscription management
  - Pagination
  - Error handling

**Create `/src/__tests__/components/research/node-progress/NodeProgressHeader.test.tsx`**:
- Test status color rendering for all statuses (running, complete, failed, pending, cancelled)
- Test type label formatting (`web_search` → `Web Search`)
- Test metrics display (visits, Q value, prior, cost)
- Test back button navigation
- Test dark mode compatibility
- Test missing node data handling

**Create `/src/__tests__/components/research/node-progress/NodeEventTimeline.test.tsx`**:
- Test event rendering with various event types
- Test expand/collapse functionality
- Test auto-scroll behavior
- Test loading state rendering
- Test empty state rendering
- Test event type icon and color mapping
- Test timestamp formatting

**Update `/src/__tests__/components/research/ResearchNodeDetailPanel.test.tsx`**:
- Test button rendering
- Test button disabled state when experimentId missing
- Test button enabled state when experimentId present
- Test `window.open()` call with correct URL
- Test button click handling

#### 2. Create E2E Tests

**Create test file** (using Playwright):
```typescript
// Test flow
test('Open node progress in new tab', async ({ page, context }) => {
  // Navigate to conversation
  await page.goto('/conversations/test-conv');
  
  // Click node in research tree
  await page.click('[data-testid="research-node-idea-0"]');
  
  // Detail panel opens
  await expect(page.locator('[data-testid="node-detail-panel"]')).toBeVisible();
  
  // Click "View Progress" button
  const [newPage] = await Promise.all([
    context.waitForEvent('page'),
    page.click('button:has-text("View Node Progress")')
  ]);
  
  // New tab opens with correct URL
  expect(newPage.url()).toContain('/nodes/idea-0');
  expect(newPage.url()).toContain('experiment_id=');
  
  // Page loads successfully
  await newPage.waitForLoadState('networkidle');
  
  // Header displays
  await expect(newPage.locator('h1')).toBeVisible();
  
  // WebSocket connects (check network)
  // Events display or empty state shows
});
```

#### 3. Update Documentation

**Create `/src/components/research/node-progress/README.md`**:
- Component overview
- Props documentation
- Usage examples
- Architecture diagram

**Update `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/NODE_PROGRESS_IMPLEMENTATION.md`**:
- Update architecture section with new tab approach
- Document URL structure and parameters
- Add data flow diagram
- Document WebSocket lifecycle
- Add troubleshooting section

**Create user guide section**:
- How to view node progress
- How to share node progress URLs
- What information is displayed

#### 4. UI Polish

**Dark Mode**:
- Verify all components work in dark mode
- Test color contrast for accessibility
- Verify gradient backgrounds
- Test border colors

**Responsive Design**:
- Test on mobile viewports (320px, 375px, 414px)
- Test on tablet viewports (768px, 1024px)
- Test header layout on small screens
- Test metrics grid wrapping
- Test timeline event cards on mobile

**Accessibility**:
- Add ARIA labels to interactive elements
- Test keyboard navigation (Tab, Enter, Escape)
- Add focus indicators
- Test with screen reader
- Verify color contrast ratios (WCAG AA)

**Loading States**:
- Add skeleton loaders for header
- Add skeleton loaders for timeline
- Add shimmer effect

**Error Handling**:
- Add error boundary component
- Add retry button for WebSocket failures
- Add toast notifications for errors
- Test offline behavior

**Performance**:
- Profile component rendering
- Add virtualization if event list > 100 items
- Lazy load event details
- Memoize expensive calculations

#### 5. Code Quality

**Linting**:
```bash
npm run lint
npm run lint:fix
```

**Type Checking**:
```bash
npx tsc --noEmit
```

**Test Coverage**:
```bash
npm run test:coverage
```

Target: Maintain or improve existing coverage (aim for >80%)

### Files Created

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/__tests__/components/research/node-progress/NodeProgressHeader.test.tsx`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/__tests__/components/research/node-progress/NodeEventTimeline.test.tsx`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/node-progress/README.md`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/NODE_PROGRESS_IMPLEMENTATION.md`

### Files Modified

- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/__tests__/state/node-event-store.test.ts`
- `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/__tests__/components/research/ResearchNodeDetailPanel.test.tsx`

### Verification Criteria

- [ ] All unit tests pass: `npm test`
- [ ] Test coverage maintained or improved (check report)
- [ ] E2E test passes: `npm run test:e2e`
- [ ] TypeScript compilation passes: `npx tsc --noEmit`
- [ ] Linting passes: `npm run lint`
- [ ] Dark mode works in all components
- [ ] Mobile layout responsive and usable
- [ ] Keyboard navigation works (Tab, Enter, ESC)
- [ ] ARIA labels present on interactive elements
- [ ] Color contrast meets WCAG AA standards
- [ ] Documentation complete and accurate
- [ ] No console errors or warnings in production build
- [ ] Performance acceptable (< 100ms render time)
- [ ] WebSocket reconnection works after network interruption

### Dependencies

- **Requires**: Task 7 (Implement Full NodeProgressPage with WebSocket)

### Notes

This task ensures quality and completeness. Tests provide confidence, documentation helps future maintenance, polish improves user experience. Can be partially parallelized with implementation.

---

## Summary

**Total Tasks**: 8

**Estimated Timeline**: 6-8 hours

**Critical Path**: Task 1 → Task 2 → Task 7 → Task 8

**Parallelizable**:
- Task 3, 5, 6 can be developed simultaneously
- Task 4 depends only on Task 3
- Task 5 and 6 are independent and can be parallel

**Risk Assessment**:
- **Low Risk**: Tasks 1-4 (simple cleanup and additions)
- **Medium Risk**: Tasks 5-6 (new components, design work)
- **High Risk**: Task 7 (WebSocket integration, state management)

**Key Milestones**:
1. After Task 4: User can open new tabs (skeleton page)
2. After Task 7: Full functionality working
3. After Task 8: Production ready with tests and docs

---

## Implementation Notes

### Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/node-progress-tabs
   ```

2. **Implement tasks in order**:
   - Complete Tasks 1-2 first (cleanup)
   - Implement Task 3 (route infrastructure)
   - Implement Tasks 4-6 in parallel (button + components)
   - Integrate in Task 7
   - Test and document in Task 8

3. **Commit after each task**:
   ```bash
   git add .
   git commit -m "Task 1: Clean up context switching implementation"
   ```

4. **Run checks before each commit**:
   ```bash
   npx tsc --noEmit
   npm test
   npm run lint
   ```

### Testing Strategy

- **Unit tests**: Test components in isolation
- **Integration tests**: Test WebSocket integration
- **E2E tests**: Test full user journey
- **Manual testing**: Test in real browser with real backend

### Rollback Plan

- Each task is atomic and can be reverted independently
- Keep old context switching code in git history
- Can revert to previous approach if needed
- No database migrations or backend API changes

### Deployment

- Feature flag recommended: `ENABLE_NODE_PROGRESS_TABS`
- Gradual rollout to users
- Monitor error rates and WebSocket connections
- Add analytics events for button clicks and page loads

---

**Created**: 2024-10-14
**Status**: Ready for Implementation
**Approved By**: User
