# React Error #185 Fix - Research Tree Component

## Problem Description

React Error #185 ("Maximum update depth exceeded") was occurring in the Research Tree component, causing an infinite loop of setState calls and preventing the component from rendering properly.

## Root Cause Analysis (from GPT-5-Pro)

The issue was NOT caused by polling or useEffect dependencies. Instead, it was caused by **ReactFlow operating in controlled mode with a synchronous write-back cycle**:

### The Feedback Loop

1. `<ReactFlow>` was used in **controlled mode** (nodes/edges passed as props)
2. ReactFlow triggers `onNodesChange` / `onEdgesChange` handlers when nodes/edges are programmatically updated
3. These handlers were writing back to the **same Zustand store** that supplies the nodes/edges props
4. This created a **synchronous feedback loop**:
   ```
   Render → ReactFlow programmatic update → onNodesChange/onEdgesChange
   → Store update → New props → Render → [LOOP REPEATS]
   ```

### Why Previous Fixes Didn't Work

- **Removing function deps from useEffect**: Didn't help because the loop was in the render path, not in effects
- **Adding memoization**: Still created new array references on every poll, triggering re-renders
- **Polling adjustments**: Didn't address the synchronous loop that React detects as Error #185

## Solutions Implemented

### 1. Made ReactFlow Read-Only (Primary Fix)

**File**: `frontend/src/components/research/ResearchTreeView.tsx`

Changed ReactFlow from controlled to read-only mode by:
- Setting `nodesDraggable={false}` - prevents dragging nodes
- Setting `nodesConnectable={false}` - prevents connecting nodes  
- Setting `elementsSelectable={true}` - still allows clicking for details
- **Removing `onNodesChange` and `onEdgesChange` handlers** (these were the write-back triggers)

```typescript
<ReactFlow
  nodes={nodes}
  edges={edges}
  nodeTypes={nodeTypes}
  onNodeClick={onNodeClick}
  onPaneClick={onPaneClick}
  fitView={false}
  minZoom={0.1}
  maxZoom={2}
  defaultViewport={{ x: 0, y: 0, zoom: 1 }}
  proOptions={{ hideAttribution: true }}
  nodesDraggable={false}      // ← NEW
  nodesConnectable={false}    // ← NEW
  elementsSelectable={true}   // ← NEW
>
```

This **breaks the controlled write-back loop** because:
- ReactFlow no longer calls change handlers on programmatic updates
- No write-backs to the store = no feedback loop
- The tree remains purely display-only (which is its intended purpose)

### 2. Optimized Data Fetchers (Performance Improvement)

**File**: `frontend/src/routes/research-tab.tsx`

#### A. Only Update Status If It Actually Changed

```typescript
const fetchStatus = useCallback(async () => {
  if (!experimentId) return;

  try {
    const statusResponse = await getExperimentStatus(experimentId);
    // Only update status if it actually changed (prevents unnecessary re-renders)
    setExperimentStatus(prev => 
      prev === statusResponse.status ? prev : statusResponse.status
    );
    // ... rest of code
  }
}, [experimentId, clearStatusPolling]);
```

This leverages React's bailout optimization - if we set the same value, React won't trigger a re-render.

#### B. Hash-Based Snapshot Comparison

```typescript
const lastTreeHashRef = useRef<string>('');

const fetchTreeSnapshot = useCallback(
  async (showSpinner = false) => {
    // ... fetch logic ...
    
    const snapshot = await response.json();
    // Use JSON.stringify as a simple hash to avoid writing identical snapshots
    const hash = JSON.stringify(snapshot?.data ?? snapshot);
    if (hash !== lastTreeHashRef.current) {
      lastTreeHashRef.current = hash;
      useResearchTreeStore.getState().setSnapshot(snapshot);
      setTreeError(null);
      setCanRenderTree(Boolean(snapshot?.data?.nodes?.length));
    }
  },
  [experimentId]
);
```

This prevents writing **identical snapshots** to the store:
- Even if the backend returns the same data, we don't update the store
- Avoids triggering Zustand subscribers unnecessarily
- Reduces wasted re-renders across all components using the store

### 3. Proper Memoization in ResearchTreeView

**File**: `frontend/src/components/research/ResearchTreeView.tsx`

Replaced direct `getFilteredNodes()` selector call with proper memoization:

```typescript
// ❌ BEFORE: getFilteredNodes() returned new array reference every render
const filteredNodes = useResearchTreeStore((state) => state.getFilteredNodes());

// ✅ AFTER: Memoized with stable dependencies
const filteredNodes = useMemo(() => {
  if (!hasFilters) {
    return Array.from(storeNodes.values());
  }
  return Array.from(storeNodes.values()).filter((node) => {
    const typeMatch = !filterType || node.type === filterType;
    const statusMatch = !filterStatus || node.status === filterStatus;
    const searchMatch = !searchQuery.trim() || 
      node.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      node.description?.toLowerCase().includes(searchQuery.toLowerCase());
    return typeMatch && statusMatch && searchMatch;
  });
}, [storeNodes, filterType, filterStatus, searchQuery, hasFilters]);
```

This ensures:
- `filteredNodes` only changes when actual dependencies change
- No new array references on every render
- Downstream `useMemo` hooks (that depend on `filteredNodes`) don't recompute unnecessarily

## Why These Fixes Work

### Primary Fix: Read-Only ReactFlow

According to ReactFlow documentation:
> "In controlled mode, programmatic updates to nodes/edges will trigger the `onNodesChange`/`onEdgesChange` handlers"

By removing these handlers and making the tree read-only:
- ✅ No synchronous write-back loop
- ✅ No Error #185
- ✅ Tree still fully functional for visualization

### Secondary Optimizations

1. **Status bailout**: Prevents re-renders when status hasn't changed
2. **Snapshot hashing**: Prevents store updates when data is identical
3. **Proper memoization**: Prevents cascading re-renders from new object references

## Testing & Verification

To verify the fix:

1. Navigate to `http://localhost:2999` in Chrome
2. Open DevTools Console
3. Go to Research Tree tab
4. Start a research session
5. **Expected**: No React Error #185 in console
6. **Expected**: Tree renders smoothly with nodes appearing as they're created

## Key Lessons

1. **React Error #185 is always a synchronous loop** - async polling can't cause it
2. **Controlled ReactFlow + write-backs = feedback loop** - be careful with controlled mode
3. **Read-only is often the right choice** for display-only visualizations
4. **Object identity matters** - new references trigger re-renders even with same content
5. **Zustand selectors that return new objects** should not be called directly in render

## References

- React Error #185: https://react.dev/errors/185
- ReactFlow Controlled Mode: https://reactflow.dev/api-reference/types/react-flow-instance
- React useState Bailout: https://legacy.reactjs.org/docs/hooks-reference.html#bailing-out-of-a-state-update
- ReactFlow State Management: https://reactflow.dev/learn/advanced-use/state-management

## Credits

Analysis and solution guidance provided by GPT-5-Pro, implemented and tested by the development team.
