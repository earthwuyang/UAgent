# Research Node Infinite Loop Fix

**Date:** 2025-10-12  
**Status:** ✅ FIXED  
**Error:** "Maximum update depth exceeded"

## Problem

The ResearchNode component was causing an infinite loop error:

```
Error: Maximum update depth exceeded. This can happen when a component repeatedly calls setState inside componentWillUpdate or componentDidUpdate. React limits the number of nested updates to prevent infinite loops.
```

## Root Cause

The issue was in `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/ResearchNode.tsx` at lines 102-117.

### Bad Pattern (BEFORE):

```typescript
export const ResearchNode = memo(({ data, selected }: NodeProps<ResearchNodeType>) => {
  const {
    expandedNodeIds,
    toggleExpanded,
    selectNode,
  } = useResearchTreeStore((state) => ({
    expandedNodeIds: state.expandedNodeIds,  // ❌ Returns new object every render!
    toggleExpanded: state.toggleExpanded,
    selectNode: state.selectNode,
  }));

  const expandedInStore = expandedNodeIds.has(data.id);
  const [isExpanded, setIsExpanded] = useState(expandedInStore);

  useEffect(() => {
    setIsExpanded(expandedInStore);  // ❌ Triggers on every render!
  }, [expandedInStore]);
  // ...
});
```

### Why This Caused an Infinite Loop:

1. **New Object Creation**: The store selector `(state) => ({ ... })` returns a **new object** on every render
2. **Reference Inequality**: Even if the values inside are the same, the object reference is different
3. **Zustand Re-render**: Zustand compares references, sees a "change", triggers component re-render
4. **useEffect Trigger**: The `expandedInStore` dependency changes, triggering `setIsExpanded()`
5. **State Update**: `setIsExpanded()` causes another render
6. **Repeat**: Back to step 1 → **infinite loop!**

## Solution

Use **separate, primitive selectors** instead of creating objects:

### Good Pattern (AFTER):

```typescript
export const ResearchNode = memo(({ data, selected }: NodeProps<ResearchNodeType>) => {
  // Use separate store selectors to avoid creating new objects on every render
  const isExpanded = useResearchTreeStore((state) => state.expandedNodeIds.has(data.id));
  const toggleExpanded = useResearchTreeStore((state) => state.toggleExpanded);
  const selectNode = useResearchTreeStore((state) => state.selectNode);
  
  // No useState needed! Read directly from store
  // No useEffect needed! Store updates trigger re-renders automatically
  // ...
});
```

### Why This Works:

1. **Primitive Values**: `state.expandedNodeIds.has(data.id)` returns a **boolean** (true/false)
2. **Stable Functions**: `state.toggleExpanded` and `state.selectNode` are stable function references
3. **Reference Equality**: Zustand only triggers re-render when the **value** changes
4. **No Local State**: Eliminates the `useState` + `useEffect` pattern that caused the loop
5. **Direct Store Access**: Component reads expanded state directly from store

## Key Principles for Zustand Selectors

### ❌ DON'T: Return new objects/arrays in selectors
```typescript
// BAD - Creates new object every time
const { a, b } = useStore((state) => ({ a: state.a, b: state.b }));

// BAD - Creates new array every time  
const items = useStore((state) => state.items.map(x => x.id));
```

### ✅ DO: Return primitive values or use shallow equality
```typescript
// GOOD - Returns primitive
const isOpen = useStore((state) => state.isOpen);

// GOOD - Multiple primitive selectors
const isOpen = useStore((state) => state.isOpen);
const count = useStore((state) => state.count);

// GOOD - Use shallow equality for objects (if needed)
import { shallow } from 'zustand/shallow';
const { a, b } = useStore((state) => ({ a: state.a, b: state.b }), shallow);
```

## Files Modified

### 1. ResearchNode.tsx (First Fix - Incomplete)
**File**: `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/ResearchNode.tsx`

**Lines Changed**: 101-117 (reduced to 101-105)

**Changes**:
- Removed: Object destructuring with store selector
- Removed: Local `useState` hook
- Removed: `useEffect` with dependency on `expandedInStore`
- Added: Three separate primitive selectors

**Note**: This fix alone was insufficient. The error persisted because the parent components had the same issue.

### 2. ResearchTreeView.tsx (Root Cause)
**File**: `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/ResearchTreeView.tsx`

**Lines Changed**: 129-155 (reduced to 129-141)

**Original Code** (BAD):
```typescript
const {
  nodes: storeNodes,
  edges: storeEdges,
  selectedNodeId,
  selectNode,
  filterType,
  filterStatus,
  searchQuery,
  setFilterType,
  setFilterStatus,
  setSearchQuery,
  clearFilters,
  isLoading,
} = useResearchTreeStore((state) => ({
  nodes: state.nodes,
  edges: state.edges,
  selectedNodeId: state.selectedNodeId,
  selectNode: state.selectNode,
  filterType: state.filterType,
  filterStatus: state.filterStatus,
  searchQuery: state.searchQuery,
  setFilterType: state.setFilterType,
  setFilterStatus: state.setFilterStatus,
  setSearchQuery: state.setSearchQuery,
  clearFilters: state.clearFilters,
  isLoading: state.isLoading,
}));
```

**Fixed Code** (GOOD):
```typescript
// Use separate store selectors to avoid creating new objects on every render
const storeNodes = useResearchTreeStore((state) => state.nodes);
const storeEdges = useResearchTreeStore((state) => state.edges);
const selectedNodeId = useResearchTreeStore((state) => state.selectedNodeId);
const selectNode = useResearchTreeStore((state) => state.selectNode);
const filterType = useResearchTreeStore((state) => state.filterType);
const filterStatus = useResearchTreeStore((state) => state.filterStatus);
const searchQuery = useResearchTreeStore((state) => state.searchQuery);
const setFilterType = useResearchTreeStore((state) => state.setFilterType);
const setFilterStatus = useResearchTreeStore((state) => state.setFilterStatus);
const setSearchQuery = useResearchTreeStore((state) => state.setSearchQuery);
const clearFilters = useResearchTreeStore((state) => state.clearFilters);
const isLoading = useResearchTreeStore((state) => state.isLoading);
```

### 3. ResearchTreePanel.tsx (Additional Fix)
**File**: `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/components/research/ResearchTreePanel.tsx`

**Lines Changed**: 41-65 (reduced to 41-52)

**Changes**: Applied the same fix as ResearchTreeView - separated object destructuring into individual selectors

## Testing

To verify the fix:

1. Navigate to a conversation with research tree data
2. Click on the "Research Tree" tab
3. Browser console should NOT show:
   - "Maximum update depth exceeded" error
   - Infinite loop errors
   - Component re-render warnings
4. Research tree should render normally with nodes visible
5. Clicking expand/collapse buttons should work smoothly

## Related Patterns

This same issue can occur with:

- **Arrays**: `useStore((state) => state.items.filter(...))` - creates new array
- **Objects**: `useStore((state) => ({ x: state.x }))` - creates new object  
- **Maps/Sets**: Direct access can create new references

**Solution**: Either use primitive selectors OR use `useMemo` to stabilize references.

## Prevention

To prevent similar issues in the future:

1. **Avoid object creation in store selectors** unless using shallow equality
2. **Use separate selectors** for each piece of state
3. **Test with React DevTools** to watch for excessive re-renders
4. **Use `useMemo`** when computing derived state from store values
5. **Be careful with Sets/Maps** - they're objects and have reference identity

## Complete Fix Summary

### What Was Wrong

Three components were using the **anti-pattern** of destructuring objects from Zustand selectors:
1. `ResearchNode.tsx` (lines 102-117)
2. `ResearchTreeView.tsx` (lines 129-155) ⚠️ **Main culprit**
3. `ResearchTreePanel.tsx` (lines 41-65)

Each time these components rendered, the selector returned a **new object** with the same values but a different reference, causing Zustand to think the state changed, triggering another render → infinite loop.

### What Was Fixed

Replaced all object destructuring patterns with **separate primitive selectors**:

```typescript
// ❌ BAD - Creates new object every render
const { a, b } = useStore((state) => ({ a: state.a, b: state.b }));

// ✅ GOOD - Returns stable references
const a = useStore((state) => state.a);
const b = useStore((state) => state.b);
```

### Why the First Fix Wasn't Enough

Fixing only `ResearchNode.tsx` didn't solve the problem because:
- The error stack trace showed `<ResearchTreeView>` as the failing component
- Parent components (`ResearchTreeView` and `ResearchTreePanel`) had the same pattern
- React error boundaries caught the error in the parent, not the child

This is a classic case where **all components in a tree must follow the same state management pattern** to prevent cascading render loops.

## Verification Steps

1. ✅ Killed old frontend process
2. ✅ Restarted frontend in `agent-frontend` tmux session
3. ✅ Frontend started successfully on port 3001
4. ✅ No "Maximum update depth exceeded" errors in console
5. ✅ Research Tree tab loads without crashing

## Conclusion

The infinite loop was caused by creating new object references in Zustand store selectors across **three components**, which triggered unnecessary re-renders. By using primitive selectors that return stable values in all three components, the entire component tree now renders efficiently without loops.

**Status**: ✅ **COMPLETELY FIXED** - All three components updated
**Frontend**: Running in `agent-frontend` tmux session on port 3001
