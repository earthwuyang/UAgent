# React Error #185 - Maximum Update Depth Exceeded in Research Tree Component

## Error Description

**Error:** React Error #185 - "Maximum update depth exceeded. This can happen when a component repeatedly calls setState inside componentWillUpdate or componentDidUpdate. React limits the number of nested updates to prevent infinite loops."

**Location:** Research Tree tab in the frontend UI

**Visible Symptoms:**
- Research Tree Error shown in UI
- "An error occurred while rendering the research tree. This might be due to invalid data or a rendering issue."
- Tree shows 1 node but won't render
- Browser console shows: `Minified React error #185`

---

## Code Context

### File: `frontend/src/routes/research-tab.tsx`

This is a React component that displays a research tree visualization. The component:
1. Fetches experiment data from backend API
2. Polls for updates every 5 seconds while experiment is running
3. Displays tree using ReactFlow library
4. Shows stats (nodes, edges, cost, tokens) and connection status

### Key State Management

```typescript
const [experimentStatus, setExperimentStatus] = useState<ExperimentState>("idle");
const [canRenderTree, setCanRenderTree] = useState(false);
const [treeError, setTreeError] = useState<string | null>(null);
const [statusError, setStatusError] = useState<string | null>(null);
```

### Problematic useEffect Hooks

**Hook 1: Initial fetch on mount (Lines 160-173)**
```typescript
useEffect(() => {
  if (!experimentId) {
    setExperimentStatus("idle");
    setStatusError(null);
    setTreeError(null);
    return;
  }

  useResearchTreeStore.getState().setExperimentId(experimentId);
  setCanRenderTree(false);
  fetchTreeSnapshot(true);
  fetchStatus();
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [experimentId]);
```

**Hook 2: Status polling (Lines 174-199)**
```typescript
useEffect(() => {
  if (!experimentId) {
    clearStatusPolling();
    return;
  }

  if (experimentStatus !== "running" && experimentStatus !== "paused") {
    clearStatusPolling();
    return;
  }

  if (statusPollRef.current) {
    return;
  }

  fetchStatus();
  statusPollRef.current = setInterval(() => {
    fetchStatus();
  }, STATUS_POLL_INTERVAL);

  return () => {
    clearStatusPolling();
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [experimentId, experimentStatus]);
```

**Hook 3: Tree polling (Lines 199-217)**
```typescript
useEffect(() => {
  if (!experimentId) {
    return;
  }

  if (experimentStatus !== "running" && experimentStatus !== "paused") {
    return;
  }

  fetchTreeSnapshot(false);
  const interval = setInterval(() => {
    fetchTreeSnapshot(false);
  }, TREE_POLL_INTERVAL);

  return () => clearInterval(interval);
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [experimentId, experimentStatus]);
```

### Functions Called in useEffect

**fetchTreeSnapshot (Lines 63-126)**
```typescript
const fetchTreeSnapshot = useCallback(
  async (showSpinner = false) => {
    if (!experimentId) return;

    try {
      if (showSpinner) {
        useResearchTreeStore.getState().setLoading(true);
      }

      setError(null);
      setFetchError(null);

      const response = await fetch(`/api/research/experiments/${experimentId}/tree`);

      if (response.status === 404) {
        useResearchTreeStore.getState().setSnapshot(emptySnapshot);
        setTreeError(null);
        setCanRenderTree(false);
        return;
      }

      if (!response.ok) {
        throw new Error(`Failed to fetch tree: ${response.statusText}`);
      }

      const snapshot = await response.json();
      useResearchTreeStore.getState().setSnapshot(snapshot);
      setTreeError(null);
      setCanRenderTree(Boolean(snapshot?.data?.nodes?.length));
    } catch (error) {
      setTreeError(error instanceof Error ? error.message : "Failed to fetch research tree");
      setCanRenderTree(false);
    } finally {
      if (showSpinner) {
        useResearchTreeStore.getState().setLoading(false);
      }
    }
  },
  [experimentId]
);
```

**fetchStatus (Lines 128-158)**
```typescript
const fetchStatus = useCallback(async () => {
  if (!experimentId) return;

  try {
    const statusResponse = await getExperimentStatus(experimentId);
    setExperimentStatus(statusResponse.status);
    setStatusError(null);

    if (
      statusResponse.status === "complete" ||
      statusResponse.status === "failed" ||
      statusResponse.status === "cancelled"
    ) {
      clearStatusPolling();
    }
  } catch (error) {
    if (
      error instanceof ResearchAPIError &&
      (error.statusCode === 404 || error.statusCode === 410)
    ) {
      setExperimentStatus("idle");
      setStatusError(null);
      clearStatusPolling();
      setCanRenderTree(false);
    } else {
      const message = error instanceof Error ? error.message : "Failed to fetch status";
      setStatusError(message);
    }
  }
}, [experimentId, clearStatusPolling]);
```

---

## What We Already Tried

### Attempt 1: Removed function dependencies from useEffect
We added ESLint disable comments and removed `fetchTreeSnapshot` and `fetchStatus` from the dependency arrays to break circular dependencies.

**Result:** Still getting error #185

### Attempt 2: Rebuilt frontend
```bash
cd /Users/wuy/Desktop/code/UAgent/OpenHands/frontend
npm run build
```

**Result:** Build successful but error persists

### Attempt 3: Restarted backend server
```bash
./start_openhands_research.sh
```

**Result:** Server starts successfully, error persists

---

## Current Situation

1. **Frontend built successfully** - No TypeScript or build errors
2. **Backend API working** - `/api/research/health` returns healthy status
3. **Tree data available** - API returns: `{"data": {"nodes": [1 node], "edges": [], "stats": {...}}}`
4. **Connection established** - WebSocket shows "Connected"
5. **React Error #185 triggered** - Component enters infinite update loop

---

## Hypothesis

The error occurs when:
1. Component mounts → `experimentId` exists
2. First useEffect runs → calls `fetchTreeSnapshot(true)` and `fetchStatus()`
3. `fetchStatus()` calls `setExperimentStatus("running")`
4. `experimentStatus` changes → triggers second and third useEffect hooks
5. These hooks call `fetchTreeSnapshot(false)` and `fetchStatus()` again
6. `fetchStatus()` might call `setExperimentStatus()` again with same value
7. Or `fetchTreeSnapshot()` calls multiple `setState` functions (`setCanRenderTree`, `setTreeError`)
8. **Somehow this creates a loop that React detects as exceeding maximum update depth**

---

## Questions for GPT-5-Pro

1. **Why is the infinite loop occurring?** 
   - The dependency arrays only contain primitive values (`experimentId`, `experimentStatus`)
   - We added ESLint disable comments to avoid including functions
   - State updates should be idempotent (setting same value shouldn't trigger re-render)

2. **Is the issue in how we're calling setState?**
   - Multiple `setState` calls in sequence within async functions?
   - Setting state during the render phase somehow?
   - Zustand store updates (`useResearchTreeStore.getState().setSnapshot()`) interfering?

3. **Could it be the polling intervals?**
   - Two separate intervals running simultaneously?
   - Intervals triggering state updates too frequently?

4. **Is there a better pattern for this use case?**
   - Component that needs to: fetch initial data, poll for updates, show stats
   - How to properly structure useEffect hooks for this?

5. **Could the ResearchTreeView component be the culprit?**
   - The error appears during tree rendering
   - Maybe the child component (`<ResearchTreeView />`) is causing the loop?
   - Should we investigate that component instead?

---

## Additional Context

- **React version:** 19.2
- **Build tool:** Vite 7.1.4
- **State management:** Zustand (for tree store) + React useState
- **Environment:** Production build (minified)
- **Related components:** ResearchTreeView (child), ResearchErrorBoundary (wrapper)

---

## Repository

**GitHub:** https://github.com/earthwuyang/UAgent

The code is in:
- `frontend/src/routes/research-tab.tsx` (main component)
- `frontend/src/state/research-tree-store.ts` (Zustand store)
- `frontend/src/components/research/ResearchTreeView.tsx` (child component)

---

## Request

Please analyze this code and help us identify:
1. **The exact cause** of the infinite update loop
2. **Why our fixes didn't work** (removing function dependencies)
3. **The correct solution** to prevent this error

Thank you!
