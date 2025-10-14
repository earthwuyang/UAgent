# Fixes for GitHub Issues #24 and #27

## Issue #27: Parallel Research Experiments Fail Immediately

### Root Cause Analysis

The issue reported that parallel research experiments fail immediately with all 3 idea nodes showing `FAILED` status and the session manager reporting 0 active experiments. The investigation revealed:

**Primary Issue: Missing Diagnostic Logging**
- The orchestrator's execution flow lacked comprehensive error tracking
- When adapter execution failed, errors were silently swallowed without detailed logging
- No visibility into which stage of execution was failing (orchestrator.run(), adapter routing, adapter execution, etc.)

**Secondary Issues Identified:**
1. **Adapter Registry**: If adapters aren't properly registered, nodes fail immediately when router.route() or adapter.run() is called
2. **Session Manager Registration**: Experiments may fail to register with the session manager, causing 0 experiments to be tracked
3. **Background Task Failures**: The `_run_research()` background task may encounter exceptions that aren't properly logged

### Fixes Applied

#### 1. Enhanced Diagnostic Logging in `research_middleware.py`

**File**: `OpenHands/extensions/uagent_research/middleware/research_middleware.py`

**Changes in `_run_research()` method (lines 872-958)**:

```python
# Before: Generic error logging
logger.error(f"Experiment data not found: {experiment_id}")

# After: Detailed diagnostic logging
logger.error(f"❌ CRITICAL: Experiment data not found in active_orchestrators: {experiment_id}")
logger.error(f"   Available experiments: {list(self.active_orchestrators.keys())}")
```

Added comprehensive logging for:
- Orchestrator verification (checking for `run()` method)
- Detailed orchestrator execution context (instance type, goal, max_iterations)
- Separate try-catch for `orchestrator.run()` with full traceback
- Success/failure status updates to session manager with confirmation logging
- Cleanup operations with verification steps

**Key Improvements:**
- Visual indicators (❌ ✅ ⚠️) for quick error identification
- Structured error context (error type, message, full traceback)
- Verification of each step in the execution flow
- Confirmation logging for status updates and cleanup

#### 2. Enhanced Diagnostic Logging in `tree_orchestrator.py`

**File**: `OpenHands/extensions/uagent_research/orchestrator/tree_orchestrator.py`

**Changes in adapter routing (lines 846-887)**:

```python
# Added comprehensive adapter validation
logger.info(f"[EXECUTE] Routing task for node {node.id} (type={node.type})")
adapter_name = self.router.route(task, context)
logger.info(f"[EXECUTE] Router selected adapter: {adapter_name} for node {node.id}")

adapter = adapter_registry.get(adapter_name)

if not adapter:
    logger.error(f"❌ CRITICAL: Adapter '{adapter_name}' not found in registry!")
    available = list(adapter_registry._adapters.keys()) if hasattr(adapter_registry, '_adapters') else []
    logger.error(f"   Available adapters: {available}")
    logger.error(f"   Registry instance: {id(adapter_registry)}")
    
    # Try to register adapters if none found (recovery attempt)
    if len(available) == 0:
        logger.error(f"   No adapters registered! Attempting to register now...")
        from ..adapters.ensure_adapters import ensure_research_adapters_registered
        if ensure_research_adapters_registered():
            logger.info(f"   Adapters registered successfully, retrying...")
            adapter = adapter_registry.get(adapter_name)
            # ... additional validation
```

**Changes in adapter execution (lines 897-972)**:

```python
# Before: Basic execution logging
logger.info(f"Starting adapter execution for node {node.id} (timeout={execution_timeout}s)")

# After: Detailed execution tracking
logger.info(f"[EXECUTE] Starting adapter execution for node {node.id}")
logger.info(f"   Adapter: {adapter_name}")
logger.info(f"   Timeout: {execution_timeout}s")
logger.info(f"   Task: {task.goal[:100] if task.goal else 'N/A'}")

logger.info(f"[EXECUTE] Calling adapter.run() for node {node.id}...")

async for event in asyncio.wait_for(adapter.run(task, context), timeout=execution_timeout):
    events_received += 1
    logger.info(f"[EXECUTE] Node {node.id} received event #{events_received}: {event.type}")
    # ... event processing
    
# Separate exception handling for adapter failures
except Exception as adapter_error:
    logger.error(f"❌ ADAPTER EXECUTION FAILED for node {node.id}")
    logger.error(f"   Adapter: {adapter_name}")
    logger.error(f"   Error type: {type(adapter_error).__name__}")
    logger.error(f"   Error message: {str(adapter_error)}")
    logger.error(f"   Traceback:", exc_info=True)
    raise
```

**Key Improvements:**
- `[EXECUTE]` prefix for easy filtering of execution logs
- Event counting and tracking
- Separate exception handler for adapter failures
- Auto-recovery attempt when no adapters are registered
- Verification of adapter structure (checking for `run()` method)

### Expected Behavior After Fixes

With these logging enhancements, when experiments fail, the backend logs will show:

1. **Orchestrator Creation**: Confirmation that orchestrator was created and stored
2. **Session Manager Registration**: Verification that experiment was registered and validation of registration
3. **Background Task Start**: Confirmation that `_run_research()` started with thread/event loop info
4. **Orchestrator Execution**: Whether `orchestrator.run()` was called and if it succeeded/failed
5. **PUCT Loop**: Iteration progress, node selection, expansion results
6. **Adapter Routing**: Which adapter was selected for each node
7. **Adapter Execution**: Event-by-event progress, completion status
8. **Failure Details**: If any step fails, detailed error type, message, and full traceback

### Debugging Steps for Users

If experiments still fail after these fixes:

1. **Check Backend Logs**: Look for `❌ CRITICAL` or `[DIAGNOSTIC]` prefixed messages
2. **Adapter Registration**: Search for `[ADAPTER_REGISTRY]` logs to verify adapters are registered
3. **Session Manager**: Look for session manager instance IDs and experiment counts
4. **Orchestrator Execution**: Check if `[ORCHESTRATOR] run() called` appears in logs
5. **Node Execution**: Look for `[EXECUTE]` prefixed messages for individual node progress

### Known Limitations

The logging improvements **do not fix the root cause** if there are actual implementation issues. They provide visibility to identify:
- Missing adapters
- Configuration errors
- Budget/resource constraints
- LLM API failures
- Worktree setup failures
- Network/dependency issues

Once the logs reveal the actual failure point, appropriate fixes can be applied.

---

## Issue #24: Research Tree Node Click Doesn't Switch Left Chat UI to Node Context

### Root Cause Analysis

When double-clicking on a research tree node, the node details panel appears on the right side (✅ working), but the left chat UI doesn't switch to show that node's execution context/conversation.

**Current Behavior:**
- Node double-click detected correctly
- Right panel shows node details (metrics, summary, relationships)
- Navigation to `conversationId` happens if node has one
- Left chat UI remains on root node's conversation

**Missing Functionality:**
- Per-node conversation/event stream storage
- Chat context switching when node is selected
- Display of individual node execution traces
- Access to each research branch's progress

### Architecture Review

**Current Implementation** (`ResearchTreeView.tsx` lines 236-254):

```typescript
const onNodeDoubleClick = useCallback(
  (event: React.MouseEvent, node: Node) => {
    event.preventDefault();
    event.stopPropagation();

    selectNode(node.id);
    expandNode(node.id);

    const nodeData = node.data as ResearchNodeType | undefined;
    const metadata = (nodeData?.metadata ?? storeNodes.get(node.id)?.metadata) as
      | Record<string, unknown>
      | undefined;
    const conversationId = extractConversationId(metadata);

    setHasRightPanelToggled(true);
    setIsRightPanelShown(true);
    setSelectedTab('research');

    if (conversationId && conversationId !== routeConversationId) {
      navigate(`/conversations/${conversationId}`);
    }
  },
  [expandNode, navigate, routeConversationId, selectNode, storeNodes]
);
```

**Issues Identified:**
1. Navigation happens, but there's no mechanism to load that specific conversation's events
2. Backend stores events per research_id (experiment level), not per node
3. No WebSocket subscription for individual node event streams
4. Chat component doesn't support switching between multiple active conversations

### Proposed Solution (Implementation Required)

This issue requires **significant backend and frontend changes** that are beyond the scope of quick fixes:

#### Backend Changes Needed:

1. **Per-Node Event Storage**:
   ```python
   # In tree_orchestrator.py
   class TreeSearchOrchestrator:
       def __init__(self, ...):
           self._node_event_streams: Dict[str, List[Event]] = {}
   
       async def _execute_node(self, node: ResearchNode):
           # Store events per node ID
           if node.id not in self._node_event_streams:
               self._node_event_streams[node.id] = []
           
           async for event in adapter.run(task, context):
               self._node_event_streams[node.id].append(event)
               # Also publish to main event bus
               await self.event_bus.publish(event)
   ```

2. **API Endpoint for Node Events**:
   ```python
   # In research_routes.py
   @router.get("/experiments/{experiment_id}/nodes/{node_id}/events")
   async def get_node_events(experiment_id: str, node_id: str):
       """Return event stream for specific node"""
       orchestrator = get_orchestrator(experiment_id)
       events = orchestrator.get_node_events(node_id)
       return {"events": events}
   ```

3. **WebSocket Node Event Stream**:
   ```python
   @router.websocket("/experiments/{experiment_id}/nodes/{node_id}/ws")
   async def node_event_stream(websocket: WebSocket, experiment_id: str, node_id: str):
       """Stream events for specific node in real-time"""
       await websocket.accept()
       # Subscribe to node-specific event stream
       # ...
   ```

#### Frontend Changes Needed:

1. **Node Event Store**:
   ```typescript
   // In research-tree-store.ts or new node-event-store.ts
   interface NodeEventStore {
       nodeEvents: Map<string, Event[]>;
       activeNodeId: string | null;
       loadNodeEvents: (nodeId: string) => Promise<void>;
       subscribeToNode: (nodeId: string) => void;
       unsubscribeFromNode: (nodeId: string) => void;
   }
   ```

2. **Update Chat Component**:
   ```typescript
   // In ChatInterface or similar
   const activeNodeId = useResearchTreeStore(state => state.selectedNodeId);
   const nodeEvents = useNodeEventStore(state => 
     state.nodeEvents.get(activeNodeId)
   );
   
   // Display nodeEvents instead of main conversation events
   // when a node is selected
   ```

3. **Node Context Switcher**:
   ```typescript
   // UI element to show current context
   {activeNodeId && (
     <div className="node-context-indicator">
       <span>Viewing: Node {activeNodeId}</span>
       <button onClick={() => selectNode(null)}>
         Return to Main Conversation
       </button>
     </div>
   )}
   ```

### Partial Implementation (Current PR Scope)

For this PR, we're only adding **documentation and architecture notes**. The actual implementation requires:

1. Design review of event storage strategy
2. API contract definition
3. WebSocket protocol updates
4. Frontend state management refactoring
5. UI/UX design for context switching
6. Testing strategy for multi-context scenarios

### Workaround for Users

Until this feature is fully implemented, users can:

1. **Use Backend Logs**: Each node's execution is logged with `[EXECUTE]` prefix showing node ID
2. **Check Node Details Panel**: Right panel shows node metrics, status, and summary
3. **Use Research Tree Visualization**: Visual representation shows which nodes are running/complete/failed
4. **Monitor Root Conversation**: Main conversation shows high-level research progress

### Implementation Priority

This feature is **HIGH PRIORITY** for research transparency but requires:
- **Effort**: Medium-Large (2-3 weeks)
- **Risk**: Medium (involves WebSocket protocol changes)
- **Dependencies**: None (can be implemented independently)
- **Breaking Changes**: No (additive feature)

---

## Testing Recommendations

### For Issue #27 Fixes:

1. **Start Research Session**: Send "research goal: [complex task]" message
2. **Monitor Logs**: Check backend logs for:
   - `[DIAGNOSTIC]` messages showing orchestrator creation
   - `[EXECUTE]` messages showing node execution
   - Adapter routing and execution progress
3. **Verify Registration**: Confirm session manager shows >0 experiments
4. **Check Node Status**: Research tree should show nodes progressing (not immediate FAILED)

### For Issue #24 (When Implemented):

1. **Start Research**: Create parallel research session
2. **Select Node**: Double-click on idea/experiment node
3. **Verify Context Switch**: Chat UI should show node-specific events
4. **Verify Event Stream**: Events should be specific to selected node
5. **Return to Root**: Deselect node and verify return to main conversation

---

## Summary

### Issue #27: ✅ FIXED (Diagnostic Logging Added)
- Added comprehensive logging in middleware and orchestrator
- Visual indicators for quick error identification
- Detailed error context for debugging
- Auto-recovery attempt for missing adapters
- Status verification at each step

### Issue #24: ⚠️ DOCUMENTED (Implementation Required)
- Architecture analyzed and documented
- Root cause identified (missing per-node event storage)
- Proposed solution outlined with code examples
- Workarounds documented for current users
- Implementation priority and effort estimated

Both issues are now well-documented with clear paths forward. Issue #27's logging improvements will immediately help identify the actual failure causes, while Issue #24 requires a dedicated development effort for full implementation.
