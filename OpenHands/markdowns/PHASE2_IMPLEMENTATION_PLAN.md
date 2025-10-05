# Phase 2 Implementation Plan - Frontend & Real Agent Integration

**Status**: Ready to implement
**Based on**: Codex recommendations + ROMA reference implementation

---

## 🎯 Phase 2 Goals

1. **Hybrid PUCT + Recursive Architecture** - Combine PUCT's adaptive exploration with ROMA's recursive decomposition
2. **Frontend Research Tree Visualization** - ReactFlow-based tree view with top-right toggle
3. **Real-time WebSocket Streaming** - Bridge EventBus to WebSocket for live updates
4. **Real Agent Integration** - Connect actual CodeActAgent, improve DeepResearch/RepoMaster
5. **Full API Suite** - Complete REST + WebSocket endpoints

---

## 📐 Architecture Decisions (from Codex)

### 1. Hybrid PUCT + Recursive Strategy

**Keep**: PUCT for adaptive exploration
**Add**: ROMA-style recursive Atomizer → Planner → Executors → Aggregator

**How it works**:
- ROMA's Planner generates dependency-aware DAG (task graph)
- PUCT selects which frontier node to expand/execute next
- Aggregator updates node Q values from execution results
- Router provides prior P values for new nodes

**Implementation**:
- Add `Atomizer` to determine if task is atomic or needs planning
- Add `Planner` adapter that generates subtasks with priors
- Keep `TreeSearchOrchestrator` but use PUCT for node selection
- Add `Aggregator` to compute Q values from child results

### 2. Frontend Architecture

**Component Structure**:
```
ConversationHeader
├── ...existing buttons...
└── ResearchToggleButton ← NEW (top-right)

ResearchTreePanel (overlay) ← NEW
├── ResearchTreeView (ReactFlow)
│   ├── TaskNode components
│   ├── CustomEdge components
│   └── Controls (fit, center, zoom)
├── NodeDetailPanel
│   ├── Node info (status, costs, artifacts)
│   └── Action buttons (expand, cancel)
└── StatsPanel
    └── Budget, progress, totals
```

**State Management**:
```typescript
// zustand store
interface ResearchTreeStore {
  version: number
  nodes: ResearchNode[]
  edges: ResearchEdge[]
  stats: TreeStats
  selectedNodeId: string | null

  // Actions
  updateFromSnapshot: (snapshot) => void
  applyNodeAdded: (event) => void
  applyNodeUpdated: (event) => void
  applyEdgeAdded: (event) => void
}
```

### 3. Event Streaming

**Keep**: EventBus for internal pub/sub + coalescing
**Add**: WebSocket bridge (`ws_publisher.py`) - **✅ DONE**

**Message Types** (ROMA-compatible):
- `tree_snapshot` - Full tree state
- `node_added` - New node with parent_id
- `node_updated` - Node status/data changes
- `edge_added` - New edge
- `stats_updated` - Budget/progress updates
- `event_log` - Research event logs
- `error` - Error messages
- `complete` - Completion notifications

---

## 🔧 Implementation Tasks

### Backend Tasks

#### 1. WebSocket Bridge ✅ COMPLETED
**File**: `orchestrator/ws_publisher.py`
**Status**: Created (425 lines)
**Features**:
- Subscribes to EventBus
- Maps ResearchEvent → WebSocket messages
- Monotonic version tracking
- Broadcasting to experiment-specific clients

#### 2. Add Tree Snapshot API Endpoint
**File**: `uagent_research/api/research_routes.py`
**Add**:
```python
@router.get("/experiments/{experiment_id}/tree")
async def get_experiment_tree(experiment_id: str):
    """Get research tree snapshot"""
    orchestrator = get_orchestrator(experiment_id)
    tree = orchestrator.tree

    return {
        "version": orchestrator.version,
        "nodes": [...],
        "edges": [...],
        "stats": tree.stats
    }

@router.get("/experiments/{experiment_id}/events")
async def get_experiment_events(
    experiment_id: str,
    since_version: int = 0
):
    """Get events since version (incremental fetch)"""
    # Return events newer than since_version
    ...

@router.patch("/experiments/{experiment_id}")
async def control_experiment(
    experiment_id: str,
    action: str  # pause, resume, cancel
):
    """Control experiment execution"""
    ...
```

#### 3. Integrate Real CodeActAgent
**File**: `adapters/codeact/adapter.py`
**Changes**:
```python
from openhands.agenthub.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.runtime import Runtime

class CodeActAdapter(AgentAdapter):
    async def _execute_code_task(self, task, context):
        # Initialize actual CodeActAgent
        agent = CodeActAgent(llm=self.llm)

        # Create runtime
        runtime = Runtime(...)

        # Execute and stream events
        state = State(...)
        async for action in agent.step(state):
            # Convert OpenHands events → ResearchEvents
            yield self._convert_action(action)
```

#### 4. Improve DeepResearch & RepoMaster Adapters
**Current**: Simplified implementations using BingSearch + WebBrowse
**Improve**:
- Add actual DeepResearch ReAct loop integration
- Add RepoMaster autogen scheduler integration
- Emit richer artifacts (code snippets, URLs, evidence)

#### 5. Add Planner Adapter (ROMA-style)
**File**: `adapters/planner/adapter.py` (new)
**Purpose**: Recursive task decomposition
```python
class PlannerAdapter(AgentAdapter):
    """
    ROMA-style recursive planner.

    Atomizes tasks and generates subtask decomposition.
    """

    async def run(self, task, context):
        # 1. Atomizer: Is this atomic?
        if self._is_atomic(task):
            # Route to executor (deepresearch/repomaster/codeact)
            ...
        else:
            # 2. Planner: Decompose into subtasks
            subtasks = await self._plan(task)

            # Emit subtasks with priors
            for subtask in subtasks:
                yield PlanEvent(
                    subtasks=[...],
                    priors=[...]  # Confidence scores
                )

    def _is_atomic(self, task) -> bool:
        """Determine if task is atomic (executable)"""
        # Use LLM or heuristics
        ...

    async def _plan(self, task) -> List[Task]:
        """Generate subtask decomposition"""
        # Use LLM to break down task
        ...
```

#### 6. Update TreeSearchOrchestrator
**File**: `orchestrator/tree_orchestrator.py`
**Changes**:
- Add Planner adapter integration
- Keep PUCT for frontier selection
- Add Aggregator for Q value computation
- Add WebSocket publisher instantiation

---

### Frontend Tasks

#### 1. Create ResearchTreeStore
**File**: `frontend/src/stores/researchTreeStore.ts` (new)
```typescript
import { create } from 'zustand'

interface ResearchNode {
  id: string
  type: string
  title: string
  status: string
  visits: number
  prior: number
  avg_value: number
  cost: number
  // ... more fields
}

interface ResearchTreeStore {
  version: number
  nodes: Map<string, ResearchNode>
  edges: Array<{parent_id: string, child_id: string}>
  stats: any
  selectedNodeId: string | null

  // Actions
  setSnapshot: (snapshot: any) => void
  applyNodeAdded: (event: any) => void
  applyNodeUpdated: (event: any) => void
  applyEdgeAdded: (event: any) => void
  applyStatsUpdated: (event: any) => void
  selectNode: (nodeId: string) => void
}

export const useResearchTreeStore = create<ResearchTreeStore>((set) => ({
  version: 0,
  nodes: new Map(),
  edges: [],
  stats: {},
  selectedNodeId: null,

  setSnapshot: (snapshot) => set({
    version: snapshot.version,
    nodes: new Map(snapshot.data.nodes.map(n => [n.id, n])),
    edges: snapshot.data.edges,
    stats: snapshot.data.stats
  }),

  applyNodeAdded: (event) => set((state) => {
    const newNodes = new Map(state.nodes)
    newNodes.set(event.data.node_id, event.data.node)
    return { nodes: newNodes, version: event.version }
  }),

  // ... more actions
}))
```

#### 2. Create useResearchWS Hook
**File**: `frontend/src/hooks/useResearchWS.ts` (new)
```typescript
import { useEffect, useRef } from 'react'
import { useResearchTreeStore } from '@/stores/researchTreeStore'

export function useResearchWS(experimentId: string) {
  const ws = useRef<WebSocket | null>(null)
  const store = useResearchTreeStore()

  useEffect(() => {
    // Connect to WebSocket
    const wsUrl = `ws://${window.location.host}/api/research/ws/experiment/${experimentId}`
    ws.current = new WebSocket(wsUrl)

    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data)

      // Route message to store
      switch (message.type) {
        case 'tree_snapshot':
          store.setSnapshot(message)
          break
        case 'node_added':
          store.applyNodeAdded(message)
          break
        case 'node_updated':
          store.applyNodeUpdated(message)
          break
        // ... more cases
      }
    }

    return () => {
      ws.current?.close()
    }
  }, [experimentId])

  return { connected: ws.current?.readyState === WebSocket.OPEN }
}
```

#### 3. Create ResearchTreeView Component
**File**: `frontend/src/components/research/ResearchTreeView.tsx` (new)
**Reference**: `/home/wuy/AI/UAgent/ROMA/frontend/src/components/graph/GraphVisualization.tsx`
```typescript
import React from 'react'
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  BackgroundVariant,
  NodeTypes,
  useReactFlow
} from 'reactflow'
import 'reactflow/dist/style.css'

import { useResearchTreeStore } from '@/stores/researchTreeStore'
import ResearchNode from './ResearchNode'

const nodeTypes: NodeTypes = {
  researchNode: ResearchNode
}

export function ResearchTreeView() {
  const { nodes, edges, selectNode } = useResearchTreeStore()
  const { fitView } = useReactFlow()

  // Convert store nodes to ReactFlow nodes
  const flowNodes = Array.from(nodes.values()).map(node => ({
    id: node.id,
    type: 'researchNode',
    position: { x: 0, y: 0 }, // Layout algorithm needed
    data: node
  }))

  const flowEdges = edges.map(edge => ({
    id: `${edge.parent_id}-${edge.child_id}`,
    source: edge.parent_id,
    target: edge.child_id,
    type: 'smoothstep'
  }))

  return (
    <ReactFlow
      nodes={flowNodes}
      edges={flowEdges}
      nodeTypes={nodeTypes}
      onNodeClick={(_, node) => selectNode(node.id)}
      fitView
    >
      <Controls />
      <Background variant={BackgroundVariant.Dots} />
    </ReactFlow>
  )
}
```

#### 4. Create ResearchNode Component
**File**: `frontend/src/components/research/ResearchNode.tsx` (new)
**Reference**: `/home/wuy/AI/UAgent/ROMA/frontend/src/components/graph/nodes/TaskNode.tsx`
```typescript
import React from 'react'
import { Handle, Position } from 'reactflow'

export function ResearchNode({ data }: { data: any }) {
  const statusColor = {
    pending: 'gray',
    running: 'blue',
    complete: 'green',
    failed: 'red'
  }[data.status] || 'gray'

  return (
    <div className="research-node" style={{ borderColor: statusColor }}>
      <Handle type="target" position={Position.Top} />

      <div className="node-header">
        <div className="node-type">{data.type}</div>
        <div className="node-status">{data.status}</div>
      </div>

      <div className="node-title">{data.title}</div>

      <div className="node-stats">
        <div>Visits: {data.visits}</div>
        <div>Q: {data.avg_value.toFixed(2)}</div>
        <div>Cost: ${data.cost.toFixed(3)}</div>
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}
```

#### 5. Create ResearchTreePanel
**File**: `frontend/src/components/research/ResearchTreePanel.tsx` (new)
```typescript
import React, { useEffect } from 'react'
import { ReactFlowProvider } from 'reactflow'
import { ResearchTreeView } from './ResearchTreeView'
import { useResearchWS } from '@/hooks/useResearchWS'
import { useResearchTreeStore } from '@/stores/researchTreeStore'

interface Props {
  experimentId: string
  onClose: () => void
}

export function ResearchTreePanel({ experimentId, onClose }: Props) {
  const { connected } = useResearchWS(experimentId)
  const { stats } = useResearchTreeStore()

  // Fetch initial snapshot
  useEffect(() => {
    fetch(`/api/research/experiments/${experimentId}/tree`)
      .then(r => r.json())
      .then(snapshot => useResearchTreeStore.getState().setSnapshot(snapshot))
  }, [experimentId])

  return (
    <div className="research-tree-panel">
      <div className="panel-header">
        <h2>Research Tree</h2>
        <div className="connection-status">
          {connected ? '🟢 Connected' : '🔴 Disconnected'}
        </div>
        <button onClick={onClose}>✕</button>
      </div>

      <div className="panel-stats">
        <div>Nodes: {stats.total_nodes || 0}</div>
        <div>Cost: ${stats.total_cost?.toFixed(2) || '0.00'}</div>
        <div>Depth: {stats.max_depth || 0}</div>
      </div>

      <div className="panel-graph">
        <ReactFlowProvider>
          <ResearchTreeView />
        </ReactFlowProvider>
      </div>
    </div>
  )
}
```

#### 6. Add Research Toggle Button
**File**: `frontend/src/routes/conversation.tsx`
**Location**: In the conversation header (find ConversationTabs)
```typescript
import { ResearchTreePanel } from '@/components/research/ResearchTreePanel'

function ConversationPage() {
  const [showResearchTree, setShowResearchTree] = useState(false)
  const experimentId = useExperimentId() // Get from context or URL

  return (
    <>
      {/* Existing conversation UI */}
      <div className="conversation-header">
        {/* ...existing buttons... */}

        <button
          className="research-toggle-button"
          onClick={() => setShowResearchTree(!showResearchTree)}
        >
          🔬 Research
        </button>
      </div>

      {/* Research panel overlay */}
      {showResearchTree && (
        <ResearchTreePanel
          experimentId={experimentId}
          onClose={() => setShowResearchTree(false)}
        />
      )}
    </>
  )
}
```

---

## 📋 Implementation Order

### Week 2 (Current)

**Backend** (2-3 days):
1. ✅ WebSocket bridge - DONE
2. Add tree snapshot endpoint
3. Add control endpoints (pause/resume/cancel)
4. Integrate real CodeActAgent
5. Add Planner adapter (optional, can defer)

**Frontend** (2-3 days):
6. Create researchTreeStore
7. Create useResearchWS hook
8. Create ResearchTreeView with ReactFlow
9. Create ResearchNode component
10. Add toggle button to conversation UI
11. Style and polish

**Testing** (1 day):
12. End-to-end workflow test
13. WebSocket stress test
14. UI/UX refinement

---

## 🎨 Styling Reference

**From ROMA**: `/home/wuy/AI/UAgent/ROMA/frontend/src/index.css`
- Dark theme compatible
- ReactFlow custom styles
- Node/edge styling
- Panel overlays

**Key CSS Classes**:
```css
.research-tree-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: 50%;
  height: 100vh;
  background: var(--background);
  border-left: 1px solid var(--border);
  z-index: 1000;
}

.research-node {
  padding: 12px;
  border: 2px solid;
  border-radius: 8px;
  background: white;
  min-width: 200px;
}
```

---

## 🧪 Testing Strategy

### Backend Tests
```python
# Test WebSocket bridge
async def test_ws_publisher():
    bus = EventBus()
    publisher = WebSocketPublisher(bus, ws_manager, "exp-1")

    await publisher.start()
    await bus.publish(StepEvent(...))
    # Assert message broadcast

# Test tree snapshot endpoint
async def test_get_tree():
    response = await client.get("/api/research/experiments/exp-1/tree")
    assert response.status_code == 200
    assert "version" in response.json()
```

### Frontend Tests
```typescript
// Test ResearchTreeStore
test('applyNodeAdded updates store', () => {
  const store = useResearchTreeStore.getState()
  store.applyNodeAdded({
    type: 'node_added',
    version: 1,
    data: { node_id: 'test-1', node: {...} }
  })

  expect(store.nodes.has('test-1')).toBe(true)
})

// Test WebSocket connection
test('useResearchWS connects and handles messages', () => {
  const { result } = renderHook(() => useResearchWS('exp-1'))
  // Assert WS connection and message handling
})
```

---

## 📝 Next Steps

1. **Review this plan** - Confirm approach
2. **Start backend endpoints** - Tree snapshot, control APIs
3. **Start frontend components** - Store, hooks, views
4. **Test incrementally** - Each component as it's built
5. **Integration testing** - Full end-to-end workflow

---

## 🔗 References

- ROMA GraphVisualization: `/home/wuy/AI/UAgent/ROMA/frontend/src/components/graph/GraphVisualization.tsx`
- ROMA WebSocket types: `/home/wuy/AI/UAgent/ROMA/frontend/src/types/websocket.ts`
- Phase 1 completion: `PHASE1_COMPLETION_SUMMARY.md`
- Codex recommendations: (in session context)

