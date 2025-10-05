# Research Tree Integration Example

This guide shows how to integrate the Research Tree visualization into OpenHands' conversation UI.

---

## 1. Install Dependencies

```bash
cd /home/wuy/AI/UAgent/OpenHands/frontend
npm install
```

The following packages have been added to `package.json`:
- `reactflow` ^11.11.0
- `dagre` ^0.8.5
- `@types/dagre` ^0.7.52

---

## 2. Import CSS (Already Done ✅)

The CSS import has been added to `/frontend/src/index.css`:

```css
@import './components/research/research-tree.css';
```

---

## 3. Add Research Toggle Button

### Option A: Quick Integration (Recommended for Testing)

Create a test page to try the research tree:

**File**: `frontend/src/routes/research-test.tsx` (new)

```typescript
import { useState } from 'react';
import { ResearchTreePanel } from '#/components/research';

export default function ResearchTestPage() {
  const [experimentId] = useState('exp-test-123');

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
      <h1 style={{ padding: '20px' }}>Research Tree Test</h1>
      <p style={{ padding: '0 20px' }}>
        Experiment ID: {experimentId}
      </p>

      <ResearchTreePanel
        experimentId={experimentId}
        onClose={() => console.log('Close clicked')}
      />
    </div>
  );
}
```

Then visit: `http://localhost:3000/research-test`

### Option B: Full Conversation Integration

Find the conversation header in your conversation route (usually `frontend/src/routes/conversation.tsx` or similar):

```typescript
import { useState } from 'react';
import { ResearchTreePanel } from '#/components/research';

function ConversationPage() {
  const [showResearchTree, setShowResearchTree] = useState(false);

  // Get experiment ID from your conversation state/context
  // This could come from Redux, context, or URL params
  const experimentId = useExperimentId(); // Implement this based on your state management

  return (
    <div className="conversation-container">
      {/* Conversation Header */}
      <div className="conversation-header">
        {/* ...existing header buttons... */}

        {/* Research Toggle Button */}
        <button
          className="research-toggle-btn"
          onClick={() => setShowResearchTree(!showResearchTree)}
          style={{
            padding: '8px 16px',
            background: showResearchTree ? '#3b82f6' : '#e5e7eb',
            color: showResearchTree ? 'white' : '#1f2937',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            transition: 'all 0.2s',
          }}
        >
          🔬 Research Tree
        </button>
      </div>

      {/* Main conversation content */}
      <div className="conversation-content">
        {/* ...existing conversation UI... */}
      </div>

      {/* Research Tree Overlay */}
      {showResearchTree && experimentId && (
        <ResearchTreePanel
          experimentId={experimentId}
          onClose={() => setShowResearchTree(false)}
        />
      )}
    </div>
  );
}
```

---

## 4. Backend Integration

### A. Register Active Orchestrators

When starting a research experiment, register the orchestrator:

**File**: `uagent_research/api/research_routes.py`

```python
from ..orchestrator.tree_orchestrator import TreeSearchOrchestrator
from ..adapters.deepresearch.adapter import DeepResearchAdapter
from ..adapters.repomaster.adapter import RepoMasterAdapter
from ..adapters.codeact.adapter import CodeActAdapter
from ..adapters.base.agent_adapter import adapter_registry

@router.post("/experiments/start")
async def start_experiment(request: StartResearchRequest, ...):
    # Create experiment in database
    experiment = Experiment(...)
    session.add(experiment)
    await session.commit()

    # Register adapters
    adapter_registry.register(DeepResearchAdapter())
    adapter_registry.register(RepoMasterAdapter())
    adapter_registry.register(CodeActAdapter())

    # Create orchestrator
    orchestrator = TreeSearchOrchestrator(
        max_parallel=3,
        budget=Budget(max_cost=1.0, max_iterations=10)
    )

    # Register in global dict
    _active_orchestrators[experiment.id] = orchestrator

    # Start in background
    background_tasks.add_task(run_experiment, experiment.id, orchestrator, request.goal)

    return ExperimentResponse(...)
```

### B. Run Experiment with WebSocket Publisher

```python
async def run_experiment(experiment_id: str, orchestrator: TreeSearchOrchestrator, goal: str):
    """Run experiment in background"""
    try:
        # Initialize WebSocket publisher
        from ..orchestrator.ws_publisher import WebSocketPublisher

        # Get WebSocket manager (this depends on your FastAPI setup)
        # ws_manager = get_websocket_manager()  # Implement this

        # publisher = WebSocketPublisher(
        #     event_bus=orchestrator.event_bus,
        #     ws_manager=ws_manager,
        #     experiment_id=experiment_id
        # )
        # await publisher.start()

        # Run tree search
        tree = await orchestrator.run(
            goal=goal,
            max_iterations=10
        )

        # Update experiment status
        # ...

    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}")
        # Update experiment with error
    finally:
        # Cleanup
        _active_orchestrators.pop(experiment_id, None)
        # if publisher:
        #     await publisher.stop()
```

---

## 5. WebSocket Endpoint Setup

Ensure WebSocket endpoint exists in `uagent_research/api/websocket_routes.py`:

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, experiment_id: str):
        await websocket.accept()
        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = []
        self.active_connections[experiment_id].append(websocket)

    def disconnect(self, websocket: WebSocket, experiment_id: str):
        if experiment_id in self.active_connections:
            self.active_connections[experiment_id].remove(websocket)

    async def broadcast(self, message: str, experiment_id: str):
        if experiment_id in self.active_connections:
            for connection in self.active_connections[experiment_id]:
                await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/experiment/{experiment_id}")
async def websocket_experiment_endpoint(
    websocket: WebSocket,
    experiment_id: str
):
    await manager.connect(websocket, experiment_id)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket, experiment_id)
```

---

## 6. Testing the Integration

### Manual Test

1. **Start Backend**:
   ```bash
   cd /home/wuy/AI/UAgent/OpenHands
   # Start your backend server
   ```

2. **Start Frontend**:
   ```bash
   cd /home/wuy/AI/UAgent/OpenHands/frontend
   npm run dev
   ```

3. **Create Test Experiment**:
   ```bash
   # Using curl or your API client
   curl -X POST http://localhost:3000/api/research/experiments/start \
     -H "Content-Type: application/json" \
     -d '{
       "goal": "Research neural architecture search",
       "session_id": "test-session",
       "research_type": "scientific"
     }'
   ```

4. **Open Research Tree**:
   - Navigate to conversation page
   - Click "🔬 Research Tree" button
   - Panel should appear on right side
   - Connection indicator should show green (connected)

### Automated Test

**File**: `frontend/src/components/research/__tests__/ResearchTreePanel.test.tsx`

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import { ResearchTreePanel } from '../ResearchTreePanel';

// Mock fetch
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({
      version: 1,
      timestamp: new Date().toISOString(),
      experiment_id: 'test-exp',
      data: {
        nodes: [
          {
            id: 'root',
            type: 'root',
            title: 'Test Root',
            status: 'complete',
            visits: 1,
            prior: 0.5,
            avg_value: 0.8,
            cost: 0.01,
            tokens_used: 100,
          }
        ],
        edges: [],
        stats: { total_nodes: 1 }
      }
    })
  })
) as jest.Mock;

// Mock WebSocket
class MockWebSocket {
  onopen: (() => void) | null = null;
  onmessage: ((event: any) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  readyState = WebSocket.OPEN;

  constructor(public url: string) {
    setTimeout(() => {
      if (this.onopen) this.onopen();
    }, 0);
  }

  close() {}
  send() {}
}

global.WebSocket = MockWebSocket as any;

describe('ResearchTreePanel', () => {
  it('fetches and displays tree snapshot', async () => {
    render(<ResearchTreePanel experimentId="test-exp" onClose={() => {}} />);

    await waitFor(() => {
      expect(screen.getByText('Research Tree')).toBeInTheDocument();
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });
  });
});
```

---

## 7. Troubleshooting

### Issue: WebSocket not connecting

**Check**:
1. WebSocket endpoint is registered: `/api/research/ws/experiment/{id}`
2. CORS settings allow WebSocket connections
3. Frontend is using correct protocol (ws:// or wss://)

**Fix**:
- Check browser console for WebSocket errors
- Verify backend logs for connection attempts

### Issue: Tree not displaying

**Check**:
1. Experiment ID is valid
2. Orchestrator is registered in `_active_orchestrators`
3. Tree has nodes (check `/api/research/experiments/{id}/tree`)

**Fix**:
- Verify API response: `curl http://localhost:3000/api/research/experiments/{id}/tree`
- Check browser console for errors

### Issue: CSS not loading

**Check**:
1. CSS import in `index.css`
2. Build system includes CSS files
3. File paths are correct

**Fix**:
- Restart frontend dev server
- Clear browser cache
- Check network tab for 404s

---

## 8. Next Steps

1. ✅ Install dependencies (`npm install`)
2. ✅ Import CSS (already done)
3. 🔄 Add toggle button to conversation UI
4. 🔄 Start experiment via API
5. 🔄 Test WebSocket connection
6. 🔄 Verify tree visualization

---

## Example: Full Working Flow

```typescript
// 1. User clicks "Start Research" button
async function startResearch(goal: string) {
  const response = await fetch('/api/research/experiments/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      goal,
      session_id: currentSessionId,
      research_type: 'scientific'
    })
  });

  const experiment = await response.json();
  return experiment.id; // "exp-abc123"
}

// 2. User clicks "🔬 Research Tree" button
function ConversationPage() {
  const [experimentId, setExperimentId] = useState<string | null>(null);
  const [showTree, setShowTree] = useState(false);

  const handleStartResearch = async () => {
    const id = await startResearch("Find papers on NAS");
    setExperimentId(id);
    setShowTree(true);
  };

  return (
    <>
      <button onClick={handleStartResearch}>Start Research</button>

      {showTree && experimentId && (
        <ResearchTreePanel
          experimentId={experimentId}
          onClose={() => setShowTree(false)}
        />
      )}
    </>
  );
}

// 3. Backend starts orchestrator
// 4. Nodes are created and streamed via WebSocket
// 5. Frontend updates in real-time
// 6. User sees tree grow as research progresses
```

---

## Summary

**Files Modified**:
- ✅ `frontend/package.json` - Added dependencies
- ✅ `frontend/src/index.css` - Imported research CSS

**Files Created**:
- ✅ All research components (`frontend/src/components/research/`)
- ✅ Research store (`frontend/src/state/research-tree-store.ts`)
- ✅ WebSocket hook (`frontend/src/hooks/useResearchWS.ts`)

**Ready to Integrate**: Follow steps 3-6 above to complete integration!
