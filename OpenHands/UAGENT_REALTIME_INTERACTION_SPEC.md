# UAgent Real-Time Interaction & CodeAct Integration - Technical Specification

**Version**: 1.0
**Date**: 2025-10-06
**Status**: Ready for Implementation

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Architecture Design](#architecture-design)
4. [Key Components](#key-components)
5. [Event Flow](#event-flow)
6. [API Interfaces](#api-interfaces)
7. [Code Structure](#code-structure)
8. [Implementation Plan](#implementation-plan)
9. [User Experience Scenarios](#user-experience-scenarios)

---

## Overview

This specification addresses two critical enhancements to the OpenHands-UAgent system:

1. **Real OpenHands CodeAct Integration**: Replace placeholder CodeActAdapter with actual OpenHands CodeActAgent
2. **Real-Time Interaction**: Enable users to monitor progress and steer research while sub-agents are running

### Design Principles

- **Non-Blocking**: Main chat remains responsive during research
- **Event-Driven**: Use signaling + heartbeats for progress updates
- **Concurrent Control**: User can interact with research mid-execution
- **Reuse OpenHands**: Leverage existing CodeActAgent via embedded sessions

---

## Requirements

### 1. Real OpenHands CodeAct Integration

**Current State**: CodeActAdapter is a placeholder with mock execution
**Target**: Use actual `openhands/agenthub/codeact_agent/CodeActAgent`

**Why**:
- OpenHands CodeAct is battle-tested for code execution
- Has robust tool ecosystem (bash, ipython, file edit, browser)
- Better error handling and safety checks
- Superior to self-written code execution

### 2. Real-Time Interaction During Research

**User Capabilities**:
- ✅ Ask "how's progress?" → Get instant summary
- ✅ Ask "what are subagents doing?" → See current tasks
- ✅ Send control commands: "pause", "resume", "cancel"
- ✅ Steer research: "focus on DARTS not ENAS"
- ✅ Cancel specific branches: "stop DeepResearch", "cancel idea-2"

**Technical Requirements**:
- Bidirectional communication: Main ↔ Orchestrator ↔ Subagents
- Heartbeat/progress from all running agents
- Main agent receives status without blocking
- Control messages routed to orchestrator/subagents

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Chat Panel  │  │ Research Tab │  │ Control Panel│       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │               │
└─────────┼─────────────────┼──────────────────┼───────────────┘
          │                 │                  │
          ║ WebSocket      ║ WebSocket        ║ REST/WS
          ║ (chat)         ║ (events)         ║ (control)
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            ResearchSessionManager                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │  │
│  │  │ Experiment   │  │ Status       │  │ Control     │  │  │
│  │  │ Registry     │  │ Aggregator   │  │ Router      │  │  │
│  │  └──────────────┘  └──────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────┐           ┌───────────────────────────┐  │
│  │  EventBus     │◄──────────┤  TreeSearchOrchestrator   │  │
│  │  (Progress)   │           │                           │  │
│  └───────┬───────┘           │  ┌─────────────────────┐  │  │
│          │                   │  │  Control Loop       │  │  │
│          │                   │  │  (ControlBus)       │  │  │
│          │                   │  └─────────────────────┘  │  │
│          │                   │                           │  │
│          │                   │  ┌─────────────────────┐  │  │
│          │                   │  │  PUCT Loop          │  │  │
│          │                   │  │  (Node Selection)   │  │  │
│          │                   │  └─────────────────────┘  │  │
│          │                   └───────────────────────────┘  │
│          │                              │                   │
│          │                              │                   │
│          ▼                              ▼                   │
│  ┌───────────────┐           ┌────────────────────────────┐│
│  │  WS Publisher │           │  Adapters (Parallel)       ││
│  └───────────────┘           │                            ││
│          │                   │  ┌──────────────────────┐  ││
│          │                   │  │ DeepResearchAdapter  │  ││
│          │                   │  │ (Web Search/Browse)  │  ││
│          │                   │  └──────────────────────┘  ││
│          │                   │                            ││
│          │                   │  ┌──────────────────────┐  ││
│          │                   │  │ RepoMasterAdapter    │  ││
│          │                   │  │ (GitHub Search)      │  ││
│          │                   │  └──────────────────────┘  ││
│          │                   │                            ││
│          │                   │  ┌──────────────────────┐  ││
│          │                   │  │ CodeActAdapter       │  ││
│          │                   │  │ ┌─────────────────┐  │  ││
│          │                   │  │ │ HeadlessAgent   │  │  ││
│          │                   │  │ │ Session         │  │  ││
│          │                   │  │ │ ┌─────────────┐ │  │  ││
│          │                   │  │ │ │ CodeActAgent│ │  │  ││
│          │                   │  │ │ │ (Real OH)   │ │  │  ││
│          │                   │  │ │ └─────────────┘ │  │  ││
│          │                   │  │ │ EventStream     │  │  ││
│          │                   │  │ │ Bridge          │  │  ││
│          │                   │  │ └─────────────────┘  │  ││
│          │                   │  └──────────────────────┘  ││
│          │                   └────────────────────────────┘│
│          │                              │                   │
│          └──────────────────────────────┘                   │
│                         (Events)                             │
└─────────────────────────────────────────────────────────────┘
```

### Event Flow Diagram

```
1. User Message → research_middleware (classify)
   ↓
2. Start TreeSearchOrchestrator (background, non-blocking)
   ↓
3. Orchestrator: PUCT Loop
   │
   ├─→ Select Node
   ├─→ Expand Node (generate children)
   ├─→ Execute Children (parallel via semaphore)
   │   ├─→ DeepResearch: search web → EventBus
   │   ├─→ RepoMaster: search GitHub → EventBus
   │   └─→ CodeAct: run code → EventBus
   │       ├─→ Create HeadlessAgentSession
   │       ├─→ Start CodeActAgent
   │       ├─→ Subscribe to EventStream
   │       └─→ Bridge OH Events → ResearchEvents
   │
   ├─→ Update Tree Stats
   ├─→ Publish to API (update_tree_state)
   └─→ Check Budget → Continue?

Meanwhile (Concurrent):

4. EventBus → WebSocketPublisher → Frontend
   │
   └─→ Research Tab displays tree + events

5. User: "how's progress?" → research_middleware
   ↓
   ResearchSessionManager.get_status()
   ↓
   Main Chat: "3 nodes complete, 2 running (DeepResearch, CodeAct)..."

6. User: "cancel idea-2" → ControlBus
   ↓
   Orchestrator Control Loop processes
   ↓
   Cancel tasks, mark node FAILED
   ↓
   EventBus → UI update
```

---

## Key Components

### 1. **ControlBus** (NEW)

**Purpose**: Route control commands from UI/main agent to orchestrator/subagents

**Location**: `extensions/uagent_research/control/control_bus.py`

**Interface**:
```python
class ControlMessage(BaseModel):
    """Control command message"""
    action: str  # pause, resume, cancel, reprioritize, steer, etc.
    target: Dict[str, str]  # {experiment_id?, branch_id?, adapter?, node_id?}
    payload: Dict[str, Any]  # Action-specific data

class ControlBus:
    """Typed command bus for research control"""

    async def publish(self, experiment_id: str, message: ControlMessage):
        """Publish control command"""

    async def subscribe(self, experiment_id: str) -> AsyncIterator[ControlMessage]:
        """Subscribe to control commands for experiment"""

    async def unsubscribe(self, experiment_id: str):
        """Cleanup subscription"""
```

**Actions**:
- `pause`: Pause research (stop scheduling new nodes)
- `resume`: Resume research
- `cancel`: Cancel entire experiment
- `cancel_node`: Cancel specific branch/node
- `reprioritize`: Adjust PUCT priors/exploration constant
- `add_node`: Inject new research direction
- `steer`: Send guidance to subagents

---

### 2. **ResearchSessionManager** (NEW)

**Purpose**: Lifecycle + state registry for research experiments

**Location**: `extensions/uagent_research/services/research_session_manager.py`

**Interface**:
```python
class ResearchSessionManager:
    """Manages active research sessions and aggregates status"""

    def __init__(self, event_bus: EventBus, control_bus: ControlBus):
        self.experiments: Dict[str, ExperimentState] = {}
        self.event_bus = event_bus
        self.control_bus = control_bus

    def register(
        self,
        experiment_id: str,
        orchestrator: TreeSearchOrchestrator,
        ws_publisher: WebSocketPublisher
    ):
        """Register new experiment"""

    def get_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get aggregated status summary

        Returns:
            {
                "experiment_id": "...",
                "status": "running|paused|complete|failed",
                "stats": {
                    "total_nodes": 10,
                    "completed": 5,
                    "failed": 1,
                    "running": 4
                },
                "adapters": {
                    "deepresearch": {
                        "status": "running",
                        "current_step": "Browsing https://...",
                        "last_event": "2025-10-06T10:30:00",
                        "cost": 0.05,
                        "tokens": 1200
                    },
                    "codeact": {...}
                },
                "active_branches": [
                    {
                        "branch_id": "idea-0-hyp-0",
                        "title": "Test using pgvector",
                        "adapter": "codeact",
                        "status": "running",
                        "progress": "Executing benchmark..."
                    }
                ]
            }
        """

    def list_active(self) -> List[str]:
        """List active experiment IDs"""

    async def send_control(self, experiment_id: str, command: ControlMessage):
        """Send control command to experiment"""

    async def _subscribe_events(self):
        """Subscribe to EventBus and update status snapshots"""
```

---

### 3. **CodeActAdapter** (REPLACE PLACEHOLDER)

**Purpose**: Run real OpenHands CodeActAgent in embedded headless session

**Location**: `extensions/uagent_research/adapters/codeact/adapter.py`

**Implementation**:
```python
class CodeActAdapter(AgentAdapter):
    """Adapter for OpenHands CodeActAgent (real integration)"""

    name = "codeact"

    async def run(self, task: Task, context: Context) -> AsyncIterator[ResearchEvent]:
        """Execute code task using real CodeActAgent"""

        # 1. Create HeadlessAgentSession
        session = await self._create_headless_session(task, context)

        # 2. Subscribe to OpenHands EventStream
        event_bridge = OpenHandsEventBridge(
            event_stream=session.event_stream,
            branch_id=context.branch_id,
            node_id=task.id
        )

        # 3. Start agent execution
        await session.start(initial_message=task.goal)

        # 4. Stream bridged events
        async for research_event in event_bridge.stream():
            if self._cancelled:
                await session.cancel()
                yield ErrorEvent(
                    branch_id=context.branch_id,
                    node_id=task.id,
                    error="Task cancelled by user"
                )
                break

            yield research_event

        # 5. Cleanup
        await session.close()

    async def _create_headless_session(
        self, task: Task, context: Context
    ) -> HeadlessAgentSession:
        """Create embedded OpenHands session for CodeAct"""

        # Use current OpenHands config
        config = self._get_openhands_config()

        # Create session with headless controller
        session = HeadlessAgentSession(
            config=config,
            agent_class=CodeActAgent,
            headless_mode=True,
            experiment_id=f"{context.branch_id}-codeact"
        )

        return session
```

---

### 4. **HeadlessAgentSession** (NEW)

**Purpose**: Wrapper around OpenHands AgentSession for embedded execution

**Location**: `extensions/uagent_research/adapters/codeact/session_runner.py`

**Interface**:
```python
class HeadlessAgentSession:
    """Headless OpenHands agent session for embedded execution"""

    def __init__(
        self,
        config: OpenHandsConfig,
        agent_class: Type[Agent],
        headless_mode: bool = True,
        experiment_id: str = None
    ):
        self.config = config
        self.agent_class = agent_class
        self.experiment_id = experiment_id

        # Create dedicated EventStream (isolated from main chat)
        self.event_stream = EventStream(experiment_id)

        # Create LLM registry
        self.llm_registry = LLMRegistry(config)

        # Create agent
        agent_config = AgentConfig.from_llm_config(config.llm)
        self.agent = agent_class(agent_config, self.llm_registry)

        # Create controller with headless mode
        self.controller = AgentController(
            agent=self.agent,
            event_stream=self.event_stream,
            headless_mode=headless_mode,
            max_iterations=config.max_iterations or 100
        )

        # Runtime will be created on start
        self.runtime = None

    async def start(self, initial_message: str):
        """Start agent execution with initial message"""

        # Create runtime
        self.runtime = await self._create_runtime()

        # Add initial message to event stream
        self.event_stream.add_event(
            MessageAction(content=initial_message),
            EventSource.USER
        )

        # Start controller loop (non-blocking)
        asyncio.create_task(self._run_controller())

    async def send_user_message(self, message: str):
        """Send user message to running agent"""
        self.event_stream.add_event(
            MessageAction(content=message),
            EventSource.USER
        )

    async def cancel(self):
        """Cancel execution and cleanup"""
        if self.controller:
            await self.controller.close()
        if self.runtime:
            await self.runtime.close()

    async def close(self):
        """Cleanup resources"""
        await self.cancel()

    async def _run_controller(self):
        """Run controller loop until completion"""
        await self.controller.run()

    async def _create_runtime(self):
        """Create sandbox runtime"""
        # Use OpenHands RemoteRuntime or LocalRuntime
        from openhands.runtime import create_runtime
        return await create_runtime(self.config)
```

---

### 5. **OpenHandsEventBridge** (NEW)

**Purpose**: Convert OpenHands events → ResearchEvents

**Location**: `extensions/uagent_research/bridges/openhands_bridge.py`

**Mapping**:
```python
class OpenHandsEventBridge:
    """Bridge OpenHands EventStream to ResearchEvent stream"""

    def __init__(self, event_stream: EventStream, branch_id: str, node_id: str):
        self.event_stream = event_stream
        self.branch_id = branch_id
        self.node_id = node_id

    async def stream(self) -> AsyncIterator[ResearchEvent]:
        """Stream bridged events"""

        async for oh_event in self.event_stream.subscribe():
            research_event = self._map_event(oh_event)
            if research_event:
                yield research_event

    def _map_event(self, oh_event: Event) -> Optional[ResearchEvent]:
        """Map OpenHands event to ResearchEvent"""

        if isinstance(oh_event, MessageAction):
            if oh_event.source == EventSource.AGENT:
                # Agent thinking/planning
                return StepEvent(
                    branch_id=self.branch_id,
                    node_id=self.node_id,
                    action=oh_event.content[:100],
                    reasoning=oh_event.content
                )

        elif isinstance(oh_event, AgentThinkAction):
            return PlanEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                steps=[oh_event.thought],
                reasoning=oh_event.thought
            )

        elif isinstance(oh_event, CmdRunAction):
            return ToolCallEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                tool="bash",
                parameters={"command": oh_event.command}
            )

        elif isinstance(oh_event, IPythonRunCellAction):
            return ToolCallEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                tool="ipython",
                parameters={"code": oh_event.code}
            )

        elif isinstance(oh_event, CmdOutputObservation):
            return ObservationEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                result={"output": oh_event.content},
                success=oh_event.exit_code == 0
            )

        elif isinstance(oh_event, ErrorObservation):
            return ErrorEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                error=oh_event.content
            )

        elif isinstance(oh_event, AgentFinishAction):
            return CompleteEvent(
                branch_id=self.branch_id,
                node_id=self.node_id,
                summary=oh_event.outputs.get("content", "Task completed"),
                artifacts=[],
                success=True
            )

        return None  # Ignore unmapped events
```

---

### 6. **TreeSearchOrchestrator** (ENHANCED)

**Add Control Loop**:

```python
class TreeSearchOrchestrator:
    def __init__(
        self,
        max_parallel: int = 3,
        budget: Optional[Budget] = None,
        router: Optional[SkillRouter] = None,
        event_bus: Optional[EventBus] = None,
        control_bus: Optional[ControlBus] = None  # NEW
    ):
        # ... existing init ...
        self.control_bus = control_bus or ControlBus()
        self._paused = False
        self._steer_map: Dict[str, str] = {}  # node_id -> steer text

    async def run(self, goal: str, context: Optional[str] = None, max_iterations: int = 10):
        """Run research with control loop"""

        # ... existing init ...

        # Start control loop (concurrent with PUCT loop)
        control_task = asyncio.create_task(self._control_loop())

        try:
            # Main PUCT loop (existing)
            for iteration in range(max_iterations):
                if self._cancelled or self._paused:
                    if self._paused:
                        logger.info("Research paused by user")
                        await asyncio.sleep(1)  # Wait for resume
                        continue
                    else:
                        break

                # ... existing PUCT logic ...

        finally:
            control_task.cancel()

    async def _control_loop(self):
        """Process control commands"""
        async for cmd in self.control_bus.subscribe(self.tree.research_id):
            if cmd.action == "pause":
                self._paused = True
                logger.info(f"Paused research: {self.tree.research_id}")

            elif cmd.action == "resume":
                self._paused = False
                logger.info(f"Resumed research: {self.tree.research_id}")

            elif cmd.action == "cancel":
                self._cancelled = True
                # Cancel all running tasks
                for task in self._running_tasks.values():
                    task.cancel()
                logger.info(f"Cancelled research: {self.tree.research_id}")

            elif cmd.action == "cancel_node":
                node_id = cmd.payload.get("node_id")
                if node_id in self._running_tasks:
                    self._running_tasks[node_id].cancel()
                    self.tree.nodes[node_id].status = NodeStatus.CANCELLED
                logger.info(f"Cancelled node: {node_id}")

            elif cmd.action == "reprioritize":
                # Adjust PUCT parameters or node priors
                delta = cmd.payload.get("delta", 0.1)
                target = cmd.target.get("adapter") or cmd.target.get("node_type")
                # ... update logic ...

            elif cmd.action == "steer":
                # Store steer directive for nodes/adapters
                target_id = cmd.target.get("node_id") or cmd.target.get("branch_id")
                self._steer_map[target_id] = cmd.payload.get("text")
                logger.info(f"Steer added for {target_id}: {cmd.payload['text']}")
```

---

### 7. **Heartbeat Mechanism**

**Option 1: Adapter-Level Heartbeats**

Each adapter emits periodic heartbeat events when idle:

```python
class DeepResearchAdapter(AgentAdapter):
    async def run(self, task: Task, context: Context):
        last_event_time = time.time()
        heartbeat_interval = 5  # seconds

        async def heartbeat_task():
            while not self._cancelled:
                await asyncio.sleep(heartbeat_interval)
                if time.time() - last_event_time > heartbeat_interval:
                    yield StepEvent(
                        branch_id=context.branch_id,
                        node_id=task.id,
                        action="heartbeat",
                        reasoning="Still running..."
                    )

        # Run heartbeat concurrently with main task
        # ...
```

**Option 2: EventBus-Level Heartbeats** (Recommended)

EventBus tracks last event per branch and emits synthetic heartbeats:

```python
class EventBus:
    def __init__(self):
        # ... existing ...
        self._last_event_time: Dict[str, float] = {}
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}

    async def publish(self, event: ResearchEvent):
        # ... existing publish logic ...

        # Update last event time
        branch_id = event.branch_id
        self._last_event_time[branch_id] = time.time()

        # Start heartbeat supervisor if not running
        if branch_id not in self._heartbeat_tasks:
            task = asyncio.create_task(self._heartbeat_supervisor(branch_id))
            self._heartbeat_tasks[branch_id] = task

    async def _heartbeat_supervisor(self, branch_id: str, interval: int = 5):
        """Emit heartbeats when branch is idle"""
        while branch_id in self._last_event_time:
            await asyncio.sleep(interval)

            if time.time() - self._last_event_time[branch_id] > interval:
                # Emit synthetic heartbeat
                heartbeat = StepEvent(
                    branch_id=branch_id,
                    node_id="heartbeat",
                    action="heartbeat",
                    reasoning=f"Branch {branch_id} still active"
                )
                # Publish to subscribers (skip heartbeat supervisor)
                await self._publish_to_subscribers(heartbeat)
```

---

## API Interfaces

### REST API

#### **1. Get Research Status** (NEW)
```http
GET /api/research/experiments/{experiment_id}/status

Response:
{
  "experiment_id": "exp_...",
  "status": "running",
  "stats": {
    "total_nodes": 10,
    "completed": 5,
    "failed": 1,
    "running": 4,
    "total_cost": 0.25,
    "total_tokens": 5000
  },
  "adapters": {
    "deepresearch": {
      "status": "running",
      "current_step": "Browsing https://...",
      "last_event": "2025-10-06T10:30:00",
      "cost": 0.05
    },
    "codeact": {
      "status": "running",
      "current_step": "Executing benchmark",
      "last_event": "2025-10-06T10:30:05",
      "cost": 0.15
    }
  },
  "active_branches": [
    {
      "branch_id": "idea-0-hyp-0",
      "title": "Test using pgvector",
      "adapter": "codeact",
      "status": "running",
      "progress": "Running tests..."
    }
  ]
}
```

#### **2. Control Experiment** (ENHANCED)
```http
PATCH /api/research/experiments/{experiment_id}

Request Body (varies by action):

# Pause
{
  "action": "pause"
}

# Resume
{
  "action": "resume"
}

# Cancel entire experiment
{
  "action": "cancel"
}

# Cancel specific node
{
  "action": "cancel_node",
  "node_id": "idea-2"
}

# Reprioritize adapter/nodes
{
  "action": "reprioritize",
  "target": {"adapter": "codeact"},
  "payload": {"delta": 0.2}  # Increase priority
}

# Add new research direction
{
  "action": "add_node",
  "parent_id": "root",
  "node": {
    "type": "IDEA",
    "title": "Try DuckDB vector extension",
    "content": "Explore DuckDB's native vector support",
    "prior": 0.8
  }
}

# Steer subagent
{
  "action": "steer",
  "target": {"branch_id": "idea-0-hyp-0"},
  "payload": {"text": "Focus on DARTS, ignore ENAS"}
}

Response:
{
  "experiment_id": "exp_...",
  "action": "pause",
  "status": "acknowledged",
  "message": "Research paused successfully"
}
```

#### **3. Get Events** (NEW)
```http
GET /api/research/experiments/{experiment_id}/events?since_version=0&limit=100

Response:
{
  "experiment_id": "exp_...",
  "since_version": 0,
  "current_version": 25,
  "events": [
    {
      "version": 1,
      "timestamp": "2025-10-06T10:25:00",
      "type": "PLAN",
      "branch_id": "idea-0",
      "data": {...}
    },
    ...
  ],
  "has_more": false
}
```

### WebSocket API

#### **Subscribe to Research Events**
```javascript
// Connect
const ws = new WebSocket('ws://localhost:3000/api/research/ws/experiment/{experimentId}');

// Receive events (server → client)
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'tree_snapshot':
      // Full tree update
      break;
    case 'node_added':
      // New node created
      break;
    case 'node_updated':
      // Node status changed
      break;
    case 'event_log':
      // Research event (PLAN, STEP, etc.)
      break;
    case 'heartbeat':
      // Keep-alive
      break;
  }
};

// Send control commands (client → server) NEW
ws.send(JSON.stringify({
  type: 'control',
  action: 'pause'
}));

ws.send(JSON.stringify({
  type: 'control',
  action: 'steer',
  target: {branch_id: 'idea-0'},
  payload: {text: 'Focus on performance optimization'}
}));
```

---

## Code Structure

### New Files

```
extensions/uagent_research/
├── control/                              # NEW
│   ├── __init__.py
│   └── control_bus.py                    # ControlBus implementation
│
├── services/                             # NEW
│   ├── __init__.py
│   └── research_session_manager.py       # ResearchSessionManager
│
├── bridges/                              # NEW
│   ├── __init__.py
│   └── openhands_bridge.py               # Event mapping OpenHands → Research
│
├── adapters/
│   └── codeact/
│       ├── adapter.py                    # REPLACE with real integration
│       └── session_runner.py             # NEW: HeadlessAgentSession
│
├── api/
│   ├── research_routes.py                # ENHANCE with control actions + status
│   └── websocket_routes.py               # ENHANCE with inbound control
│
└── middleware/
    └── research_middleware.py            # ENHANCE for "how's progress?" queries
```

### Modified Files

```
extensions/uagent_research/
├── orchestrator/
│   └── tree_orchestrator.py              # Add control_bus, control_loop
│
├── orchestrator/
│   └── event_bus.py                      # Add heartbeat supervisor (optional)
│
└── api/
    └── research_routes.py                # Add /status, /events, control actions
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

#### Tasks:
1. **ControlBus** (1 day)
   - [ ] Implement `ControlMessage` model
   - [ ] Implement `ControlBus` with asyncio.Queue per experiment
   - [ ] Add subscription/publish methods
   - [ ] Unit tests

2. **ResearchSessionManager** (2 days)
   - [ ] Implement experiment registry
   - [ ] Implement status aggregation
   - [ ] Subscribe to EventBus for status updates
   - [ ] Add `get_status()` method
   - [ ] Integration tests

3. **HeadlessAgentSession** (2 days)
   - [ ] Create wrapper around OpenHands AgentSession
   - [ ] Implement headless controller mode
   - [ ] Add `start()`, `send_user_message()`, `cancel()` methods
   - [ ] Test with simple CodeActAgent task

4. **OpenHandsEventBridge** (1 day)
   - [ ] Implement event mapping logic
   - [ ] Handle all OpenHands event types
   - [ ] Unit tests for each mapping

### Phase 2: CodeAct Integration (Week 2)

#### Tasks:
1. **Replace CodeActAdapter** (2 days)
   - [ ] Implement real CodeActAdapter using HeadlessAgentSession
   - [ ] Subscribe to event bridge
   - [ ] Handle cancellation and cleanup
   - [ ] Integration tests

2. **Orchestrator Control Loop** (2 days)
   - [ ] Add ControlBus to TreeSearchOrchestrator
   - [ ] Implement `_control_loop()` method
   - [ ] Handle pause/resume/cancel
   - [ ] Handle node cancellation
   - [ ] Handle reprioritization
   - [ ] Handle steering

3. **Heartbeat System** (1 day)
   - [ ] Choose approach (adapter-level or EventBus-level)
   - [ ] Implement heartbeat emission
   - [ ] Test heartbeat delivery to frontend

### Phase 3: API Enhancements (Week 3)

#### Tasks:
1. **REST API** (2 days)
   - [ ] Implement `GET /status` endpoint
   - [ ] Implement `GET /events` endpoint
   - [ ] Enhance `PATCH /experiments/{id}` with all control actions
   - [ ] API documentation
   - [ ] Integration tests

2. **WebSocket Control** (1 day)
   - [ ] Extend WebSocket to accept control messages
   - [ ] Wire to ControlBus
   - [ ] Test bidirectional flow

3. **Research Middleware Enhancement** (2 days)
   - [ ] Detect "how's progress?" queries
   - [ ] Call ResearchSessionManager for summaries
   - [ ] Detect control intents ("pause", "cancel X")
   - [ ] Route to ControlBus
   - [ ] Integration tests

### Phase 4: Testing & Documentation (Week 4)

#### Tasks:
1. **End-to-End Tests** (3 days)
   - [ ] Test full flow: research trigger → CodeAct execution → progress query → steer → cancel
   - [ ] Test concurrent research + chat interaction
   - [ ] Test all control actions
   - [ ] Performance testing

2. **Documentation** (2 days)
   - [ ] Update UAGENT_MECHANISM_EXPLAINED.md
   - [ ] Create user guide for control features
   - [ ] API documentation
   - [ ] Troubleshooting guide

---

## User Experience Scenarios

### Scenario 1: Monitor Progress

**User Action**:
```
User: "how's progress on the research?"
```

**System Flow**:
1. Message → `research_middleware.process_message()`
2. Classify as progress query
3. Call `ResearchSessionManager.get_status(experiment_id)`
4. Format summary:
   ```
   Research Progress:

   Overall: 5/10 nodes complete, 3 running, 2 pending
   Cost: $0.25

   Active Tasks:
   - DeepResearch: Browsing documentation for pgvector (running)
   - CodeAct: Executing benchmark comparison (running)
   - RepoMaster: Analyzing GitHub repos (complete)

   Recent Events:
   - [10:30:05] CodeAct: Running pytest for sorting algorithms
   - [10:30:00] DeepResearch: Found 5 relevant articles
   ```
5. Return to user in main chat (< 300ms, no blocking)

---

### Scenario 2: Steer Research

**User Action**:
```
User: "focus the codeact agent on DARTS, ignore ENAS"
```

**System Flow**:
1. Message → `research_middleware.process_message()`
2. Classify as steer intent
3. Create `ControlMessage`:
   ```python
   ControlMessage(
       action="steer",
       target={"adapter": "codeact"},
       payload={"text": "Focus on DARTS algorithm, deprioritize ENAS"}
   )
   ```
4. Publish to ControlBus
5. Orchestrator control loop receives command
6. Update `self._steer_map["codeact"] = "Focus on DARTS..."`
7. Forward to active CodeAct sessions:
   ```python
   await codeact_session.send_user_message(
       "[System] Steering: Focus on DARTS algorithm, deprioritize ENAS"
   )
   ```
8. CodeAct agent adjusts execution accordingly
9. EventBus emits update → Frontend shows steer applied
10. Main chat confirms: "✓ Steered CodeAct to focus on DARTS"

---

### Scenario 3: Cancel Branch

**User Action**:
```
User: "cancel idea-2, it's not working"
```

**System Flow**:
1. Message → `research_middleware.process_message()`
2. Classify as cancel intent, extract `node_id="idea-2"`
3. Create `ControlMessage`:
   ```python
   ControlMessage(
       action="cancel_node",
       target={"node_id": "idea-2"},
       payload={}
   )
   ```
4. Publish to ControlBus
5. Orchestrator control loop receives
6. Cancel running task for `idea-2`:
   ```python
   self._running_tasks["idea-2"].cancel()
   ```
7. Mark node as CANCELLED:
   ```python
   self.tree.nodes["idea-2"].status = NodeStatus.CANCELLED
   ```
8. Publish tree update
9. EventBus → Frontend updates tree visualization
10. Main chat confirms: "✓ Cancelled branch idea-2"

---

### Scenario 4: Pause and Resume

**User Action**:
```
User: "pause the research, I need to check something"
... (user investigates)
User: "ok resume"
```

**System Flow**:

**Pause**:
1. `ControlMessage(action="pause")` → ControlBus
2. Orchestrator sets `self._paused = True`
3. PUCT loop stops scheduling new nodes (running tasks continue)
4. EventBus emits pause event
5. Main chat: "✓ Research paused"

**Resume**:
1. `ControlMessage(action="resume")` → ControlBus
2. Orchestrator sets `self._paused = False`
3. PUCT loop continues from where it stopped
4. EventBus emits resume event
5. Main chat: "✓ Research resumed"

---

## Configuration

### Environment Variables

```bash
# .env

# Research heartbeat interval (seconds)
RESEARCH_HEARTBEAT_INTERVAL=5

# Max concurrent subagents
RESEARCH_MAX_PARALLEL=3

# Enable/disable auto-trigger
ENABLE_AUTO_RESEARCH_TRIGGER=true

# Confidence threshold
RESEARCH_CONFIDENCE_THRESHOLD=0.7

# Budget limits
RESEARCH_MAX_ITERATIONS=50
RESEARCH_MAX_COST=10.0

# Control settings
RESEARCH_ENABLE_STEERING=true
RESEARCH_ENABLE_RUNTIME_CONTROL=true
```

### Orchestrator Config

```python
# Start research with control enabled
orchestrator = TreeSearchOrchestrator(
    max_parallel=3,
    budget=Budget(max_cost=10.0, max_iterations=50),
    event_bus=event_bus,
    control_bus=control_bus  # Enable control
)
```

---

## Testing Strategy

### Unit Tests

```python
# Test ControlBus
async def test_control_bus_publish_subscribe():
    bus = ControlBus()

    # Publish
    await bus.publish("exp-1", ControlMessage(action="pause"))

    # Subscribe
    async for msg in bus.subscribe("exp-1"):
        assert msg.action == "pause"
        break

# Test OpenHands event bridge
def test_event_bridge_message_action():
    bridge = OpenHandsEventBridge(event_stream, "branch-1", "node-1")

    oh_event = MessageAction(content="Running benchmark...")
    research_event = bridge._map_event(oh_event)

    assert isinstance(research_event, StepEvent)
    assert research_event.action == "Running benchmark..."

# Test ResearchSessionManager
async def test_session_manager_get_status():
    manager = ResearchSessionManager(event_bus, control_bus)
    manager.register("exp-1", orchestrator, ws_publisher)

    status = manager.get_status("exp-1")

    assert status["experiment_id"] == "exp-1"
    assert "stats" in status
    assert "adapters" in status
```

### Integration Tests

```python
# Test full flow: start → progress query → steer → cancel
async def test_end_to_end_control_flow():
    # 1. Start research
    experiment_id = await research_middleware.start_research(
        goal="Compare sorting algorithms",
        session_id="test-session"
    )

    # 2. Wait for some progress
    await asyncio.sleep(5)

    # 3. Query progress
    status = research_session_manager.get_status(experiment_id)
    assert status["stats"]["running"] > 0

    # 4. Steer CodeAct
    await research_session_manager.send_control(
        experiment_id,
        ControlMessage(
            action="steer",
            target={"adapter": "codeact"},
            payload={"text": "Focus on quicksort"}
        )
    )

    # 5. Verify steer applied
    await asyncio.sleep(2)
    # Check EventBus for steer event

    # 6. Cancel experiment
    await research_session_manager.send_control(
        experiment_id,
        ControlMessage(action="cancel")
    )

    # 7. Verify cancelled
    status = research_session_manager.get_status(experiment_id)
    assert status["status"] == "cancelled"
```

---

## Migration Path

### Backward Compatibility

✅ **Fully backward compatible**:
- Existing research functionality unchanged
- Control features are opt-in
- Old experiments continue working without modification

### Gradual Rollout

1. **Week 1-2**: Deploy core infrastructure (ControlBus, ResearchSessionManager, HeadlessAgentSession)
   - No user-facing changes yet
   - Internal testing

2. **Week 3**: Deploy CodeAct integration
   - CodeAct now uses real OpenHands agent
   - Better code execution quality
   - Still no control features exposed

3. **Week 4**: Enable control features
   - Add control UI in frontend
   - Enable "how's progress?" queries
   - Enable pause/resume/cancel
   - Full steering capabilities

---

## Success Metrics

### Performance Targets

- **Progress Query Response**: < 300ms
- **Control Command Latency**: < 500ms (command → effect)
- **Heartbeat Interval**: 5s (configurable 2-10s)
- **Event Delivery**: < 1s (EventBus → Frontend)

### Quality Targets

- **CodeAct Success Rate**: > 90% (vs current placeholder)
- **Concurrent Research + Chat**: No blocking, < 10ms impact
- **Control Accuracy**: 100% (commands execute as intended)

---

## Risks & Mitigation

### Risk 1: HeadlessAgentSession Complexity

**Risk**: Embedded OpenHands sessions may be resource-intensive

**Mitigation**:
- Reuse runtime containers when possible
- Implement session pooling
- Set timeout limits on CodeAct execution
- Monitor memory/CPU usage

### Risk 2: Event Mapping Gaps

**Risk**: Some OpenHands events may not map cleanly to ResearchEvents

**Mitigation**:
- Start with core event types (MessageAction, CmdRunAction, Observations)
- Add mappings incrementally as needed
- Log unmapped events for analysis
- Provide fallback "generic" event type

### Risk 3: Control Race Conditions

**Risk**: User sends control while node is executing

**Mitigation**:
- Use asyncio.Queue for ordered command processing
- Implement state machine for orchestrator (running/paused/cancelled)
- Add command validation before execution
- Proper locking for shared state

---

## Appendix

### Glossary

- **HeadlessAgentSession**: Embedded OpenHands session for subagent execution (no UI)
- **ControlBus**: Typed message bus for runtime control commands
- **EventBridge**: Converts OpenHands events → ResearchEvents
- **ResearchSessionManager**: Central registry for experiment state and control
- **PUCT**: Predictor + UCT algorithm for tree search (from AlphaZero)

### References

- [OpenHands CodeActAgent](../../openhands/agenthub/codeact_agent/codeact_agent.py)
- [OpenHands AgentController](../../openhands/controller/agent_controller.py)
- [OpenHands EventStream](../../openhands/events/stream.py)
- [UAGENT_MECHANISM_EXPLAINED.md](./UAGENT_MECHANISM_EXPLAINED.md)
- [UAGENT_DOCUMENTATION_REVIEW.md](./UAGENT_DOCUMENTATION_REVIEW.md)

---

## Next Steps

1. **Review this specification** with the team
2. **Approve architecture** and implementation plan
3. **Begin Phase 1**: Core Infrastructure
4. **Set up CI/CD** for continuous testing
5. **Monitor progress** using the 4-week timeline

---

**Specification Complete** ✅

Ready for implementation!
