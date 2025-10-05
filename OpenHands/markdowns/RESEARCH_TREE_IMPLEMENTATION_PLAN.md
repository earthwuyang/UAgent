# Research Tree with Parallel Search - Implementation Plan

## Executive Summary

This plan implements a research tree visualization with parallel tree search capabilities, MCP integration (Tongyi DeepResearch, RepoMaster), and intelligent request routing for scientific research workflows.

**Key Features:**
1. ✅ Research tree visualization (top-right toggle)
2. ✅ Parallel tree search in CodeAct framework
3. ✅ Tongyi DeepResearch + RepoMaster MCP integration
4. ✅ Scientific research pipeline: Ideas → Hypotheses → Experiments
5. ✅ Real-time updates via WebSocket

**Architecture Principles:**
- Minimal changes to OpenHands core
- Feature-flagged and modular
- Clean separation of concerns
- Real-time updates with backpressure handling
- Scalable parallel execution

---

## Architecture Overview

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  ┌──────────────────┐  ┌─────────────────────────────────────┐ │
│  │ ResearchTreeIcon │  │     ResearchTreePanel               │ │
│  │  (Top Right)     │→ │  - Tree Visualization               │ │
│  └──────────────────┘  │  - Node Details                     │ │
│                         │  - Real-time Updates (WS)           │ │
│                         └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                    ↕ REST + WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI + SQLAlchemy)               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐│
│  │IntelligentRouter│→ │ ResearchService  │  │WebSocketGateway││
│  │  (Classifier)   │  │ - CRUD           │← │- Event Stream  ││
│  └─────────────────┘  │ - Orchestration  │  └────────────────┘│
│                        └────────┬─────────┘                     │
│                                 ↓                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         TreeSearchOrchestrator (TSO)                      │  │
│  │  - Parallel Expansion (asyncio tasks)                    │  │
│  │  - Beam Search / UCB1 Scoring                            │  │
│  │  - Scientific Pipeline: Ideas → Hypotheses → Experiments │  │
│  └────────────┬───────────────────────────────────┬─────────┘  │
│               ↓                                     ↓            │
│  ┌───────────────────────┐          ┌──────────────────────┐   │
│  │   MCP Tool Adapters   │          │  ResearchEventBus    │   │
│  │ - Tongyi DeepResearch │          │  - Pub/Sub           │   │
│  │ - RepoMaster Deep     │          │  - Event Coalescing  │   │
│  │ - Standard LLM        │          │  - Backpressure      │   │
│  └───────────────────────┘          └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                    ↕
┌─────────────────────────────────────────────────────────────────┐
│              Database (SQLite with JSON field)                   │
│  research_tree: {                                                │
│    nodes: Map<node_id, Node>,                                    │
│    edges: Array<{from, to}>,                                     │
│    stats: {created, expanded, complete}                          │
│  }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Backend Core Infrastructure

### 1.1 Data Models Enhancement

**File:** `extensions/uagent_research/uagent_research/models/tree_node.py` (NEW)

```python
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel

class NodeType(str, Enum):
    ROOT = "root"
    IDEA = "idea"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    RESULT = "result"

class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"

class TreeNode(BaseModel):
    id: str
    parent_id: Optional[str] = None
    type: NodeType
    prompt: str
    summary: str
    score: Optional[float] = None
    status: NodeStatus = NodeStatus.PENDING
    metadata: Dict[str, Any] = {}
    version: int = 0
    created_at: datetime
    updated_at: datetime

class ResearchTree(BaseModel):
    research_id: str
    nodes: Dict[str, TreeNode] = {}
    edges: List[Dict[str, str]] = []
    stats: Dict[str, Any] = {
        "created": 0,
        "expanded": 0,
        "complete": False
    }
    version: int = 0
```

**File:** `extensions/uagent_research/uagent_research/models/research_session.py` (UPDATE)

Add fields:
- `research_tree_version: int`
- `parallel_config: dict` (beam_width, max_parallel, timeout)
- `active_nodes: list[str]` (currently expanding)

### 1.2 ResearchEventBus

**File:** `extensions/uagent_research/uagent_research/core/event_bus.py` (NEW)

```python
import asyncio
from typing import Dict, Set, Callable, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class ResearchEventBus:
    """In-memory async pub/sub for research events"""

    def __init__(self):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._coalescing: Dict[str, Dict[str, Any]] = {}
        self._coalesce_interval = 0.1  # 100ms

    async def subscribe(self, research_id: str) -> asyncio.Queue:
        """Subscribe to events for a research session"""
        queue = asyncio.Queue(maxsize=100)
        self._subscribers[research_id].add(queue)
        logger.info(f"Subscriber added for research {research_id}")
        return queue

    def unsubscribe(self, research_id: str, queue: asyncio.Queue):
        """Unsubscribe from events"""
        if research_id in self._subscribers:
            self._subscribers[research_id].discard(queue)

    async def publish(self, research_id: str, event: Dict[str, Any]):
        """Publish event to all subscribers with coalescing"""
        event_type = event.get("type")
        node_id = event.get("node", {}).get("id") if event.get("node") else None

        # Coalesce node_updated events by node_id
        if event_type == "node_updated" and node_id:
            key = f"{research_id}:{node_id}"
            self._coalescing[key] = event
            # Will be flushed by periodic task
            return

        # Immediate publish for other events
        await self._publish_immediate(research_id, event)

    async def _publish_immediate(self, research_id: str, event: Dict[str, Any]):
        """Publish event immediately to all subscribers"""
        if research_id not in self._subscribers:
            return

        dead_queues = set()
        for queue in self._subscribers[research_id]:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Queue full for {research_id}, dropping event")
                dead_queues.add(queue)
            except Exception as e:
                logger.error(f"Error publishing event: {e}")
                dead_queues.add(queue)

        # Clean up dead queues
        for queue in dead_queues:
            self.unsubscribe(research_id, queue)

    async def flush_coalesced(self):
        """Periodic task to flush coalesced events"""
        while True:
            await asyncio.sleep(self._coalesce_interval)

            if not self._coalescing:
                continue

            # Flush all coalesced events
            events_to_send = list(self._coalescing.items())
            self._coalescing.clear()

            for key, event in events_to_send:
                research_id = key.split(":")[0]
                await self._publish_immediate(research_id, event)

# Global event bus instance
event_bus = ResearchEventBus()
```

### 1.3 TreeSearchOrchestrator (TSO)

**File:** `extensions/uagent_research/uagent_research/core/tree_search_orchestrator.py` (NEW)

```python
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
import numpy as np

from ..models.tree_node import TreeNode, NodeType, NodeStatus, ResearchTree
from .event_bus import event_bus
from .mcp_adapters import get_mcp_adapter

logger = logging.getLogger(__name__)

class TreeSearchOrchestrator:
    """
    Parallel tree search orchestrator for research.

    Implements beam search with UCB1 scoring for exploration/exploitation.
    """

    def __init__(
        self,
        research_id: str,
        research_service,
        config: Dict[str, Any]
    ):
        self.research_id = research_id
        self.research_service = research_service

        # Configuration
        self.beam_width = config.get("beam_width", 4)
        self.max_parallel = config.get("max_parallel", 8)
        self.max_nodes = config.get("max_nodes", 300)
        self.timeout_s = config.get("timeout_s", 600)
        self.ucb_c = config.get("ucb_c", 1.414)  # UCB1 exploration constant

        # State
        self.tree: Optional[ResearchTree] = None
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(self.max_parallel)
        self.start_time: Optional[float] = None

    async def run(self, initial_prompt: str):
        """
        Main orchestration loop.

        1. Create root node
        2. Generate ideas (parallel)
        3. For each idea, generate hypotheses (parallel)
        4. For each hypothesis, run experiment (parallel)
        5. Emit results
        """
        self.start_time = time.time()

        try:
            # Initialize tree
            self.tree = await self.research_service.get_tree(self.research_id)

            # Create root node
            root = await self._create_node(
                None,
                NodeType.ROOT,
                initial_prompt,
                initial_prompt
            )

            # Scientific research pipeline
            await self._scientific_pipeline(root)

            # Mark complete
            await self._emit_complete()

        except asyncio.CancelledError:
            logger.info(f"Research {self.research_id} cancelled")
            raise
        except Exception as e:
            logger.error(f"Research {self.research_id} failed: {e}", exc_info=True)
            await self._emit_error(str(e))
            raise

    async def _scientific_pipeline(self, root: TreeNode):
        """
        Scientific research pipeline: Ideas → Hypotheses → Experiments
        """
        # Step 1: Generate ideas (parallel)
        logger.info(f"[{self.research_id}] Generating ideas...")
        ideas = await self._parallel_expand(
            root,
            NodeType.IDEA,
            self._generate_ideas,
            count=self.beam_width
        )

        if not ideas:
            raise ValueError("No ideas generated")

        # Step 2: Generate hypotheses for each idea (parallel)
        logger.info(f"[{self.research_id}] Generating hypotheses for {len(ideas)} ideas...")
        all_hypotheses = []
        for idea in ideas:
            hypotheses = await self._parallel_expand(
                idea,
                NodeType.HYPOTHESIS,
                self._generate_hypotheses,
                count=3  # 3 hypotheses per idea
            )
            all_hypotheses.extend(hypotheses)

        if not all_hypotheses:
            raise ValueError("No hypotheses generated")

        # Step 3: Score and select top hypotheses (beam search)
        scored_hypotheses = await self._score_nodes(all_hypotheses)
        top_hypotheses = scored_hypotheses[:self.beam_width]

        logger.info(f"[{self.research_id}] Selected {len(top_hypotheses)} top hypotheses")

        # Step 4: Run experiments for top hypotheses (parallel)
        logger.info(f"[{self.research_id}] Running experiments...")
        experiments = []
        for hypothesis in top_hypotheses:
            experiment = await self._run_experiment(hypothesis)
            experiments.append(experiment)

        logger.info(f"[{self.research_id}] Completed {len(experiments)} experiments")

    async def _parallel_expand(
        self,
        parent: TreeNode,
        node_type: NodeType,
        generator_func,
        count: int
    ) -> List[TreeNode]:
        """
        Expand a node in parallel.

        Returns list of created child nodes.
        """
        # Check budget
        if len(self.tree.nodes) + count > self.max_nodes:
            logger.warning(f"Node budget exceeded")
            return []

        # Check timeout
        if time.time() - self.start_time > self.timeout_s:
            logger.warning(f"Timeout exceeded")
            return []

        # Generate children in parallel
        tasks = []
        for i in range(count):
            task = asyncio.create_task(
                self._expand_one(parent, node_type, generator_func, i)
            )
            tasks.append(task)

        # Wait for all with timeout
        try:
            children = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Parallel expansion failed: {e}")
            return []

        # Filter out errors
        valid_children = [
            child for child in children
            if isinstance(child, TreeNode)
        ]

        return valid_children

    async def _expand_one(
        self,
        parent: TreeNode,
        node_type: NodeType,
        generator_func,
        index: int
    ) -> TreeNode:
        """
        Expand one node using generator function.
        """
        async with self.semaphore:
            try:
                # Call generator function
                prompt, summary, metadata = await generator_func(parent, index)

                # Create node
                node = await self._create_node(
                    parent.id,
                    node_type,
                    prompt,
                    summary,
                    metadata
                )

                return node

            except Exception as e:
                logger.error(f"Expansion failed: {e}")
                raise

    async def _generate_ideas(self, parent: TreeNode, index: int):
        """Generate a single idea using LLM or MCP"""
        # Use Tongyi DeepResearch MCP if available
        adapter = get_mcp_adapter("tongyi")
        if adapter:
            result = await adapter.generate_idea(parent.prompt, index)
            return (
                result["prompt"],
                result["summary"],
                {"provider": "tongyi", **result.get("metadata", {})}
            )

        # Fallback to standard LLM
        # TODO: Implement LLM call
        return (
            f"Idea {index}: {parent.prompt}",
            f"Generated idea based on: {parent.prompt}",
            {"provider": "llm"}
        )

    async def _generate_hypotheses(self, parent: TreeNode, index: int):
        """Generate hypothesis from idea"""
        adapter = get_mcp_adapter("tongyi")
        if adapter:
            result = await adapter.generate_hypothesis(parent.summary, index)
            return (
                result["prompt"],
                result["summary"],
                {"provider": "tongyi", **result.get("metadata", {})}
            )

        return (
            f"Hypothesis {index} for {parent.summary[:50]}",
            f"Testable hypothesis derived from idea",
            {"provider": "llm"}
        )

    async def _run_experiment(self, hypothesis: TreeNode) -> TreeNode:
        """Run experiment to test hypothesis"""
        async with self.semaphore:
            # Use RepoMaster MCP if code-related
            adapter = get_mcp_adapter("repomaster")
            if adapter and "code" in hypothesis.summary.lower():
                result = await adapter.run_experiment(hypothesis.summary)
                node = await self._create_node(
                    hypothesis.id,
                    NodeType.EXPERIMENT,
                    hypothesis.summary,
                    result["summary"],
                    {"provider": "repomaster", **result.get("metadata", {})}
                )
            else:
                # Use standard experiment execution
                # TODO: Integrate with OpenHands agent for code execution
                node = await self._create_node(
                    hypothesis.id,
                    NodeType.EXPERIMENT,
                    hypothesis.summary,
                    f"Experiment results for: {hypothesis.summary[:50]}",
                    {"provider": "openhands"}
                )

            # Mark experiment as done
            await self._update_node(node.id, status=NodeStatus.DONE, score=0.8)

            return node

    async def _score_nodes(self, nodes: List[TreeNode]) -> List[TreeNode]:
        """
        Score nodes using UCB1 algorithm.

        UCB1 = avg_score + c * sqrt(ln(N) / n)
        """
        total_expanded = self.tree.stats.get("expanded", 1)

        scored = []
        for node in nodes:
            # Base score from metadata
            base_score = node.score or 0.5

            # Exploration bonus
            node_visits = node.metadata.get("visits", 1)
            exploration = self.ucb_c * np.sqrt(np.log(total_expanded) / node_visits)

            ucb_score = base_score + exploration

            scored.append((ucb_score, node))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return [node for score, node in scored]

    async def _create_node(
        self,
        parent_id: Optional[str],
        node_type: NodeType,
        prompt: str,
        summary: str,
        metadata: Dict[str, Any] = None
    ) -> TreeNode:
        """Create and persist a new node"""
        node_id = f"{node_type.value}_{len(self.tree.nodes)}_{int(time.time() * 1000)}"

        node = TreeNode(
            id=node_id,
            parent_id=parent_id,
            type=node_type,
            prompt=prompt,
            summary=summary,
            status=NodeStatus.RUNNING,
            metadata=metadata or {},
            version=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        # Add to tree
        self.tree.nodes[node_id] = node
        if parent_id:
            self.tree.edges.append({"from": parent_id, "to": node_id})
        self.tree.stats["created"] += 1
        self.tree.version += 1

        # Persist
        await self.research_service.update_tree(self.research_id, self.tree)

        # Emit event
        await event_bus.publish(self.research_id, {
            "type": "node_added",
            "node": node.dict()
        })

        logger.info(f"Created node {node_id} ({node_type.value})")

        return node

    async def _update_node(
        self,
        node_id: str,
        **updates
    ):
        """Update node and emit event"""
        if node_id not in self.tree.nodes:
            return

        node = self.tree.nodes[node_id]

        # Apply updates
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)

        node.version += 1
        node.updated_at = datetime.utcnow()
        self.tree.version += 1

        # Persist
        await self.research_service.update_tree(self.research_id, self.tree)

        # Emit event
        await event_bus.publish(self.research_id, {
            "type": "node_updated",
            "node": {
                "id": node_id,
                "status": updates.get("status"),
                "score": updates.get("score"),
                "version": node.version
            },
            "version": node.version
        })

    async def _emit_complete(self):
        """Emit completion event"""
        self.tree.stats["complete"] = True
        await self.research_service.update_tree(self.research_id, self.tree)

        # Find best nodes
        best_nodes = sorted(
            self.tree.nodes.values(),
            key=lambda n: n.score or 0,
            reverse=True
        )[:5]

        await event_bus.publish(self.research_id, {
            "type": "complete",
            "summary": {
                "best_nodes": [n.id for n in best_nodes],
                "total_nodes": len(self.tree.nodes),
                "duration_s": time.time() - self.start_time
            }
        })

    async def _emit_error(self, message: str):
        """Emit error event"""
        await event_bus.publish(self.research_id, {
            "type": "error",
            "error": {"message": message}
        })
```

**This is getting very long. Let me continue with the plan structure...**

---

## Phase 2: MCP Integration

### 2.1 MCP Adapter Interface

**File:** `extensions/uagent_research/uagent_research/core/mcp_adapters/__init__.py`

### 2.2 Tongyi DeepResearch Adapter

**File:** `extensions/uagent_research/uagent_research/core/mcp_adapters/tongyi_adapter.py`

### 2.3 RepoMaster Deep Research Adapter

**File:** `extensions/uagent_research/uagent_research/core/mcp_adapters/repomaster_adapter.py`

---

## Phase 3: API & Service Layer

### 3.1 ResearchService Enhancement

**File:** `extensions/uagent_research/uagent_research/services/research_service.py`

### 3.2 WebSocket Gateway

**File:** `extensions/uagent_research/uagent_research/api/websocket_routes.py` (UPDATE)

### 3.3 REST Endpoints

**File:** `extensions/uagent_research/uagent_research/api/research_routes.py` (UPDATE)

---

## Phase 4: Frontend Implementation

### 4.1 Research Tree Toggle Icon

**File:** `frontend/src/components/research/ResearchTreeToggle.tsx`

### 4.2 Research Tree Panel

**File:** `frontend/src/components/research/ResearchTreePanel.tsx`

### 4.3 Tree Node Component

**File:** `frontend/src/components/research/TreeNode.tsx`

### 4.4 Hooks

**File:** `frontend/src/hooks/use-research-tree.ts`
**File:** `frontend/src/hooks/use-research-websocket.ts`

---

## Phase 5: Integration & Testing

### 5.1 Intelligent Router

**File:** `extensions/uagent_research/uagent_research/core/intelligent_router.py`

### 5.2 Integration Tests

---

## Implementation Timeline

### Week 1: Backend Core
- [ ] Day 1-2: Data models, EventBus
- [ ] Day 3-5: TreeSearchOrchestrator core
- [ ] Day 6-7: Testing

### Week 2: MCP Integration
- [ ] Day 1-3: MCP adapter framework
- [ ] Day 4-5: Tongyi & RepoMaster adapters
- [ ] Day 6-7: Testing

### Week 3: API & Services
- [ ] Day 1-3: ResearchService, WebSocket
- [ ] Day 4-5: REST endpoints
- [ ] Day 6-7: Integration testing

### Week 4: Frontend
- [ ] Day 1-2: Toggle icon, basic panel
- [ ] Day 3-4: Tree visualization
- [ ] Day 5-6: WebSocket integration
- [ ] Day 7: E2E testing

---

## Next Steps for Review

1. **Architecture Validation**
   - Is the component separation clear?
   - Are there missing pieces?

2. **Technology Choices**
   - Beam search vs other algorithms?
   - WebSocket vs SSE?
   - Tree visualization library?

3. **Priority**
   - Which phase to start first?
   - Can we parallelize development?

4. **Concerns**
   - Performance bottlenecks?
   - Scalability issues?
   - Security considerations?

Please review and provide feedback on:
- Architecture design
- Implementation approach
- Timeline feasibility
- Missing components
