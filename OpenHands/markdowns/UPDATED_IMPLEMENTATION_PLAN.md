# Research Tree Implementation Plan v2.0
## Based on Codex Review & Recommendations

## 🎯 Key Changes from v1

### Critical Improvements (Based on Codex Review)

1. **Parallel Execution** → Use **TaskGroup + PriorityQueue** instead of plain Semaphore
2. **Search Algorithm** → Use **PUCT (AlphaZero-style)** instead of plain UCB1
3. **Database Schema** → **Normalized tables** instead of pure JSON
4. **Event Bus** → **Typed events with backpressure** and per-key coalescing
5. **MCP Adapters** → Add **circuit breakers, retries with jitter, idempotency**
6. **Frontend State** → **Normalized entity storage** with virtualization
7. **Observability** → Add **OpenTelemetry, Prometheus metrics**

---

## 📐 Updated Architecture

### 1. TreeSearchOrchestrator v2 (PUCT + TaskGroup)

**Changes:**
- ❌ ~~Plain asyncio.Semaphore~~
- ✅ **TaskGroup for structured concurrency** (Python 3.11+)
- ✅ **PriorityQueue** for beam scheduling
- ✅ **PUCT scoring**: `Q + c_puct * P * sqrt(N)/(1+n)`
- ✅ **Progressive widening**: Limit children until visit count justifies

**New File:** `extensions/uagent_research/uagent_research/core/tree_search_orchestrator_v2.py`

```python
import asyncio
from asyncio import TaskGroup
from queue import PriorityQueue
import math
import hashlib
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PUCT_Scorer:
    """
    PUCT (Predictor + UCT) scoring from AlphaZero.

    score = Q + c_puct * P * sqrt(N) / (1 + n)

    where:
    - Q: average value of node (exploitation)
    - P: prior probability (from policy network or heuristic)
    - N: parent visit count
    - n: this node's visit count
    - c_puct: exploration constant (default 1.25)
    """

    def __init__(self, c_puct: float = 1.25):
        self.c_puct = c_puct

    def score(self, node: TreeNode, parent_visits: int) -> float:
        Q = node.metadata.get("avg_value", 0.0)
        P = node.metadata.get("prior", 0.5)
        n = node.metadata.get("visits", 0)

        exploration = self.c_puct * P * math.sqrt(parent_visits) / (1 + n)

        return Q + exploration

class ProgressiveWidener:
    """
    Limits child expansion based on visit count.

    max_children = floor(k * n^alpha)

    Default: k=1, alpha=0.5 (square root growth)
    """

    def __init__(self, k: float = 1.0, alpha: float = 0.5):
        self.k = k
        self.alpha = alpha

    def should_expand(self, node: TreeNode) -> bool:
        visits = node.metadata.get("visits", 0)
        current_children = len(node.metadata.get("children", []))

        max_allowed = math.floor(self.k * (visits ** self.alpha))

        return current_children < max_allowed

class TreeSearchOrchestratorV2:
    """
    Production-ready parallel tree search with:
    - TaskGroup for structured concurrency
    - PriorityQueue for beam selection
    - PUCT scoring with progressive widening
    - Per-provider rate limiting
    - Circuit breakers
    - Idempotent deduplication
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
        self.c_puct = config.get("c_puct", 1.25)

        # Components
        self.scorer = PUCT_Scorer(self.c_puct)
        self.widener = ProgressiveWidener()
        self.frontier = PriorityQueue()

        # Deduplication cache
        self.content_hashes: Dict[str, str] = {}  # hash -> node_id

        # Provider rate limiters
        self.provider_limiters: Dict[str, TokenBucket] = {}

        # Metrics
        self.metrics = ResearchMetrics(research_id)

    async def run(self, initial_prompt: str):
        """
        Main orchestration loop with TaskGroup.
        """
        start_time = asyncio.get_event_loop().time()

        try:
            async with asyncio.timeout(self.timeout_s):
                async with TaskGroup() as tg:
                    # Create root
                    root = await self._create_node(
                        None, NodeType.ROOT, initial_prompt, initial_prompt
                    )

                    # Scientific pipeline
                    await self._scientific_pipeline_v2(root, tg)

        except TimeoutError:
            logger.warning(f"Research {self.research_id} timed out")
            await self._emit_timeout()
        except* Exception as eg:
            # TaskGroup collects exceptions
            logger.error(f"Research failed with {len(eg.exceptions)} errors")
            for e in eg.exceptions:
                logger.error(f"  - {e}")
            await self._emit_error(str(eg))

        duration = asyncio.get_event_loop().time() - start_time
        self.metrics.record_duration(duration)
        await self._emit_complete()

    async def _scientific_pipeline_v2(self, root: TreeNode, tg: TaskGroup):
        """
        Scientific pipeline with structured concurrency.
        """
        # Step 1: Generate ideas
        idea_tasks = []
        for i in range(self.beam_width):
            task = tg.create_task(
                self._expand_with_retry(root, NodeType.IDEA, i)
            )
            idea_tasks.append(task)

        # Wait for ideas (implicit with TaskGroup)
        # No need for gather, TaskGroup handles it

        # Get created ideas from database
        ideas = await self._get_children(root.id, NodeType.IDEA)

        # Step 2: Score and select top ideas
        scored_ideas = self._score_nodes(ideas, root.metadata.get("visits", 1))
        top_ideas = scored_ideas[:self.beam_width]

        # Step 3: Generate hypotheses for each idea
        for idea in top_ideas:
            if self.widener.should_expand(idea):
                for i in range(3):  # 3 hypotheses per idea
                    tg.create_task(
                        self._expand_with_retry(idea, NodeType.HYPOTHESIS, i)
                    )

        # Get all hypotheses
        all_hypotheses = []
        for idea in top_ideas:
            hyps = await self._get_children(idea.id, NodeType.HYPOTHESIS)
            all_hypotheses.extend(hyps)

        # Step 4: Score and select top hypotheses
        scored_hypotheses = self._score_nodes(all_hypotheses, sum(i.metadata.get("visits", 1) for i in top_ideas))
        top_hypotheses = scored_hypotheses[:self.beam_width]

        # Step 5: Run experiments
        for hypothesis in top_hypotheses:
            tg.create_task(
                self._run_experiment_with_retry(hypothesis)
            )

    async def _expand_with_retry(
        self,
        parent: TreeNode,
        node_type: NodeType,
        index: int
    ) -> Optional[TreeNode]:
        """
        Expand node with retries and deduplication.
        """
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Check budget
                if await self._is_budget_exceeded():
                    return None

                # Generate content via provider
                provider = self._select_provider(node_type)

                # Rate limit
                await self.provider_limiters[provider].acquire()

                # Call provider with timeout
                async with asyncio.timeout(30):
                    content = await self._call_provider(
                        provider, parent, node_type, index
                    )

                # Deduplication
                content_hash = self._hash_content(content["summary"])
                if content_hash in self.content_hashes:
                    logger.info(f"Duplicate content detected, skipping")
                    self.metrics.record_duplicate()
                    return None

                # Create node
                node = await self._create_node(
                    parent.id,
                    node_type,
                    content["prompt"],
                    content["summary"],
                    {
                        "provider": provider,
                        "prior": content.get("prior", 0.5),
                        **content.get("metadata", {})
                    }
                )

                # Cache hash
                self.content_hashes[content_hash] = node.id

                # Update parent visits
                await self._increment_visits(parent.id)

                self.metrics.record_expansion(provider)

                return node

            except TimeoutError:
                logger.warning(f"Provider timeout on attempt {attempt + 1}")
                self.metrics.record_timeout(provider)

            except Exception as e:
                logger.error(f"Expansion failed: {e}")
                self.metrics.record_error(provider)

            # Exponential backoff with jitter
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) * (0.5 + random.random() * 0.5)
                await asyncio.sleep(delay)

        return None

    def _score_nodes(self, nodes: List[TreeNode], parent_visits: int) -> List[TreeNode]:
        """
        Score nodes using PUCT and sort descending.
        """
        scored = [
            (self.scorer.score(node, parent_visits), node)
            for node in nodes
        ]

        scored.sort(key=lambda x: x[0], reverse=True)

        return [node for score, node in scored]

    def _hash_content(self, content: str) -> str:
        """Hash content for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _is_budget_exceeded(self) -> bool:
        """Check if budget limits reached"""
        stats = await self.research_service.get_stats(self.research_id)

        if stats["created"] >= self.max_nodes:
            return True

        if stats["total_cost"] >= self.config.get("max_cost", float("inf")):
            return True

        return False

class TokenBucket:
    """
    Token bucket rate limiter.

    Allows burst of requests up to capacity, then limits to rate.
    """

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary"""
        async with self.lock:
            now = asyncio.get_event_loop().time()

            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # If no tokens available, wait
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
```

---

### 2. Normalized Database Schema

**Changes:**
- ❌ ~~Pure JSON storage~~
- ✅ **Normalized tables** with JSON for unstructured data
- ✅ **WAL mode** for better concurrency
- ✅ **Indexes** on common queries
- ✅ **FTS5** for text search

**Migration:** `extensions/uagent_research/migrations/001_normalized_schema.sql`

```sql
-- Enable WAL mode for better concurrency
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- Nodes table (normalized)
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL,
    parent_id TEXT,
    depth INTEGER NOT NULL DEFAULT 0,
    type TEXT NOT NULL CHECK(type IN ('root', 'idea', 'hypothesis', 'experiment', 'result')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'done', 'error')),

    -- Scoring fields
    score REAL,
    prior REAL DEFAULT 0.5,
    visits INTEGER DEFAULT 0,
    avg_value REAL DEFAULT 0.0,

    -- Content
    prompt TEXT NOT NULL,
    summary TEXT NOT NULL,
    content_hash TEXT NOT NULL,

    -- Metadata (JSON for flexibility)
    metadata_json TEXT DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign keys
    FOREIGN KEY (research_id) REFERENCES research_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES nodes(id) ON DELETE CASCADE
);

-- Indexes for common queries
CREATE INDEX idx_nodes_research_id ON nodes(research_id);
CREATE INDEX idx_nodes_parent_id ON nodes(parent_id);
CREATE INDEX idx_nodes_research_status ON nodes(research_id, status);
CREATE INDEX idx_nodes_type_score ON nodes(research_id, type, score DESC);
CREATE INDEX idx_nodes_content_hash ON nodes(content_hash);

-- Edges table (explicit graph structure)
CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_id TEXT NOT NULL,
    from_node_id TEXT NOT NULL,
    to_node_id TEXT NOT NULL,
    label TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (research_id) REFERENCES research_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (from_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (to_node_id) REFERENCES nodes(id) ON DELETE CASCADE,

    UNIQUE(from_node_id, to_node_id)
);

CREATE INDEX idx_edges_research_id ON edges(research_id);
CREATE INDEX idx_edges_from ON edges(from_node_id);
CREATE INDEX idx_edges_to ON edges(to_node_id);

-- Provider calls table (for analytics and idempotency)
CREATE TABLE provider_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_id TEXT NOT NULL,
    node_id TEXT,
    provider TEXT NOT NULL,

    -- Request deduplication
    request_hash TEXT NOT NULL,

    -- Performance tracking
    latency_ms INTEGER,
    cost REAL DEFAULT 0.0,

    -- Result
    status TEXT NOT NULL CHECK(status IN ('success', 'error', 'timeout')),
    error_json TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (research_id) REFERENCES research_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE SET NULL
);

CREATE INDEX idx_provider_calls_request_hash ON provider_calls(request_hash);
CREATE INDEX idx_provider_calls_research_id ON provider_calls(research_id);
CREATE INDEX idx_provider_calls_provider ON provider_calls(provider, created_at);

-- Full-text search for node content
CREATE VIRTUAL TABLE nodes_fts USING fts5(
    node_id UNINDEXED,
    prompt,
    summary,
    content='nodes',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER nodes_fts_insert AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(node_id, prompt, summary)
    VALUES (new.id, new.prompt, new.summary);
END;

CREATE TRIGGER nodes_fts_update AFTER UPDATE ON nodes BEGIN
    UPDATE nodes_fts
    SET prompt = new.prompt, summary = new.summary
    WHERE node_id = new.id;
END;

CREATE TRIGGER nodes_fts_delete AFTER DELETE ON nodes BEGIN
    DELETE FROM nodes_fts WHERE node_id = old.id;
END;

-- Research sessions table (update)
ALTER TABLE research_sessions ADD COLUMN config_json TEXT DEFAULT '{}';
ALTER TABLE research_sessions ADD COLUMN stats_json TEXT DEFAULT '{}';
```

---

### 3. Typed Event Bus with Backpressure

**Changes:**
- ❌ ~~Untyped dict events~~
- ✅ **Pydantic models** for type safety
- ✅ **Per-key coalescing** (by node_id)
- ✅ **Adaptive batching** based on queue depth
- ✅ **Overflow policy** with metrics

**File:** `extensions/uagent_research/uagent_research/core/event_bus_v2.py`

```python
from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

# Typed event models
class NodeAddedEvent(BaseModel):
    type: Literal["node_added"] = "node_added"
    node_id: str
    parent_id: Optional[str]
    node_type: str
    summary: str
    status: str
    version: int
    timestamp: datetime

class NodeUpdatedEvent(BaseModel):
    type: Literal["node_updated"] = "node_updated"
    node_id: str
    status: Optional[str] = None
    score: Optional[float] = None
    visits: Optional[int] = None
    version: int
    timestamp: datetime

class ProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    expanded: int
    total: int
    timestamp: datetime

class CompleteEvent(BaseModel):
    type: Literal["complete"] = "complete"
    best_nodes: List[str]
    total_nodes: int
    duration_s: float
    timestamp: datetime

class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    node_id: Optional[str] = None
    timestamp: datetime

ResearchEvent = NodeAddedEvent | NodeUpdatedEvent | ProgressEvent | CompleteEvent | ErrorEvent

class ResearchEventBusV2:
    """
    Production event bus with:
    - Typed events (Pydantic)
    - Per-key coalescing for node_updated
    - Adaptive batching
    - Backpressure handling
    - Overflow metrics
    """

    def __init__(
        self,
        queue_size: int = 100,
        coalesce_interval_ms: int = 100,
        batch_size: int = 10
    ):
        self.queue_size = queue_size
        self.coalesce_interval = coalesce_interval_ms / 1000.0
        self.batch_size = batch_size

        # Subscribers: research_id -> Set[Queue]
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}

        # Coalescing buffer: research_id -> node_id -> event
        self._coalescing: Dict[str, Dict[str, NodeUpdatedEvent]] = {}

        # Metrics
        self._metrics = EventBusMetrics()

        # Start flush task
        self._flush_task = None

    async def start(self):
        """Start background flush task"""
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        """Stop background tasks"""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

    async def subscribe(self, research_id: str) -> asyncio.Queue:
        """Subscribe to events for a research session"""
        queue = asyncio.Queue(maxsize=self.queue_size)

        if research_id not in self._subscribers:
            self._subscribers[research_id] = set()

        self._subscribers[research_id].add(queue)

        logger.info(f"Subscriber added for {research_id} (total: {len(self._subscribers[research_id])})")

        return queue

    def unsubscribe(self, research_id: str, queue: asyncio.Queue):
        """Unsubscribe from events"""
        if research_id in self._subscribers:
            self._subscribers[research_id].discard(queue)

            if not self._subscribers[research_id]:
                del self._subscribers[research_id]

    async def publish(self, research_id: str, event: ResearchEvent):
        """
        Publish event with smart routing:
        - node_updated: coalesce by node_id
        - others: immediate publish
        """
        if isinstance(event, NodeUpdatedEvent):
            # Coalesce
            if research_id not in self._coalescing:
                self._coalescing[research_id] = {}

            self._coalescing[research_id][event.node_id] = event
            self._metrics.record_coalesced()
        else:
            # Immediate
            await self._publish_immediate(research_id, event)

    async def _publish_immediate(self, research_id: str, event: ResearchEvent):
        """Publish event to all subscribers immediately"""
        if research_id not in self._subscribers:
            return

        event_dict = event.model_dump(mode="json")

        dead_queues = set()

        for queue in self._subscribers[research_id]:
            try:
                queue.put_nowait(event_dict)
                self._metrics.record_sent()
            except asyncio.QueueFull:
                logger.warning(f"Queue full for {research_id}, dropping event")
                self._metrics.record_dropped()
                dead_queues.add(queue)
            except Exception as e:
                logger.error(f"Error publishing: {e}")
                dead_queues.add(queue)

        # Clean up dead queues
        for queue in dead_queues:
            self.unsubscribe(research_id, queue)

    async def _flush_loop(self):
        """Periodic flush of coalesced events"""
        while True:
            try:
                await asyncio.sleep(self.coalesce_interval)
                await self._flush_coalesced()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush error: {e}")

    async def _flush_coalesced(self):
        """Flush all coalesced events"""
        if not self._coalescing:
            return

        # Collect all events
        to_flush: List[tuple[str, NodeUpdatedEvent]] = []

        for research_id, events_by_node in self._coalescing.items():
            for node_id, event in events_by_node.items():
                to_flush.append((research_id, event))

        # Clear buffer
        self._coalescing.clear()

        # Publish in batches
        for i in range(0, len(to_flush), self.batch_size):
            batch = to_flush[i:i + self.batch_size]

            tasks = [
                self._publish_immediate(research_id, event)
                for research_id, event in batch
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

        self._metrics.record_flushed(len(to_flush))
```

---

## 📊 Observability & Metrics

### OpenTelemetry Integration

**File:** `extensions/uagent_research/uagent_research/observability/tracing.py`

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from prometheus_client import start_http_server

# Initialize tracer
tracer = trace.get_tracer("uagent_research")

# Initialize metrics
meter = metrics.get_meter("uagent_research")

# Counters
node_expansions = meter.create_counter(
    "research.node.expansions",
    description="Number of node expansions",
    unit="1"
)

provider_calls = meter.create_counter(
    "research.provider.calls",
    description="Provider API calls",
    unit="1"
)

provider_errors = meter.create_counter(
    "research.provider.errors",
    description="Provider errors",
    unit="1"
)

# Histograms
provider_latency = meter.create_histogram(
    "research.provider.latency",
    description="Provider call latency",
    unit="ms"
)

tree_depth = meter.create_histogram(
    "research.tree.depth",
    description="Tree depth distribution",
    unit="1"
)

# Gauges
active_researches = meter.create_up_down_counter(
    "research.active",
    description="Active research sessions",
    unit="1"
)
```

---

## 🧪 Updated Testing Strategy

### 1. Unit Tests

```python
# tests/test_puct_scorer.py
def test_puct_prefers_high_value():
    """PUCT should prefer nodes with high Q"""

def test_puct_explores_unvisited():
    """PUCT should explore nodes with low visit count"""

def test_progressive_widening():
    """Progressive widening should limit children"""
```

### 2. Integration Tests

```python
# tests/test_mcp_circuit_breaker.py
async def test_circuit_breaker_opens_on_failures():
    """Circuit breaker should open after N failures"""

async def test_circuit_breaker_half_open_recovery():
    """Circuit breaker should try recovery after timeout"""
```

### 3. Load Tests

```python
# tests/load/test_event_bus_throughput.py
async def test_event_bus_handles_1000_events_per_second():
    """Event bus should handle high throughput"""
```

---

## 📋 Updated Implementation Timeline

### Week 1: Core Infrastructure
- **Day 1-2**: Database schema migration, normalized tables, WAL mode
- **Day 3-4**: PUCT scorer, progressive widening, TaskGroup orchestrator
- **Day 5-6**: Typed event bus with backpressure
- **Day 7**: Unit tests

### Week 2: Resilience & MCP
- **Day 1-2**: Token bucket rate limiters, circuit breakers
- **Day 3-4**: MCP adapters with retries, idempotency
- **Day 5-6**: Observability (metrics, tracing)
- **Day 7**: Integration tests

### Week 3: API & WebSocket
- **Day 1-2**: ResearchService with new schema
- **Day 3-4**: WebSocket with versioned deltas, reconnection
- **Day 5-6**: REST endpoints, snapshot API
- **Day 7**: API tests

### Week 4: Frontend
- **Day 1-2**: Normalized state store, entity adapter
- **Day 3-4**: Tree visualization with virtualization
- **Day 5-6**: WebSocket integration, reconnection protocol
- **Day 7**: E2E tests, polish

---

## 🎯 Summary of Improvements

| Aspect | v1 (Original) | v2 (Updated) | Benefit |
|--------|---------------|--------------|---------|
| **Concurrency** | Semaphore | TaskGroup + PriorityQueue | Structured lifecycles, better cancellation |
| **Search** | UCB1 | PUCT + Progressive Widening | Better exploration, prevents explosion |
| **Database** | JSON blob | Normalized + WAL | Better performance, queryability |
| **Events** | Untyped dicts | Pydantic models | Type safety, validation |
| **Backpressure** | Basic drop | Adaptive batching + coalescing | Smooth UI updates |
| **Resilience** | Basic retry | Circuit breaker + jitter | Handles flaky providers |
| **Dedup** | None | Content hashing | Prevents duplicate work |
| **Observability** | Logging | OpenTelemetry + Prometheus | Production monitoring |
| **State** | JSON tree | Normalized entities | Efficient updates |

---

## ✅ Ready to Proceed

This updated plan addresses all of Codex's recommendations and provides a production-ready architecture. The implementation is now:

1. **More robust**: Circuit breakers, retries, backpressure
2. **More scalable**: Normalized DB, batching, rate limiting
3. **More maintainable**: Typed events, structured concurrency, observability
4. **More efficient**: PUCT scoring, progressive widening, deduplication

**Next step:** Please review and approve to begin implementation!
