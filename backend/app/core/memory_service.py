"""
Memory Management Service for UAgent
Implements working memory, persistent memory, and context assembly
Based on ChatGPT 5 Pro's memory management plan
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryEventKind(Enum):
    """Types of memory events"""
    OBSERVATION = "observation"
    DECISION = "decision"
    ERROR = "error"
    NOTE = "note"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REFLECTION = "reflection"


class MemorySnapshotType(Enum):
    """Types of memory snapshots"""
    ROLLING = "rolling"
    MILESTONE = "milestone"
    POSTMORTEM = "postmortem"
    CONTEXT_SWITCH = "context_switch"


@dataclass
class MemoryEvent:
    """A single memory event"""
    run_id: str
    node_id: str
    source: str  # 'llm', 'tool:<name>', 'system', 'user'
    kind: MemoryEventKind
    body: Dict[str, Any]
    importance: float = 0.0  # 0.0 to 1.0
    token_cost: int = 0
    artifact_uri: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "node_id": self.node_id,
            "source": self.source,
            "kind": self.kind.value,
            "body": self.body,
            "importance": self.importance,
            "token_cost": self.token_cost,
            "artifact_uri": self.artifact_uri,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MemorySnapshot:
    """A memory snapshot/summary"""
    run_id: str
    node_id: str
    window_seq: int
    summary_type: MemorySnapshotType
    summary_text: str
    stats: Dict[str, Any]
    token_count: int = 0
    context_window_start: Optional[int] = None
    context_window_end: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SemanticChunk:
    """A reusable semantic memory chunk"""
    text: str
    title: Optional[str] = None
    org_scope: str = "default"
    meta: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    avg_relevance: float = 0.0
    embedding: Optional[List[float]] = None
    content_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContextConfig:
    """Configuration for context assembly"""
    max_tokens: int = 8000
    task_header_budget: float = 0.35  # 35% for task instructions
    rolling_summary_budget: float = 0.20  # 20% for latest summary
    semantic_budget: float = 0.25  # 25% for retrieved semantic memory
    recent_events_budget: float = 0.10  # 10% for high-salience recent events
    tool_schemas_budget: float = 0.10  # 10% for tool schemas


class MemoryService:
    """
    Core memory management service implementing working and persistent memory
    """

    def __init__(self, db_path: str = "data/research_history.db", artifact_store_path: str = "data/artifacts"):
        self.db_path = Path(db_path)
        self.artifact_store_path = Path(artifact_store_path)
        self.artifact_store_path.mkdir(parents=True, exist_ok=True)

        # Memory configuration
        self.max_event_body_size = 4096  # 4KB before offloading to artifacts
        self.rolling_summary_trigger_events = 15
        self.rolling_summary_trigger_tokens = 2500
        self.importance_threshold_salient = 0.7

        # Initialize semantic memory engine
        from .semantic_memory import SemanticMemoryEngine
        self.semantic_engine = SemanticMemoryEngine()

        logger.info(f"Memory service initialized with DB: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash for content deduplication"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)"""
        return len(text) // 4

    async def _offload_to_artifact_store(self, run_id: str, node_id: str, content: Any) -> str:
        """Offload large content to artifact store and return URI"""
        content_str = json.dumps(content) if not isinstance(content, str) else content
        content_hash = self._compute_hash(content_str)

        # Create artifact file path
        artifact_dir = self.artifact_store_path / run_id / node_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{content_hash}.json"

        # Write content to file
        with open(artifact_path, 'w', encoding='utf-8') as f:
            f.write(content_str)

        return f"file://{artifact_path.absolute()}"

    async def _load_from_artifact_store(self, artifact_uri: str) -> Any:
        """Load content from artifact store"""
        if artifact_uri.startswith("file://"):
            file_path = artifact_uri[7:]  # Remove "file://" prefix
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.loads(f.read())
            except Exception as e:
                logger.error(f"Failed to load artifact from {artifact_uri}: {e}")
                return None
        return None

    async def log_event(
        self,
        run_id: str,
        node_id: str,
        source: str,
        kind: MemoryEventKind,
        body: Dict[str, Any],
        importance: float = 0.0,
        maybe_artifact: bool = False
    ) -> int:
        """
        Log a memory event with optional artifact offloading
        Returns the event_id
        """
        try:
            # Prepare body content
            body_json = json.dumps(body)
            content_hash = self._compute_hash(body_json)
            token_cost = self._estimate_tokens(body_json)
            artifact_uri = None

            # Offload large content to artifact store
            if maybe_artifact and len(body_json) > self.max_event_body_size:
                artifact_uri = await self._offload_to_artifact_store(run_id, node_id, body)
                # Store only metadata in database
                body_json = json.dumps({
                    "artifact_reference": True,
                    "content_type": "offloaded",
                    "size_bytes": len(body_json),
                    "summary": str(body)[:200] + "..." if len(str(body)) > 200 else str(body)
                })

            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO memory_events
                    (run_id, node_id, source, kind, body_json, token_cost, importance,
                     artifact_uri, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, node_id, source, kind.value, body_json,
                    token_cost, importance, artifact_uri, content_hash
                ))
                event_id = cursor.lastrowid
                conn.commit()

                logger.debug(f"Logged memory event {event_id} for {run_id}/{node_id}")

                # Check if we should trigger summarization
                await self._maybe_trigger_summarization(run_id, node_id)

                return event_id

        except Exception as e:
            logger.error(f"Failed to log memory event: {e}")
            return -1

    async def _maybe_trigger_summarization(self, run_id: str, node_id: str):
        """Check if we should trigger rolling summarization"""
        try:
            stats = await self._get_memory_stats_since_last_snapshot(run_id, node_id)

            should_summarize = (
                stats["event_count"] >= self.rolling_summary_trigger_events or
                stats["total_tokens"] >= self.rolling_summary_trigger_tokens or
                stats["has_tool_boundary"]
            )

            if should_summarize:
                logger.info(f"Triggering summarization for {run_id}/{node_id}: {stats}")
                await self.create_rolling_summary(run_id, node_id)

        except Exception as e:
            logger.error(f"Error checking summarization trigger: {e}")

    async def _get_memory_stats_since_last_snapshot(self, run_id: str, node_id: str) -> Dict[str, Any]:
        """Get memory statistics since last snapshot"""
        with self._get_connection() as conn:
            # Get last snapshot window_seq
            cursor = conn.execute("""
                SELECT MAX(window_seq) as last_seq, MAX(context_window_end) as last_end_id
                FROM memory_snapshots
                WHERE run_id = ? AND node_id = ?
            """, (run_id, node_id))
            result = cursor.fetchone()

            last_seq = result[0] if result[0] is not None else -1
            last_end_id = result[1] if result[1] is not None else 0

            # Get events since last snapshot
            cursor = conn.execute("""
                SELECT COUNT(*) as event_count, SUM(token_cost) as total_tokens,
                       COUNT(CASE WHEN source LIKE 'tool:%' THEN 1 END) as tool_events,
                       MAX(event_id) as max_event_id
                FROM memory_events
                WHERE run_id = ? AND node_id = ? AND event_id > ?
            """, (run_id, node_id, last_end_id))
            stats = cursor.fetchone()

            return {
                "event_count": stats[0] or 0,
                "total_tokens": stats[1] or 0,
                "tool_events": stats[2] or 0,
                "has_tool_boundary": (stats[2] or 0) > 0,
                "last_sequence": last_seq,
                "max_event_id": stats[3] or last_end_id
            }

    async def create_rolling_summary(self, run_id: str, node_id: str, summary_type: MemorySnapshotType = MemorySnapshotType.ROLLING) -> int:
        """Create a rolling summary of recent memory events"""
        try:
            stats = await self._get_memory_stats_since_last_snapshot(run_id, node_id)

            if stats["event_count"] == 0:
                logger.debug(f"No new events to summarize for {run_id}/{node_id}")
                return -1

            # Fetch events to summarize
            events = await self.get_events_since_last_snapshot(run_id, node_id)

            # Generate summary using LLM (placeholder - would integrate with actual LLM)
            summary_text = await self._generate_summary(events, summary_type)

            # Calculate next window sequence
            next_seq = stats["last_sequence"] + 1

            # Store snapshot
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO memory_snapshots
                    (run_id, node_id, window_seq, summary_type, summary_text, stats_json,
                     token_count, context_window_start, context_window_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, node_id, next_seq, summary_type.value, summary_text,
                    json.dumps(stats), self._estimate_tokens(summary_text),
                    events[0]["event_id"] if events else None,
                    events[-1]["event_id"] if events else None
                ))
                snapshot_id = cursor.lastrowid
                conn.commit()

                logger.info(f"Created rolling summary {snapshot_id} for {run_id}/{node_id}")
                return snapshot_id

        except Exception as e:
            logger.error(f"Failed to create rolling summary: {e}")
            return -1

    async def get_events_since_last_snapshot(self, run_id: str, node_id: str) -> List[Dict[str, Any]]:
        """Get events since the last snapshot"""
        with self._get_connection() as conn:
            # Get last snapshot end
            cursor = conn.execute("""
                SELECT MAX(context_window_end) as last_end_id
                FROM memory_snapshots
                WHERE run_id = ? AND node_id = ?
            """, (run_id, node_id))
            result = cursor.fetchone()
            last_end_id = result[0] if result[0] is not None else 0

            # Fetch events
            cursor = conn.execute("""
                SELECT * FROM memory_events
                WHERE run_id = ? AND node_id = ? AND event_id > ?
                ORDER BY event_id
            """, (run_id, node_id, last_end_id))

            events = []
            for row in cursor:
                event_data = dict(row)
                # Load artifact content if needed
                if event_data.get("artifact_uri"):
                    artifact_content = await self._load_from_artifact_store(event_data["artifact_uri"])
                    if artifact_content:
                        event_data["body_content"] = artifact_content
                events.append(event_data)

            return events

    async def _generate_summary(self, events: List[Dict[str, Any]], summary_type: MemorySnapshotType) -> str:
        """Generate summary of events using LLM-based summarizer"""
        if not events:
            return "No events to summarize."

        try:
            # Import summarizer (avoid circular imports)
            from .memory_summarizer import RollingSummarizer, SummaryContext

            summarizer = RollingSummarizer()

            # Prepare context if available
            context = await self._build_summary_context(events[0]["run_id"], events[0]["node_id"])

            if summary_type == MemorySnapshotType.POSTMORTEM:
                # For postmortem, get final results
                final_results = await self._get_final_results(events[0]["run_id"], events[0]["node_id"])
                structured_summary = await summarizer.create_postmortem_summary(events, final_results, context)
            else:
                # Regular rolling summary
                max_tokens = 800 if summary_type == MemorySnapshotType.ROLLING else 1200
                structured_summary = await summarizer.create_rolling_summary(events, context, max_tokens)

            # Format structured summary as text
            summary_parts = [
                f"**CONTEXT**: {structured_summary.context}",
                f"**STATE**: {structured_summary.state}",
            ]

            if structured_summary.evidence:
                evidence_text = "\n".join([f"- {ev.get('text', ev)}" for ev in structured_summary.evidence[:3]])
                summary_parts.append(f"**EVIDENCE**:\n{evidence_text}")

            if structured_summary.actions:
                actions_text = "\n".join([f"- {action}" for action in structured_summary.actions[:5]])
                summary_parts.append(f"**ACTIONS**:\n{actions_text}")

            if structured_summary.key_insights:
                insights_text = "\n".join([f"- {insight}" for insight in structured_summary.key_insights[:3]])
                summary_parts.append(f"**INSIGHTS**:\n{insights_text}")

            summary_parts.append(f"\n*Compression: {structured_summary.compression_ratio:.2f}, Tokens: {structured_summary.token_count}*")

            return "\n\n".join(summary_parts)

        except Exception as e:
            logger.error(f"LLM summarization failed, using fallback: {e}")
            return self._generate_fallback_summary(events)

    async def _build_summary_context(self, run_id: str, node_id: str):
        """Build context for summarization"""
        try:
            from .memory_summarizer import SummaryContext

            # Get task information from database
            with self._get_connection() as conn:
                # Get goal information
                cursor = conn.execute("""
                    SELECT g.title, g.description, g.success_criteria, g.constraints
                    FROM research_goals g
                    WHERE g.id = ?
                """, (run_id,))
                goal_row = cursor.fetchone()

                # Get node information
                cursor = conn.execute("""
                    SELECT n.title, n.description, n.hypothesis, n.context
                    FROM research_nodes n
                    WHERE n.id = ?
                """, (node_id,))
                node_row = cursor.fetchone()

                if goal_row and node_row:
                    success_criteria = json.loads(goal_row[2]) if goal_row[2] else []
                    constraints = json.loads(goal_row[3]) if goal_row[3] else {}
                    node_context = json.loads(node_row[3]) if node_row[3] else {}

                    return SummaryContext(
                        task_description=f"{goal_row[1]} - {node_row[1]}",
                        success_criteria=success_criteria,
                        constraints=constraints,
                        current_state=node_row[2] or "In progress",
                        open_questions=node_context.get("open_questions", [])
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to build summary context: {e}")
            return None

    async def _get_final_results(self, run_id: str, node_id: str) -> Dict[str, Any]:
        """Get final results for postmortem summary"""
        try:
            with self._get_connection() as conn:
                # Get latest experiment results
                cursor = conn.execute("""
                    SELECT success, confidence, data, insights
                    FROM experiment_results
                    WHERE node_id = ?
                    ORDER BY id DESC LIMIT 1
                """, (node_id,))

                result = cursor.fetchone()
                if result:
                    return {
                        "success": bool(result[0]),
                        "confidence": result[1],
                        "data": json.loads(result[2]) if result[2] else {},
                        "insights": json.loads(result[3]) if result[3] else []
                    }

            return {"success": False, "confidence": 0.0, "data": {}, "insights": []}

        except Exception as e:
            logger.error(f"Failed to get final results: {e}")
            return {"success": False, "confidence": 0.0, "data": {}, "insights": []}

    def _generate_fallback_summary(self, events: List[Dict[str, Any]]) -> str:
        """Generate simple fallback summary when LLM is unavailable"""
        event_types = {}
        tool_calls = []
        decisions = []
        errors = []

        for event in events:
            kind = event["kind"]
            event_types[kind] = event_types.get(kind, 0) + 1

            body_json = event.get("body_json", "{}")
            try:
                body = json.loads(body_json)
            except:
                body = {"text": body_json}

            if kind == "tool_call":
                tool_calls.append(body.get("tool_name", "unknown"))
            elif kind == "decision":
                decisions.append(body.get("decision", ""))
            elif kind == "error":
                errors.append(body.get("error", ""))

        summary_parts = [
            f"**Summary Window ({len(events)} events)**",
            f"Event types: {dict(event_types)}",
        ]

        if tool_calls:
            summary_parts.append(f"Tools used: {list(set(tool_calls))}")

        if decisions:
            summary_parts.append(f"Key decisions: {decisions[:3]}")  # Top 3

        if errors:
            summary_parts.append(f"Errors encountered: {len(errors)}")

        return "\n".join(summary_parts)

    async def get_latest_snapshot(self, run_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest memory snapshot"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM memory_snapshots
                WHERE run_id = ? AND node_id = ?
                ORDER BY window_seq DESC LIMIT 1
            """, (run_id, node_id))

            result = cursor.fetchone()
            return dict(result) if result else None

    async def get_salient_events(
        self,
        run_id: str,
        node_id: str,
        min_importance: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent high-importance events"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM memory_events
                WHERE run_id = ? AND node_id = ? AND importance >= ?
                ORDER BY ts DESC LIMIT ?
            """, (run_id, node_id, min_importance, limit))

            events = []
            for row in cursor:
                event_data = dict(row)
                # Load artifact content if needed
                if event_data.get("artifact_uri"):
                    artifact_content = await self._load_from_artifact_store(event_data["artifact_uri"])
                    if artifact_content:
                        event_data["body_content"] = artifact_content
                events.append(event_data)

            return events

    async def promote_to_semantic_memory(
        self,
        text: str,
        title: Optional[str] = None,
        meta: Dict[str, Any] = None,
        org_scope: str = "default"
    ) -> int:
        """Promote content to reusable semantic memory with embeddings"""
        try:
            with self._get_connection() as conn:
                chunk_id = await self.semantic_engine.store_semantic_chunk(
                    conn, text, title, meta, org_scope
                )

                logger.info(f"Promoted content to semantic memory: chunk_id={chunk_id}")
                return chunk_id

        except Exception as e:
            logger.error(f"Failed to promote to semantic memory: {e}")
            return -1

    async def search_semantic_memory(
        self,
        query_terms: List[str],
        k: int = 10,
        org_scope: str = "default",
        use_mmr: bool = True,
        mmr_lambda: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search semantic memory using vector similarity and MMR diversification"""
        try:
            # Join query terms into a single query
            query_text = " ".join(query_terms) if query_terms else ""

            with self._get_connection() as conn:
                from .semantic_memory import RetrievalConfig

                config = RetrievalConfig(
                    k=k,
                    mmr_lambda=mmr_lambda if use_mmr else 1.0,  # 1.0 = no diversification
                    min_score_threshold=0.1,
                    boost_recent=True,
                    boost_usage=True
                )

                matches = await self.semantic_engine.search_semantic_chunks(
                    conn, query_text, config, org_scope
                )

                # Convert to legacy format for compatibility
                results = []
                for match in matches:
                    results.append({
                        "chunk_id": match.chunk_id,
                        "title": match.title,
                        "text": match.text,
                        "meta": match.meta,
                        "usage_count": match.usage_count,
                        "score": match.score,
                        "embedding": match.embedding
                    })

                # Update usage counts for retrieved chunks
                for result in results:
                    conn.execute("""
                        UPDATE semantic_chunks
                        SET usage_count = usage_count + 1
                        WHERE chunk_id = ?
                    """, (result["chunk_id"],))

                conn.commit()

                logger.debug(f"Semantic search returned {len(results)} results for query: {query_text[:50]}")
                return results

        except Exception as e:
            logger.error(f"Failed to search semantic memory: {e}")
            return []

    async def assemble_context(
        self,
        run_id: str,
        node_id: str,
        query_terms: List[str] = None,
        config: ContextConfig = None
    ) -> Dict[str, Any]:
        """Assemble context for LLM prompt with budget management"""
        config = config or ContextConfig()
        query_terms = query_terms or []

        try:
            context = {
                "run_id": run_id,
                "node_id": node_id,
                "components": {},
                "token_budget": config.max_tokens,
                "token_usage": {}
            }

            # 1. Get task header (35% budget)
            header_budget = int(config.max_tokens * config.task_header_budget)
            task_header = await self._get_task_header(run_id, node_id)
            context["components"]["task_header"] = task_header[:header_budget * 4]  # Rough chars to tokens
            context["token_usage"]["task_header"] = self._estimate_tokens(context["components"]["task_header"])

            # 2. Get latest rolling summary (20% budget)
            summary_budget = int(config.max_tokens * config.rolling_summary_budget)
            latest_snapshot = await self.get_latest_snapshot(run_id, node_id)
            if latest_snapshot:
                summary_text = latest_snapshot["summary_text"]
                context["components"]["rolling_summary"] = summary_text[:summary_budget * 4]
                context["token_usage"]["rolling_summary"] = self._estimate_tokens(context["components"]["rolling_summary"])
            else:
                context["components"]["rolling_summary"] = ""
                context["token_usage"]["rolling_summary"] = 0

            # 3. Retrieve semantic memory (25% budget)
            semantic_budget = int(config.max_tokens * config.semantic_budget)
            semantic_chunks = await self.search_semantic_memory(query_terms, k=8)
            semantic_content = []
            semantic_tokens = 0

            for chunk in semantic_chunks:
                chunk_text = f"**{chunk.get('title', 'Untitled')}**\n{chunk['text']}"
                chunk_tokens = self._estimate_tokens(chunk_text)

                if semantic_tokens + chunk_tokens <= semantic_budget:
                    semantic_content.append(chunk_text)
                    semantic_tokens += chunk_tokens
                else:
                    break

            context["components"]["semantic_memory"] = "\n\n".join(semantic_content)
            context["token_usage"]["semantic_memory"] = semantic_tokens

            # 4. Get high-salience recent events (10% budget)
            events_budget = int(config.max_tokens * config.recent_events_budget)
            salient_events = await self.get_salient_events(run_id, node_id, min_importance=0.7, limit=10)
            events_content = []
            events_tokens = 0

            for event in salient_events:
                event_text = f"[{event['source']}] {event['kind']}: {event.get('body_json', '')[:200]}"
                event_tokens = self._estimate_tokens(event_text)

                if events_tokens + event_tokens <= events_budget:
                    events_content.append(event_text)
                    events_tokens += event_tokens
                else:
                    break

            context["components"]["recent_events"] = "\n".join(events_content)
            context["token_usage"]["recent_events"] = events_tokens

            # 5. Tool schemas (10% budget) - placeholder
            tool_budget = int(config.max_tokens * config.tool_schemas_budget)
            context["components"]["tool_schemas"] = "Available tools: search, code_analysis, file_operations"
            context["token_usage"]["tool_schemas"] = self._estimate_tokens(context["components"]["tool_schemas"])

            # Calculate total usage
            total_tokens = sum(context["token_usage"].values())
            context["total_tokens_used"] = total_tokens
            context["budget_utilization"] = total_tokens / config.max_tokens

            logger.debug(f"Assembled context for {run_id}/{node_id}: {total_tokens} tokens ({context['budget_utilization']:.1%} of budget)")

            return context

        except Exception as e:
            logger.error(f"Failed to assemble context: {e}")
            return {"error": str(e), "components": {}}

    async def _get_task_header(self, run_id: str, node_id: str) -> str:
        """Get task header with instructions and constraints"""
        # In production, this would fetch from research goals and node context
        return f"""
        **Research Task: {run_id}**
        Node: {node_id}

        Objective: Complete the assigned research task with high quality results.
        Constraints: Maintain accuracy, cite sources, document methodology.
        Success Criteria: Achieve confidence score > 0.7, provide actionable insights.
        """

    async def record_memory_stats(
        self,
        run_id: str,
        stat_type: str,
        stat_data: Dict[str, Any],
        node_id: Optional[str] = None
    ):
        """Record memory system statistics"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO memory_stats (run_id, node_id, stat_type, stat_data)
                    VALUES (?, ?, ?, ?)
                """, (run_id, node_id, stat_type, json.dumps(stat_data)))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record memory stats: {e}")

    async def get_memory_system_metrics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory system performance metrics"""
        try:
            with self._get_connection() as conn:
                metrics = {}

                # Event counts
                where_clause = "WHERE run_id = ?" if run_id else ""
                params = [run_id] if run_id else []

                cursor = conn.execute(f"""
                    SELECT COUNT(*) as total_events,
                           COUNT(CASE WHEN artifact_uri IS NOT NULL THEN 1 END) as offloaded_events,
                           AVG(token_cost) as avg_token_cost,
                           AVG(importance) as avg_importance
                    FROM memory_events {where_clause}
                """, params)

                event_stats = cursor.fetchone()
                metrics["events"] = dict(event_stats)

                # Snapshot counts
                cursor = conn.execute(f"""
                    SELECT COUNT(*) as total_snapshots,
                           AVG(token_count) as avg_snapshot_tokens,
                           COUNT(DISTINCT run_id || '|' || node_id) as unique_contexts
                    FROM memory_snapshots {where_clause}
                """, params)

                snapshot_stats = cursor.fetchone()
                metrics["snapshots"] = dict(snapshot_stats)

                # Semantic memory
                cursor = conn.execute("""
                    SELECT COUNT(*) as total_chunks,
                           AVG(usage_count) as avg_usage,
                           COUNT(DISTINCT org_scope) as unique_scopes
                    FROM semantic_chunks
                """)

                semantic_stats = cursor.fetchone()
                metrics["semantic_memory"] = dict(semantic_stats)

                return metrics

        except Exception as e:
            logger.error(f"Failed to get memory metrics: {e}")
            return {"error": str(e)}

    async def cleanup_expired_memory(self, max_age_days: int = 30):
        """Clean up old memory data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)

            with self._get_connection() as conn:
                # Delete old low-importance events
                cursor = conn.execute("""
                    DELETE FROM memory_events
                    WHERE ts < ? AND importance < 0.3
                """, (cutoff_date.isoformat(),))

                deleted_events = cursor.rowcount

                # Clean up expired KV entries
                cursor = conn.execute("""
                    DELETE FROM kv_memory
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                """, (datetime.now().isoformat(),))

                deleted_kv = cursor.rowcount

                conn.commit()

                logger.info(f"Memory cleanup: deleted {deleted_events} events, {deleted_kv} KV entries")

        except Exception as e:
            logger.error(f"Failed to cleanup memory: {e}")