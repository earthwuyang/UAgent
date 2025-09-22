"""Agent memory store built on AgentVectorDB (LanceDB-backed)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agentvectordb import AgentVectorDBStore
from agentvectordb.embeddings import DefaultTextEmbeddingFunction


@dataclass
class AVDBConfig:
    """Configuration for AgentVectorDB-backed memory."""

    db_path: str
    dim: int = 384
    importance_min: float = 0.35
    max_age_days: int = 30

    @property
    def max_age_seconds(self) -> int:
        return int(self.max_age_days * 24 * 60 * 60)


class AgentMemory:
    """High-level memory helper built on AgentVectorDB.

    Collections:
      * episodic   – concrete events / traces
      * semantic   – distilled insights & summaries
      * procedural – how-to knowledge, tool usage
      * preferences – agent policies / preferences
    """

    def __init__(self, cfg: AVDBConfig):
        self.cfg = cfg
        Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.store = AgentVectorDBStore(db_path=cfg.db_path)
        embedding_fn = DefaultTextEmbeddingFunction(dimension=cfg.dim)
        self.collections = {
            "episodic": self.store.get_or_create_collection("episodic", embedding_fn),
            "semantic": self.store.get_or_create_collection("semantic", embedding_fn),
            "procedural": self.store.get_or_create_collection("procedural", embedding_fn),
            "preferences": self.store.get_or_create_collection("preferences", embedding_fn),
        }

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------
    async def add_episodic(self, content: str, *, importance: float = 0.5, tags: Optional[Dict[str, Any]] = None) -> None:
        await self._add("episodic", content, importance, tags)

    async def add_semantic(self, content: str, *, importance: float = 0.6, tags: Optional[Dict[str, Any]] = None) -> None:
        await self._add("semantic", content, importance, tags)

    async def add_procedural(self, content: str, *, importance: float = 0.5, tags: Optional[Dict[str, Any]] = None) -> None:
        await self._add("procedural", content, importance, tags)

    async def add_preference(self, content: str, *, importance: float = 0.4, tags: Optional[Dict[str, Any]] = None) -> None:
        await self._add("preferences", content, importance, tags)

    async def save_debate(self, topic: str, transcripts: Iterable[Dict[str, Any]], verdict: Dict[str, Any]) -> None:
        summary = verdict.get("summary") or verdict.get("verdict") or ""
        text = f"[DEBATE] {topic}\nSUMMARY: {summary}\n"
        await self.add_episodic(text, importance=0.7, tags={"topic": topic, "type": "debate"})
        if summary:
            await self.add_semantic(summary, importance=0.6, tags={"topic": topic, "source": "debate"})

    async def _add(self, collection: str, content: str, importance: float, tags: Optional[Dict[str, Any]]) -> None:
        record = {"content": content, "importance_score": float(importance)}
        if tags:
            record.update(tags)
        await asyncio.to_thread(self.collections[collection].add, record)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------
    async def query(self, collection: str, query_text: str, *, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        coll = self.collections[collection]
        result = await asyncio.to_thread(coll.query, query_text=query_text, k=k, filter=filter)
        return list(result or [])

    async def retrieve_context(self, topic: str, *, k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Return top memories across collections for a topic string."""
        episodic = await self.query("episodic", topic, k=k)
        semantic = await self.query("semantic", topic, k=k)
        procedural = await self.query("procedural", topic, k=k)
        preferences = await self.query("preferences", topic, k=min(3, k))
        return {
            "episodic": episodic,
            "semantic": semantic,
            "procedural": procedural,
            "preferences": preferences,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    async def prune(self) -> None:
        for coll in self.collections.values():
            await asyncio.to_thread(
                coll.prune_memories,
                max_age_seconds=self.cfg.max_age_seconds,
                min_importance_score=self.cfg.importance_min,
            )
