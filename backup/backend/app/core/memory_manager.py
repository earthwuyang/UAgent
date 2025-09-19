"""
Memory Management System for uagent Research Tree
Stores experiment results, findings, and knowledge in vector database for better retrieval and README generation
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Represents a memory entry in the system"""
    id: str
    goal_id: str
    node_id: str
    timestamp: datetime
    entry_type: str  # 'experiment', 'finding', 'error', 'success', 'insight'
    title: str
    content: str
    context: Dict[str, Any]
    embedding: Optional[List[float]] = None
    tags: List[str] = None
    workspace_path: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class MemoryManager:
    """Manages experiment memory using vector database for semantic search and retrieval"""

    def __init__(self, db_path: str = "./memory_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

        # Initialize sentence transformer for embeddings
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded SentenceTransformer model successfully")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            self.encoder = None

        # Initialize ChromaDB
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            self.collection = self.client.get_or_create_collection(
                name="uagent_memory",
                metadata={"description": "uagent research tree memory system"}
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None

    def _generate_id(self, content: str, goal_id: str, node_id: str) -> str:
        """Generate unique ID for memory entry"""
        data = f"{goal_id}:{node_id}:{content[:100]}"
        return hashlib.md5(data.encode()).hexdigest()

    def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for text content"""
        if not self.encoder:
            return None

        try:
            embedding = self.encoder.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return None

    async def store_memory(
        self,
        goal_id: str,
        node_id: str,
        entry_type: str,
        title: str,
        content: str,
        context: Dict[str, Any] = None,
        tags: List[str] = None,
        workspace_path: str = None
    ) -> str:
        """Store a memory entry in the vector database"""

        if context is None:
            context = {}
        if tags is None:
            tags = []

        # Generate unique ID
        memory_id = self._generate_id(content, goal_id, node_id)

        # Create embedding
        embedding = self._create_embedding(f"{title}\n{content}")

        # Create memory entry
        entry = MemoryEntry(
            id=memory_id,
            goal_id=goal_id,
            node_id=node_id,
            timestamp=datetime.now(),
            entry_type=entry_type,
            title=title,
            content=content,
            context=context,
            embedding=embedding,
            tags=tags,
            workspace_path=workspace_path
        )

        # Store in ChromaDB
        if self.collection:
            try:
                self.collection.add(
                    ids=[memory_id],
                    documents=[f"{title}\n{content}"],
                    embeddings=[embedding] if embedding else None,
                    metadatas=[{
                        "goal_id": goal_id,
                        "node_id": node_id,
                        "entry_type": entry_type,
                        "title": title,
                        "timestamp": entry.timestamp.isoformat(),
                        "tags": ",".join(tags),
                        "workspace_path": workspace_path or "",
                        "context": json.dumps(context)
                    }]
                )
                logger.info(f"Stored memory entry: {memory_id}")
            except Exception as e:
                logger.error(f"Failed to store memory entry: {e}")

        return memory_id

    async def retrieve_memories(
        self,
        query: str = None,
        goal_id: str = None,
        entry_type: str = None,
        tags: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memories based on query and filters"""

        if not self.collection:
            return []

        try:
            # Build filters
            where_conditions = {}
            if goal_id:
                where_conditions["goal_id"] = goal_id
            if entry_type:
                where_conditions["entry_type"] = entry_type

            # If we have a query, do semantic search
            if query and self.encoder:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=where_conditions if where_conditions else None
                )
            else:
                # Otherwise, get recent entries
                results = self.collection.get(
                    where=where_conditions if where_conditions else None,
                    limit=limit
                )

            # Process results
            memories = []
            if 'documents' in results:
                for i, doc in enumerate(results['documents']):
                    if isinstance(doc, list):
                        doc = doc[0] if doc else ""

                    metadata = results['metadatas'][i] if 'metadatas' in results else {}
                    if isinstance(metadata, list):
                        metadata = metadata[0] if metadata else {}

                    memory = {
                        'id': results['ids'][i] if 'ids' in results else "",
                        'content': doc,
                        'goal_id': metadata.get('goal_id', ''),
                        'node_id': metadata.get('node_id', ''),
                        'entry_type': metadata.get('entry_type', ''),
                        'title': metadata.get('title', ''),
                        'timestamp': metadata.get('timestamp', ''),
                        'tags': metadata.get('tags', '').split(',') if metadata.get('tags') else [],
                        'workspace_path': metadata.get('workspace_path', ''),
                        'context': json.loads(metadata.get('context', '{}'))
                    }
                    memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    async def get_goal_summary(self, goal_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of all memories for a research goal"""

        memories = await self.retrieve_memories(goal_id=goal_id, limit=100)

        # Categorize memories
        summary = {
            'goal_id': goal_id,
            'total_entries': len(memories),
            'experiments': [],
            'findings': [],
            'errors': [],
            'successes': [],
            'insights': [],
            'workspace_paths': set(),
            'key_learnings': []
        }

        for memory in memories:
            entry_type = memory['entry_type']
            if entry_type in summary:
                summary[entry_type].append(memory)

            if memory['workspace_path']:
                summary['workspace_paths'].add(memory['workspace_path'])

        # Convert set to list for JSON serialization
        summary['workspace_paths'] = list(summary['workspace_paths'])

        # Generate key learnings from successful experiments
        success_content = []
        for success in summary['successes']:
            success_content.append(success['content'])

        if success_content:
            # Use semantic search to find most important learnings
            key_learnings = await self.retrieve_memories(
                query="key learning important result outcome",
                goal_id=goal_id,
                limit=5
            )
            summary['key_learnings'] = key_learnings

        return summary

# Global memory manager instance
memory_manager = MemoryManager()