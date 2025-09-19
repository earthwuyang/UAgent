"""
Semantic Memory Module for UAgent
Implements vector-based semantic search and memory retrieval with MMR diversification
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SemanticMatch:
    """A semantic search match"""
    chunk_id: int
    text: str
    title: Optional[str]
    score: float
    meta: Dict[str, Any]
    usage_count: int
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalConfig:
    """Configuration for semantic retrieval"""
    k: int = 10  # Number of candidates to retrieve
    mmr_lambda: float = 0.5  # Balance between relevance and diversity (0.0 = max diversity, 1.0 = max relevance)
    min_score_threshold: float = 0.1  # Minimum similarity score
    boost_recent: bool = True  # Boost recently used chunks
    boost_usage: bool = True  # Boost frequently used chunks


class SemanticMemoryEngine:
    """
    Semantic memory engine with vector embeddings and MMR-based retrieval
    """

    def __init__(self, embedding_dimension: int = 384, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_dimension = embedding_dimension
        self.model_name = model_name

        # Initialize Hugging Face embedding model
        self._embedding_model = None
        self._model_available = False

        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(model_name)
            self._model_available = True
            # Update dimension based on actual model
            self.embedding_dimension = self._embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Initialized embedding model {model_name} with dimension {self.embedding_dimension}")
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to simple text matching")
            self._model_available = False
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            self._model_available = False

        # Cache for embeddings
        self._embedding_cache: Dict[str, List[float]] = {}

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using Hugging Face model"""
        if not self._model_available:
            return None

        try:
            # Check cache first
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if text_hash in self._embedding_cache:
                return self._embedding_cache[text_hash]

            # Generate embedding using Hugging Face model
            if self._embedding_model:
                # Run in thread pool to avoid blocking
                import asyncio
                loop = asyncio.get_event_loop()

                embedding = await loop.run_in_executor(
                    None,
                    lambda: self._embedding_model.encode([text], normalize_embeddings=True)[0].tolist()
                )

                # Cache the result
                self._embedding_cache[text_hash] = embedding
                return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")

        return None

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            return float(dot_product / (norm_v1 * norm_v2))

        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0

    def text_similarity(self, text1: str, text2: str) -> float:
        """Simple text-based similarity when embeddings are not available"""
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
            return 0.0

        return intersection / union

    async def search_semantic_chunks(
        self,
        db_connection,
        query_text: str,
        config: RetrievalConfig = None,
        org_scope: str = "default"
    ) -> List[SemanticMatch]:
        """
        Search semantic memory chunks with vector similarity and MMR diversification
        """
        config = config or RetrievalConfig()

        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query_text)

            # Fetch all chunks from database
            cursor = db_connection.execute("""
                SELECT chunk_id, title, text, meta_json, usage_count, avg_relevance,
                       embedding_vector, created_at, updated_at
                FROM semantic_chunks
                WHERE org_scope = ?
                ORDER BY usage_count DESC, avg_relevance DESC
            """, (org_scope,))

            chunks = []
            for row in cursor:
                chunk_data = dict(row)
                chunk_data["meta"] = json.loads(chunk_data.get("meta_json", "{}"))

                # Parse embedding if available
                embedding_json = chunk_data.get("embedding_vector")
                embedding = None
                if embedding_json:
                    try:
                        embedding = json.loads(embedding_json)
                    except:
                        pass

                chunks.append({
                    "chunk_id": chunk_data["chunk_id"],
                    "title": chunk_data["title"],
                    "text": chunk_data["text"],
                    "meta": chunk_data["meta"],
                    "usage_count": chunk_data["usage_count"],
                    "avg_relevance": chunk_data["avg_relevance"],
                    "embedding": embedding,
                    "created_at": chunk_data["created_at"],
                    "updated_at": chunk_data["updated_at"]
                })

            # Calculate similarities
            candidates = []
            for chunk in chunks:
                # Calculate similarity score
                if query_embedding and chunk["embedding"]:
                    # Vector similarity
                    score = self.cosine_similarity(query_embedding, chunk["embedding"])
                else:
                    # Fallback to text similarity
                    score = self.text_similarity(query_text, chunk["text"])

                # Apply boosting factors
                boosted_score = score

                if config.boost_usage and chunk["usage_count"] > 0:
                    usage_boost = min(chunk["usage_count"] / 10.0, 0.2)  # Max 20% boost
                    boosted_score += usage_boost

                if config.boost_recent:
                    # Boost items updated recently
                    try:
                        updated_date = datetime.fromisoformat(chunk["updated_at"].replace("Z", "+00:00"))
                        days_old = (datetime.now() - updated_date).days
                        if days_old < 7:  # Less than a week old
                            recent_boost = (7 - days_old) / 70.0  # Max 10% boost
                            boosted_score += recent_boost
                    except:
                        pass

                if boosted_score >= config.min_score_threshold:
                    candidates.append(SemanticMatch(
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        title=chunk["title"],
                        score=boosted_score,
                        meta=chunk["meta"],
                        usage_count=chunk["usage_count"],
                        embedding=chunk["embedding"]
                    ))

            # Sort by score and take top candidates
            candidates.sort(key=lambda x: x.score, reverse=True)
            top_candidates = candidates[:config.k * 2]  # Get more candidates for MMR

            # Apply MMR diversification
            if len(top_candidates) > config.k:
                selected = self.mmr_diversify(top_candidates, config.k, config.mmr_lambda)
            else:
                selected = top_candidates

            logger.debug(f"Semantic search: {len(candidates)} candidates -> {len(selected)} selected")

            return selected

        except Exception as e:
            logger.error(f"Failed to search semantic chunks: {e}")
            return []

    def mmr_diversify(
        self,
        candidates: List[SemanticMatch],
        k: int,
        lambda_param: float = 0.5
    ) -> List[SemanticMatch]:
        """
        Apply Maximal Marginal Relevance (MMR) for result diversification
        """
        if not candidates:
            return []

        selected = []
        remaining = candidates.copy()

        # Always select the highest scoring item first
        if remaining:
            selected.append(remaining.pop(0))

        # Select remaining items using MMR
        while remaining and len(selected) < k:
            best_score = float('-inf')
            best_item = None
            best_idx = -1

            for i, candidate in enumerate(remaining):
                # Relevance score (normalized)
                relevance = candidate.score

                # Diversity score (maximum similarity to already selected items)
                max_similarity = 0.0
                for selected_item in selected:
                    if candidate.embedding and selected_item.embedding:
                        similarity = self.cosine_similarity(candidate.embedding, selected_item.embedding)
                    else:
                        similarity = self.text_similarity(candidate.text, selected_item.text)
                    max_similarity = max(max_similarity, similarity)

                # MMR score: λ * relevance - (1-λ) * max_similarity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = candidate
                    best_idx = i

            if best_item:
                selected.append(best_item)
                remaining.pop(best_idx)

        return selected

    async def store_semantic_chunk(
        self,
        db_connection,
        text: str,
        title: Optional[str] = None,
        meta: Dict[str, Any] = None,
        org_scope: str = "default"
    ) -> int:
        """
        Store a semantic chunk with embedding
        """
        try:
            meta = meta or {}
            content_hash = hashlib.sha256(text.encode()).hexdigest()

            # Check if chunk already exists
            cursor = db_connection.execute("""
                SELECT chunk_id FROM semantic_chunks
                WHERE content_hash = ? AND org_scope = ?
            """, (content_hash, org_scope))

            existing = cursor.fetchone()
            if existing:
                # Update usage count
                db_connection.execute("""
                    UPDATE semantic_chunks
                    SET usage_count = usage_count + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ?
                """, (existing[0],))
                db_connection.commit()
                return existing[0]

            # Generate embedding
            embedding = await self.generate_embedding(text)
            embedding_json = json.dumps(embedding) if embedding else None

            # Insert new chunk
            cursor = db_connection.execute("""
                INSERT INTO semantic_chunks
                (org_scope, title, text, meta_json, content_hash, usage_count, embedding_vector)
                VALUES (?, ?, ?, ?, ?, 1, ?)
            """, (org_scope, title, text, json.dumps(meta), content_hash, embedding_json))

            chunk_id = cursor.lastrowid
            db_connection.commit()

            logger.info(f"Stored semantic chunk {chunk_id} with embedding: {embedding is not None}")
            return chunk_id

        except Exception as e:
            logger.error(f"Failed to store semantic chunk: {e}")
            return -1

    async def update_chunk_relevance(
        self,
        db_connection,
        chunk_id: int,
        relevance_score: float
    ):
        """
        Update the average relevance score for a chunk
        """
        try:
            # Get current stats
            cursor = db_connection.execute("""
                SELECT usage_count, avg_relevance FROM semantic_chunks
                WHERE chunk_id = ?
            """, (chunk_id,))

            result = cursor.fetchone()
            if not result:
                return

            usage_count, current_avg = result

            # Calculate new average relevance
            total_relevance = current_avg * usage_count
            new_avg_relevance = (total_relevance + relevance_score) / (usage_count + 1)

            # Update the chunk
            db_connection.execute("""
                UPDATE semantic_chunks
                SET avg_relevance = ?, usage_count = usage_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            """, (new_avg_relevance, chunk_id))

            db_connection.commit()

        except Exception as e:
            logger.error(f"Failed to update chunk relevance: {e}")

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics"""
        return {
            "cached_embeddings": len(self._embedding_cache),
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name,
            "model_available": self._model_available,
            "cache_size_mb": sum(len(str(emb)) for emb in self._embedding_cache.values()) / (1024 * 1024)
        }

    async def batch_generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts efficiently"""
        if not self._model_available:
            return [None] * len(texts)

        try:
            # Check cache for existing embeddings
            embeddings = []
            texts_to_generate = []
            indices_to_generate = []

            for i, text in enumerate(texts):
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                if text_hash in self._embedding_cache:
                    embeddings.append(self._embedding_cache[text_hash])
                else:
                    embeddings.append(None)
                    texts_to_generate.append(text)
                    indices_to_generate.append(i)

            # Generate missing embeddings in batch
            if texts_to_generate and self._embedding_model:
                # Run in thread pool to avoid blocking
                import asyncio
                loop = asyncio.get_event_loop()

                batch_embeddings = await loop.run_in_executor(
                    None,
                    lambda: self._embedding_model.encode(texts_to_generate, normalize_embeddings=True).tolist()
                )

                for i, embedding in enumerate(batch_embeddings):
                    original_index = indices_to_generate[i]
                    embeddings[original_index] = embedding

                    # Cache the result
                    text_hash = hashlib.sha256(texts_to_generate[i].encode()).hexdigest()
                    self._embedding_cache[text_hash] = embedding

            return embeddings

        except Exception as e:
            logger.error(f"Failed to batch generate embeddings: {e}")
            return [None] * len(texts)

    async def recompute_all_embeddings(self, db_connection, batch_size: int = 50):
        """Recompute embeddings for all chunks that don't have them"""
        try:
            # Find chunks without embeddings
            cursor = db_connection.execute("""
                SELECT chunk_id, text FROM semantic_chunks
                WHERE embedding_vector IS NULL OR embedding_vector = ''
                ORDER BY chunk_id
            """)

            chunks_to_update = [(row[0], row[1]) for row in cursor]

            if not chunks_to_update:
                logger.info("All chunks already have embeddings")
                return

            logger.info(f"Recomputing embeddings for {len(chunks_to_update)} chunks")

            # Process in batches
            for i in range(0, len(chunks_to_update), batch_size):
                batch = chunks_to_update[i:i + batch_size]
                texts = [chunk[1] for chunk in batch]
                chunk_ids = [chunk[0] for chunk in batch]

                # Generate embeddings
                embeddings = await self.batch_generate_embeddings(texts)

                # Update database
                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    if embedding:
                        embedding_json = json.dumps(embedding)
                        db_connection.execute("""
                            UPDATE semantic_chunks
                            SET embedding_vector = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE chunk_id = ?
                        """, (embedding_json, chunk_id))

                db_connection.commit()
                logger.info(f"Updated embeddings for batch {i//batch_size + 1}/{(len(chunks_to_update) + batch_size - 1)//batch_size}")

                # Small delay to avoid rate limiting
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Failed to recompute embeddings: {e}")