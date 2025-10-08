"""Caching layer for dependency analysis results."""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from cachetools import TTLCache

from .models import FileDependencies
from .exceptions import CacheError


class DependencyCacheManager:
    """Manages caching of dependency analysis results."""

    def __init__(
        self,
        cache_backend: str = 'memory',
        redis_url: Optional[str] = None,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize cache manager.
        
        Args:
            cache_backend: 'memory' or 'redis'
            redis_url: Redis connection URL (if using Redis)
            ttl_seconds: Time-to-live for cache entries
            max_size: Maximum cache size (for memory cache)
            logger: Optional logger
        """
        self.cache_backend = cache_backend
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize cache backend
        self._memory_cache: Optional[TTLCache] = None
        self._redis_client: Optional[Any] = None
        self._cache_lock = threading.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0
        }
        
        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize the cache backend."""
        if self.cache_backend == 'memory':
            self._initialize_memory_cache()
        elif self.cache_backend == 'redis':
            self._initialize_redis_cache()
        else:
            raise ValueError(f"Unsupported cache backend: {self.cache_backend}")

    def _initialize_memory_cache(self) -> None:
        """Initialize in-memory cache using cachetools."""
        try:
            self._memory_cache = TTLCache(maxsize=self.max_size, ttl=self.ttl_seconds)
            self.logger.debug(f"Initialized memory cache with max_size={self.max_size}, ttl={self.ttl_seconds}s")
        except Exception as e:
            self.logger.error(f"Failed to initialize memory cache: {e}")
            raise CacheError(f"Memory cache initialization failed: {e}")

    def _initialize_redis_cache(self) -> None:
        """Initialize Redis cache."""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available, falling back to memory cache")
            self.cache_backend = 'memory'
            self._initialize_memory_cache()
            return
        
        if not self.redis_url:
            self.logger.warning("Redis URL not provided, falling back to memory cache")
            self.cache_backend = 'memory'
            self._initialize_memory_cache()
            return
        
        try:
            self._redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self._redis_client.ping()
            self.logger.debug(f"Initialized Redis cache at {self.redis_url}")
        
        except Exception as e:
            self.logger.warning(f"Failed to connect to Redis, falling back to memory cache: {e}")
            self.cache_backend = 'memory'
            self._initialize_memory_cache()

    async def get(self, file_path: str, file_hash: str) -> Optional[FileDependencies]:
        """Get cached dependencies for file.
        
        Args:
            file_path: Absolute file path
            file_hash: SHA256 hash of file content
            
        Returns:
            FileDependencies if found in cache, None otherwise
        """
        cache_key = self._make_cache_key(file_path, file_hash)
        
        try:
            if self.cache_backend == 'memory':
                return await self._get_from_memory(cache_key)
            elif self.cache_backend == 'redis':
                return await self._get_from_redis(cache_key)
        except Exception as e:
            self.logger.warning(f"Cache get error for {file_path}: {e}")
            self._stats['errors'] += 1
        
        self._stats['misses'] += 1
        return None

    async def set(
        self,
        file_path: str,
        file_hash: str,
        dependencies: FileDependencies
    ) -> None:
        """Store dependencies in cache.
        
        Args:
            file_path: Absolute file path
            file_hash: SHA256 hash of file content
            dependencies: FileDependencies to cache
        """
        cache_key = self._make_cache_key(file_path, file_hash)
        
        try:
            if self.cache_backend == 'memory':
                await self._set_to_memory(cache_key, dependencies)
            elif self.cache_backend == 'redis':
                await self._set_to_redis(cache_key, dependencies)
            
            self._stats['sets'] += 1
            self.logger.debug(f"Cached dependencies for {file_path}")
        
        except Exception as e:
            self.logger.warning(f"Cache set error for {file_path}: {e}")
            self._stats['errors'] += 1

    async def invalidate(self, file_path: str) -> None:
        """Remove file from cache.
        
        Args:
            file_path: Absolute file path to remove
        """
        try:
            if self.cache_backend == 'memory':
                await self._invalidate_from_memory(file_path)
            elif self.cache_backend == 'redis':
                await self._invalidate_from_redis(file_path)
            
            self.logger.debug(f"Invalidated cache for {file_path}")
        
        except Exception as e:
            self.logger.warning(f"Cache invalidation error for {file_path}: {e}")
            self._stats['errors'] += 1

    async def clear(self) -> None:
        """Clear entire cache."""
        try:
            if self.cache_backend == 'memory':
                with self._cache_lock:
                    if self._memory_cache:
                        self._memory_cache.clear()
            elif self.cache_backend == 'redis':
                if self._redis_client:
                    # Delete all keys with our prefix
                    pattern = "dep_cache:*"
                    keys = self._redis_client.keys(pattern)
                    if keys:
                        self._redis_client.delete(*keys)
            
            self.logger.debug("Cleared dependency cache")
        
        except Exception as e:
            self.logger.warning(f"Cache clear error: {e}")
            self._stats['errors'] += 1

    async def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_ratio = self._stats['hits'] / total_requests if total_requests > 0 else 0
        
        stats = {
            'backend': self.cache_backend,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_ratio': hit_ratio,
            'sets': self._stats['sets'],
            'errors': self._stats['errors'],
            'total_requests': total_requests
        }
        
        # Add backend-specific stats
        if self.cache_backend == 'memory' and self._memory_cache:
            with self._cache_lock:
                stats.update({
                    'current_size': len(self._memory_cache),
                    'max_size': self._memory_cache.maxsize,
                    'ttl_seconds': self._memory_cache.ttl
                })
        elif self.cache_backend == 'redis' and self._redis_client:
            try:
                info = self._redis_client.info('memory')
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'unknown'),
                    'redis_connected': True
                })
            except Exception:
                stats['redis_connected'] = False
        
        return stats

    async def _get_from_memory(self, cache_key: str) -> Optional[FileDependencies]:
        """Get from memory cache."""
        with self._cache_lock:
            if self._memory_cache and cache_key in self._memory_cache:
                data = self._memory_cache[cache_key]
                # Validate that it's a FileDependencies object
                if isinstance(data, FileDependencies):
                    self._stats['hits'] += 1
                    return data
        return None

    async def _set_to_memory(self, cache_key: str, dependencies: FileDependencies) -> None:
        """Set to memory cache."""
        with self._cache_lock:
            if self._memory_cache:
                self._memory_cache[cache_key] = dependencies

    async def _invalidate_from_memory(self, file_path: str) -> None:
        """Invalidate from memory cache."""
        with self._cache_lock:
            if self._memory_cache:
                # Find keys that contain the file path
                keys_to_remove = [
                    key for key in self._memory_cache.keys()
                    if file_path in key
                ]
                for key in keys_to_remove:
                    del self._memory_cache[key]

    async def _get_from_redis(self, cache_key: str) -> Optional[FileDependencies]:
        """Get from Redis cache."""
        if not self._redis_client:
            return None
        
        try:
            data = self._redis_client.get(cache_key)
            if data:
                # Deserialize JSON to FileDependencies
                import json
                json_data = json.loads(data)
                dependencies = FileDependencies(**json_data)
                self._stats['hits'] += 1
                return dependencies
        except Exception as e:
            self.logger.debug(f"Redis get error for {cache_key}: {e}")
        
        return None

    async def _set_to_redis(self, cache_key: str, dependencies: FileDependencies) -> None:
        """Set to Redis cache."""
        if not self._redis_client:
            return
        
        try:
            # Serialize FileDependencies to JSON
            json_data = dependencies.model_dump_json()
            
            # Set with TTL
            self._redis_client.setex(
                name=cache_key,
                time=self.ttl_seconds,
                value=json_data
            )
        except Exception as e:
            self.logger.debug(f"Redis set error for {cache_key}: {e}")
            raise

    async def _invalidate_from_redis(self, file_path: str) -> None:
        """Invalidate from Redis cache."""
        if not self._redis_client:
            return
        
        try:
            # Find keys that contain the file path
            pattern = f"dep_cache:*{file_path.replace('/', '_')}*"
            keys = self._redis_client.keys(pattern)
            if keys:
                self._redis_client.delete(*keys)
        except Exception as e:
            self.logger.debug(f"Redis invalidation error for {file_path}: {e}")
            raise

    def _make_cache_key(self, file_path: str, file_hash: str) -> str:
        """Create cache key from file path and hash.
        
        Args:
            file_path: Absolute file path
            file_hash: SHA256 hash of file content
            
        Returns:
            Cache key string
        """
        # Create key that includes file hash for invalidation
        # Replace path separators to avoid key issues in Redis
        safe_path = file_path.replace('/', '_').replace('\\', '_')
        return f"dep_cache:{safe_path}:{file_hash}"
