"""Simple cache implementation for UAgent"""

import json
import time
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod


class Cache(ABC):
    """Abstract cache interface"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries"""
        pass


class MemoryCache(Cache):
    """Simple in-memory cache implementation"""

    def __init__(self):
        """Initialize memory cache"""
        self._data: Dict[str, Dict[str, Any]] = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self._data:
            return None

        entry = self._data[key]

        # Check if expired
        if entry.get("expires_at") and time.time() > entry["expires_at"]:
            await self.delete(key)
            return None

        return entry["value"]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        entry = {
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + ttl if ttl else None
        }
        self._data[key] = entry

    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        if key in self._data:
            del self._data[key]

    async def clear(self) -> None:
        """Clear all cache entries"""
        self._data.clear()

    def size(self) -> int:
        """Get number of cache entries"""
        return len(self._data)


class RedisCache(Cache):
    """Redis cache implementation"""

    def __init__(self, redis_client):
        """Initialize Redis cache

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.redis.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        try:
            serialized = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
        except Exception:
            # Silently fail for cache errors
            pass

    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        try:
            await self.redis.delete(key)
        except Exception:
            # Silently fail for cache errors
            pass

    async def clear(self) -> None:
        """Clear all cache entries"""
        try:
            await self.redis.flushdb()
        except Exception:
            # Silently fail for cache errors
            pass


def create_cache(cache_type: str = "memory", **kwargs) -> Cache:
    """Factory function to create cache instance

    Args:
        cache_type: Type of cache ('memory', 'redis')
        **kwargs: Additional parameters for cache initialization

    Returns:
        Cache instance
    """
    if cache_type == "memory":
        return MemoryCache()
    elif cache_type == "redis":
        redis_client = kwargs.get("redis_client")
        if not redis_client:
            raise ValueError("redis_client is required for RedisCache")
        return RedisCache(redis_client)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")