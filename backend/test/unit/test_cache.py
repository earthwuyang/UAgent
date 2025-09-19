"""
Unit tests for cache functionality
"""

import asyncio
import time
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.cache import MemoryCache, RedisCache, create_cache


class TestMemoryCache:
    """Test MemoryCache functionality"""

    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = MemoryCache()
        assert hasattr(cache, '_data')
        assert isinstance(cache._data, dict)
        assert len(cache._data) == 0

    @pytest.mark.asyncio
    async def test_set_and_get_basic(self):
        """Test basic set and get operations"""
        cache = MemoryCache()

        # Set a value
        await cache.set("test_key", "test_value")

        # Get the value
        result = await cache.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self):
        """Test getting a non-existent key returns None"""
        cache = MemoryCache()

        result = await cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self):
        """Test setting with custom TTL"""
        cache = MemoryCache()

        await cache.set("test_key", "test_value", ttl=100)
        result = await cache.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_set_with_no_ttl(self):
        """Test setting with no TTL (should not expire)"""
        cache = MemoryCache()

        await cache.set("test_key", "test_value")
        result = await cache.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that items expire after TTL"""
        cache = MemoryCache()

        # Set with very short TTL
        await cache.set("test_key", "test_value", ttl=1)

        # Should be available immediately
        result = await cache.get("test_key")
        assert result == "test_value"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired now
        result = await cache.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_key(self):
        """Test deleting a key"""
        cache = MemoryCache()

        await cache.set("test_key", "test_value")
        await cache.delete("test_key")

        result = await cache.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self):
        """Test deleting a non-existent key (should not raise error)"""
        cache = MemoryCache()

        # Should not raise an exception
        await cache.delete("nonexistent_key")

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing the entire cache"""
        cache = MemoryCache()

        # Add multiple items
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Clear cache
        await cache.clear()

        # All items should be gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_overwrite_existing_key(self):
        """Test overwriting an existing key"""
        cache = MemoryCache()

        await cache.set("test_key", "original_value")
        await cache.set("test_key", "new_value")

        result = await cache.get("test_key")
        assert result == "new_value"

    def test_cache_size(self):
        """Test cache size method"""
        cache = MemoryCache()

        assert cache.size() == 0

    @pytest.mark.asyncio
    async def test_cache_size_tracking(self):
        """Test that cache size is tracked correctly"""
        cache = MemoryCache()

        assert cache.size() == 0

        await cache.set("key1", "value1")
        assert cache.size() == 1

        await cache.set("key2", "value2")
        assert cache.size() == 2

        await cache.delete("key1")
        assert cache.size() == 1

        await cache.clear()
        assert cache.size() == 0

    @pytest.mark.asyncio
    async def test_cache_value_types(self):
        """Test cache with different value types"""
        cache = MemoryCache()

        # String value
        await cache.set("str", "string_value")
        assert await cache.get("str") == "string_value"

        # Dictionary value
        dict_value = {"key": "value", "number": 42}
        await cache.set("dict", dict_value)
        assert await cache.get("dict") == dict_value

        # List value
        list_value = [1, 2, 3, "four"]
        await cache.set("list", list_value)
        assert await cache.get("list") == list_value

        # None value
        await cache.set("none", None)
        assert await cache.get("none") is None

        # Integer value
        await cache.set("int", 42)
        assert await cache.get("int") == 42

        # Boolean value
        await cache.set("bool", True)
        assert await cache.get("bool") is True

    @pytest.mark.asyncio
    async def test_cache_entry_structure(self):
        """Test that cache entries have the correct structure"""
        cache = MemoryCache()

        await cache.set("test_key", "test_value", ttl=100)

        # Access internal data structure
        entry = cache._data["test_key"]
        assert "value" in entry
        assert "created_at" in entry
        assert "expires_at" in entry
        assert entry["value"] == "test_value"
        assert isinstance(entry["created_at"], float)
        assert isinstance(entry["expires_at"], float)

    @pytest.mark.asyncio
    async def test_cache_entry_without_ttl(self):
        """Test cache entries without TTL"""
        cache = MemoryCache()

        await cache.set("test_key", "test_value")

        entry = cache._data["test_key"]
        assert entry["expires_at"] is None

    @pytest.mark.asyncio
    async def test_expired_item_cleanup(self):
        """Test that expired items are removed when accessed"""
        cache = MemoryCache()

        # Set item with short TTL
        await cache.set("test_key", "test_value", ttl=1)

        # Verify it exists in internal storage
        assert "test_key" in cache._data

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Access expired item - should remove it and return None
        result = await cache.get("test_key")
        assert result is None

        # Should be removed from internal storage
        assert "test_key" not in cache._data

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent cache access"""
        cache = MemoryCache()

        async def set_items(start_idx):
            for i in range(start_idx, start_idx + 5):
                await cache.set(f"key_{i}", f"value_{i}")

        async def get_items(start_idx):
            results = []
            for i in range(start_idx, start_idx + 5):
                result = await cache.get(f"key_{i}")
                results.append(result)
            return results

        # Run concurrent set operations
        await asyncio.gather(
            set_items(0),
            set_items(10),
            set_items(20)
        )

        # Run concurrent get operations
        results = await asyncio.gather(
            get_items(0),
            get_items(10),
            get_items(20)
        )

        # Verify items were set and retrieved
        all_results = [item for sublist in results for item in sublist if item is not None]
        assert len(all_results) == 15  # 3 groups * 5 items each


class TestRedisCache:
    """Test RedisCache functionality"""

    def test_redis_cache_initialization(self):
        """Test RedisCache initialization"""
        mock_redis = MagicMock()
        cache = RedisCache(mock_redis)
        assert cache.redis == mock_redis

    @pytest.mark.asyncio
    async def test_redis_get_success(self):
        """Test successful Redis get operation"""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps("test_value")

        cache = RedisCache(mock_redis)
        result = await cache.get("test_key")

        assert result == "test_value"
        mock_redis.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_redis_get_not_found(self):
        """Test Redis get when key doesn't exist"""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None

        cache = RedisCache(mock_redis)
        result = await cache.get("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_redis_get_exception(self):
        """Test Redis get with exception"""
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = Exception("Redis error")

        cache = RedisCache(mock_redis)
        result = await cache.get("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_redis_set_with_ttl(self):
        """Test Redis set with TTL"""
        mock_redis = AsyncMock()
        cache = RedisCache(mock_redis)

        await cache.set("test_key", "test_value", ttl=100)

        mock_redis.setex.assert_called_once_with("test_key", 100, json.dumps("test_value"))

    @pytest.mark.asyncio
    async def test_redis_set_without_ttl(self):
        """Test Redis set without TTL"""
        mock_redis = AsyncMock()
        cache = RedisCache(mock_redis)

        await cache.set("test_key", "test_value")

        mock_redis.set.assert_called_once_with("test_key", json.dumps("test_value"))

    @pytest.mark.asyncio
    async def test_redis_set_exception(self):
        """Test Redis set with exception (should not raise)"""
        mock_redis = AsyncMock()
        mock_redis.set.side_effect = Exception("Redis error")

        cache = RedisCache(mock_redis)

        # Should not raise exception
        await cache.set("test_key", "test_value")

    @pytest.mark.asyncio
    async def test_redis_delete(self):
        """Test Redis delete operation"""
        mock_redis = AsyncMock()
        cache = RedisCache(mock_redis)

        await cache.delete("test_key")

        mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_redis_delete_exception(self):
        """Test Redis delete with exception"""
        mock_redis = AsyncMock()
        mock_redis.delete.side_effect = Exception("Redis error")

        cache = RedisCache(mock_redis)

        # Should not raise exception
        await cache.delete("test_key")

    @pytest.mark.asyncio
    async def test_redis_clear(self):
        """Test Redis clear operation"""
        mock_redis = AsyncMock()
        cache = RedisCache(mock_redis)

        await cache.clear()

        mock_redis.flushdb.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_clear_exception(self):
        """Test Redis clear with exception"""
        mock_redis = AsyncMock()
        mock_redis.flushdb.side_effect = Exception("Redis error")

        cache = RedisCache(mock_redis)

        # Should not raise exception
        await cache.clear()


class TestCacheFactory:
    """Test cache factory function"""

    def test_create_memory_cache(self):
        """Test creating memory cache"""
        cache = create_cache("memory")
        assert isinstance(cache, MemoryCache)

    def test_create_redis_cache(self):
        """Test creating Redis cache"""
        mock_redis = MagicMock()
        cache = create_cache("redis", redis_client=mock_redis)
        assert isinstance(cache, RedisCache)
        assert cache.redis == mock_redis

    def test_create_redis_cache_without_client(self):
        """Test creating Redis cache without client raises error"""
        with pytest.raises(ValueError, match="redis_client is required"):
            create_cache("redis")

    def test_create_invalid_cache_type(self):
        """Test creating cache with invalid type raises error"""
        with pytest.raises(ValueError, match="Unknown cache type"):
            create_cache("invalid_type")

    def test_create_memory_cache_ignores_extra_kwargs(self):
        """Test that memory cache creation ignores extra kwargs"""
        cache = create_cache("memory", extra_param="ignored")
        assert isinstance(cache, MemoryCache)