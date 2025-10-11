"""Redis cache implementation for recommendation system."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

LOGGER = logging.getLogger(__name__)


class RedisCache:
    """Redis cache manager with fallback support."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        decode_responses: bool = True,
    ):
        """Initialize Redis connection."""
        self.host = host
        self.port = port
        self.db = db
        self.enabled = True

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=decode_responses,
            )
            # Test connection
            self.client.ping()
            LOGGER.info("Redis connection established: %s:%s", host, port)
        except (ConnectionError, TimeoutError) as exc:
            LOGGER.warning("Redis connection failed: %s. Cache disabled.", exc)
            self.enabled = False
            self.client = None
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Unexpected error connecting to Redis: %s", exc)
            self.enabled = False
            self.client = None

    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if not self.enabled or not self.client:
            return None

        try:
            return self.client.get(key)
        except RedisError as exc:
            LOGGER.warning("Redis GET failed for key %s: %s", key, exc)
            return None

    def set(
        self,
        key: str,
        value: Union[str, bytes],
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache with optional TTL."""
        if not self.enabled or not self.client:
            return False

        try:
            if ttl:
                return self.client.setex(key, ttl, value)
            return self.client.set(key, value)
        except RedisError as exc:
            LOGGER.warning("Redis SET failed for key %s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.enabled or not self.client:
            return False

        try:
            return bool(self.client.delete(key))
        except RedisError as exc:
            LOGGER.warning("Redis DELETE failed for key %s: %s", key, exc)
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.enabled or not self.client:
            return False

        try:
            return bool(self.client.exists(key))
        except RedisError as exc:
            LOGGER.warning("Redis EXISTS failed for key %s: %s", key, exc)
            return False

    def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from cache."""
        value = self.get(key)
        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to decode JSON for key %s: %s", key, exc)
            return None

    def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set JSON value in cache."""
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            return self.set(key, serialized, ttl)
        except (TypeError, ValueError) as exc:
            LOGGER.warning("Failed to serialize JSON for key %s: %s", key, exc)
            return False

    def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Add members to sorted set."""
        if not self.enabled or not self.client:
            return 0

        try:
            return self.client.zadd(key, mapping)
        except RedisError as exc:
            LOGGER.warning("Redis ZADD failed for key %s: %s", key, exc)
            return 0

    def zrevrange(
        self,
        key: str,
        start: int = 0,
        end: int = -1,
        withscores: bool = False,
    ) -> List[Any]:
        """Get members from sorted set in descending order."""
        if not self.enabled or not self.client:
            return []

        try:
            return self.client.zrevrange(key, start, end, withscores=withscores)
        except RedisError as exc:
            LOGGER.warning("Redis ZREVRANGE failed for key %s: %s", key, exc)
            return []

    def zincrby(self, key: str, amount: float, value: str) -> float:
        """Increment score of member in sorted set."""
        if not self.enabled or not self.client:
            return 0.0

        try:
            return self.client.zincrby(key, amount, value)
        except RedisError as exc:
            LOGGER.warning("Redis ZINCRBY failed for key %s: %s", key, exc)
            return 0.0

    def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on key."""
        if not self.enabled or not self.client:
            return False

        try:
            return bool(self.client.expire(key, ttl))
        except RedisError as exc:
            LOGGER.warning("Redis EXPIRE failed for key %s: %s", key, exc)
            return False

    def pipeline(self):
        """Create Redis pipeline."""
        if not self.enabled or not self.client:
            return None

        return self.client.pipeline()

    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self.enabled or not self.client:
            return 0

        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except RedisError as exc:
            LOGGER.warning("Redis clear_pattern failed for %s: %s", pattern, exc)
            return 0


# Global cache instance
_cache: Optional[RedisCache] = None


def get_cache() -> Optional[RedisCache]:
    """Get global cache instance."""
    global _cache

    if _cache is None:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            # Parse Redis URL
            import urllib.parse

            parsed = urllib.parse.urlparse(redis_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 6379
            password = parsed.password
            db = int(parsed.path.lstrip("/")) if parsed.path else 0
        else:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            password = os.getenv("REDIS_PASSWORD")
            db = int(os.getenv("REDIS_DB", "0"))

        _cache = RedisCache(
            host=host,
            port=port,
            db=db,
            password=password,
        )

    return _cache


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(
    ttl: int = 300,
    key_prefix: str = "",
    key_func: Optional[Callable] = None,
):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        key_func: Custom function to generate cache key
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            if not cache or not cache.enabled:
                return func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key_str = key_func(*args, **kwargs)
            else:
                cache_key_str = cache_key(func.__name__, *args, **kwargs)

            if key_prefix:
                cache_key_str = f"{key_prefix}:{cache_key_str}"

            # Try to get from cache
            cached_value = cache.get_json(cache_key_str)
            if cached_value is not None:
                LOGGER.debug("Cache HIT: %s", cache_key_str)
                return cached_value

            # Cache miss - compute value
            LOGGER.debug("Cache MISS: %s", cache_key_str)
            result = func(*args, **kwargs)

            # Store in cache
            if result is not None:
                cache.set_json(cache_key_str, result, ttl=ttl)

            return result

        return wrapper

    return decorator


class HotItemTracker:
    """Track and retrieve hot/trending items."""

    def __init__(self, cache: RedisCache):
        """Initialize tracker."""
        self.cache = cache
        self.hourly_key = "trending:1h"
        self.daily_key = "trending:24h"

    def track_view(self, dataset_id: int) -> None:
        """Track a dataset view."""
        if not self.cache or not self.cache.enabled:
            return

        try:
            pipeline = self.cache.pipeline()
            if pipeline:
                # Increment hourly counter
                pipeline.zincrby(self.hourly_key, 1, str(dataset_id))
                pipeline.expire(self.hourly_key, 3600)

                # Increment daily counter
                pipeline.zincrby(self.daily_key, 1, str(dataset_id))
                pipeline.expire(self.daily_key, 86400)

                pipeline.execute()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to track view for dataset %s: %s", dataset_id, exc)

    def track_click(self, dataset_id: int) -> None:
        """Track a dataset click (higher weight)."""
        if not self.cache or not self.cache.enabled:
            return

        try:
            pipeline = self.cache.pipeline()
            if pipeline:
                # Clicks have higher weight (3x)
                pipeline.zincrby(self.hourly_key, 3, str(dataset_id))
                pipeline.expire(self.hourly_key, 3600)

                pipeline.zincrby(self.daily_key, 3, str(dataset_id))
                pipeline.expire(self.daily_key, 86400)

                pipeline.execute()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to track click for dataset %s: %s", dataset_id, exc)

    def get_hot_items(self, limit: int = 100, timeframe: str = "1h") -> List[int]:
        """
        Get hot/trending items.

        Args:
            limit: Number of items to return
            timeframe: '1h' or '24h'

        Returns:
            List of dataset IDs sorted by popularity
        """
        if not self.cache or not self.cache.enabled:
            return []

        key = self.hourly_key if timeframe == "1h" else self.daily_key

        try:
            items = self.cache.zrevrange(key, 0, limit - 1)
            return [int(item) for item in items]
        except (ValueError, TypeError) as exc:
            LOGGER.warning("Failed to parse hot items: %s", exc)
            return []


# Initialize hot item tracker
def get_hot_tracker() -> Optional[HotItemTracker]:
    """Get global hot item tracker."""
    cache = get_cache()
    if cache:
        return HotItemTracker(cache)
    return None
