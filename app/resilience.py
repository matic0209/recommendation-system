"""Resilience and fault tolerance for recommendation service."""
from __future__ import annotations

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from circuitbreaker import circuit

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class FallbackResult:
    items: List[int]
    level: int
    source: str


class FallbackStrategy:
    """Multi-level fallback strategy for recommendations."""

    def __init__(
        self,
        cache=None,
        precomputed_dir: Optional[Path] = None,
        static_popular: Optional[List[int]] = None,
    ):
        """
        Initialize fallback strategy.

        Args:
            cache: Redis cache instance
            precomputed_dir: Directory with precomputed recommendations
            static_popular: Static popular items list
        """
        self.cache = cache
        self.precomputed_dir = precomputed_dir
        self.static_popular = static_popular or []
        self.precomputed_cache: Dict[int, List[int]] = {}

        # Load precomputed recommendations if available
        if precomputed_dir and precomputed_dir.exists():
            self._load_precomputed()

    def _load_precomputed(self) -> None:
        """Load precomputed recommendations from disk."""
        if not self.precomputed_dir:
            return

        try:
            precomputed_file = self.precomputed_dir / "precomputed_recommendations.pkl"
            if precomputed_file.exists():
                with open(precomputed_file, "rb") as f:
                    self.precomputed_cache = pickle.load(f)
                LOGGER.info(
                    "Loaded precomputed recommendations for %d items",
                    len(self.precomputed_cache),
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load precomputed recommendations: %s", exc)

    def get_recommendations(
        self,
        dataset_id: int,
        limit: int = 10,
        user_id: Optional[int] = None,
    ) -> List[int]:
        """
        Get recommendations with fallback levels.

        Fallback order:
        1. Redis cache
        2. Precomputed results
        3. Static popular items

        Args:
            dataset_id: Target dataset ID
            limit: Number of recommendations
            user_id: Optional user ID for personalized cache

        Returns:
            List of recommended dataset IDs
        """
        result = self.get_with_metadata(dataset_id, limit=limit, user_id=user_id)
        return result.items

    def get_with_metadata(
        self,
        dataset_id: int,
        limit: int = 10,
        user_id: Optional[int] = None,
    ) -> FallbackResult:
        """Same as get_recommendations but with metadata about source level."""
        # Level 1: Try Redis cache
        if self.cache and self.cache.enabled:
            try:
                cache_key = (
                    f"fallback:recommend:{dataset_id}:{user_id}:{limit}"
                    if user_id
                    else f"fallback:similar:{dataset_id}:{limit}"
                )
                cached = self.cache.get_json(cache_key)
                if cached and isinstance(cached, list):
                    LOGGER.debug("Fallback Level 1 (Redis): %s", cache_key)
                    return FallbackResult(cached[:limit], level=1, source="redis")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Redis fallback failed: %s", exc)

        # Level 2: Try precomputed results
        if dataset_id in self.precomputed_cache:
            LOGGER.debug("Fallback Level 2 (Precomputed): dataset_id=%s", dataset_id)
            return FallbackResult(
                self.precomputed_cache[dataset_id][:limit], level=2, source="precomputed"
            )

        # Level 3: Return static popular items
        LOGGER.debug("Fallback Level 3 (Popular): dataset_id=%s", dataset_id)
        items = [item for item in self.static_popular if item != dataset_id][:limit]
        return FallbackResult(items, level=3, source="popular")

    def save_to_cache(
        self,
        dataset_id: int,
        recommendations: List[int],
        limit: int = 10,
        user_id: Optional[int] = None,
        ttl: int = 3600,
    ) -> None:
        """Save recommendations to cache for fallback."""
        if not self.cache or not self.cache.enabled:
            return

        try:
            cache_key = (
                f"fallback:recommend:{dataset_id}:{user_id}:{limit}"
                if user_id
                else f"fallback:similar:{dataset_id}:{limit}"
            )
            self.cache.set_json(cache_key, recommendations, ttl=ttl)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to save fallback cache: %s", exc)


def with_timeout(seconds: float):
    """
    Decorator to add timeout to async functions.

    Args:
        seconds: Timeout in seconds
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                LOGGER.warning(
                    "Function %s timed out after %s seconds",
                    func.__name__,
                    seconds,
                )
                raise

        return wrapper

    return decorator


def with_fallback(fallback_value: Any = None, log_errors: bool = True):
    """
    Decorator to provide fallback value on exception.

    Args:
        fallback_value: Value to return on exception
        log_errors: Whether to log errors
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if log_errors:
                    LOGGER.warning(
                        "Function %s failed with error: %s. Returning fallback value.",
                        func.__name__,
                        exc,
                    )
                return fallback_value

        return wrapper

    return decorator


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
):
    """
    Decorator to add circuit breaker pattern.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying again
        expected_exception: Exception type to catch
    """

    def decorator(func: Callable) -> Callable:
        @circuit(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class TimeoutManager:
    """Manage timeouts for different operations."""

    # Default timeouts (in seconds)
    TIMEOUTS = {
        "redis_get": 0.1,
        "redis_set": 0.2,
        "model_inference": 1.0,
        "database_query": 2.0,
        "recommendation_total": 2.0,
        "feature_fetch": 0.5,
    }

    @classmethod
    def get_timeout(cls, operation: str) -> float:
        """Get timeout for operation."""
        return cls.TIMEOUTS.get(operation, 1.0)

    @classmethod
    def set_timeout(cls, operation: str, seconds: float) -> None:
        """Set custom timeout for operation."""
        cls.TIMEOUTS[operation] = seconds


class HealthChecker:
    """Health check for service dependencies."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, bool] = {}

    def check_redis(self, cache) -> bool:
        """Check Redis health."""
        if not cache or not cache.enabled:
            self.checks["redis"] = False
            return False

        try:
            cache.client.ping()
            self.checks["redis"] = True
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Redis health check failed: %s", exc)
            self.checks["redis"] = False
            return False

    def check_models(self, state) -> bool:
        """Check if models are loaded."""
        models_loaded = getattr(state, "models_loaded", False)
        self.checks["models"] = models_loaded
        return models_loaded

    def get_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        all_healthy = all(self.checks.values()) if self.checks else False
        return {
            "healthy": all_healthy,
            "checks": self.checks,
        }


def precompute_recommendations(
    model_bundle,
    dataset_ids: List[int],
    limit: int = 50,
    output_path: Optional[Path] = None,
) -> Dict[int, List[int]]:
    """
    Precompute recommendations for all datasets.

    This can be run offline to generate fallback recommendations.

    Args:
        model_bundle: Model bundle with behavior/content/vector models
        dataset_ids: List of dataset IDs to precompute
        limit: Number of recommendations per item
        output_path: Path to save precomputed results

    Returns:
        Dictionary mapping dataset_id to list of recommendations
    """
    LOGGER.info("Precomputing recommendations for %d datasets", len(dataset_ids))

    precomputed = {}

    for dataset_id in dataset_ids:
        # Combine scores from all models
        candidates = set()

        # Behavior-based
        if dataset_id in model_bundle.behavior:
            behavior_items = list(model_bundle.behavior[dataset_id].keys())[:limit]
            candidates.update(behavior_items)

        # Content-based
        if dataset_id in model_bundle.content:
            content_items = list(model_bundle.content[dataset_id].keys())[:limit]
            candidates.update(content_items)

        # Vector-based
        if dataset_id in model_bundle.vector:
            vector_items = [
                entry["dataset_id"] for entry in model_bundle.vector[dataset_id][:limit]
            ]
            candidates.update(vector_items)

        # Remove self
        candidates.discard(dataset_id)

        # Convert to list
        precomputed[dataset_id] = list(candidates)[:limit]

    # Save to disk if path provided
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(precomputed, f)
            LOGGER.info("Saved precomputed recommendations to %s", output_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to save precomputed recommendations: %s", exc)

    return precomputed
