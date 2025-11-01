"""Prometheus metrics for recommendation service."""
from __future__ import annotations

import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict

from prometheus_client import Counter, Gauge, Histogram, Info

# Service info
service_info = Info("recommendation_service", "Recommendation service information")

# Request metrics
recommendation_requests_total = Counter(
    "recommendation_requests_total",
    "Total number of recommendation requests",
    ["endpoint", "status"],
)

recommendation_latency_seconds = Histogram(
    "recommendation_latency_seconds",
    "Recommendation request latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

recommendation_count = Histogram(
    "recommendation_count",
    "Number of items returned in recommendation",
    ["endpoint"],
    buckets=(0, 1, 5, 10, 20, 50, 100),
)

# Timeout metrics
recommendation_timeouts_total = Counter(
    "recommendation_timeouts_total",
    "Number of recommendation timeouts",
    ["endpoint", "operation"],
)

# Thread pool queue size
thread_pool_queue_gauge = Gauge(
    "recommendation_thread_pool_queue_size",
    "Current size of recommendation thread pool queue",
)

# Degradation metrics
recommendation_degraded_total = Counter(
    "recommendation_degraded_total",
    "Number of degraded responses served",
    ["endpoint", "reason"],
)

# Cache metrics
cache_requests_total = Counter(
    "cache_requests_total",
    "Total number of cache requests",
    ["operation", "status"],
)

cache_hit_rate = Gauge(
    "cache_hit_rate",
    "Cache hit rate (hits / total requests)",
)

# Model metrics
model_inference_latency_seconds = Histogram(
    "model_inference_latency_seconds",
    "Model inference latency in seconds",
    ["model_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

recall_candidates_count = Histogram(
    "recall_candidates_count",
    "Number of candidates from recall stage",
    ["recall_type"],
    buckets=(0, 10, 50, 100, 200, 500, 1000),
)

# Fallback metrics
fallback_triggered_total = Counter(
    "fallback_triggered_total",
    "Number of times fallback was triggered",
    ["reason", "level"],
)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open)",
    ["breaker_name"],
)

circuit_breaker_failures_total = Counter(
    "circuit_breaker_failures_total",
    "Total number of circuit breaker failures",
    ["breaker_name"],
)

# Data quality metrics
feature_missing_rate = Gauge(
    "feature_missing_rate",
    "Rate of missing features",
    ["feature_name"],
)

user_coverage_ratio = Gauge(
    "user_coverage_ratio",
    "Ratio of users with recommendations available",
)

item_coverage_ratio = Gauge(
    "item_coverage_ratio",
    "Ratio of items being recommended",
)

# Business metrics
user_interaction_total = Counter(
    "user_interaction_total",
    "Total user interactions with recommendations",
    ["interaction_type"],
)

recommendation_click_total = Counter(
    "recommendation_click_total",
    "Total clicks on recommendations",
    ["source"],
)

# Error metrics
error_total = Counter(
    "error_total",
    "Total number of errors",
    ["error_type", "endpoint"],
)

# Exposure / fallback metrics
recommendation_exposures_total = Counter(
    "recommendation_exposures_total",
    "Number of recommendation items exposed",
    ["endpoint", "variant", "experiment_variant", "degrade_reason"],
)

fallback_ratio_gauge = Gauge(
    "recommendation_fallback_ratio",
    "Ratio of fallback items served per endpoint (rolling)",
    ["endpoint"],
)


class MetricsTracker:
    """Track metrics for monitoring."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.exposure_totals: Dict[str, int] = defaultdict(int)
        self.fallback_totals: Dict[str, int] = defaultdict(int)

    def track_cache_hit(self) -> None:
        """Track cache hit."""
        self.cache_hits += 1
        cache_requests_total.labels(operation="get", status="hit").inc()
        self._update_cache_hit_rate()

    def track_cache_miss(self) -> None:
        """Track cache miss."""
        self.cache_misses += 1
        cache_requests_total.labels(operation="get", status="miss").inc()
        self._update_cache_hit_rate()

    def _update_cache_hit_rate(self) -> None:
        """Update cache hit rate gauge."""
        total = self.cache_hits + self.cache_misses
        if total > 0:
            rate = self.cache_hits / total
            cache_hit_rate.set(rate)

    def track_fallback(self, reason: str, level: int) -> None:
        """Track fallback usage."""
        fallback_triggered_total.labels(reason=reason, level=level).inc()

    def track_error(self, error_type: str, endpoint: str) -> None:
        """Track error occurrence."""
        error_total.labels(error_type=error_type, endpoint=endpoint).inc()

    def track_exposure(self, endpoint: str, degrade_reason: str, count: int) -> None:
        """Track exposure counts and update fallback ratio gauge."""
        if count <= 0:
            return
        self.exposure_totals[endpoint] += count
        if degrade_reason and degrade_reason != "none":
            self.fallback_totals[endpoint] += count
        total = self.exposure_totals[endpoint]
        if total > 0:
            ratio = self.fallback_totals[endpoint] / total
            fallback_ratio_gauge.labels(endpoint=endpoint).set(ratio)


# Global metrics tracker
_metrics_tracker: MetricsTracker | None = None


def get_metrics_tracker() -> MetricsTracker:
    """Get global metrics tracker instance."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    return _metrics_tracker


def track_request_metrics(endpoint: str):
    """
    Decorator to track request metrics.

    Args:
        endpoint: Name of the endpoint being tracked
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                get_metrics_tracker().track_error(error_type, endpoint)
                raise
            finally:
                # Track latency
                latency = time.time() - start_time
                recommendation_latency_seconds.labels(endpoint=endpoint).observe(latency)

                # Track request count
                recommendation_requests_total.labels(
                    endpoint=endpoint, status=status
                ).inc()

        return wrapper

    return decorator


def track_model_inference(model_type: str):
    """
    Decorator to track model inference time.

    Args:
        model_type: Type of model (e.g., 'ranking', 'recall')
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                latency = time.time() - start_time
                model_inference_latency_seconds.labels(model_type=model_type).observe(
                    latency
                )

        return wrapper

    return decorator
