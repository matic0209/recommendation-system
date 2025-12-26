"""FastAPI service exposing dataset detail recommendations."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import sqlite3
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from datetime import datetime, timezone
from threading import Lock, Thread

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sklearn.pipeline import Pipeline
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config.settings import BASE_DIR, DATA_DIR, FEATURE_STORE_PATH, MODEL_REGISTRY_PATH, MODELS_DIR
from pipeline.feature_store_redis import RedisFeatureStore
from app.telemetry import record_exposure
from app.model_manager import get_current_version, load_run_id_from_dir, deploy_from_source
from app.cache import get_cache, get_hot_tracker, cached
from app.experiments import assign_variant, load_experiments
from app.resilience import (
    FallbackStrategy,
    HealthChecker,
    TimeoutManager,
    with_circuit_breaker,
    with_fallback,
)
from app.metrics import (
    get_metrics_tracker,
    recommendation_count,
    recommendation_latency_seconds,
    recommendation_requests_total,
    recommendation_exposures_total,
    service_info,
    track_request_metrics,
    recommendation_degraded_total,
    recommendation_timeouts_total,
    thread_pool_queue_gauge,
)
from app.sentry_config import (
    init_sentry,
    set_user_context,
    set_request_context,
    set_recommendation_context,
    capture_exception_with_context,
    add_breadcrumb,
)

EXECUTOR_MAX_WORKERS = int(os.getenv("RECO_THREAD_POOL_WORKERS", "4"))
EXECUTOR = ThreadPoolExecutor(max_workers=EXECUTOR_MAX_WORKERS)
SLOW_OPERATION_THRESHOLD = float(os.getenv("SLOW_OPERATION_THRESHOLD", "0.5"))
STAGE_LOG_THRESHOLD = float(os.getenv("STAGE_LOG_THRESHOLD", "0.25"))
USER_FEATURE_CACHE_TTL = float(os.getenv("USER_FEATURE_CACHE_TTL", "30.0"))
TimeoutManager.configure_from_env()


class ExperimentFileHandler(FileSystemEventHandler):
    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path.resolve()

    def on_modified(self, event):
        if event.is_directory:
            return
        try:
            changed = Path(event.src_path).resolve()
        except FileNotFoundError:
            return
        if changed != self.config_path:
            return
        try:
            app.state.experiments = load_experiments(self.config_path)
            LOGGER.info(
                "Experiments config reloaded automatically (%d experiments)",
                len(app.state.experiments),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to reload experiments after file change: %s", exc)


def _executor_queue_size() -> int:
    queue = getattr(EXECUTOR, "_work_queue", None)
    if queue is None:
        return 0
    return queue.qsize()


async def _run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    thread_pool_queue_gauge.set(_executor_queue_size())
    result = await loop.run_in_executor(EXECUTOR, partial(func, *args, **kwargs))
    thread_pool_queue_gauge.set(_executor_queue_size())
    return result

@dataclass
class ModelBundle:
    behavior: Dict[int, Dict[int, float]]
    content: Dict[int, Dict[int, float]]
    vector: Dict[int, List[Dict[str, float]]]
    popular: List[int]
    rank_model: Optional[Pipeline]
    run_id: Optional[str]

def _load_model_bundle(base_dir: Path, run_id: Optional[str] = None) -> ModelBundle:
    behavior = _load_pickle(base_dir / "item_sim_behavior.pkl")
    content = _load_pickle(base_dir / "item_sim_content.pkl")
    vector = _load_vector_recall(base_dir / "item_recall_vector.json")
    popular = _load_popular(base_dir / "top_items.json")
    rank_model = _load_rank_model(base_dir / "rank_model.pkl")
    if run_id is None:
        run_id = load_run_id_from_dir(base_dir)
    popular = [int(item) for item in popular]
    return ModelBundle(behavior=behavior, content=content, vector=vector, popular=popular, rank_model=rank_model, run_id=run_id)


import random

def _set_bundle(state, bundle: ModelBundle, *, prefix: str) -> None:
    def _attr(name: str) -> str:
        return f"{prefix}_{name}" if prefix else name

    setattr(state, _attr("behavior"), bundle.behavior)
    setattr(state, _attr("content"), bundle.content)
    setattr(state, _attr("vector_recall"), bundle.vector)
    setattr(state, _attr("rank_model"), bundle.rank_model)
    setattr(state, _attr("popular"), bundle.popular)
    setattr(state, _attr("model_run_id"), bundle.run_id)
    setattr(state, _attr("bundle"), bundle)


def _choose_bundle(state) -> tuple[ModelBundle, str]:
    rollout = getattr(state, "shadow_rollout", 0.0) or 0.0
    shadow_bundle = getattr(state, "shadow_bundle", None)
    if shadow_bundle and rollout > 0 and random.random() < rollout:
        return shadow_bundle, "shadow"
    primary = getattr(state, "bundle", None)
    if primary is None:
        raise RuntimeError("Primary model bundle not loaded")
    return primary, "primary"


def _collect_dataset_ids(bundle: ModelBundle) -> Set[int]:
    dataset_ids: Set[int] = set(bundle.popular)
    for source, neighbors in bundle.behavior.items():
        dataset_ids.add(int(source))
        dataset_ids.update(int(neighbor) for neighbor in neighbors.keys())
    for source, neighbors in bundle.content.items():
        dataset_ids.add(int(source))
        dataset_ids.update(int(neighbor) for neighbor in neighbors.keys())
    for source, entries in bundle.vector.items():
        dataset_ids.add(int(source))
        dataset_ids.update(int(entry.get("dataset_id", 0)) for entry in entries)
    return {item for item in dataset_ids if item}


LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Dataset Recommendation API")

# Set service info for Prometheus
service_info.info({
    "version": "1.0.0",
    "service": "recommendation-api",
    "environment": "production",
})

REQUEST_ID_HEADER = "X-Request-ID"
DEFAULT_CHANNEL_WEIGHTS = {
    "behavior": 1.5,  # Increased: strongest personalization signal
    "content": 0.8,   # Increased: important for relevance
    "vector": 0.5,    # Slightly increased but relatively lower priority
    "popular": 0.05,  # Increased: provides diversity
}


class HotUserData:
    """Manage user-centric datasets with TTL-based refresh."""

    def __init__(self, dataset_tags: Dict[int, List[str]], ttl_seconds: int = 300):
        self.dataset_tags = dataset_tags
        self.ttl_seconds = ttl_seconds
        self._lock = Lock()
        self._last_refresh = 0.0
        self._user_history: Dict[int, List[Dict[str, float]]] = {}
        self._user_history_sets: Dict[int, Set[int]] = {}
        self._user_tag_preferences: Dict[int, Dict[str, float]] = {}
        self._user_profiles: Dict[int, Dict[str, Optional[str]]] = {}
        self._refreshing = False

    def update_dataset_tags(self, dataset_tags: Dict[int, List[str]]) -> None:
        self.dataset_tags = dataset_tags
        # Force refresh so tag preferences align with new tags
        self._last_refresh = 0.0

    def bootstrap(
        self,
        history: Dict[int, List[Dict[str, float]]],
        profiles: Dict[int, Dict[str, Optional[str]]],
    ) -> None:
        self._user_history = history or {}
        self._user_profiles = profiles or {}
        self._user_history_sets = {
            user_id: {record["dataset_id"] for record in records}
            for user_id, records in self._user_history.items()
        }
        self._user_tag_preferences = _build_user_tag_preferences(self._user_history, self.dataset_tags)
        self._last_refresh = time.time()

    def _is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self._last_refresh) > self.ttl_seconds

    def _run_refresh(self) -> None:
        """Build a fresh snapshot and atomically swap it in."""
        try:
            fresh_history = _load_user_history()
            fresh_profiles = _load_user_profile()
            fresh_sets = {
                user_id: {record["dataset_id"] for record in records}
                for user_id, records in fresh_history.items()
            }
            fresh_tag_preferences = _build_user_tag_preferences(fresh_history, self.dataset_tags)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Hot user data refresh failed: %s", exc)
            with self._lock:
                self._refreshing = False
            return

        with self._lock:
            self._user_history = fresh_history
            self._user_profiles = fresh_profiles
            self._user_history_sets = fresh_sets
            self._user_tag_preferences = fresh_tag_preferences
            self._last_refresh = time.time()
            self._refreshing = False
        LOGGER.info("Hot user data refreshed (users=%d)", len(fresh_history))

    def _refresh_blocking(self) -> None:
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True
        self._run_refresh()

    def _schedule_async_refresh(self) -> None:
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True
        Thread(target=self._run_refresh, daemon=True).start()

    def ensure_fresh(self, *, force: bool = False) -> None:
        if not self._user_history:
            self._refresh_blocking()
            return
        if force:
            self._schedule_async_refresh()
            return
        if self._is_expired():
            self._schedule_async_refresh()

    def get_history(self) -> Dict[int, List[Dict[str, float]]]:
        self.ensure_fresh()
        return self._user_history

    def get_history_for_user(self, user_id: int) -> Optional[List[Dict[str, float]]]:
        history = self.get_history()
        return history.get(int(user_id))

    def get_history_sets(self) -> Dict[int, Set[int]]:
        self.ensure_fresh()
        return self._user_history_sets

    def get_tag_preferences(self) -> Dict[int, Dict[str, float]]:
        self.ensure_fresh()
        return self._user_tag_preferences

    def get_tag_preferences_for_user(self, user_id: int) -> Dict[str, float]:
        preferences = self.get_tag_preferences()
        return preferences.get(int(user_id), {})

    def get_profiles(self) -> Dict[int, Dict[str, Optional[str]]]:
        self.ensure_fresh()
        return self._user_profiles


def _detect_device_type(user_agent: str) -> str:
    if not user_agent:
        return "unknown"
    ua = user_agent.lower()
    if "ipad" in ua or "tablet" in ua:
        return "tablet"
    if any(keyword in ua for keyword in ["iphone", "android", "mobile"]):
        return "mobile"
    if any(keyword in ua for keyword in ["windows", "macintosh", "linux"]):
        return "desktop"
    return "unknown"


def _extract_request_context(request: Request) -> Dict[str, str]:
    headers = request.headers
    source = headers.get("X-Recommend-Source") or request.query_params.get("source") or "unknown"
    device_type = _detect_device_type(headers.get("User-Agent", ""))
    locale = headers.get("Accept-Language", "unknown").split(",")[0].strip().lower() or "unknown"
    client_app = headers.get("X-Client-App") or ""
    return {
        "source": source,
        "device_type": device_type,
        "locale": locale,
        "client_app": client_app,
    }


def _compute_mmr_lambda(*, endpoint: str, request_context: Optional[Dict[str, str]]) -> float:
    base = 0.7 if endpoint == "recommend_detail" else 0.5
    context = request_context or {}
    source = context.get("source")
    if source in {"search", "landing"}:
        base = 0.5
    device = context.get("device_type")
    if device == "mobile":
        base = min(base + 0.1, 0.85)
    return base


def _resolve_experiment_config_path() -> Path:
    """Resolve experiment config path from ENV or default repository location."""
    env_path = os.getenv("EXPERIMENT_CONFIG_PATH")
    if not env_path:
        return (BASE_DIR / "config" / "experiments.yaml").resolve()
    candidate = Path(env_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (BASE_DIR / candidate).resolve()


def _start_experiment_watcher(config_path: Path) -> Optional[Observer]:
    """Start watchdog observer for experiment config if directory exists."""
    watch_dir = config_path.parent
    if not watch_dir.exists():
        LOGGER.warning(
            "Experiment config directory %s does not exist; automatic reload disabled.",
            watch_dir,
        )
        return None

    observer = Observer()
    handler = ExperimentFileHandler(config_path)
    try:
        observer.schedule(handler, watch_dir.as_posix(), recursive=False)
        observer.daemon = True
        observer.start()
        LOGGER.info("Watching %s for experiment updates", config_path)
        return observer
    except OSError as exc:
        LOGGER.warning("Unable to watch experiment config %s: %s", config_path, exc)
    return None


def _load_channel_weight_overrides() -> Dict[str, float]:
    path = MODELS_DIR / "channel_weights.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to parse channel weight artifact %s: %s", path, exc)
        return {}
    weights = payload.get("weights") if isinstance(payload, dict) else None
    if not isinstance(weights, dict):
        LOGGER.warning("Channel weight artifact missing 'weights' key; ignoring.")
        return {}
    normalized: Dict[str, float] = {}
    for key, value in weights.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _get_channel_weight_baseline(state) -> Dict[str, float]:
    overrides = getattr(state, "channel_weights", None)
    weights = DEFAULT_CHANNEL_WEIGHTS.copy()
    if overrides:
        for channel, value in overrides.items():
            try:
                weights[channel] = float(value)
            except (TypeError, ValueError):
                continue
    return weights


def _get_user_data_manager(state) -> Optional[HotUserData]:
    return getattr(state, "user_data", None)


def _get_user_history_records(state, user_id: int) -> Optional[List[Dict[str, float]]]:
    manager = _get_user_data_manager(state)
    if manager:
        return manager.get_history_for_user(int(user_id))
    history = getattr(state, "user_history", {})
    return history.get(int(user_id))


def _get_user_tag_preferences(state, user_id: int) -> Dict[str, float]:
    manager = _get_user_data_manager(state)
    if manager:
        return manager.get_tag_preferences_for_user(int(user_id))
    preferences = getattr(state, "user_tag_preferences", {})
    return preferences.get(int(user_id), {})


def _get_user_history_sets(state) -> Dict[int, Set[int]]:
    manager = _get_user_data_manager(state)
    if manager:
        return manager.get_history_sets()
    return getattr(state, "user_history_sets", {})


def _get_user_features(state, user_id: Optional[int]) -> Dict[str, float]:
    if not user_id:
        return {}
    feature_store = getattr(state, "feature_store", None)
    if not feature_store:
        return {}
    cache = getattr(state, "_user_feature_cache", None)
    now = time.time()
    user_key = int(user_id)
    if cache:
        cached_entry = cache.get(user_key)
        if cached_entry and now - cached_entry["ts"] < USER_FEATURE_CACHE_TTL:
            return cached_entry["data"].copy()
    try:
        raw = feature_store.get_user_features(user_key)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to fetch user features for %s: %s", user_id, exc)
        return {}
    normalized: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            normalized[f"user_{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    if cache is None:
        cache = {}
        setattr(state, "_user_feature_cache", cache)
    cache[user_key] = {"ts": now, "data": normalized.copy()}
    return normalized


def _log_stage_duration(
    stage: str,
    start_time: float,
    *,
    dataset_id: Optional[int],
    user_id: Optional[int],
    request_id: Optional[str],
) -> None:
    duration = time.perf_counter() - start_time
    if duration < STAGE_LOG_THRESHOLD:
        LOGGER.debug(
            "Stage %s finished quickly (%.3fs) dataset=%s user=%s request=%s",
            stage,
            duration,
            dataset_id,
            user_id,
            request_id,
        )
        return
    LOGGER.info(
        "Stage %s duration=%.3fs dataset=%s user=%s request=%s",
        stage,
        duration,
        dataset_id,
        user_id,
        request_id,
    )


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach request ID and basic timing information to each request."""
    raw_request_id = request.headers.get(REQUEST_ID_HEADER)
    request_id = raw_request_id if raw_request_id and raw_request_id.startswith("req_") else f"req_{uuid.uuid4()}"
    request.state.request_id = request_id

    # 设置 Sentry 请求上下文
    endpoint = request.url.path
    set_request_context(
        request_id=request_id,
        endpoint=endpoint,
        method=request.method,
        url=str(request.url),
    )

    # 添加面包屑
    add_breadcrumb(
        message=f"{request.method} {endpoint}",
        category="http.request",
        level="info",
        data={"request_id": request_id},
    )

    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    response.headers.setdefault("X-Response-Time", f"{duration:.4f}")
    return response


class RecommendationItem(BaseModel):
    dataset_id: int
    title: Optional[str]
    price: Optional[float]
    cover_image: Optional[str]
    score: float
    reason: str


class RecommendationResponse(BaseModel):
    dataset_id: int
    recommendations: List[RecommendationItem]
    request_id: str  # 用于前端埋点追踪
    algorithm_version: Optional[str] = None  # 算法版本，用于A/B对比
    variant: str = "primary"
    experiment_variant: Optional[str] = None
    request_context: Optional[Dict[str, str]] = None


class SimilarResponse(BaseModel):
    dataset_id: int
    similar_items: List[RecommendationItem]
    request_id: str  # 用于前端埋点追踪
    algorithm_version: Optional[str] = None  # 算法版本，用于A/B对比
    variant: str = "primary"
    experiment_variant: Optional[str] = None
    request_context: Optional[Dict[str, str]] = None


class ReloadRequest(BaseModel):
    mode: str = "primary"
    source: Optional[str] = None
    run_id: Optional[str] = None
    rollout: Optional[float] = None


def _load_pickle(path: Path) -> Dict[int, Dict[int, float]]:
    if not path.exists():
        LOGGER.warning("Model file missing: %s", path)
        return {}
    with open(path, "rb") as stream:
        return pickle.load(stream)


def _load_popular(path: Path) -> List[int]:
    if not path.exists():
        LOGGER.warning("Popular list missing: %s", path)
        return []
    return json.loads(path.read_text())


def _load_vector_recall(path: Path) -> Dict[int, List[Dict[str, float]]]:
    if not path.exists():
        LOGGER.warning("Vector recall file missing: %s", path)
        return {}
    raw = json.loads(path.read_text())
    result: Dict[int, List[Dict[str, float]]] = {}
    for key, value in raw.items():
        try:
            dataset_id = int(key)
        except ValueError:
            continue
        entries: List[Dict[str, float]] = []
        for entry in value:
            if isinstance(entry, dict):
                neighbor_id = int(entry.get("dataset_id", 0))
                score = float(entry.get("score", 0.0))
                entries.append({"dataset_id": neighbor_id, "score": score})
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                entries.append({"dataset_id": int(entry[0]), "score": float(entry[1])})
        result[dataset_id] = entries
    return result


def _load_rank_model(path: Path) -> Optional[Pipeline]:
    if not path.exists():
        LOGGER.warning("Ranking model missing: %s", path)
        return None
    with open(path, "rb") as stream:
        try:
            model = pickle.load(stream)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to load ranking model: %s", exc)
            return None
    if isinstance(model, Pipeline):
        return model
    if isinstance(model, dict) and model.get("type") == "lightgbm_ranker":
        return model
    LOGGER.warning("Ranking artifact at %s has unexpected type %s", path, type(model))
    return None


def _load_dataset_stats(
    *,
    feature_store: Optional[RedisFeatureStore] = None,
    dataset_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    frame = pd.DataFrame()
    if feature_store and dataset_ids:
        frame = _load_dataset_stats_from_redis(feature_store, dataset_ids)
        if not frame.empty:
            return _normalize_dataset_stats_frame(frame, source="redis")

    frame = _read_feature_store("SELECT * FROM dataset_stats", parse_dates=["last_event_time"])
    source_label = "sqlite" if not frame.empty else "parquet"
    if frame.empty:
        stats_path = DATA_DIR / "processed" / "dataset_stats.parquet"
        if stats_path.exists():
            frame = pd.read_parquet(stats_path)
            source_label = "parquet"
        else:
            frame = pd.DataFrame(columns=["dataset_id", "interaction_count", "total_weight", "last_event_time"])
            source_label = "sqlite"

    if frame.empty:
        frame = pd.DataFrame(columns=["dataset_id", "interaction_count", "total_weight", "last_event_time"])

    return _normalize_dataset_stats_frame(frame, source=source_label)


def _load_pickle_file(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as stream:
            return pickle.load(stream)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load pickle %s: %s", path, exc)
        return None


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse JSON %s: %s", path, exc)
        return None


def _load_recall_artifacts(base_dir: Path) -> Dict[str, Any]:
    recall_assets: Dict[str, Any] = {}

    user_similarity = _load_pickle_file(base_dir / "user_similarity.pkl")
    if user_similarity:
        recall_assets["user_similarity"] = user_similarity

    for name in ["tag_to_items", "item_to_tags", "category_index", "price_bucket_index"]:
        data = _load_json_file(base_dir / f"{name}.json")
        if data:
            # Convert lists back to sets for faster lookup where needed
            if name in {"tag_to_items", "category_index", "price_bucket_index"}:
                normalized = {}
                for key, value in data.items():
                    if isinstance(value, list):
                        normalized[key] = {int(v) for v in value if v is not None}
                    else:
                        normalized[key] = value
                recall_assets[name] = normalized
            else:
                recall_assets[name] = data

    # Optional Faiss index info (metadata file saved by recall engine)
    faiss_meta = _load_json_file(base_dir / "faiss_recall.meta.json")
    if faiss_meta:
        recall_assets["faiss_meta"] = faiss_meta

    return recall_assets


def _parse_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tag.strip().lower() for tag in str(raw).split(";") if tag.strip()]


def _read_feature_store(query: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if not FEATURE_STORE_PATH.exists():
        return pd.DataFrame()
    try:
        uri = f"file:{FEATURE_STORE_PATH}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            return pd.read_sql_query(query, conn, parse_dates=parse_dates)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Feature store query failed: %s", exc)
        return pd.DataFrame()


def _load_model_run_id(source: Optional[Path] = None) -> Optional[str]:
    registry_path = MODEL_REGISTRY_PATH if source is None else Path(source) / "model_registry.json"
    if not registry_path.exists():
        return None
    try:
        registry = json.loads(registry_path.read_text())
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse model registry: %s", registry_path)
        return None
    current = registry.get("current")
    if isinstance(current, dict):
        return current.get("run_id")
    return None


def _load_dataset_metadata_from_redis(
    feature_store: RedisFeatureStore, dataset_ids: Iterable[int]
) -> pd.DataFrame:
    try:
        features_map = feature_store.get_batch_dataset_features(list(dataset_ids))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to fetch dataset features from Redis: %s", exc)
        return pd.DataFrame()

    if not features_map:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for dataset_id, fields in features_map.items():
        record: Dict[str, Any] = {"dataset_id": int(dataset_id)}
        record.update(fields)
        records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        LOGGER.warning("Redis feature store returned no dataset metadata")
    return frame


def _parse_dataset_metadata_frame(
    frame: pd.DataFrame,
    *,
    source: str,
) -> Tuple[Dict[int, Dict[str, Optional[str]]], Dict[int, List[str]], pd.DataFrame]:
    if frame.empty:
        return {}, {}, pd.DataFrame()

    frame = frame.copy()
    rename_map = {
        "dataset_id": "dataset_id",
        "dataset_name": "title",
        "name": "title",
    }
    frame = frame.rename(columns=rename_map)

    if "dataset_id" not in frame.columns:
        LOGGER.warning("Dataset metadata frame missing dataset_id column (source=%s)", source)
        return {}, {}, pd.DataFrame()

    metadata: Dict[int, Dict[str, Optional[str]]] = {}
    dataset_tags: Dict[int, List[str]] = {}
    for row in frame.to_dict(orient="records"):
        dataset_id = int(pd.to_numeric(row.get("dataset_id"), errors="coerce") or 0)
        if not dataset_id:
            continue
        title_value = row.get("title")
        if title_value in (None, ""):
            title = None
        else:
            title = str(title_value)
        metadata[dataset_id] = {
            "title": title,
            "price": row.get("price"),
            "cover_image": row.get("cover_image"),
            "company": row.get("create_company_name"),
        }
        dataset_tags[dataset_id] = _parse_tags(row.get("tag"))

    raw_columns = [col for col in ["dataset_id", "price", "description", "tag"] if col in frame.columns]
    raw_features = frame[raw_columns].copy()
    raw_features["dataset_id"] = pd.to_numeric(
        raw_features.get("dataset_id"), errors="coerce"
    ).fillna(0).astype(int)
    if "price" in raw_features.columns:
        raw_features["price"] = pd.to_numeric(
            raw_features.get("price"), errors="coerce"
        ).fillna(0.0)
    if "description" in raw_features.columns:
        raw_features["description"] = raw_features.get("description", "").fillna("").astype(str)
    if "tag" in raw_features.columns:
        raw_features["tag"] = raw_features.get("tag", "").fillna("").astype(str)

    LOGGER.info(
        "Loaded dataset metadata from %s (%d items)", source, len(metadata)
    )
    return metadata, dataset_tags, raw_features


def _load_dataset_stats_from_redis(
    feature_store: RedisFeatureStore, dataset_ids: Iterable[int]
) -> pd.DataFrame:
    try:
        stats_map = feature_store.get_dataset_stats(list(dataset_ids))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to fetch dataset stats from Redis: %s", exc)
        return pd.DataFrame()

    if not stats_map:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for dataset_id, fields in stats_map.items():
        record: Dict[str, Any] = {"dataset_id": int(dataset_id)}
        record.update(fields)
        records.append(record)

    return pd.DataFrame(records)


def _normalize_dataset_stats_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["dataset_id"] = pd.to_numeric(frame.get("dataset_id"), errors="coerce").fillna(0).astype(int)
    if "interaction_count" in frame.columns:
        frame["interaction_count"] = pd.to_numeric(
            frame.get("interaction_count"), errors="coerce"
        ).fillna(0.0)
    if "total_weight" in frame.columns:
        frame["total_weight"] = pd.to_numeric(
            frame.get("total_weight"), errors="coerce"
        ).fillna(0.0)
    if "last_event_time" in frame.columns:
        frame["last_event_time"] = pd.to_datetime(
            frame.get("last_event_time"), errors="coerce"
        )

    LOGGER.info(
        "Loaded dataset stats from %s (%d items)", source, len(frame.index)
    )
    return frame


def _load_dataset_metadata(
    *,
    feature_store: Optional[RedisFeatureStore] = None,
    dataset_ids: Optional[Iterable[int]] = None,
) -> Tuple[Dict[int, Dict[str, Optional[str]]], Dict[int, List[str]], pd.DataFrame]:
    frame = pd.DataFrame()
    if feature_store and dataset_ids:
        frame = _load_dataset_metadata_from_redis(feature_store, dataset_ids)
        if not frame.empty:
            return _parse_dataset_metadata_frame(frame, source="redis")

    frame = _read_feature_store("SELECT * FROM dataset_features")
    source_label = "sqlite" if not frame.empty else "parquet"
    if frame.empty:
        # Try to load enriched features with text embeddings first
        enriched_path = DATA_DIR / "processed" / "dataset_features_with_embeddings.parquet"
        meta_path = DATA_DIR / "processed" / "dataset_features.parquet"

        if enriched_path.exists():
            frame = pd.read_parquet(enriched_path)
            source_label = "parquet-enriched"
            LOGGER.info("Loaded enriched dataset features with text embeddings")
        elif meta_path.exists():
            frame = pd.read_parquet(meta_path)
            source_label = "parquet"
        else:
            LOGGER.warning("Dataset feature file missing: %s", meta_path)
            return {}, {}, pd.DataFrame()

    if frame.empty:
        return {}, {}, pd.DataFrame()

    return _parse_dataset_metadata_frame(frame, source=source_label)


def _load_feature_versions() -> Dict[str, str]:
    frame = _read_feature_store("SELECT view_name, refreshed_at FROM feature_metadata")
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["view_name"] = frame["view_name"].astype(str)
    frame["refreshed_at"] = frame["refreshed_at"].fillna("").astype(str)
    versions: Dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        view_name = row.get("view_name")
        refreshed_at = row.get("refreshed_at")
        if view_name:
            versions[view_name] = refreshed_at or ""
    return versions


def _load_slot_metrics() -> pd.DataFrame:
    """Load and aggregate slot metrics for ranking features.

    This mirrors the _aggregate_slot_metrics function from train_models.py
    to ensure consistency between training and inference.
    """
    slot_metrics_path = DATA_DIR / "processed" / "recommend_slot_metrics.parquet"
    if not slot_metrics_path.exists():
        LOGGER.warning("Slot metrics file missing: %s. Ranking will use zero features.", slot_metrics_path)
        return pd.DataFrame()

    try:
        metrics = pd.read_parquet(slot_metrics_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load slot metrics: %s", exc)
        return pd.DataFrame()

    if metrics.empty:
        return pd.DataFrame()

    # Ensure required columns exist
    required_cols = ["dataset_id", "position", "exposure_count", "ctr", "cvr"]
    for col in required_cols:
        if col not in metrics.columns:
            LOGGER.warning("Missing column %s in slot metrics", col)
            return pd.DataFrame()

    metrics["dataset_id"] = metrics["dataset_id"].astype(int)
    metrics["position"] = metrics["position"].astype(int)

    # Aggregate across positions (same logic as train_models._aggregate_slot_metrics)
    grouped = (
        metrics.groupby("dataset_id")
        .agg(
            slot_total_exposures=("exposure_count", "sum"),
            slot_total_clicks=("click_count", "sum") if "click_count" in metrics.columns else ("exposure_count", lambda x: 0),
            slot_total_conversions=("conversion_count", "sum") if "conversion_count" in metrics.columns else ("exposure_count", lambda x: 0),
            slot_total_revenue=("conversion_revenue", "sum") if "conversion_revenue" in metrics.columns else ("exposure_count", lambda x: 0),
            slot_mean_ctr=("ctr", "mean"),
            slot_max_ctr=("ctr", "max"),
            slot_mean_cvr=("cvr", "mean"),
            slot_position_coverage=("position", "nunique"),
        )
        .reset_index()
    )

    # Get top position stats
    top1 = metrics[metrics["position"] == 1].groupby("dataset_id").agg(
        slot_ctr_top1=("ctr", "mean"),
        slot_cvr_top1=("cvr", "mean"),
    )
    top3 = metrics[metrics["position"] <= 3].groupby("dataset_id").agg(
        slot_ctr_top3=("ctr", "mean"),
        slot_cvr_top3=("cvr", "mean"),
    )

    # Merge all stats
    merged = grouped.merge(top1, on="dataset_id", how="left").merge(top3, on="dataset_id", how="left")

    # Fill missing values
    for col in [
        "slot_ctr_top1", "slot_cvr_top1", "slot_ctr_top3", "slot_cvr_top3",
        "slot_total_exposures", "slot_total_clicks", "slot_total_conversions",
        "slot_total_revenue", "slot_mean_ctr", "slot_max_ctr", "slot_mean_cvr",
        "slot_position_coverage",
    ]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)

    LOGGER.info("Loaded slot metrics for %d datasets", len(merged))
    return merged


def _compute_feature_snapshot_id(versions: Dict[str, str]) -> Optional[str]:
    if not versions:
        return None
    parts = [f"{name}:{versions[name]}" for name in sorted(versions)]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest


def _load_user_history() -> Dict[int, List[Dict[str, float]]]:
    frame = _read_feature_store(
        "SELECT user_id, dataset_id, weight, last_event_time FROM interactions",
        parse_dates=["last_event_time"],
    )
    if frame.empty:
        interactions_path = DATA_DIR / "processed" / "interactions.parquet"
        if not interactions_path.exists():
            LOGGER.warning("Interactions file missing: %s", interactions_path)
            return {}
        frame = pd.read_parquet(interactions_path)

    if frame.empty:
        return {}

    frame["last_event_time"] = pd.to_datetime(frame["last_event_time"], errors="coerce")
    history: Dict[int, List[Dict[str, float]]] = {}
    for user_id, group in frame.groupby("user_id"):
        group = group.sort_values("last_event_time", ascending=False)
        total = group["weight"].sum()
        normalized = group["weight"] / total if total else group["weight"]
        records = []
        for row in group.assign(norm_weight=normalized).to_dict(orient="records"):
            records.append(
                {
                    "dataset_id": int(row["dataset_id"]),
                    "weight": float(row["norm_weight"]),
                    "last_event_time": row["last_event_time"],
                }
            )
        history[int(user_id)] = records
    return history


def _load_user_profile() -> Dict[int, Dict[str, Optional[str]]]:
    frame = _read_feature_store("SELECT * FROM user_profile")
    if frame.empty:
        profile_path = DATA_DIR / "processed" / "user_profile.parquet"
        if not profile_path.exists():
            LOGGER.warning("User profile file missing: %s", profile_path)
            return {}
        frame = pd.read_parquet(profile_path)

    if frame.empty:
        return {}

    profiles: Dict[int, Dict[str, Optional[str]]] = {}
    for row in frame.to_dict(orient="records"):
        user_id = int(row.get("user_id"))
        profiles[user_id] = {
            "company_name": row.get("company_name"),
            "province": row.get("province"),
            "city": row.get("city"),
            "is_consumption": row.get("is_consumption"),
        }
    return profiles


def _build_user_tag_preferences(
    user_history: Dict[int, List[Dict[str, float]]],
    dataset_tags: Dict[int, List[str]],
) -> Dict[int, Dict[str, float]]:
    preferences: Dict[int, Dict[str, float]] = {}
    for user_id, records in user_history.items():
        counter: Counter = Counter()
        for record in records:
            tags = dataset_tags.get(record["dataset_id"], [])
            if not tags:
                continue
            weight = float(record["weight"])
            for tag in tags:
                counter[tag] += weight
        total = sum(counter.values())
        if total > 0:
            preferences[user_id] = {tag: weight / total for tag, weight in counter.items()}
        else:
            preferences[user_id] = {}
    return preferences


def _sorted_items(scores: Dict[int, float]) -> List[int]:
    return [item for item, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def _jaccard_similarity(tags1: List[str], tags2: List[str]) -> float:
    """Calculate Jaccard similarity between two tag lists."""
    if not tags1 or not tags2:
        return 0.0
    set1 = set(t.lower().strip() for t in tags1 if t)
    set2 = set(t.lower().strip() for t in tags2 if t)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _apply_mmr_reranking(
    scores: Dict[int, float],
    dataset_tags: Dict[int, List[str]],
    lambda_param: float = 0.7,
    limit: int = 10,
) -> List[int]:
    """
    Apply MMR (Maximal Marginal Relevance) reranking for diversity.

    MMR Score = λ * Relevance - (1-λ) * max_similarity_to_selected

    Args:
        scores: Dataset ID to relevance score mapping
        dataset_tags: Dataset ID to tags list mapping
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
        limit: Number of items to select

    Returns:
        List of dataset IDs in MMR order
    """
    if not scores:
        return []

    # Normalize scores to [0, 1] range
    max_score = max(scores.values()) if scores else 1.0
    min_score = min(scores.values()) if scores else 0.0
    score_range = max_score - min_score if max_score > min_score else 1.0

    normalized_scores = {
        item_id: (score - min_score) / score_range
        for item_id, score in scores.items()
    }

    selected = []
    candidates = set(scores.keys())

    while len(selected) < limit and candidates:
        mmr_scores = {}

        for candidate in candidates:
            relevance = normalized_scores[candidate]

            # Calculate maximum similarity to already selected items
            if selected:
                candidate_tags = dataset_tags.get(candidate, [])
                max_sim = max(
                    _jaccard_similarity(candidate_tags, dataset_tags.get(s, []))
                    for s in selected
                )
            else:
                max_sim = 0.0

            # MMR score: balance relevance and diversity
            mmr_scores[candidate] = lambda_param * relevance - (1 - lambda_param) * max_sim

        # Select item with highest MMR score
        if mmr_scores:
            best = max(mmr_scores.items(), key=lambda x: x[1])[0]
            selected.append(best)
            candidates.remove(best)
        else:
            break

    return selected


def _apply_exploration(
    ranked_ids: List[int],
    all_dataset_ids: Set[int],
    epsilon: float = 0.1,
) -> List[int]:
    """
    Apply epsilon-greedy exploration strategy.

    Replace some highly-ranked items with random exploratory items.

    Args:
        ranked_ids: Already ranked dataset IDs
        all_dataset_ids: All available dataset IDs for exploration
        epsilon: Exploration rate (0.0 = no exploration, 1.0 = full random)

    Returns:
        List of dataset IDs with exploration applied
    """
    import random

    if epsilon <= 0 or not all_dataset_ids:
        return ranked_ids

    n_total = len(ranked_ids)
    n_explore = min(int(n_total * epsilon), n_total)
    n_exploit = n_total - n_explore

    # Keep top (1-epsilon) items for exploitation
    exploit_ids = ranked_ids[:n_exploit]

    # Randomly sample for exploration (exclude already selected items)
    explore_pool = list(all_dataset_ids - set(exploit_ids))
    if explore_pool:
        n_explore = min(n_explore, len(explore_pool))
        explore_ids = random.sample(explore_pool, n_explore)
    else:
        # If no items left to explore, just return exploit items
        explore_ids = []

    return exploit_ids + explore_ids


def _build_response_items(
    candidate_scores: Dict[int, float],
    reasons: Dict[int, str],
    limit: int,
    metadata: Dict[int, Dict[str, Optional[str]]],
    dataset_tags: Optional[Dict[int, List[str]]] = None,
    apply_mmr: bool = True,
    mmr_lambda: float = 0.7,
    apply_exploration: bool = False,
    exploration_epsilon: float = 0.1,
    all_dataset_ids: Optional[Set[int]] = None,
) -> List[RecommendationItem]:
    """
    Build response items with optional MMR reranking and exploration.

    Args:
        candidate_scores: Dataset ID to score mapping
        reasons: Dataset ID to reason mapping
        limit: Maximum number of items to return
        metadata: Dataset metadata
        dataset_tags: Dataset tags for MMR (optional)
        apply_mmr: Whether to apply MMR reranking
        mmr_lambda: MMR lambda parameter (relevance vs diversity trade-off)
        apply_exploration: Whether to apply epsilon-greedy exploration
        exploration_epsilon: Exploration rate (0.0-1.0)
        all_dataset_ids: All available dataset IDs for exploration pool

    Returns:
        List of recommendation items
    """
    if not candidate_scores:
        return []

    # Apply MMR reranking if enabled and tags available
    if apply_mmr and dataset_tags:
        ranked_ids = _apply_mmr_reranking(
            candidate_scores,
            dataset_tags,
            lambda_param=mmr_lambda,
            limit=limit,
        )
    else:
        # Fallback to score-based ranking
        ranked_ids = [
            dataset_id for dataset_id, _ in
            sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)
        ][:limit]

    # Apply exploration if enabled
    if apply_exploration and all_dataset_ids:
        ranked_ids = _apply_exploration(
            ranked_ids,
            all_dataset_ids,
            epsilon=exploration_epsilon,
        )

    result: List[RecommendationItem] = []
    for dataset_id in ranked_ids:
        info = metadata.get(dataset_id, {})
        # Use score from candidate_scores if available, otherwise use a default score
        score = candidate_scores.get(dataset_id, 0.5)
        reason = reasons.get(dataset_id, "exploration" if dataset_id not in candidate_scores else "unknown")

        result.append(
            RecommendationItem(
                dataset_id=dataset_id,
                title=info.get("title"),
                price=info.get("price"),
                cover_image=info.get("cover_image"),
                score=score,
                reason=reason,
            )
        )

    return result


def _build_fallback_items(
    dataset_ids: List[int],
    metadata: Dict[int, Dict[str, Optional[str]]],
    reason: str,
) -> List[RecommendationItem]:
    items: List[RecommendationItem] = []
    for idx, dataset_id in enumerate(dataset_ids):
        info = metadata.get(dataset_id, {})
        score = max(0.0, 0.05 - idx * 0.005)
        items.append(
            RecommendationItem(
                dataset_id=int(dataset_id),
                title=info.get("title"),
                price=info.get("price"),
                cover_image=info.get("cover_image"),
                score=score,
                reason=reason,
            )
        )
    return items


def _create_feature_store() -> Optional[RedisFeatureStore]:
    """Create Redis feature store client if configuration is provided."""
    redis_url = (
        os.getenv("FEATURE_REDIS_URL")
        or os.getenv("FEATURE_STORE_REDIS_URL")
        or os.getenv("REDIS_FEATURE_URL")
        or os.getenv("REDIS_URL")
    )
    if not redis_url:
        return None

    parsed = urllib.parse.urlparse(redis_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    password = parsed.password
    if parsed.path and parsed.path != "/":
        try:
            db = int(parsed.path.lstrip("/"))
        except ValueError:
            db = 1
    else:
        db = 1

    try:
        store = RedisFeatureStore(host=host, port=port, db=db, password=password)
        LOGGER.info(
            "Connected to Redis feature store (host=%s, port=%s, db=%s)", host, port, db
        )
        return store
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to initialize Redis feature store: %s", exc)
        return None


def _fetch_dataset_features_from_store(
    feature_store: Optional[RedisFeatureStore],
    dataset_ids: List[int],
) -> Dict[int, Dict[str, Any]]:
    if not feature_store or not dataset_ids:
        return {}
    try:
        return feature_store.get_batch_dataset_features(dataset_ids)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Feature store dataset fetch failed: %s", exc)
        return {}


def _fetch_dataset_stats_from_store(
    feature_store: Optional[RedisFeatureStore],
    dataset_ids: List[int],
) -> Dict[int, Dict[str, Any]]:
    if not feature_store or not dataset_ids:
        return {}
    try:
        return feature_store.get_dataset_stats(dataset_ids)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Feature store stats fetch failed: %s", exc)
        return {}


async def _call_blocking(
    func: Callable,
    *args,
    endpoint: str,
    operation: str,
    timeout: float,
    **kwargs,
):
    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            _run_in_executor(func, *args, **kwargs),
            timeout=timeout,
        )
        duration = time.perf_counter() - start
        if duration >= SLOW_OPERATION_THRESHOLD:
            LOGGER.info(
                "Slow blocking call (endpoint=%s, operation=%s, duration=%.3fs)",
                endpoint,
                operation,
                duration,
            )
        return result
    except asyncio.TimeoutError as exc:
        LOGGER.warning(
            "Operation timed out (endpoint=%s, operation=%s)", endpoint, operation
        )
        recommendation_timeouts_total.labels(endpoint=endpoint, operation=operation).inc()
        raise exc


def _serve_fallback(
    state,
    *,
    dataset_id: int,
    limit: int,
    endpoint: str,
    degrade_cause: str,
    user_id: Optional[int] = None,
) -> tuple[List[RecommendationItem], Dict[int, str], str]:
    fallback = getattr(state, "fallback_strategy", None)
    fallback_reason = f"{degrade_cause}:no_fallback"
    if not fallback:
        recommendation_degraded_total.labels(
            endpoint=endpoint, reason=fallback_reason
        ).inc()
        return [], {}, fallback_reason

    result = fallback.get_with_metadata(dataset_id=dataset_id, limit=limit, user_id=user_id)
    reason_label = f"{degrade_cause}:{result.source}"
    metrics_tracker = get_metrics_tracker()
    metrics_tracker.track_fallback(reason_label, result.level)

    if not result.items:
        empty_reason = f"{reason_label}:empty"
        recommendation_degraded_total.labels(endpoint=endpoint, reason=empty_reason).inc()
        return [], {}, empty_reason

    recommendation_degraded_total.labels(endpoint=endpoint, reason=reason_label).inc()
    items = _build_fallback_items(result.items, state.metadata, f"fallback:{result.source}")
    reasons = {item.dataset_id: f"fallback:{result.source}" for item in items}
    return items, reasons, reason_label


def _augment_with_multi_channel(
    state,
    *,
    target_id: int,
    scores: Dict[int, float],
    reasons: Dict[int, str],
    limit: int,
    user_id: Optional[int] = None,
) -> None:
    recall = getattr(state, "recall_indices", None)
    if not recall:
        return

    def _bump(dataset_id: int, score: float, label: str) -> None:
        if dataset_id == target_id or score <= 0:
            return
        if dataset_id not in scores:
            scores[dataset_id] = score
            reasons[dataset_id] = label
        else:
            scores[dataset_id] += score
            if label not in reasons[dataset_id]:
                reasons[dataset_id] = f"{reasons[dataset_id]}+{label}"

    # Tag-based recall
    item_to_tags = recall.get("item_to_tags", {})
    tag_to_items = recall.get("tag_to_items", {})
    target_tags = item_to_tags.get(str(target_id)) or item_to_tags.get(target_id)
    if not target_tags:
        target_tags = state.dataset_tags.get(target_id, [])
    if target_tags and tag_to_items:
        candidate_scores: Dict[int, float] = {}
        target_set = set(tag.lower() for tag in target_tags if tag)
        for tag in target_set:
            for candidate in tag_to_items.get(tag, set()):
                if candidate == target_id:
                    continue
                overlap = len(target_set & set(state.dataset_tags.get(candidate, [])))
                if overlap:
                    candidate_scores[int(candidate)] = candidate_scores.get(int(candidate), 0.0) + overlap

        # 归一化tag分数到[0, 1]，然后乘以权重
        if candidate_scores:
            normalized_tag_scores = _normalize_channel_scores(candidate_scores)
            for dataset_id, norm_score in sorted(normalized_tag_scores.items(), key=lambda x: x[1], reverse=True)[: limit * 2]:
                _bump(int(dataset_id), norm_score * 0.4, "tag")

    # Category recall
    category_index = recall.get("category_index", {})
    company = state.metadata.get(target_id, {}).get("company")
    if company:
        candidates = category_index.get(str(company).lower(), set())
        for dataset_id in list(candidates)[: limit * 2]:
            if dataset_id == target_id:
                continue
            _bump(int(dataset_id), 0.3, "category")

    # Price bucket recall
    price_bucket_index = recall.get("price_bucket_index", {})
    raw_price = state.metadata.get(target_id, {}).get("price", 0.0)
    try:
        price = float(raw_price)
    except (TypeError, ValueError):
        price = 0.0
    if price_bucket_index:
        if price < 100:
            bucket = "0"
        elif price < 500:
            bucket = "1"
        elif price < 1000:
            bucket = "2"
        elif price < 5000:
            bucket = "3"
        else:
            bucket = "4"
        candidates = price_bucket_index.get(bucket) or price_bucket_index.get(int(bucket)) or set()
        for dataset_id in list(candidates)[: limit * 2]:
            _bump(int(dataset_id), 0.25, "price")

    # UserCF recall
    if user_id:
        user_similarity = recall.get("user_similarity", {})
        history_sets = _get_user_history_sets(state)
        if user_similarity and user_id in user_similarity:
            target_history = history_sets.get(int(user_id), set())
            similar_users = user_similarity.get(int(user_id), [])
            candidate_scores: Dict[int, float] = {}
            for other_id, similarity in similar_users:
                candidate_set = history_sets.get(int(other_id), set())
                for dataset in candidate_set:
                    if dataset in target_history or dataset == target_id:
                        continue
                    candidate_scores[dataset] = candidate_scores.get(dataset, 0.0) + float(similarity)
            for dataset_id, score in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[: limit * 2]:
                _bump(int(dataset_id), float(score) * 0.6, "usercf")


def _combine_scores(
    target_id: int,
    behavior: Dict[int, Dict[int, float]],
    content: Dict[int, Dict[int, float]],
    vector: Dict[int, List[Dict[str, float]]],
    popular: List[int],
    limit: int,
) -> tuple[Dict[int, float], Dict[int, str]]:
    return _combine_scores_with_weights(
        target_id,
        behavior,
        content,
        vector,
        popular,
        limit,
        DEFAULT_CHANNEL_WEIGHTS,
    )


def _normalize_channel_scores(channel_scores: Dict[int, float]) -> Dict[int, float]:
    """Normalize channel scores to [0, 1] range using Min-Max scaling.

    This ensures all recall channels compete on the same scale, preventing
    channels with inherently larger scores (e.g., vector: 15-22) from
    dominating channels with smaller scores (e.g., content: 0.2-0.9).

    Args:
        channel_scores: Dict mapping item_id to raw channel score

    Returns:
        Dict mapping item_id to normalized score in [0, 1]
    """
    if not channel_scores:
        return {}

    max_val = max(channel_scores.values())
    min_val = min(channel_scores.values())
    range_val = max_val - min_val if max_val > min_val else 1.0

    return {
        item_id: (score - min_val) / range_val
        for item_id, score in channel_scores.items()
    }


def _combine_scores_with_weights(
    target_id: int,
    behavior: Dict[int, Dict[int, float]],
    content: Dict[int, Dict[int, float]],
    vector: Dict[int, List[Dict[str, float]]],
    popular: List[int],
    limit: int,
    weights: Dict[str, float],
) -> tuple[Dict[int, float], Dict[int, str]]:
    """Combine scores from multiple recall channels with normalization.

    Each channel is independently normalized to [0, 1] before applying weights,
    ensuring fair competition across channels with different score magnitudes.

    Args:
        target_id: Target dataset ID for recall
        behavior: Behavior-based recall results
        content: Content-based recall results
        vector: Vector-based recall results
        popular: Popular items fallback
        limit: Result limit
        weights: Channel weights

    Returns:
        Tuple of (scores dict, reasons dict)
    """
    scores: Dict[int, float] = {}
    reasons: Dict[int, str] = {}

    # ========== Behavior召回（归一化） ==========
    behavior_scores = behavior.get(target_id, {})
    if behavior_scores:
        normalized_behavior = _normalize_channel_scores(behavior_scores)
        for item_id, norm_score in normalized_behavior.items():
            scores[int(item_id)] = norm_score * weights.get("behavior", 1.0)
            reasons[int(item_id)] = "behavior"

    # ========== Content召回（归一化） ==========
    content_scores = content.get(target_id, {})
    if content_scores:
        normalized_content = _normalize_channel_scores(content_scores)
        for item_id, norm_score in normalized_content.items():
            item_id = int(item_id)
            if item_id not in scores:
                scores[item_id] = norm_score * weights.get("content", 0.5)
                reasons[item_id] = "content"
            else:
                # 如果已有分数（来自其他渠道），累加
                scores[item_id] += norm_score * weights.get("content", 0.5)
                reasons[item_id] = f"{reasons[item_id]}+content"

    # ========== Vector召回（归一化） ==========
    vector_scores_dict = {}
    for entry in vector.get(target_id, []):
        item_id = int(entry.get("dataset_id", 0))
        if item_id == target_id:
            continue
        score = float(entry.get("score", 0.0))
        if score > 0:
            vector_scores_dict[item_id] = score
        if len(vector_scores_dict) >= limit * 4:
            break

    if vector_scores_dict:
        normalized_vector = _normalize_channel_scores(vector_scores_dict)
        for item_id, norm_score in normalized_vector.items():
            if item_id not in scores:
                scores[item_id] = norm_score * weights.get("vector", 0.4)
                reasons[item_id] = "vector"
            else:
                scores[item_id] += norm_score * weights.get("vector", 0.4)
                reasons[item_id] = f"{reasons[item_id]}+vector"

    # ========== Popular召回（归一化） ==========
    # popular是列表，按排序给分，线性衰减
    popular_scores = {}
    for idx, item_id in enumerate(popular):
        if item_id == target_id or item_id in scores:
            continue
        # 线性衰减：第1个=1.0, 最后一个=0.1
        popular_scores[item_id] = 1.0 - (idx / max(len(popular), 1)) * 0.9
        if len(popular_scores) >= limit * 5:
            break

    for item_id, norm_score in popular_scores.items():
        scores[item_id] = norm_score * weights.get("popular", 0.01)
        reasons[item_id] = "popular"

    scores.pop(target_id, None)
    return scores, reasons


def _compute_dynamic_channel_weights(
    base_weights: Dict[str, float],
    *,
    dataset_id: int,
    user_id: Optional[int],
    bundle: ModelBundle,
    state,
) -> Dict[str, float]:
    """Adjust channel weights based on available signals for this request."""
    adjusted = {key: max(float(value), 0.0) for key, value in base_weights.items()}

    def _shift(source: str, targets: List[str], fraction: float) -> None:
        current = adjusted.get(source, 0.0)
        if current <= 0 or not targets or fraction <= 0:
            return
        amount = current * min(fraction, 1.0)
        adjusted[source] = max(current - amount, 0.0)
        share = amount / len(targets)
        for target in targets:
            adjusted[target] = max(adjusted.get(target, 0.0) + share, 0.0)

    def _boost(target: str, amount: float) -> None:
        if amount <= 0:
            return
        adjusted[target] = max(adjusted.get(target, 0.0) + amount, 0.0)

    user_history = _get_user_history_records(state, user_id) if user_id else None
    if not user_history:
        # No personalization history → rely more on content/vector/popular
        _shift("behavior", ["content", "vector", "popular"], 0.5)

    behavior_neighbors = bundle.behavior.get(dataset_id) or {}
    if len(behavior_neighbors) < 3:
        _shift("behavior", ["content", "vector"], 0.3)

    content_neighbors = bundle.content.get(dataset_id) or {}
    if len(content_neighbors) < 3:
        _shift("content", ["behavior", "vector"], 0.3)

    vector_entries = bundle.vector.get(dataset_id) or []
    if not vector_entries:
        _shift("vector", ["behavior", "content"], 0.5)
    elif len(vector_entries) < 5:
        _boost("vector", 0.1)

    if not state.dataset_tags.get(dataset_id):
        _shift("content", ["behavior", "vector"], 0.2)

    return adjusted


def _normalize_event_time(value: Any) -> Optional[datetime]:
    """Convert different timestamp representations to timezone-aware UTC datetimes."""
    if value is None:
        return None
    parsed: Optional[datetime]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.fromtimestamp(float(text), tz=timezone.utc)
            except (ValueError, TypeError):
                return None
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_channel_from_reason(reason: Optional[str]) -> str:
    if not reason:
        return "unknown"
    text = str(reason).strip().lower()
    if not text:
        return "unknown"
    parts = [token for token in text.split("+") if token]
    if not parts:
        return "unknown"
    channel = parts[0]
    if channel.startswith("fallback"):
        return "fallback"
    return channel


def _apply_personalization(
    user_id: Optional[int],
    scores: Dict[int, float],
    reasons: Dict[int, str],
    state,
    behavior: Dict[int, Dict[int, float]],
    history_limit: int = 50,  # Increased to capture more history
    decay_half_life_days: float = 7.0,  # Interest decays by half every 7 days
) -> None:
    """
    Apply personalization with time-based interest decay.

    Args:
        user_id: User ID
        scores: Current recommendation scores
        reasons: Recommendation reasons
        state: Application state
        behavior: Behavior similarity matrix
        history_limit: Number of historical interactions to consider
        decay_half_life_days: Days for interest to decay by half
    """
    if not user_id:
        return
    user_history = _get_user_history_records(state, int(user_id))
    if not user_history:
        return

    now = datetime.now(timezone.utc)

    recent_history = user_history[:history_limit]
    history_ids = {record["dataset_id"] for record in recent_history}

    # Remove already interacted items
    for dataset_id in history_ids:
        scores.pop(dataset_id, None)
        reasons.pop(dataset_id, None)

    tag_pref = _get_user_tag_preferences(state, int(user_id))

    # Apply time-decayed personalization boost
    for dataset_id in list(scores.keys()):
        boost = 0.0
        for record in recent_history:
            source_id = record["dataset_id"]

            # Calculate time decay factor
            timestamp = record.get("last_event_time")
            event_time = _normalize_event_time(timestamp)
            if event_time:
                days_ago = (now - event_time).total_seconds() / 86400  # seconds to days
                decay_factor = 0.5 ** (days_ago / decay_half_life_days)
            else:
                decay_factor = 1.0  # No decay if timestamp not available
            weight = record.get("weight", 1.0)
            sim = behavior.get(source_id, {}).get(dataset_id, 0.0)
            if sim:
                # Apply time decay to the similarity boost
                boost += sim * weight * decay_factor * 0.5

        # Tag preference boost (with overall decay based on oldest interaction)
        candidate_tags = state.dataset_tags.get(dataset_id, [])
        if candidate_tags and tag_pref:
            tag_boost = sum(tag_pref.get(tag, 0.0) for tag in candidate_tags)
            # Use average decay factor from recent history
            if recent_history:
                decay_samples = []
                for rec in recent_history[:5]:
                    event_time = _normalize_event_time(rec.get("last_event_time"))
                    if event_time:
                        delta_days = (now - event_time).total_seconds() / 86400
                        decay_samples.append(0.5 ** (delta_days / decay_half_life_days))
                    else:
                        decay_samples.append(1.0)
                avg_decay = sum(decay_samples) / len(decay_samples)
            else:
                avg_decay = 1.0
            boost += tag_boost * avg_decay * 0.2

        if boost > 0:
            scores[dataset_id] += boost
            base_reason = reasons.get(dataset_id, "unknown")
            if "personalized" not in base_reason:
                reasons[dataset_id] = f"{base_reason}+personalized"


def _ensure_dataset_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame is indexed by dataset_id without mutating original."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    if frame.index.name == "dataset_id":
        return frame
    if "dataset_id" in frame.columns:
        return frame.set_index("dataset_id")
    return frame


def _build_static_ranking_features(
    dataset_ids: List[int],
    raw_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    slot_metrics_aggregated: pd.DataFrame,
    feature_overrides: Optional[pd.DataFrame] = None,
    stats_overrides: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute dataset-level static ranking features."""
    if not dataset_ids:
        return pd.DataFrame()

    raw_indexed = _ensure_dataset_index(raw_features)
    selected = raw_indexed.reindex(dataset_ids)
    if selected.empty:
        selected = pd.DataFrame(index=dataset_ids)
    selected = selected.copy()
    selected["price"] = pd.to_numeric(selected.get("price"), errors="coerce").fillna(0.0)
    selected["description"] = selected.get("description", "").fillna("").astype(str)
    selected["tag"] = selected.get("tag", "").fillna("").astype(str)

    if feature_overrides is not None and not feature_overrides.empty:
        overrides = _ensure_dataset_index(feature_overrides).reindex(dataset_ids)
        selected = overrides.combine_first(selected)

    selected["description_length"] = selected.get("description", "").str.len().fillna(0.0).astype(float)
    selected["tag_count"] = selected.get("tag", "").apply(
        lambda text: float(len([t for t in str(text).split(';') if t.strip()])) if isinstance(text, str) else 0.0
    )

    stats_indexed = _ensure_dataset_index(dataset_stats)
    stats = stats_indexed.reindex(dataset_ids)
    if stats.empty:
        stats = pd.DataFrame(index=dataset_ids)
    stats = stats.copy()
    stats["interaction_count"] = pd.to_numeric(stats.get("interaction_count"), errors="coerce").fillna(0.0)
    stats["total_weight"] = pd.to_numeric(stats.get("total_weight"), errors="coerce").fillna(0.0)

    if stats_overrides is not None and not stats_overrides.empty:
        overrides = _ensure_dataset_index(stats_overrides).reindex(dataset_ids)
        overrides = overrides.apply(pd.to_numeric, errors="coerce")
        stats = overrides.combine_first(stats).fillna(0.0)

    features = pd.DataFrame(index=dataset_ids)
    features["price_log"] = np.log1p(selected["price"].clip(lower=0.0))
    features["description_length"] = selected["description_length"].fillna(0.0)
    features["tag_count"] = selected["tag_count"].fillna(0.0)
    features["weight_log"] = np.log1p(stats["total_weight"].clip(lower=0.0))
    features["interaction_count"] = stats["interaction_count"].fillna(0.0)

    if not stats.empty and "interaction_count" in stats.columns:
        features["popularity_rank"] = stats["interaction_count"].rank(ascending=False, method="dense").fillna(0.0)
        features["popularity_percentile"] = stats["interaction_count"].rank(pct=True).fillna(0.5)
    else:
        features["popularity_rank"] = 0.0
        features["popularity_percentile"] = 0.5

    features["price_bucket"] = pd.cut(
        selected["price"],
        bins=[-np.inf, 0.5, 1.0, 2.0, 5.0, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(0.0)

    features["days_since_last_interaction"] = 30.0
    features["interaction_density"] = features["interaction_count"] / 30.0
    features["has_description"] = (features["description_length"] > 0).astype(float)
    features["has_tags"] = (features["tag_count"] > 0).astype(float)
    features["content_richness"] = features["description_length"] * features["tag_count"]

    optional_columns = ["image_richness_score", "image_embed_norm", "has_images", "has_cover"]
    for col in optional_columns:
        if col in selected.columns:
            features[col] = pd.to_numeric(selected[col], errors="coerce").fillna(0.0)
        else:
            features[col] = 0.0

    slot_indexed = _ensure_dataset_index(slot_metrics_aggregated)
    slot_columns = [
        "slot_total_exposures",
        "slot_total_clicks",
        "slot_total_conversions",
        "slot_total_revenue",
        "slot_mean_ctr",
        "slot_max_ctr",
        "slot_mean_cvr",
        "slot_position_coverage",
        "slot_ctr_top1",
        "slot_ctr_top3",
        "slot_cvr_top1",
        "slot_cvr_top3",
    ]
    if not slot_indexed.empty:
        slot_data = slot_indexed.reindex(dataset_ids)
        for col in slot_columns:
            if col in slot_data.columns:
                features[col] = pd.to_numeric(slot_data[col], errors="coerce").fillna(0.0)
            else:
                features[col] = 0.0
    else:
        for col in slot_columns:
            features[col] = 0.0

    text_embedding_columns = ["text_embed_norm", "text_embed_mean", "text_embed_std"]
    for col in text_embedding_columns:
        if col in selected.columns:
            features[col] = pd.to_numeric(selected[col], errors="coerce").fillna(0.0)
        else:
            features[col] = 0.0

    pca_columns = [col for col in selected.columns if col.startswith("text_pca_")]
    for col in pca_columns:
        features[col] = pd.to_numeric(selected[col], errors="coerce").fillna(0.0)

    return features


def _compute_ranking_features(
    dataset_ids: List[int],
    raw_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    slot_metrics_aggregated: pd.DataFrame,
    feature_store: Optional[RedisFeatureStore] = None,
    *,
    scores: Optional[Dict[int, float]] = None,
    reasons: Optional[Dict[int, str]] = None,
    channel_weights: Optional[Dict[str, float]] = None,
    endpoint: str = "recommend_detail",
    variant: str = "primary",
    experiment_variant: Optional[str] = None,
    request_context: Optional[Dict[str, str]] = None,
    user_features: Optional[Dict[str, float]] = None,
    precomputed_static: Optional[pd.DataFrame] = None,
    raw_features_indexed: Optional[pd.DataFrame] = None,
    dataset_stats_indexed: Optional[pd.DataFrame] = None,
    slot_metrics_indexed: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if not dataset_ids:
        return pd.DataFrame()

    indexed_raw = raw_features_indexed if raw_features_indexed is not None else raw_features
    indexed_stats = dataset_stats_indexed if dataset_stats_indexed is not None else dataset_stats
    indexed_slot = slot_metrics_indexed if slot_metrics_indexed is not None else slot_metrics_aggregated

    if precomputed_static is not None and not precomputed_static.empty:
        features = precomputed_static.reindex(dataset_ids).copy()
        missing_mask = features.isnull().all(axis=1)
        missing_ids = features.index[missing_mask].tolist()
        if missing_ids:
            filled = _build_static_ranking_features(
                missing_ids,
                indexed_raw,
                indexed_stats,
                indexed_slot,
            )
            features.loc[missing_ids] = filled
    else:
        features = _build_static_ranking_features(
            dataset_ids,
            indexed_raw,
            indexed_stats,
            indexed_slot,
        )

    override_feature_df = None
    override_stats_df = None
    override_ids: Set[int] = set()
    realtime_features = _fetch_dataset_features_from_store(feature_store, dataset_ids)
    if realtime_features:
        override_feature_df = (
            pd.DataFrame.from_dict(realtime_features, orient="index")
            .rename_axis("dataset_id")
        )
        override_feature_df.index = override_feature_df.index.astype(int)
        override_ids.update(int(idx) for idx in override_feature_df.index)
    realtime_stats = _fetch_dataset_stats_from_store(feature_store, dataset_ids)
    if realtime_stats:
        override_stats_df = (
            pd.DataFrame.from_dict(realtime_stats, orient="index")
            .rename_axis("dataset_id")
        )
        override_stats_df.index = override_stats_df.index.astype(int)
        override_ids.update(int(idx) for idx in override_stats_df.index)
    if override_ids:
        refreshed = _build_static_ranking_features(
            sorted(override_ids),
            indexed_raw,
            indexed_stats,
            indexed_slot,
            feature_overrides=override_feature_df,
            stats_overrides=override_stats_df,
        )
        if features.empty:
            features = refreshed
        else:
            features.loc[refreshed.index] = refreshed

    # Request-level dynamic features
    score_lookup = scores or {}
    reason_lookup = reasons or {}
    context = request_context or {}
    channel_weights = channel_weights or {}

    if dataset_ids:
        sorted_scores = sorted(score_lookup.items(), key=lambda kv: kv[1], reverse=True)
        position_lookup = {dataset_id: idx for idx, (dataset_id, _) in enumerate(sorted_scores)}
    else:
        position_lookup = {}

    features["score"] = features.index.to_series().map(lambda dataset_id: float(score_lookup.get(dataset_id, 0.0))).values
    features["position"] = features.index.to_series().map(lambda dataset_id: position_lookup.get(dataset_id, -1)).fillna(-1).astype(int)

    def _reason_channel(dataset_id: int) -> str:
        return _extract_channel_from_reason(reason_lookup.get(dataset_id))

    features["channel"] = features.index.to_series().map(_reason_channel).fillna("unknown").astype(str)

    def _channel_weight(channel: str) -> float:
        if channel in channel_weights:
            try:
                return float(channel_weights[channel])
            except (TypeError, ValueError):
                return DEFAULT_CHANNEL_WEIGHTS.get(channel, 0.1)
        return DEFAULT_CHANNEL_WEIGHTS.get(channel, 0.1)

    features["channel_weight"] = features["channel"].map(_channel_weight).astype(float)

    features["endpoint"] = endpoint or context.get("endpoint", "recommend_detail")
    features["variant"] = variant or context.get("variant", "primary")
    features["experiment_variant"] = (experiment_variant or context.get("experiment_variant") or "control")
    features["source"] = context.get("source", "unknown")
    features["device_type"] = context.get("device_type", "unknown")
    features["locale"] = context.get("locale", "unknown")

    user_features = user_features or {}
    for key, value in user_features.items():
        try:
            features[key] = float(value)
        except (TypeError, ValueError):
            features[key] = 0.0

    return features


def _prepare_ranker_features(rank_model, features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(rank_model, dict):
        return features
    required_columns = rank_model.get("feature_columns") or list(features.columns)
    feature_types = rank_model.get("feature_types", {})
    category_mappings = rank_model.get("category_mappings", {})

    working = features.copy()
    for column, kind in feature_types.items():
        if column not in working.columns:
            if kind == "categorical":
                working[column] = "unknown"
            else:
                working[column] = 0.0
        if kind == "numeric":
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
        else:
            working[column] = working[column].fillna("unknown").astype(str)

    for column, categories in category_mappings.items():
        if column in working.columns:
            working[column] = pd.Categorical(working[column].astype(str), categories=categories)

    return working[required_columns] if required_columns else working


def _align_features_to_estimator(rank_model, features: pd.DataFrame) -> pd.DataFrame:
    """Align features to estimators that expect specific training columns."""
    feature_names = getattr(rank_model, "feature_names_in_", None)
    if feature_names is None:
        return features
    required = list(feature_names)
    working = features.copy()
    for column in required:
        if column not in working.columns:
            working[column] = 0.0
    return working[required]


def _predict_rank_scores(rank_model, features: pd.DataFrame) -> pd.Series:
    if rank_model is None or features.empty:
        return pd.Series(dtype=float)
    try:
        if isinstance(rank_model, dict) and rank_model.get("type") == "lightgbm_ranker":
            prepared = _prepare_ranker_features(rank_model, features)
            scores = rank_model["model"].predict(prepared)
            return pd.Series(scores, index=features.index, dtype=float)
        aligned = _align_features_to_estimator(rank_model, features)
        if hasattr(rank_model, "predict_proba"):
            scores = rank_model.predict_proba(aligned)[:, 1]
            return pd.Series(scores, index=features.index, dtype=float)
        scores = rank_model.predict(aligned)
        return pd.Series(scores, index=features.index, dtype=float)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Ranking model prediction failed: %s", exc)
        return pd.Series(dtype=float)


@with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
def _apply_ranking_with_circuit_breaker(
    scores: Dict[int, float],
    reasons: Dict[int, str],
    rank_model,
    features: pd.DataFrame,
) -> None:
    """Apply ranking with circuit breaker protection."""
    probabilities = _predict_rank_scores(rank_model, features)
    if probabilities.empty:
        return
    for dataset_id, prob in zip(features.index.astype(int), probabilities.values):
        if dataset_id not in scores:
            continue
        prob = float(prob)
        scores[dataset_id] += prob
        base_reason = reasons.get(dataset_id, "unknown")
        if "rank" not in base_reason:
            reasons[dataset_id] = f"{base_reason}+rank"


def _apply_ranking(
    scores: Dict[int, float],
    reasons: Dict[int, str],
    rank_model,
    raw_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    slot_metrics_aggregated: pd.DataFrame,
    feature_store: Optional[RedisFeatureStore] = None,
    *,
    endpoint: str,
    variant: str,
    experiment_variant: Optional[str],
    request_context: Optional[Dict[str, str]],
    channel_weights: Dict[str, float],
    user_features: Optional[Dict[str, float]],
) -> None:
    if rank_model is None or not scores:
        return

    try:
        current_state = _get_app_state()
    except RuntimeError:
        current_state = None

    dataset_ids = list(scores.keys())
    features = _compute_ranking_features(
        dataset_ids,
        raw_features,
        dataset_stats,
        slot_metrics_aggregated,
        feature_store=feature_store,
        scores=scores,
        reasons=reasons,
        channel_weights=channel_weights,
        endpoint=endpoint,
        variant=variant,
        experiment_variant=experiment_variant,
        request_context=request_context,
        user_features=user_features,
        precomputed_static=getattr(current_state, "ranking_static_features", None),
        raw_features_indexed=getattr(current_state, "raw_features_indexed", None),
        dataset_stats_indexed=getattr(current_state, "dataset_stats_indexed", None),
        slot_metrics_indexed=getattr(current_state, "slot_metrics_indexed", None),
    )
    if features.empty:
        return

    try:
        _apply_ranking_with_circuit_breaker(scores, reasons, rank_model, features)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Ranking model failed (circuit breaker or error): %s", exc)
        # Continue without ranking - scores already populated from recall


def _log_exposure(
    event: str,
    *,
    user_id: Optional[int],
    page_id: int,
    items: List[RecommendationItem],
    algorithm_version: Optional[str],
    variant: str,
    reasons: Dict[int, str],
    request_id: str,
    degrade_reason: Optional[str],
    experiment_variant: Optional[str] = None,
    request_context: Optional[Dict[str, str]] = None,
) -> None:
    exposure_items = [
        {
            "dataset_id": item.dataset_id,
            "score": item.score,
            "reason": reasons.get(item.dataset_id, item.reason),
        }
        for item in items
    ]
    context = {"endpoint": event, "variant": variant}
    if request_context:
        for key, value in request_context.items():
            if value not in (None, ""):
                context[key] = value
    if degrade_reason:
        context["degrade_reason"] = degrade_reason
    if experiment_variant:
        context["experiment_variant"] = experiment_variant
    try:
        state = _get_app_state()
    except RuntimeError:
        state = None
    if state:
        model_run_id = getattr(state, "model_run_id", None)
        if model_run_id and "model_run_id" not in context:
            context["model_run_id"] = model_run_id
        feature_snapshot_id = getattr(state, "feature_snapshot_id", None)
        if feature_snapshot_id:
            context["feature_snapshot_id"] = feature_snapshot_id
        feature_versions = getattr(state, "feature_versions", None)
        if feature_versions:
            context["feature_versions"] = feature_versions

    endpoint_label = context.get("endpoint", event)
    variant_label = context.get("variant", "primary") or "primary"
    experiment_label = context.get("experiment_variant", "control") or "control"
    degrade_label = context.get("degrade_reason", "none") or "none"
    exposure_count = len(exposure_items)

    recommendation_exposures_total.labels(
        endpoint=endpoint_label,
        variant=variant_label,
        experiment_variant=experiment_label,
        degrade_reason=degrade_label,
    ).inc(exposure_count)

    metrics_tracker = get_metrics_tracker()
    metrics_tracker.track_exposure(endpoint_label, degrade_label, exposure_count)

    record_exposure(
        request_id=request_id,
        user_id=user_id,
        page_id=page_id,
        algorithm_version=algorithm_version,
        items=exposure_items,
        context=context,
    )


def _get_app_state():
    if not hasattr(app.state, "models_loaded"):
        raise RuntimeError("Models are not loaded yet.")
    return app.state


@app.on_event("startup")
def load_models() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Initialize Sentry
    sentry_enabled = init_sentry(
        service_name="recommendation-api",
        enable_tracing=True,
        traces_sample_rate=0.1,  # 10% 采样率
        profiles_sample_rate=0.1,
    )
    if sentry_enabled:
        LOGGER.info("Sentry monitoring enabled for recommendation-api")
    else:
        LOGGER.warning("Sentry monitoring disabled (SENTRY_DSN not configured)")

    # Initialize cache
    cache = get_cache()
    if cache and cache.enabled:
        LOGGER.info("Redis cache initialized successfully")
        app.state.cache = cache
        app.state.hot_tracker = get_hot_tracker()
    else:
        LOGGER.warning("Redis cache not available, running without cache")
        app.state.cache = None
        app.state.hot_tracker = None

    bundle = _load_model_bundle(MODELS_DIR, run_id=_load_model_run_id())
    dataset_ids = _collect_dataset_ids(bundle)
    feature_store = _create_feature_store()
    app.state.feature_store = feature_store
    experiment_config_path = _resolve_experiment_config_path()
    app.state.experiments = load_experiments(experiment_config_path)
    existing_observer = getattr(app.state, "experiment_observer", None)
    if existing_observer:
        existing_observer.stop()
        existing_observer.join(timeout=2)
    app.state.experiment_observer = _start_experiment_watcher(experiment_config_path)
    app.state.experiment_config_path = experiment_config_path

    metadata, dataset_tags, raw_features = _load_dataset_metadata(
        feature_store=feature_store,
        dataset_ids=dataset_ids,
    )
    dataset_stats = _load_dataset_stats(
        feature_store=feature_store,
        dataset_ids=dataset_ids,
    )
    slot_metrics_aggregated = _load_slot_metrics()
    if not raw_features.empty and "dataset_id" in raw_features.columns:
        raw_features_indexed = raw_features.set_index("dataset_id")
    else:
        raw_features_indexed = pd.DataFrame()
    if not dataset_stats.empty and "dataset_id" in dataset_stats.columns:
        dataset_stats_indexed = dataset_stats.set_index("dataset_id")
    else:
        dataset_stats_indexed = pd.DataFrame()
    if not slot_metrics_aggregated.empty and "dataset_id" in slot_metrics_aggregated.columns:
        slot_metrics_indexed = slot_metrics_aggregated.set_index("dataset_id")
    else:
        slot_metrics_indexed = pd.DataFrame()
    ranking_static_features = _build_static_ranking_features(
        sorted(dataset_ids),
        raw_features_indexed,
        dataset_stats_indexed,
        slot_metrics_indexed,
    )
    feature_versions = _load_feature_versions()
    feature_snapshot_id = _compute_feature_snapshot_id(feature_versions)
    user_history = _load_user_history()
    user_profiles = _load_user_profile()
    user_tag_preferences = _build_user_tag_preferences(user_history, dataset_tags)
    channel_weight_overrides = _load_channel_weight_overrides()
    hot_user_ttl = int(os.getenv("HOT_USER_DATA_TTL_SECONDS", "300"))
    user_data_manager = HotUserData(dataset_tags, ttl_seconds=hot_user_ttl)
    user_data_manager.bootstrap(user_history, user_profiles)

    _set_bundle(app.state, bundle, prefix="")
    app.state.shadow_bundle = None
    app.state.shadow_rollout = 0.0
    app.state.metadata = metadata
    app.state.raw_features = raw_features
    app.state.raw_features_indexed = raw_features_indexed
    app.state.dataset_tags = dataset_tags
    app.state.dataset_stats = dataset_stats
    app.state.dataset_stats_indexed = dataset_stats_indexed
    app.state.slot_metrics_aggregated = slot_metrics_aggregated
    app.state.slot_metrics_indexed = slot_metrics_indexed
    app.state.ranking_static_features = ranking_static_features
    app.state.channel_weights = channel_weight_overrides or {}
    app.state.user_data = user_data_manager
    app.state.user_history = user_history  # Backward compatibility
    app.state.user_profiles = user_profiles
    app.state.user_tag_preferences = user_tag_preferences
    app.state.personalization_history_limit = 20
    app.state.feature_versions = feature_versions
    app.state.feature_snapshot_id = feature_snapshot_id
    app.state.models_loaded = True

    app.state.recall_indices = _load_recall_artifacts(MODELS_DIR)
    app.state.user_history_sets = user_data_manager.get_history_sets()

    # Initialize fallback strategy
    precomputed_dir = MODELS_DIR / "precomputed"
    app.state.fallback_strategy = FallbackStrategy(
        cache=cache,
        precomputed_dir=precomputed_dir if precomputed_dir.exists() else None,
        static_popular=bundle.popular,
    )

    # Initialize health checker
    app.state.health_checker = HealthChecker()

    LOGGER.info(
        "Model artifacts loaded (variant=primary, behavior=%d, content=%d, vector=%d, users=%d, run=%s)",
        len(bundle.behavior),
        len(bundle.content),
        len(bundle.vector),
        len(user_history),
        bundle.run_id or "unknown",
    )


@app.on_event("shutdown")
def shutdown_event() -> None:
    observer = getattr(app.state, "experiment_observer", None)
    if observer:
        observer.stop()
        observer.join(timeout=2)
        app.state.experiment_observer = None


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint with detailed status."""
    try:
        state = _get_app_state()
        health_checker = getattr(state, "health_checker", None)

        if health_checker:
            # Perform health checks
            cache = getattr(state, "cache", None)
            health_checker.check_redis(cache)
            health_checker.check_models(state)
            status = health_checker.get_status()

            return {
                "status": "healthy" if status["healthy"] else "degraded",
                "cache": "enabled" if cache and cache.enabled else "disabled",
                "models_loaded": getattr(state, "models_loaded", False),
                "checks": status["checks"],
            }

        # Fallback if health checker not initialized
        cache_status = "enabled" if getattr(state, "cache", None) and state.cache.enabled else "disabled"
        return {
            "status": "ok",
            "cache": cache_status,
            "models_loaded": getattr(state, "models_loaded", False),
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Health check failed: %s", exc)
        return {
            "status": "unhealthy",
            "error": str(exc),
        }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/test-sentry")
def test_sentry(error_type: str = "exception") -> Dict[str, Any]:
    """
    测试 Sentry 错误捕获的端点（仅用于测试）

    参数:
    - error_type: 错误类型 (exception, message, warning)

    示例:
    - GET /test-sentry?error_type=exception  # 触发异常
    - GET /test-sentry?error_type=message    # 发送消息
    - GET /test-sentry?error_type=warning    # 发送警告
    """
    try:
        from app.sentry_config import capture_exception_with_context, capture_message_with_context

        if error_type == "exception":
            # 触发一个测试异常
            try:
                raise ValueError("Sentry 测试异常：这是一个用于测试监控的错误")
            except ValueError as e:
                capture_exception_with_context(
                    e,
                    level="error",
                    fingerprint=["test", "sentry", "exception"],
                    test_trigger=True,
                    endpoint="/test-sentry",
                )
                return {
                    "status": "error_captured",
                    "message": "测试异常已发送到 Sentry",
                    "error_type": error_type,
                }

        elif error_type == "message":
            # 发送测试消息
            capture_message_with_context(
                "Sentry 测试消息：监控系统运行正常",
                level="info",
                test_trigger=True,
                endpoint="/test-sentry",
            )
            return {
                "status": "message_sent",
                "message": "测试消息已发送到 Sentry",
                "error_type": error_type,
            }

        elif error_type == "warning":
            # 发送警告
            capture_message_with_context(
                "Sentry 测试警告：这是一个测试警告",
                level="warning",
                test_trigger=True,
                endpoint="/test-sentry",
            )
            return {
                "status": "warning_sent",
                "message": "测试警告已发送到 Sentry",
                "error_type": error_type,
            }

        else:
            return {
                "status": "invalid_type",
                "message": f"未知的错误类型: {error_type}",
                "supported_types": ["exception", "message", "warning"],
            }

    except ImportError:
        return {
            "status": "sentry_not_available",
            "message": "Sentry 未配置或不可用",
        }


@app.get("/hot/trending")
def get_trending(
    limit: int = Query(20, ge=1, le=100),
    timeframe: str = Query("1h", regex="^(1h|24h)$"),
) -> Dict[str, Any]:
    """Get trending/hot datasets."""
    state = _get_app_state()
    hot_tracker = getattr(state, "hot_tracker", None)

    if not hot_tracker:
        # Fallback to static popular list
        bundle, _ = _choose_bundle(state)
        hot_items = bundle.popular[:limit]
    else:
        hot_items = hot_tracker.get_hot_items(limit=limit, timeframe=timeframe)
        # Fallback to static popular if no trending data
        if not hot_items:
            bundle, _ = _choose_bundle(state)
            hot_items = bundle.popular[:limit]

    # Enrich with metadata
    items = []
    for dataset_id in hot_items:
        info = state.metadata.get(dataset_id, {})
        items.append({
            "dataset_id": dataset_id,
            "title": info.get("title"),
            "price": info.get("price"),
            "cover_image": info.get("cover_image"),
        })

    return {
        "timeframe": timeframe,
        "items": items,
    }


@app.get("/similar/{dataset_id}", response_model=SimilarResponse)
async def get_similar(
    request: Request,
    dataset_id: int,
    limit: int = Query(10, ge=1, le=50),
) -> SimilarResponse:
    endpoint = "similar"
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.perf_counter()
    status = "success"
    degrade_reason: Optional[str] = None
    variant = "primary"
    request_context = _extract_request_context(request)

    metrics_tracker = get_metrics_tracker()
    state = _get_app_state()
    cache = getattr(state, "cache", None)
    channel_weights = _get_channel_weight_baseline(state)
    applied_channel_weights = channel_weights

    # 设置 Sentry 上下文
    set_request_context(
        request_id=request_id,
        endpoint=endpoint,
        dataset_id=dataset_id,
        limit=limit,
    )

    try:
        if cache and cache.enabled:
            cache_key = f"similar:{dataset_id}:{limit}"
            try:
                cached_result = await _call_blocking(
                    cache.get_json,
                    cache_key,
                    endpoint=endpoint,
                    operation="redis_get",
                    timeout=TimeoutManager.get_timeout("redis_get"),
                )
            except asyncio.TimeoutError:
                cached_result = None
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Cache fetch failed for key %s: %s", cache_key, exc)
                cached_result = None
            else:
                if cached_result:
                    metrics_tracker.track_cache_hit()
                    response = SimilarResponse(**cached_result)
                    hot_tracker = getattr(state, "hot_tracker", None)
                    if hot_tracker:
                        await _run_in_executor(hot_tracker.track_view, dataset_id)
                    recommendation_count.labels(endpoint=endpoint).observe(
                        len(response.similar_items)
                    )
                    return response
                metrics_tracker.track_cache_miss()

        async def _compute() -> tuple[List[RecommendationItem], Dict[int, str], str, Optional[str], Dict[str, float]]:
            local_bundle, local_variant = _choose_bundle(state)
            effective_weights = _compute_dynamic_channel_weights(
                channel_weights,
                dataset_id=dataset_id,
                user_id=None,
                bundle=local_bundle,
                state=state,
            )
            stage_start = time.perf_counter()
            scores, reasons = _combine_scores_with_weights(
                dataset_id,
                local_bundle.behavior,
                local_bundle.content,
                local_bundle.vector,
                local_bundle.popular,
                limit,
                effective_weights,
            )
            _augment_with_multi_channel(
                state,
                target_id=dataset_id,
                scores=scores,
                reasons=reasons,
                limit=limit,
            )
            user_feature_map: Dict[str, float] = {}
            await _call_blocking(
                partial(
                    _apply_ranking,
                    scores,
                    reasons,
                    local_bundle.rank_model,
                    state.raw_features,
                    state.dataset_stats,
                    state.slot_metrics_aggregated,
                    state.feature_store,
                    endpoint=endpoint,
                    variant=local_variant,
                    experiment_variant=None,
                    request_context=request_context,
                    channel_weights=effective_weights,
                    user_features=user_feature_map,
                ),
                endpoint=endpoint,
                operation="model_inference",
                timeout=TimeoutManager.get_timeout("model_inference"),
            )
            mmr_lambda = _compute_mmr_lambda(endpoint=endpoint, request_context=request_context)
            items = _build_response_items(
                scores, reasons, limit, state.metadata,
                dataset_tags=state.dataset_tags,
                apply_mmr=True,
                mmr_lambda=mmr_lambda,
            )
            return items, reasons, local_variant, local_bundle.run_id, effective_weights

        compute_started = time.perf_counter()
        compute_started = time.perf_counter()
        try:
            items, reasons, variant, run_id, applied_channel_weights = await asyncio.wait_for(
                _compute(),
                timeout=TimeoutManager.get_timeout("recommendation_total"),
            )
            compute_duration = time.perf_counter() - compute_started
            LOGGER.info(
                "Recommendation compute completed (endpoint=%s, dataset=%s, user=%s, elapsed=%.3fs, items=%d)",
                endpoint,
                dataset_id,
                user_id,
                compute_duration,
                len(items),
            )
            compute_duration = time.perf_counter() - compute_started
            LOGGER.info(
                "Recommendation compute completed (endpoint=%s, dataset=%s, user=%s, elapsed=%.3fs, items=%d)",
                endpoint,
                dataset_id,
                None,
                compute_duration,
                len(items),
            )
        except asyncio.TimeoutError:
            recommendation_timeouts_total.labels(endpoint=endpoint, operation="total").inc()
            LOGGER.warning("Similar request timed out (dataset=%s, request=%s)", dataset_id, request_id)
            degrade_reason = "timeout"

            # Sentry: 记录超时事件
            add_breadcrumb(
                message=f"Recommendation timeout for dataset {dataset_id}",
                category="timeout",
                level="warning",
                data={"dataset_id": dataset_id, "request_id": request_id},
            )

            items, reasons, degrade_reason = _serve_fallback(
                state,
                dataset_id=dataset_id,
                limit=limit,
                endpoint=endpoint,
                degrade_cause=degrade_reason,
            )
            variant = "fallback"
            run_id = getattr(state, "model_run_id", None)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Error in get_similar (dataset=%s, request=%s): %s", dataset_id, request_id, exc
            )
            degrade_reason = "error"

            # Sentry: 捕获异常
            capture_exception_with_context(
                exc,
                level="error",
                fingerprint=["get_similar", type(exc).__name__],
                dataset_id=dataset_id,
                request_id=request_id,
                endpoint=endpoint,
            )

            items, reasons, degrade_reason = _serve_fallback(
                state,
                dataset_id=dataset_id,
                limit=limit,
                endpoint=endpoint,
                degrade_cause=degrade_reason,
            )
            variant = "fallback"
            run_id = getattr(state, "model_run_id", None)

        if not items:
            if degrade_reason is None:
                degrade_reason = "empty"
                items, reasons, degrade_reason = _serve_fallback(
                    state,
                    dataset_id=dataset_id,
                    limit=limit,
                    endpoint=endpoint,
                    degrade_cause=degrade_reason,
                )
                variant = "fallback"
                if items:
                    LOGGER.info(
                        "Served fallback recommendations for dataset %s (reason=%s)",
                        dataset_id,
                        degrade_reason,
                    )
            if not items:
                status = "error"
                detail = "No similar datasets found"
                if degrade_reason:
                    detail = f"{detail} (degraded={degrade_reason})"
                raise HTTPException(status_code=503 if degrade_reason else 404, detail=detail)

        response = SimilarResponse(
            dataset_id=dataset_id,
            similar_items=items[:limit],
            request_id=request_id,
            algorithm_version=run_id,
            variant=variant,
            request_context=request_context,
        )

        # 设置推荐上下文到 Sentry
        set_recommendation_context(
            algorithm_version=run_id,
            variant=variant,
            experiment_variant=None,
            degrade_reason=degrade_reason,
            channel_weights=applied_channel_weights,
        )

        _log_exposure(
            "similar",
            user_id=None,
            page_id=dataset_id,
            items=items[:limit],
            algorithm_version=run_id,
            variant=variant,
            reasons=reasons,
            request_id=request_id,
            degrade_reason=degrade_reason,
            experiment_variant=None,
            request_context=request_context,
        )

        if cache and cache.enabled and degrade_reason is None:
            cache_key = f"similar:{dataset_id}:{limit}"
            try:
                await _call_blocking(
                    cache.set_json,
                    cache_key,
                    response.dict(),
                    ttl=300,
                    endpoint=endpoint,
                    operation="redis_set",
                    timeout=TimeoutManager.get_timeout("redis_set"),
                )
            except asyncio.TimeoutError:
                LOGGER.warning("Cache set timed out for key %s", cache_key)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Cache set failed for key %s: %s", cache_key, exc)

        hot_tracker = getattr(state, "hot_tracker", None)
        if hot_tracker:
            await _run_in_executor(hot_tracker.track_view, dataset_id)

        recommendation_count.labels(endpoint=endpoint).observe(len(response.similar_items))
        if degrade_reason:
            status = "success"

        return response
    except HTTPException:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start_time
        recommendation_latency_seconds.labels(endpoint=endpoint).observe(duration)
        recommendation_requests_total.labels(endpoint=endpoint, status=status).inc()


@app.get("/recommend/detail/{dataset_id}", response_model=RecommendationResponse)
async def recommend_for_detail(
    request: Request,
    dataset_id: int,
    user_id: Optional[int] = None,
    limit: int = Query(10, ge=1, le=50),
) -> RecommendationResponse:
    endpoint = "recommend_detail"
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.perf_counter()
    status = "success"
    degrade_reason: Optional[str] = None
    variant = "primary"
    request_context = _extract_request_context(request)

    state = _get_app_state()
    metrics_tracker = get_metrics_tracker()
    cache = getattr(state, "cache", None)

    experiments = getattr(state, "experiments", {})
    experiment_variant, experiment_params = assign_variant(
        experiments,
        "recommendation_detail",
        user_id=user_id,
        request_id=request_id,
    )
    channel_weights = _get_channel_weight_baseline(state)
    applied_channel_weights = channel_weights
    for key, value in experiment_params.items():
        if key.endswith("_weight"):
            channel = key.replace("_weight", "")
            channel_weights[channel] = float(value)

    # 设置 Sentry 上下文
    set_request_context(
        request_id=request_id,
        endpoint=endpoint,
        dataset_id=dataset_id,
        limit=limit,
    )
    if user_id:
        set_user_context(user_id)

    try:
        if cache and cache.enabled and user_id:
            cache_key = f"recommend:{dataset_id}:{user_id}:{limit}"
            try:
                cached_result = await _call_blocking(
                    cache.get_json,
                    cache_key,
                    endpoint=endpoint,
                    operation="redis_get",
                    timeout=TimeoutManager.get_timeout("redis_get"),
                )
            except asyncio.TimeoutError:
                cached_result = None
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Cache fetch failed for key %s: %s", cache_key, exc)
                cached_result = None
            else:
                if cached_result:
                    LOGGER.debug("Cache hit for recommend:%s:%s", dataset_id, user_id)
                    hot_tracker = getattr(state, "hot_tracker", None)
                    if hot_tracker:
                        await _run_in_executor(hot_tracker.track_view, dataset_id)
                    return RecommendationResponse(**cached_result)

        async def _compute() -> tuple[List[RecommendationItem], Dict[int, str], str, Optional[str], Dict[str, float]]:
            local_bundle, local_variant = _choose_bundle(state)
            effective_weights = _compute_dynamic_channel_weights(
                channel_weights,
                dataset_id=dataset_id,
                user_id=user_id,
                bundle=local_bundle,
                state=state,
            )
            stage_start = time.perf_counter()
            scores, reasons = _combine_scores_with_weights(
                dataset_id,
                local_bundle.behavior,
                local_bundle.content,
                local_bundle.vector,
                local_bundle.popular,
                limit,
                effective_weights,
            )
            _log_stage_duration(
                "score_fusion",
                stage_start,
                dataset_id=dataset_id,
                user_id=user_id,
                request_id=request_id,
            )

            stage_start = time.perf_counter()
            _apply_personalization(
                user_id,
                scores,
                reasons,
                state,
                local_bundle.behavior,
                state.personalization_history_limit,
            )
            _log_stage_duration(
                "personalization",
                stage_start,
                dataset_id=dataset_id,
                user_id=user_id,
                request_id=request_id,
            )

            stage_start = time.perf_counter()
            _augment_with_multi_channel(
                state,
                target_id=dataset_id,
                scores=scores,
                reasons=reasons,
                limit=limit,
                user_id=user_id,
            )
            _log_stage_duration(
                "multi_channel",
                stage_start,
                dataset_id=dataset_id,
                user_id=user_id,
                request_id=request_id,
            )

            stage_start = time.perf_counter()
            user_feature_map = _get_user_features(state, user_id)
            _log_stage_duration(
                "user_features",
                stage_start,
                dataset_id=dataset_id,
                user_id=user_id,
                request_id=request_id,
            )

            stage_start = time.perf_counter()
            await _call_blocking(
                partial(
                    _apply_ranking,
                    scores,
                    reasons,
                    local_bundle.rank_model,
                    state.raw_features,
                    state.dataset_stats,
                    state.slot_metrics_aggregated,
                    state.feature_store,
                    endpoint=endpoint,
                    variant=local_variant,
                    experiment_variant=experiment_variant,
                    request_context=request_context,
                    channel_weights=effective_weights,
                    user_features=user_feature_map,
                ),
                endpoint=endpoint,
                operation="model_inference",
                timeout=TimeoutManager.get_timeout("model_inference"),
            )
            _log_stage_duration(
                "ranking",
                stage_start,
                dataset_id=dataset_id,
                user_id=user_id,
                request_id=request_id,
            )
            mmr_lambda = _compute_mmr_lambda(endpoint=endpoint, request_context=request_context)
            stage_start = time.perf_counter()
            items = _build_response_items(
                scores, reasons, limit, state.metadata,
                dataset_tags=state.dataset_tags,
                apply_mmr=True,
                mmr_lambda=mmr_lambda,
                apply_exploration=True,  # 启用探索机制
                exploration_epsilon=0.15,  # 15%探索率
                all_dataset_ids=set(state.metadata.keys()),  # 全量dataset池
            )
            _log_stage_duration(
                "response_build",
                stage_start,
                dataset_id=dataset_id,
                user_id=user_id,
                request_id=request_id,
            )
            return items, reasons, local_variant, local_bundle.run_id, effective_weights

        try:
            items, reasons, variant, run_id, applied_channel_weights = await asyncio.wait_for(
                _compute(),
                timeout=TimeoutManager.get_timeout("recommendation_total"),
            )
        except asyncio.TimeoutError:
            recommendation_timeouts_total.labels(endpoint=endpoint, operation="total").inc()
            LOGGER.warning(
                "Recommendation request timed out (dataset=%s, user=%s, request=%s)",
                dataset_id,
                user_id,
                request_id,
            )
            degrade_reason = "timeout"

            # Sentry: 记录超时事件
            add_breadcrumb(
                message=f"Recommendation timeout for dataset {dataset_id}, user {user_id}",
                category="timeout",
                level="warning",
                data={"dataset_id": dataset_id, "user_id": user_id, "request_id": request_id},
            )

            items, reasons, degrade_reason = _serve_fallback(
                state,
                dataset_id=dataset_id,
                limit=limit,
                endpoint=endpoint,
                degrade_cause=degrade_reason,
                user_id=user_id,
            )
            variant = "fallback"
            run_id = getattr(state, "model_run_id", None)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Error in recommend_for_detail (dataset=%s, user=%s, request=%s): %s",
                dataset_id,
                user_id,
                request_id,
                exc,
            )
            degrade_reason = "error"

            # Sentry: 捕获异常
            capture_exception_with_context(
                exc,
                level="error",
                fingerprint=["recommend_for_detail", type(exc).__name__],
                dataset_id=dataset_id,
                user_id=user_id,
                request_id=request_id,
                endpoint=endpoint,
                experiment_variant=experiment_variant,
            )

            items, reasons, degrade_reason = _serve_fallback(
                state,
                dataset_id=dataset_id,
                limit=limit,
                endpoint=endpoint,
                degrade_cause=degrade_reason,
                user_id=user_id,
            )
            variant = "fallback"
            run_id = getattr(state, "model_run_id", None)

        if not items:
            if degrade_reason is None:
                degrade_reason = "empty"
                items, reasons, degrade_reason = _serve_fallback(
                    state,
                    dataset_id=dataset_id,
                    limit=limit,
                    endpoint=endpoint,
                    degrade_cause=degrade_reason,
                    user_id=user_id,
                )
                variant = "fallback"
                if items:
                    LOGGER.info(
                        "Served fallback recommendations for dataset %s user %s (reason=%s)",
                        dataset_id,
                        user_id,
                        degrade_reason,
                    )
            if not items:
                status = "error"
                detail = "No recommendations available"
                if degrade_reason:
                    detail = f"{detail} (degraded={degrade_reason})"
                raise HTTPException(status_code=503 if degrade_reason else 404, detail=detail)

        response = RecommendationResponse(
            dataset_id=dataset_id,
            recommendations=items[:limit],
            request_id=request_id,
            algorithm_version=run_id,
            variant=variant,
            experiment_variant=experiment_variant,
            request_context=request_context,
        )

        # 设置推荐上下文到 Sentry
        set_recommendation_context(
            algorithm_version=run_id,
            variant=variant,
            experiment_variant=experiment_variant,
            degrade_reason=degrade_reason,
            channel_weights=applied_channel_weights,
        )

        _log_exposure(
            "recommend_detail",
            user_id=int(user_id) if user_id is not None else None,
            page_id=dataset_id,
            items=items[:limit],
            algorithm_version=run_id,
            variant=variant,
            reasons=reasons,
            request_id=request_id,
            degrade_reason=degrade_reason,
            experiment_variant=experiment_variant,
            request_context=request_context,
        )

        if cache and cache.enabled and user_id and degrade_reason is None:
            cache_key = f"recommend:{dataset_id}:{user_id}:{limit}"
            try:
                await _call_blocking(
                    cache.set_json,
                    cache_key,
                    response.dict(),
                    ttl=180,
                    endpoint=endpoint,
                    operation="redis_set",
                    timeout=TimeoutManager.get_timeout("redis_set"),
                )
            except asyncio.TimeoutError:
                LOGGER.warning("Cache set timed out for key %s", cache_key)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Cache set failed for key %s: %s", cache_key, exc)

        hot_tracker = getattr(state, "hot_tracker", None)
        if hot_tracker:
            await _run_in_executor(hot_tracker.track_view, dataset_id)

        recommendation_count.labels(endpoint=endpoint).observe(len(response.recommendations))
        if degrade_reason:
            status = "success"

        return response
    except HTTPException:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start_time
        recommendation_latency_seconds.labels(endpoint=endpoint).observe(duration)
        recommendation_requests_total.labels(endpoint=endpoint, status=status).inc()


@app.post("/models/reload")
def reload_models(request: ReloadRequest) -> Dict[str, object]:
    try:
        state = _get_app_state()
    except RuntimeError:
        load_models()
        state = _get_app_state()

    mode = request.mode.lower()
    if mode not in {"primary", "shadow"}:
        raise HTTPException(status_code=400, detail="mode must be 'primary' or 'shadow'")

    source_dir = Path(request.source).resolve() if request.source else MODELS_DIR
    run_id = request.run_id or _load_model_run_id(source_dir)
    bundle = _load_model_bundle(source_dir, run_id=run_id)

    if mode == "primary":
        if request.source and source_dir != MODELS_DIR:
            deploy_from_source(source_dir)
            bundle = _load_model_bundle(MODELS_DIR, run_id=_load_model_run_id())
        _set_bundle(state, bundle, prefix="")
        message = "Primary model reloaded"
    else:
        _set_bundle(state, bundle, prefix="shadow")
        rollout = request.rollout if request.rollout is not None else getattr(state, "shadow_rollout", 0.0)
        state.shadow_rollout = max(0.0, min(1.0, rollout))
        message = "Shadow model loaded"

    if mode == "primary" and request.rollout is not None:
        state.shadow_rollout = max(0.0, min(1.0, request.rollout))

    return {
        "status": "ok",
        "mode": mode,
        "run_id": bundle.run_id,
        "shadow_rollout": getattr(state, "shadow_rollout", 0.0),
        "message": message,
    }
import urllib.parse
