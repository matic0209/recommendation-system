"""FastAPI service exposing dataset detail recommendations."""
from __future__ import annotations

import asyncio
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
    service_info,
    track_request_metrics,
    recommendation_degraded_total,
    recommendation_timeouts_total,
    thread_pool_queue_gauge,
)

EXECUTOR_MAX_WORKERS = int(os.getenv("RECO_THREAD_POOL_WORKERS", "4"))
EXECUTOR = ThreadPoolExecutor(max_workers=EXECUTOR_MAX_WORKERS)


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
    "behavior": 1.0,
    "content": 0.5,
    "vector": 0.4,
    "popular": 0.01,
}


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach request ID and basic timing information to each request."""
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
    request.state.request_id = request_id
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


class SimilarResponse(BaseModel):
    dataset_id: int
    similar_items: List[RecommendationItem]
    request_id: str  # 用于前端埋点追踪
    algorithm_version: Optional[str] = None  # 算法版本，用于A/B对比


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
    if not isinstance(model, Pipeline):
        LOGGER.warning("Ranking artifact at %s is not a sklearn Pipeline", path)
        return None
    return model


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
        metadata[dataset_id] = {
            "title": row.get("title"),
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
        meta_path = DATA_DIR / "processed" / "dataset_features.parquet"
        if meta_path.exists():
            frame = pd.read_parquet(meta_path)
            source_label = "parquet"
        else:
            LOGGER.warning("Dataset feature file missing: %s", meta_path)
            return {}, {}, pd.DataFrame()

    if frame.empty:
        return {}, {}, pd.DataFrame()

    return _parse_dataset_metadata_frame(frame, source=source_label)


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


def _build_response_items(
    candidate_scores: Dict[int, float],
    reasons: Dict[int, str],
    limit: int,
    metadata: Dict[int, Dict[str, Optional[str]]],
) -> List[RecommendationItem]:
    result: List[RecommendationItem] = []
    for dataset_id, score in sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True):
        info = metadata.get(dataset_id, {})
        result.append(
            RecommendationItem(
                dataset_id=dataset_id,
                title=info.get("title"),
                price=info.get("price"),
                cover_image=info.get("cover_image"),
                score=float(score),
                reason=reasons.get(dataset_id, "unknown"),
            )
        )
        if len(result) >= limit:
            break
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


async def _call_blocking(
    func: Callable,
    *args,
    endpoint: str,
    operation: str,
    timeout: float,
    **kwargs,
):
    try:
        return await asyncio.wait_for(
            _run_in_executor(func, *args, **kwargs),
            timeout=timeout,
        )
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
        for dataset_id, score in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[: limit * 2]:
            _bump(int(dataset_id), float(score) * 0.4, "tag")

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
    price = state.metadata.get(target_id, {}).get("price", 0.0)
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
        history_sets = getattr(state, "user_history_sets", {})
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


def _combine_scores_with_weights(
    target_id: int,
    behavior: Dict[int, Dict[int, float]],
    content: Dict[int, Dict[int, float]],
    vector: Dict[int, List[Dict[str, float]]],
    popular: List[int],
    limit: int,
    weights: Dict[str, float],
) -> tuple[Dict[int, float], Dict[int, str]]:
    scores: Dict[int, float] = {}
    reasons: Dict[int, str] = {}

    behavior_scores = behavior.get(target_id, {})
    for item_id, score in behavior_scores.items():
        scores[int(item_id)] = float(score) * weights.get("behavior", 1.0)
        reasons[int(item_id)] = "behavior"

    content_scores = content.get(target_id, {})
    for item_id, score in content_scores.items():
        item_id = int(item_id)
        if item_id not in scores:
            scores[item_id] = float(score) * weights.get("content", 0.5)
            reasons[item_id] = "content"
        else:
            scores[item_id] += float(score) * weights.get("content", 0.5)

    vector_scores = vector.get(target_id, [])
    for entry in vector_scores:
        item_id = int(entry.get("dataset_id", 0))
        if item_id == target_id:
            continue
        score = float(entry.get("score", 0.0))
        if score <= 0:
            continue
        if item_id not in scores:
            scores[item_id] = score * weights.get("vector", 0.4)
            reasons[item_id] = "vector"
        else:
            scores[item_id] += score * weights.get("vector", 0.4)
        if len(scores) >= limit * 4:
            break

    for idx, item_id in enumerate(popular):
        if item_id == target_id or item_id in scores:
            continue
        base = max(weights.get("popular", 0.01), 0.0)
        scores[item_id] = max(base, base - idx * base * 0.01)
        reasons[item_id] = "popular"
        if len(scores) >= limit * 5:
            break
    scores.pop(target_id, None)
    return scores, reasons


def _apply_personalization(
    user_id: Optional[int],
    scores: Dict[int, float],
    reasons: Dict[int, str],
    state,
    behavior: Dict[int, Dict[int, float]],
    history_limit: int = 20,
) -> None:
    if not user_id:
        return
    user_history = state.user_history.get(int(user_id))
    if not user_history:
        return

    recent_history = user_history[:history_limit]
    history_ids = {record["dataset_id"] for record in recent_history}
    for dataset_id in history_ids:
        scores.pop(dataset_id, None)
        reasons.pop(dataset_id, None)

    tag_pref = state.user_tag_preferences.get(int(user_id), {})

    for dataset_id in list(scores.keys()):
        boost = 0.0
        for record in recent_history:
            source_id = record["dataset_id"]
            weight = record["weight"]
            sim = behavior.get(source_id, {}).get(dataset_id, 0.0)
            if sim:
                boost += sim * weight * 0.5
        candidate_tags = state.dataset_tags.get(dataset_id, [])
        if candidate_tags and tag_pref:
            tag_boost = sum(tag_pref.get(tag, 0.0) for tag in candidate_tags)
            boost += tag_boost * 0.2

        if boost > 0:
            scores[dataset_id] += boost
            base_reason = reasons.get(dataset_id, "unknown")
            if "personalized" not in base_reason:
                reasons[dataset_id] = f"{base_reason}+personalized"


def _compute_ranking_features(
    dataset_ids: List[int],
    raw_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
) -> pd.DataFrame:
    if not dataset_ids:
        return pd.DataFrame()

    if raw_features.empty:
        selected = pd.DataFrame(index=dataset_ids)
        selected["price"] = 0.0
        selected["description"] = ""
        selected["tag"] = ""
    else:
        selected = raw_features.set_index("dataset_id").reindex(dataset_ids)
        selected["price"] = pd.to_numeric(selected.get("price"), errors="coerce").fillna(0.0)
        selected["description"] = selected.get("description", "").fillna("").astype(str)
        selected["tag"] = selected.get("tag", "").fillna("").astype(str)

    selected["description_length"] = selected.get("description", "").str.len().astype(float)
    selected["tag_count"] = selected.get("tag", "").apply(
        lambda text: float(len([t for t in text.split(';') if t.strip()])) if isinstance(text, str) else 0.0
    )

    if dataset_stats.empty:
        stats = pd.DataFrame(index=dataset_ids)
        stats["interaction_count"] = 0.0
        stats["total_weight"] = 0.0
    else:
        stats = dataset_stats.set_index("dataset_id").reindex(dataset_ids)
        stats["interaction_count"] = pd.to_numeric(stats.get("interaction_count"), errors="coerce").fillna(0.0)
        stats["total_weight"] = pd.to_numeric(stats.get("total_weight"), errors="coerce").fillna(0.0)

    features = pd.DataFrame(index=dataset_ids)
    features["price_log"] = np.log1p(selected["price"].clip(lower=0.0))
    features["description_length"] = selected["description_length"].fillna(0.0)
    features["tag_count"] = selected["tag_count"].fillna(0.0)
    features["weight_log"] = np.log1p(stats["total_weight"].clip(lower=0.0))
    features["interaction_count"] = stats["interaction_count"].fillna(0.0)

    optional_columns = ["image_richness_score", "image_embed_norm", "has_images", "has_cover"]
    for col in optional_columns:
        if col in selected.columns:
            features[col] = pd.to_numeric(selected[col], errors="coerce").fillna(0.0)
        else:
            features[col] = 0.0
    return features


@with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
def _apply_ranking_with_circuit_breaker(
    scores: Dict[int, float],
    reasons: Dict[int, str],
    rank_model: Pipeline,
    features: pd.DataFrame,
) -> None:
    """Apply ranking with circuit breaker protection."""
    probabilities = rank_model.predict_proba(features)[:, 1]

    for dataset_id, prob in zip(features.index.astype(int), probabilities):
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
    rank_model: Optional[Pipeline],
    raw_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
) -> None:
    if rank_model is None or not scores:
        return

    dataset_ids = list(scores.keys())
    features = _compute_ranking_features(dataset_ids, raw_features, dataset_stats)
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
    if degrade_reason:
        context["degrade_reason"] = degrade_reason
    if experiment_variant:
        context["experiment_variant"] = experiment_variant
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
    experiment_config_path = Path(os.getenv("EXPERIMENT_CONFIG_PATH", BASE_DIR / "config" / "experiments.yaml"))
    app.state.experiments = load_experiments(experiment_config_path)

    metadata, dataset_tags, raw_features = _load_dataset_metadata(
        feature_store=feature_store,
        dataset_ids=dataset_ids,
    )
    dataset_stats = _load_dataset_stats(
        feature_store=feature_store,
        dataset_ids=dataset_ids,
    )
    user_history = _load_user_history()
    user_profiles = _load_user_profile()
    user_tag_preferences = _build_user_tag_preferences(user_history, dataset_tags)

    _set_bundle(app.state, bundle, prefix="")
    app.state.shadow_bundle = None
    app.state.shadow_rollout = 0.0
    app.state.metadata = metadata
    app.state.raw_features = raw_features
    app.state.dataset_tags = dataset_tags
    app.state.dataset_stats = dataset_stats
    app.state.user_history = user_history
    app.state.user_profiles = user_profiles
    app.state.user_tag_preferences = user_tag_preferences
    app.state.personalization_history_limit = 20
    app.state.models_loaded = True

    app.state.recall_indices = _load_recall_artifacts(MODELS_DIR)
    app.state.user_history_sets = {
        user_id: {record["dataset_id"] for record in records}
        for user_id, records in user_history.items()
    }

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

    metrics_tracker = get_metrics_tracker()
    state = _get_app_state()
    cache = getattr(state, "cache", None)
    channel_weights = DEFAULT_CHANNEL_WEIGHTS.copy()

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

        async def _compute() -> tuple[List[RecommendationItem], Dict[int, str], str, Optional[str]]:
            local_bundle, local_variant = _choose_bundle(state)
            scores, reasons = _combine_scores_with_weights(
                dataset_id,
                local_bundle.behavior,
                local_bundle.content,
                local_bundle.vector,
                local_bundle.popular,
                limit,
                channel_weights,
            )
            _augment_with_multi_channel(
                state,
                target_id=dataset_id,
                scores=scores,
                reasons=reasons,
                limit=limit,
            )
            await _call_blocking(
                _apply_ranking,
                scores,
                reasons,
                local_bundle.rank_model,
                state.raw_features,
                state.dataset_stats,
                endpoint=endpoint,
                operation="model_inference",
                timeout=TimeoutManager.get_timeout("model_inference"),
            )
            items = _build_response_items(scores, reasons, limit, state.metadata)
            return items, reasons, local_variant, local_bundle.run_id

        try:
            items, reasons, variant, run_id = await asyncio.wait_for(
                _compute(),
                timeout=TimeoutManager.get_timeout("recommendation_total"),
            )
        except asyncio.TimeoutError:
            recommendation_timeouts_total.labels(endpoint=endpoint, operation="total").inc()
            LOGGER.warning("Similar request timed out (dataset=%s, request=%s)", dataset_id, request_id)
            degrade_reason = "timeout"
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
            algorithm_version=run_id
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
    channel_weights = DEFAULT_CHANNEL_WEIGHTS.copy()
    for key, value in experiment_params.items():
        if key.endswith("_weight"):
            channel = key.replace("_weight", "")
            channel_weights[channel] = float(value)

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

        async def _compute() -> tuple[List[RecommendationItem], Dict[int, str], str, Optional[str]]:
            local_bundle, local_variant = _choose_bundle(state)
            scores, reasons = _combine_scores(
                dataset_id, local_bundle.behavior, local_bundle.content, local_bundle.vector, local_bundle.popular, limit
            )
            _apply_personalization(
                user_id,
                scores,
                reasons,
                state,
                local_bundle.behavior,
                state.personalization_history_limit,
            )
            _augment_with_multi_channel(
                state,
                target_id=dataset_id,
                scores=scores,
                reasons=reasons,
                limit=limit,
                user_id=user_id,
            )
            await _call_blocking(
                _apply_ranking,
                scores,
                reasons,
                local_bundle.rank_model,
                state.raw_features,
                state.dataset_stats,
                endpoint=endpoint,
                operation="model_inference",
                timeout=TimeoutManager.get_timeout("model_inference"),
            )
            items = _build_response_items(scores, reasons, limit, state.metadata)
            return items, reasons, local_variant, local_bundle.run_id

        try:
            items, reasons, variant, run_id = await asyncio.wait_for(
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
            algorithm_version=run_id
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
