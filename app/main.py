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
from functools import lru_cache, partial
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


def _env_float(
    name: str,
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """读取浮点型环境变量，非法值回退默认值并告警。

    这样做的目的：调参写错一个环境变量时，只退化为旧行为，而不是让线上
    每个请求都抛异常。
    """
    raw = os.getenv(name)
    if raw in (None, ""):
        value = float(default)
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            LOGGER.warning("环境变量 %s=%r 不是合法数字，回退默认值 %s", name, raw, default)
            value = float(default)
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


app = FastAPI(title="Dataset Recommendation API")

# Set service info for Prometheus
service_info.info({
    "version": "1.0.0",
    "service": "recommendation-api",
    "environment": "production",
})

REQUEST_ID_HEADER = "X-Request-ID"

# 融合层渠道权重：分数 = Σ(渠道内 Min-Max 归一化分 × 权重)
# 说明：
#   1) 该表是"兜底/默认值"，服务启动时会用 models/channel_weights.json 覆盖
#      behavior/content/vector/popular 四项（该文件由 pipeline.train_channel_weights
#      按真实 CTR/CVR 每天重算）。
#   2) 覆盖结果会经过 _enforce_channel_weight_order() 处理：既限制单渠道浮动幅度，
#      又强制保证 vector=content > behavior > popular 的业务次序不被反转。
#   3) popular 无论权重多少，最终列表里最多占 MAX_POPULAR_SHARE（默认 25%）。
#   4) "12cat" 不参与分数融合，仅作为排序特征 channel_weight 的取值来源，
#      类别召回的加分系数见 DEFAULT_AUGMENT_WEIGHTS。
DEFAULT_CHANNEL_WEIGHTS = {
    # 权重次序（业务规则，2026-02 与业务方确认）：
    #     vector(语义) = content(内容/标签)  >  behavior(行为协同)  >  popular(全局热门)
    # _enforce_channel_weight_order() 会强制保证这个次序，即使
    # models/channel_weights.json（每日 CTR 训练结果）给出相反的值。
    "vector": 1.2,    # SBERT 语义向量 —— 一级
    "content": 1.2,   # 内容/标签相似 —— 一级（与 vector 同级）
    "behavior": 0.9,  # 行为协同 —— 二级
    "popular": 0.05,  # 全局热门 —— 三级（仅作兜底，且最终列表有配额限制）
    "12cat": 0.8,     # 12类别召回：仅用于排序特征 channel_weight
}

# 权重保序规则参数
CHANNEL_WEIGHT_TIERS = (("vector", "content"), ("behavior",), ("popular",))
CHANNEL_WEIGHT_OVERRIDE_BAND = _env_float("CHANNEL_WEIGHT_OVERRIDE_BAND", 0.4, minimum=0.0, maximum=1.0)
CHANNEL_WEIGHT_MIN_GAP = _env_float("CHANNEL_WEIGHT_MIN_GAP", 0.15, minimum=0.0, maximum=0.9)

# 最终推荐列表的配额（解决"某类热销商品刷屏"和"热门渠道占比过高"）
# 12 条推荐时：popular 来源最多 3 条，非目标行业的单一类别最多 3 条
MAX_POPULAR_SHARE = _env_float("MAX_POPULAR_SHARE", 0.25, minimum=0.0, maximum=1.0)
MAX_OTHER_CATEGORY_SHARE = _env_float("MAX_OTHER_CATEGORY_SHARE", 0.25, minimum=0.0, maximum=1.0)

# 辅助渠道加分系数（绝对分量纲，与上面"归一化×权重"不是同一套刻度）
# 可通过 config/experiments.yaml 覆盖，写法为 augment_<渠道>_weight，例如：
#   parameters: {augment_12cat_weight: 0.8}
# 注意：models/channel_weights.json 不会包含 augment_ 前缀的键，因此这里的系数
# 不会被 CTR 训练结果意外覆盖。
DEFAULT_AUGMENT_WEIGHTS = {
    "tag": 0.4,      # 标签重叠召回
    "12cat": 0.6,    # 12类别召回（原硬编码 0.5，小幅加强类别相关性）
    "category": 0.3,  # 同公司/同来源（遗留渠道）
    "price": 0.2,    # 价格分桶召回（原硬编码 0.25）
    "usercf": 0.6,   # 相似用户召回
}

# 行业类别词表：与 pipeline/enhance_tags.py 的 13 个类别保持一致，
# 并保留历史短名（金融/科技/农业 等）以兼容更早的标签数据。
# 用于排序特征 same_top_category（"目标与候选是否同属一个行业类别"）。
TOP_CATEGORY_TOKENS = frozenset([
    # 当前 13 个类别（enhance_tags.py TOP_CATEGORIES）
    "政府政务", "金融财经", "医疗健康", "交通物流", "教育科研", "工业制造",
    "商业零售", "能源环保", "文化娱乐", "农业农村", "互联网科技", "社会民生",
    "数字产品",
    # 历史短名（旧标签里可能出现）
    "金融", "交通运输", "教育培训", "农业", "科技",
])

# 类别配额里"未分类 / 仅属于目标自身类别"候选使用的计数桶名
UNCATEGORIZED_BUCKET = "__uncategorized__"

# 探索率（epsilon-greedy）：默认 0.10（原硬编码 0.15），
# 15% 的随机替换对点击率的负向影响大于其发现价值
EXPLORATION_EPSILON = _env_float("EXPLORATION_EPSILON", 0.10, minimum=0.0, maximum=0.5)



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


def _get_time_bucket(bucket_hours: int = 1) -> str:
    """Generate time bucket identifier for cache key.

    This ensures cached results refresh periodically rather than being
    permanently fixed for the same request parameters.

    Args:
        bucket_hours: Time bucket granularity in hours (1=hourly, 24=daily)

    Returns:
        Time bucket string like "2025-12-26-14" (hourly) or "2025-12-26" (daily)

    Examples:
        >>> _get_time_bucket(1)  # at 14:30
        "2025-12-26-14"
        >>> _get_time_bucket(24)
        "2025-12-26"
    """
    now = datetime.now()
    if bucket_hours == 1:
        return now.strftime("%Y-%m-%d-%H")  # Hourly bucket
    else:
        return now.strftime("%Y-%m-%d")  # Daily bucket


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
    """Compute MMR lambda parameter for diversity-relevance tradeoff.

    Lambda=0.5 means 50% relevance, 50% diversity (balanced).
    Lower lambda = more diversity, higher lambda = more relevance.

    可通过环境变量调整基准值（默认值保持不变）：
      MMR_LAMBDA         recommend_detail 基准值，默认 0.5
      MMR_LAMBDA_SIMILAR similar 接口基准值，默认 0.4

    Args:
        endpoint: Recommendation endpoint
        request_context: Request context with source, device_type, etc.

    Returns:
        Lambda value in [0.2, 0.6] range
    """
    # 基准值：detail 偏相关性（0.5），similar 偏多样性（0.4）
    if endpoint == "recommend_detail":
        base = _env_float("MMR_LAMBDA", 0.5, minimum=0.1, maximum=0.9)
    else:
        base = _env_float("MMR_LAMBDA_SIMILAR", 0.4, minimum=0.1, maximum=0.9)
    context = request_context or {}
    source = context.get("source")

    # 搜索场景：用户有明确意图，适度提升相关性
    if source == "search":
        base = 0.6
    # 浏览场景（landing/home）：强化多样性
    elif source in {"landing", "home"}:
        base = 0.3

    # 移动端用户倾向浏览多样化内容，降低相关性权重
    device = context.get("device_type")
    if device == "mobile":
        base = max(base - 0.1, 0.2)  # 降低而非提升，上限改为下限

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


def _enforce_channel_weight_order(weights: Dict[str, float]) -> Dict[str, float]:
    """把渠道权重约束到业务规则：vector = content > behavior > popular。

    两步处理（都可调）：
      1) 限幅：每个渠道夹在默认值的 [1-band, 1+band] 内（默认 ±40%），
         防止每日 CTR 训练结果把某渠道压到接近 0 或抬到失真；
      2) 保序：上层渠道至少比下层高 gap（默认 15%），
         vector/content 视为同一层，取较低者作为该层的上限基准。

    这样即使 models/channel_weights.json 出现"popular 比 content 还高"这种
    与业务预期相反的结果，线上次序也不会被反转。

    Args:
        weights: 已应用覆盖值的权重表

    Returns:
        修正后的权重表（同时在日志里记录被修正的渠道）
    """
    adjusted = {key: max(float(value), 0.0) for key, value in weights.items()}
    corrections: Dict[str, Tuple[float, float]] = {}

    band = max(CHANNEL_WEIGHT_OVERRIDE_BAND, 0.0)
    for channel, default in DEFAULT_CHANNEL_WEIGHTS.items():
        if channel not in adjusted or channel == "12cat":
            continue
        lower = default * (1 - band)
        upper = default * (1 + band)
        value = adjusted[channel]
        clamped = min(max(value, lower), upper)
        if abs(clamped - value) > 1e-9:
            corrections[channel] = (value, clamped)
            adjusted[channel] = clamped

    gap = min(max(CHANNEL_WEIGHT_MIN_GAP, 0.0), 0.9)
    for index in range(1, len(CHANNEL_WEIGHT_TIERS)):
        upper_tier = CHANNEL_WEIGHT_TIERS[index - 1]
        current_tier = CHANNEL_WEIGHT_TIERS[index]
        upper_ceiling = min(adjusted.get(name, 0.0) for name in upper_tier)
        ceiling = upper_ceiling / (1 + gap)
        for name in current_tier:
            if name not in adjusted:
                continue
            if adjusted[name] > ceiling:
                corrections[name] = (adjusted[name], ceiling)
                adjusted[name] = max(ceiling, 0.0)

    if corrections:
        LOGGER.info(
            "Channel weight order enforced (vector=content>behavior>popular): %s",
            ", ".join(f"{name}: {old:.4f}->{new:.4f}" for name, (old, new) in corrections.items()),
        )
    return adjusted


def _load_excluded_dataset_ids() -> Set[int]:
    """加载"不可售/已下架"数据集黑名单。

    来源（可同时使用）：
      1) 环境变量 RECO_EXCLUDED_DATASET_IDS="123,456"；
      2) models/excluded_dataset_ids.json，内容为 [123, 456] 或 {"ids": [123, 456]}。

    背景：业务库的状态字段（is_delete / publish_status / status）**从未同步**到推荐
    系统——特征表与 models 文件里都没有状态列，所以推荐链路此前没有任何状态过滤。
    实测证据：数据集 15769 的线上页面显示"该数据已删除"，业务库最新状态为
    publish_status=0，但它仍留在推荐候选池里（全站共 22 个这类 id）。

    这里提供一条不依赖 ETL 改动的兜底通道；名单可直接用
    scripts/export_excluded_ids.py 从业务库增量导出目录（/dianshu/backup/.../jsons）
    自动生成。根治办法是让特征表带上在售标记（见 docs/推荐系统问题总结与修复方案.md）。
    """
    excluded: Set[int] = set()

    raw = os.getenv("RECO_EXCLUDED_DATASET_IDS", "") or ""
    for token in raw.replace(";", ",").replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            excluded.add(int(token))
        except (TypeError, ValueError):
            LOGGER.warning("RECO_EXCLUDED_DATASET_IDS 含非法项 %r，已忽略", token)

    path = MODELS_DIR / "excluded_dataset_ids.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                payload = payload.get("ids") or []
            for item in payload or []:
                try:
                    excluded.add(int(item))
                except (TypeError, ValueError):
                    continue
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("解析排除名单 %s 失败: %s", path, exc)
    return excluded


def _prune_unavailable_datasets(
    *,
    bundle: ModelBundle,
    recall_indices: Dict[str, Any],
    metadata: Dict[int, Dict[str, Optional[str]]],
    dataset_tags: Dict[int, List[str]],
    excluded: Set[int],
) -> int:
    """把不可售/已下架数据集从内存索引中剔除（就地修改，保持引用一致）。

    覆盖：元数据、标签、行为/内容/向量/热门召回表、12cat 类别索引、
    标签倒排、价格分桶、item_to_tags。剔除后这些 ID 既不会被召回，
    也不会进入探索池（探索池取自 metadata.keys()）。
    """
    if not excluded:
        return 0
    removed = 0

    for mapping in (metadata, dataset_tags):
        for dataset_id in list(mapping.keys()):
            if dataset_id in excluded:
                mapping.pop(dataset_id, None)
                removed += 1

    for attr in ("behavior", "content"):
        neighbors_map = getattr(bundle, attr, None)
        if not isinstance(neighbors_map, dict):
            continue
        for source_id in list(neighbors_map.keys()):
            if source_id in excluded:
                neighbors_map.pop(source_id, None)
                removed += 1
                continue
            neighbors = neighbors_map[source_id]
            if isinstance(neighbors, dict):
                neighbors_map[source_id] = {
                    key: value for key, value in neighbors.items() if key not in excluded
                }
                removed += len(neighbors) - len(neighbors_map[source_id])

    vector_map = getattr(bundle, "vector", None)
    if isinstance(vector_map, dict):
        for source_id in list(vector_map.keys()):
            if source_id in excluded:
                vector_map.pop(source_id, None)
                removed += 1
                continue
            entries = vector_map[source_id]
            if isinstance(entries, list):
                kept = []
                for entry in entries:
                    try:
                        entry_id = int(entry.get("dataset_id", 0) or 0)
                    except (AttributeError, TypeError, ValueError):
                        kept.append(entry)          # 结构异常时保守保留，不能让启动失败
                        continue
                    if entry_id not in excluded:
                        kept.append(entry)
                removed += len(entries) - len(kept)
                vector_map[source_id] = kept

    popular = getattr(bundle, "popular", None)
    if isinstance(popular, list):
        before = len(popular)
        popular[:] = [dataset_id for dataset_id in popular if dataset_id not in excluded]
        removed += before - len(popular)

    for name in ("tag_to_items", "category_to_items", "price_bucket_index"):
        index = recall_indices.get(name)
        if not isinstance(index, dict):
            continue
        for key, value in list(index.items()):
            if isinstance(value, set):
                index[key] = {item for item in value if item not in excluded}
            elif isinstance(value, list):
                index[key] = [item for item in value if item not in excluded]

    # 注意：这里必须把字符串形式的排除集在循环外算好。
    # 早期版本写成 `str(dataset_id) in {str(item) for item in excluded}`，
    # 会在每个索引键上重建一次集合（5000 键 × 11000 排除 id ≈ 5500 万次字符串转换），
    # 让启动多花几十秒 —— 已修正。
    excluded_str = {str(item) for item in excluded}
    for name in ("item_to_tags", "item_to_categories"):
        index = recall_indices.get(name)
        if isinstance(index, dict):
            for dataset_id in list(index.keys()):
                if dataset_id in excluded or str(dataset_id) in excluded_str:
                    index.pop(dataset_id, None)

    user_similarity = recall_indices.get("user_similarity")
    if isinstance(user_similarity, dict):
        for user_id, entries in list(user_similarity.items()):
            if not isinstance(entries, list):
                continue
            kept_entries = []
            for entry in entries:
                if not isinstance(entry, (list, tuple)) or not entry:
                    kept_entries.append(entry)
                    continue
                try:
                    if int(entry[0]) in excluded:
                        continue
                except (TypeError, ValueError):
                    kept_entries.append(entry)   # 结构异常时保守保留
                    continue
                kept_entries.append(entry)
            user_similarity[user_id] = kept_entries
    return removed


def _drop_excluded_from_scores(
    scores: Dict[int, float],
    reasons: Dict[int, str],
    excluded: Set[int],
) -> int:
    """请求内兜底：把不可售/已下架的候选从候选集中剔除（防止命中陈旧索引）。"""
    if not excluded or not scores:
        return 0
    dropped = 0
    for dataset_id in list(scores.keys()):
        if dataset_id in excluded:
            scores.pop(dataset_id, None)
            reasons.pop(dataset_id, None)
            dropped += 1
    return dropped


def _get_channel_weight_baseline(state) -> Dict[str, float]:
    overrides = getattr(state, "channel_weights", None)
    weights = DEFAULT_CHANNEL_WEIGHTS.copy()
    if overrides:
        for channel, value in overrides.items():
            try:
                weights[channel] = float(value)
            except (TypeError, ValueError):
                continue
    return _enforce_channel_weight_order(weights)


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
        # Prefer enhanced version for tag indices (from enhance_tags.py)
        if name in {"tag_to_items", "item_to_tags"}:
            data = _load_json_file(base_dir / f"{name}_enhanced.json")
            if data:
                LOGGER.info("Using enhanced %s index", name)
            else:
                data = _load_json_file(base_dir / f"{name}.json")
        else:
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

    # Load 12-category indices (clean categories from enhance_tags.py)
    category_to_items = _load_json_file(base_dir / "category_to_items.json")
    if category_to_items:
        # Convert lists to sets for faster lookup
        recall_assets["category_to_items"] = {
            key: {int(v) for v in value} for key, value in category_to_items.items()
        }
        LOGGER.info("Loaded category_to_items with %d categories", len(category_to_items))

    item_to_categories = _load_json_file(base_dir / "item_to_categories.json")
    if item_to_categories:
        recall_assets["item_to_categories"] = {
            int(k): v for k, v in item_to_categories.items()
        }
        LOGGER.info("Loaded item_to_categories with %d items", len(item_to_categories))

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


def _apply_list_quotas(
    ranked_ids: List[int],
    reasons: Dict[int, str],
    *,
    limit: int,
    target_dataset_id: Optional[int] = None,
    item_to_categories: Optional[Dict[Any, Any]] = None,
    max_popular_share: float = MAX_POPULAR_SHARE,
    max_other_category_share: float = MAX_OTHER_CATEGORY_SHARE,
) -> List[int]:
    """对最终推荐列表施加"来源渠道 + 行业类别"配额。

    背景（实测）：平台上的低价热销商品（例如鼠标指针皮肤）长期占据热门榜前列，
    在"看了又看"里会造成与当前数据集完全无关的商品刷屏——实测 12 条里 8~10 条
    都是鼠标指针，同一批商品出现在两个毫不相关的数据集页面上。业务规则：
      1) popular（全局热门）来源最多占 limit × max_popular_share（12 条 → 3 条）；
      2) 除"目标数据集自身所属行业"之外的单一行业类别，最多占
         limit × max_other_category_share（12 条 → 3 条）；目标自身行业不受限
         （例如鼠标指针详情页返回多个鼠标指针是合理的）；
      3) 没有类别的候选按"未分类"单独计数，避免未分类商品刷屏；
      4) 为保证返回条数不减少，被配额挡下的候选会在末尾按原顺序补齐。

    Args:
        ranked_ids: 已按 MMR/分数排好序的候选 ID（建议长度 > limit，便于补位）
        reasons: 数据集 ID → 召回来源（用于识别 popular 渠道）
        limit: 最终需要返回的条数
        target_dataset_id: 当前页面数据集 ID（其所属行业豁免类别配额）
        item_to_categories: 数据集 → 行业类别列表
        max_popular_share: popular 渠道占比上限
        max_other_category_share: 非目标行业单一类别占比上限

    Returns:
        选中的数据集 ID 列表（长度尽量为 limit）
    """
    if limit <= 0 or not ranked_ids:
        return ranked_ids[: max(limit, 0)]

    max_popular = max(1, int(limit * max(0.0, min(max_popular_share, 1.0)) + 0.5))
    max_other_category = max(1, int(limit * max(0.0, min(max_other_category_share, 1.0)) + 0.5))

    target_categories: Set[str] = set()
    if item_to_categories and target_dataset_id is not None:
        target_categories = _as_category_set(
            item_to_categories.get(target_dataset_id, item_to_categories.get(str(target_dataset_id)))
        )

    selected: List[int] = []
    blocked: List[int] = []
    popular_count = 0
    category_counts: Dict[str, int] = {}

    for dataset_id in ranked_ids:
        if len(selected) >= limit:
            break

        channel = _extract_channel_from_reason(reasons.get(dataset_id))
        is_popular = channel in {"popular", "fallback"}

        candidate_categories: Set[str] = set()
        if item_to_categories:
            candidate_categories = _as_category_set(item_to_categories.get(dataset_id))
        if candidate_categories and target_categories and (candidate_categories & target_categories):
            # 命中"目标自身行业"：豁免类别配额（例如鼠标指针详情页返回多个鼠标指针是合理的）
            bucket_categories: Set[str] = set()
        elif candidate_categories:
            bucket_categories = candidate_categories - target_categories
        else:
            # 完全没有类别信息：单独用"未分类"桶计数，避免未分类商品刷屏
            bucket_categories = {UNCATEGORIZED_BUCKET}

        if is_popular and popular_count >= max_popular:
            blocked.append(dataset_id)
            continue
        if bucket_categories and any(
            category_counts.get(name, 0) >= max_other_category for name in bucket_categories
        ):
            blocked.append(dataset_id)
            continue

        selected.append(dataset_id)
        if is_popular:
            popular_count += 1
        for name in bucket_categories:
            category_counts[name] = category_counts.get(name, 0) + 1

    if len(selected) < limit and blocked:
        need = limit - len(selected)
        # 放行顺序：先放行"非热门来源"（内容/语义/标签/价格/行为等），热门来源排到最后。
        # 原因（实测）：若直接按原排名顺序放行，凑够条数的往往正是排名靠前的低价热销品，
        # 会把刚被配额挡下的鼠标指针类商品又填回列表，使 popular 上限形同虚设
        # （实测 12 条里出现 4 条 popular，超过 max_popular=3）。
        non_popular_blocked = [
            dataset_id
            for dataset_id in blocked
            if _extract_channel_from_reason(reasons.get(dataset_id)) not in {"popular", "fallback"}
        ]
        non_popular_set = set(non_popular_blocked)
        popular_blocked = [dataset_id for dataset_id in blocked if dataset_id not in non_popular_set]
        relaxed = (non_popular_blocked + popular_blocked)[:need]
        selected.extend(relaxed)
        LOGGER.info(
            "List quota relaxed: 候选不足，放行 %d 条以保持 limit=%d（其中非热门来源 %d 条）",
            len(relaxed),
            limit,
            sum(1 for dataset_id in relaxed if dataset_id in non_popular_set),
        )
    if blocked:
        LOGGER.info(
            "List quota applied (limit=%d, max_popular=%d, max_other_category=%d): "
            "blocked=%d, selected=%d, popular_in_result=%d",
            limit,
            max_popular,
            max_other_category,
            len(blocked),
            len(selected),
            sum(
                1
                for dataset_id in selected
                if _extract_channel_from_reason(reasons.get(dataset_id)) in {"popular", "fallback"}
            ),
        )
    return selected


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
    ranking_scores: Optional[Dict[int, float]] = None,
    target_dataset_id: Optional[int] = None,
    item_to_categories: Optional[Dict[Any, Any]] = None,
    max_popular_share: float = MAX_POPULAR_SHARE,
    max_other_category_share: float = MAX_OTHER_CATEGORY_SHARE,
) -> List[RecommendationItem]:
    """
    Build response items with optional MMR reranking and exploration.

    Args:
        candidate_scores: Dataset ID to score mapping (for MMR ranking)
        reasons: Dataset ID to reason mapping
        limit: Maximum number of items to return
        metadata: Dataset metadata
        dataset_tags: Dataset tags for MMR (optional)
        apply_mmr: Whether to apply MMR reranking
        mmr_lambda: MMR lambda parameter (relevance vs diversity trade-off)
        apply_exploration: Whether to apply epsilon-greedy exploration
        exploration_epsilon: Exploration rate (0.0-1.0)
        all_dataset_ids: All available dataset IDs for exploration pool
        ranking_scores: Optional ranking model scores for display (used instead of candidate_scores)
        target_dataset_id: 当前页面数据集 ID（用于列表配额，判定"目标自身行业"）
        item_to_categories: 数据集 → 行业类别列表（用于列表配额）
        max_popular_share: popular 来源占比上限（默认 25%，12 条 → 3 条）
        max_other_category_share: 非目标行业单一类别占比上限（默认 25%）

    Returns:
        List of recommendation items
    """
    if not candidate_scores:
        return []

    # 先取出比 limit 更多的候选（供列表配额补位使用）
    pool_limit = max(limit, limit * 3)

    # Apply MMR reranking if enabled and tags available
    if apply_mmr and dataset_tags:
        ranked_ids = _apply_mmr_reranking(
            candidate_scores,
            dataset_tags,
            lambda_param=mmr_lambda,
            limit=pool_limit,
        )
    else:
        # Fallback to score-based ranking
        ranked_ids = [
            dataset_id for dataset_id, _ in
            sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)
        ][:pool_limit]

    # 业务配额：限制 popular 渠道占比，以及"非目标行业"的单一类别占比
    ranked_ids = _apply_list_quotas(
        ranked_ids,
        reasons,
        limit=limit,
        target_dataset_id=target_dataset_id,
        item_to_categories=item_to_categories,
        max_popular_share=max_popular_share,
        max_other_category_share=max_other_category_share,
    )

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
        # Use ranking score for display if available, otherwise use candidate score
        if ranking_scores and dataset_id in ranking_scores:
            score = ranking_scores[dataset_id]
        else:
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


def _augment_weight(name: str, weights: Optional[Dict[str, float]] = None) -> float:
    """读取辅助渠道的加分系数。

    优先取 weights 里的 "augment_<渠道>"（可由 config/experiments.yaml 的
    augment_<渠道>_weight 参数注入），否则回退 DEFAULT_AUGMENT_WEIGHTS。
    单独的 augment_ 前缀可以避免被 models/channel_weights.json（CTR 训练结果）
    意外覆盖——两者的量纲本来就不同。
    """
    if weights:
        override = weights.get(f"augment_{name}")
        if override is not None:
            try:
                return float(override)
            except (TypeError, ValueError):
                LOGGER.warning("辅助渠道系数 augment_%s=%r 非法，使用默认值", name, override)
    return float(DEFAULT_AUGMENT_WEIGHTS.get(name, 0.0))


def _augment_with_multi_channel(
    state,
    *,
    target_id: int,
    scores: Dict[int, float],
    reasons: Dict[int, str],
    limit: int,
    user_id: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
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
                candidate_tags = set(tag.lower() for tag in state.dataset_tags.get(candidate, []) if tag)
                overlap = len(target_set & candidate_tags)
                if overlap:
                    candidate_scores[int(candidate)] = candidate_scores.get(int(candidate), 0.0) + overlap

        # 归一化tag分数到[0, 1]，然后乘以权重
        if candidate_scores:
            normalized_tag_scores = _normalize_channel_scores(candidate_scores)
            for dataset_id, norm_score in sorted(normalized_tag_scores.items(), key=lambda x: x[1], reverse=True)[: limit * 2]:
                _bump(int(dataset_id), norm_score * _augment_weight("tag", weights), "tag")

    # 12-category recall (standard categories from zero-shot classification)
    item_to_categories = recall.get("item_to_categories", {})
    category_to_items = recall.get("category_to_items", {})
    target_categories = item_to_categories.get(target_id, [])
    if target_categories and category_to_items:
        category_candidate_scores: Dict[int, float] = {}
        target_cat_set = set(target_categories)
        for cat in target_categories:
            for candidate in category_to_items.get(cat, set()):
                if candidate == target_id:
                    continue
                candidate_cats = set(item_to_categories.get(candidate, []))
                overlap = len(target_cat_set & candidate_cats)
                if overlap:
                    category_candidate_scores[int(candidate)] = category_candidate_scores.get(int(candidate), 0.0) + overlap

        # 归一化并加入候选
        if category_candidate_scores:
            normalized_cat_scores = _normalize_channel_scores(category_candidate_scores)
            for dataset_id, norm_score in sorted(normalized_cat_scores.items(), key=lambda x: x[1], reverse=True)[: limit * 2]:
                _bump(int(dataset_id), norm_score * _augment_weight("12cat", weights), "12cat")

    # Company category recall (legacy)
    category_index = recall.get("category_index", {})
    company = state.metadata.get(target_id, {}).get("company")
    if company:
        candidates = category_index.get(str(company).lower(), set())
        for dataset_id in list(candidates)[: limit * 2]:
            if dataset_id == target_id:
                continue
            _bump(int(dataset_id), _augment_weight("category", weights), "category")

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
            _bump(int(dataset_id), _augment_weight("price", weights), "price")

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
                _bump(int(dataset_id), float(score) * _augment_weight("usercf", weights), "usercf")


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
    *,
    dataset_features: Optional[pd.DataFrame] = None,
) -> tuple[Dict[int, float], Dict[int, str]]:
    """Combine scores from multiple recall channels with normalization.

    Each channel is independently normalized to [0, 1] before applying weights,
    ensuring fair competition across channels with different score magnitudes.

    Popular recall includes quality filtering (added 2025-12-27) to remove low-quality items:
    - Low price + low interaction: price < 1.90 AND interaction_count < 66
    - Long inactive + low interaction: days_inactive > 180 AND interaction_count < 30

    Args:
        target_id: Target dataset ID for recall
        behavior: Behavior-based recall results
        content: Content-based recall results
        vector: Vector-based recall results
        popular: Popular items fallback (global hot list)
        limit: Result limit
        weights: Channel weights
        dataset_features: Optional DataFrame (indexed by dataset_id) for Popular quality filtering.
            Must contain columns: ['price', 'interaction_count', 'days_since_last_purchase']

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

    # ========== Popular召回（归一化 + 质量过滤） ==========
    # popular是列表，按排序给分，线性衰减
    popular_scores = {}
    popular_filtered_count = 0

    # 性能优化：批量预查询Popular item的features
    popular_features_batch = None
    if dataset_features is not None and not dataset_features.empty:
        try:
            popular_ids_in_features = [item_id for item_id in popular if item_id in dataset_features.index]
            if popular_ids_in_features:
                popular_features_batch = dataset_features.loc[
                    popular_ids_in_features,
                    ['price', 'interaction_count', 'days_since_last_purchase']
                ]
        except (KeyError, ValueError):
            popular_features_batch = None

    for idx, item_id in enumerate(popular):
        if item_id == target_id or item_id in scores:
            continue

        # 质量过滤：过滤低质量Popular item
        if popular_features_batch is not None and item_id in popular_features_batch.index:
            try:
                features = popular_features_batch.loc[item_id]
                price = float(features['price'])
                interaction_count = int(features['interaction_count'])
                days_inactive = float(features['days_since_last_purchase'])

                # 过滤规则：组合低质量信号（更精准的AND逻辑）
                # 1. 低价且无人气：price < 1.90 AND interaction < 66
                # 2. 长期不活跃且交互少：days_inactive > 180 AND interaction < 30
                low_price_low_interaction = (price < 1.90 and interaction_count < 66)
                inactive_low_interaction = (days_inactive > 180 and interaction_count < 30)

                if low_price_low_interaction or inactive_low_interaction:
                    popular_filtered_count += 1
                    continue
            except (KeyError, ValueError, TypeError):
                # 数据缺失或类型错误，跳过过滤（保留item）
                pass

        # 线性衰减：第1个=1.0, 最后一个=0.1
        popular_scores[item_id] = 1.0 - (idx / max(len(popular), 1)) * 0.9
        if len(popular_scores) >= limit * 5:
            break

    # 记录过滤统计
    if popular_filtered_count > 0:
        LOGGER.info(
            f"Popular recall quality filter: filtered {popular_filtered_count} low-quality items, "
            f"kept {len(popular_scores)} items"
        )

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
        """把 source 渠道的一部分权重转移给 targets。

        转移量按各 target 当前权重的占比分配，而不是等额均分：
        等额均分会让本来权重极低的渠道白捡一个绝对量（历史上 popular 会因此
        从配置的 0.02 被抬到 0.22），导致实际生效权重与配置表严重不符。
        占比为 0 的目标渠道不参与分配，权重不会凭空消失。
        """
        current = adjusted.get(source, 0.0)
        if current <= 0 or not targets or fraction <= 0:
            return
        amount = current * min(fraction, 1.0)
        adjusted[source] = max(current - amount, 0.0)

        weighted_targets = [t for t in targets if adjusted.get(t, 0.0) > 0]
        total = sum(adjusted[t] for t in weighted_targets)
        if not weighted_targets or total <= 0:
            # 目标渠道权重全为 0：退化为等额分配，避免这部分权重无处可去
            share = amount / len(targets)
            for target in targets:
                adjusted[target] = max(adjusted.get(target, 0.0) + share, 0.0)
            return

        for target in weighted_targets:
            share = amount * adjusted[target] / total
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


def _as_category_set(value: Any) -> Set[str]:
    """把 item_to_categories.json 的取值统一成类别集合（兼容 list/tuple/set/字符串）。"""
    if value is None:
        return set()
    if isinstance(value, str):
        return {token.strip() for token in value.replace(";", ",").split(",") if token.strip()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(token).strip() for token in value if str(token).strip()}
    return set()


@lru_cache(maxsize=8192)
def _split_tags_cached(raw_text: str) -> Tuple[str, ...]:
    """标签串 → 小写标签元组（带缓存）。

    候选之间大量重复标签串，缓存可以避免在每次请求里反复 split；
    缓存只按原始字符串做键，命中率高、无副作用。
    """
    return tuple(_parse_tags(raw_text))


def _tag_set(value: Any) -> Set[str]:
    """把分号分隔的标签串解析成小写集合。

    口径与训练侧 pipeline/train_models.py 的 _parse_tags 保持一致：
    None / NaN / 空串一律视为"无标签"，避免把 NaN 解析成字符串 "nan"
    造成两个都没有标签的数据集被误判为"标签相同"。
    """
    if value is None:
        return set()
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return set()
        except (TypeError, ValueError):
            pass
        value = str(value)
    return set(_split_tags_cached(value))


def _compute_relevance_features(
    dataset_ids: List[int],
    raw_indexed: pd.DataFrame,
    target_dataset_id: Optional[int],
    item_to_categories: Optional[Dict[Any, Any]] = None,
) -> pd.DataFrame:
    """计算"目标数据集 ↔ 候选数据集"之间的相关度特征。

    产出 5 列（与 pipeline/train_models.py 训练时使用的特征同名同口径）：
      tag_overlap_count        标签交集个数
      tag_jaccard_similarity   标签 Jaccard 相似度
      same_top_category        是否同属一个行业类别（0/1）
      category_overlap         13 类行业类别交集个数
      category_match           是否有共同的行业类别（0/1）

    这些特征依赖"当前请求的目标数据集"，因此必须在每次请求时按目标重算。
    历史上只有启动阶段算过一次（target 为 None，全部填 0），导致排序模型训练
    时见过的类别/标签相关度信号在推理阶段恒为 0（训练/推理不一致）。

    Args:
        dataset_ids: 候选数据集 ID 列表
        raw_indexed: 以 dataset_id 为索引的原始特征（需包含 tag 列）
        target_dataset_id: 目标（当前页面）数据集 ID，None 表示无目标上下文
        item_to_categories: 数据集 → 行业类别列表（来自 models/item_to_categories.json）

    Returns:
        DataFrame，index 为 dataset_ids，列为上述 5 个特征
    """
    columns = [
        "tag_overlap_count",
        "tag_jaccard_similarity",
        "same_top_category",
        "category_overlap",
        "category_match",
    ]
    if not dataset_ids:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(0.0, index=pd.Index(dataset_ids, name="dataset_id"), columns=columns)

    indexed = _ensure_dataset_index(raw_indexed)
    tag_lookup: Dict[int, Any] = {}
    if not indexed.empty and "tag" in indexed.columns:
        tag_lookup = indexed["tag"].to_dict()

    target_tags: Set[str] = set()
    if target_dataset_id is not None:
        target_tags = _tag_set(tag_lookup.get(target_dataset_id, ""))
    target_top_categories = target_tags & TOP_CATEGORY_TOKENS

    target_categories: Set[str] = set()
    if item_to_categories and target_dataset_id is not None:
        target_categories = _as_category_set(
            item_to_categories.get(target_dataset_id, item_to_categories.get(str(target_dataset_id)))
        )

    tag_overlap_count: List[float] = []
    tag_jaccard: List[float] = []
    same_top_category: List[float] = []
    category_overlap: List[float] = []
    category_match: List[float] = []

    for dataset_id in dataset_ids:
        candidate_tags = _tag_set(tag_lookup.get(dataset_id, ""))
        if target_tags and candidate_tags:
            intersection = len(target_tags & candidate_tags)
            union = len(target_tags | candidate_tags)
            tag_overlap_count.append(float(intersection))
            tag_jaccard.append(float(intersection) / union if union > 0 else 0.0)
        else:
            tag_overlap_count.append(0.0)
            tag_jaccard.append(0.0)

        same_top_category.append(
            1.0 if (target_top_categories & (candidate_tags & TOP_CATEGORY_TOKENS)) else 0.0
        )

        if item_to_categories:
            candidate_categories = _as_category_set(
                item_to_categories.get(dataset_id, item_to_categories.get(str(dataset_id)))
            )
            overlap = len(target_categories & candidate_categories)
            category_overlap.append(float(overlap))
            category_match.append(1.0 if overlap > 0 else 0.0)
        else:
            category_overlap.append(0.0)
            category_match.append(0.0)

    frame["tag_overlap_count"] = tag_overlap_count
    frame["tag_jaccard_similarity"] = tag_jaccard
    frame["same_top_category"] = same_top_category
    frame["category_overlap"] = category_overlap
    frame["category_match"] = category_match
    return frame


def _build_static_ranking_features(
    dataset_ids: List[int],
    raw_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    slot_metrics_aggregated: pd.DataFrame,
    feature_overrides: Optional[pd.DataFrame] = None,
    stats_overrides: Optional[pd.DataFrame] = None,
    target_dataset_id: Optional[int] = None,
    item_to_categories: Optional[Dict[Any, Any]] = None,
) -> pd.DataFrame:
    """Compute dataset-level static ranking features.

    Args:
        target_dataset_id: The target/page dataset ID for computing category relevance features.
        item_to_categories: 数据集 → 行业类别列表，用于 category_overlap / category_match 特征。
    """
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

    # === 相关度特征（标签重叠 + 行业类别重叠）===
    # 由 _compute_relevance_features 统一计算：这里只在"启动预计算/补齐缺失数据集"
    # 时算一次（无目标数据集时全为 0），请求内的实际取值由
    # _compute_ranking_features 按当前目标数据集重算。
    relevance = _compute_relevance_features(
        list(features.index),
        raw_indexed,
        target_dataset_id,
        item_to_categories,
    )
    if not relevance.empty:
        for column in relevance.columns:
            features[column] = relevance[column].reindex(features.index).fillna(0.0).astype(float)

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
    target_dataset_id: Optional[int] = None,
    item_to_categories: Optional[Dict[Any, Any]] = None,
) -> pd.DataFrame:
    if not dataset_ids:
        return pd.DataFrame()

    # Extract target_dataset_id from request_context if not provided
    if target_dataset_id is None and request_context:
        try:
            target_dataset_id = int(request_context.get("target_dataset_id", 0)) or None
        except (TypeError, ValueError):
            target_dataset_id = None

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
                target_dataset_id=target_dataset_id,
                item_to_categories=item_to_categories,
            )
            features.loc[missing_ids] = filled
    else:
        features = _build_static_ranking_features(
            dataset_ids,
            indexed_raw,
            indexed_stats,
            indexed_slot,
            target_dataset_id=target_dataset_id,
            item_to_categories=item_to_categories,
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
            target_dataset_id=target_dataset_id,
            item_to_categories=item_to_categories,
        )
        if features.empty:
            features = refreshed
        else:
            features.loc[refreshed.index] = refreshed

    # Add freshness features from dataset metadata
    if "days_since_created" in indexed_raw.columns:
        days_map = indexed_raw["days_since_created"].to_dict()
        features["content_age_days"] = features.index.to_series().map(
            lambda dataset_id: float(days_map.get(dataset_id, 999))
        ).fillna(999.0).astype(float)

        def _compute_freshness_score(days: float) -> float:
            """Compute freshness score: 1.0 for <=7 days, 0.5 for <=30 days, 0.2 for >30 days."""
            if days <= 7:
                return 1.0
            elif days <= 30:
                return 0.5
            else:
                return 0.2

        features["freshness_score"] = features["content_age_days"].map(_compute_freshness_score).astype(float)
    else:
        # Fallback if field not available
        features["content_age_days"] = 999.0
        features["freshness_score"] = 0.2

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

    # === 按当前目标数据集重算"请求相关"特征 ===
    # tag_overlap_count / tag_jaccard_similarity / same_top_category /
    # category_overlap / category_match 都取决于"当前页面的目标数据集"，
    # 而 precomputed_static 是启动时按 target=None 预计算的（这些列恒为 0）。
    # 若不在这里重算，排序模型训练时学到的相关度信号在推理阶段会全部为 0。
    if target_dataset_id is not None:
        relevance = _compute_relevance_features(
            list(features.index),
            indexed_raw,
            target_dataset_id,
            item_to_categories,
        )
        if not relevance.empty:
            for column in relevance.columns:
                features[column] = relevance[column].reindex(features.index).fillna(0.0).astype(float)

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

    # Prepare freshness boost lookup
    freshness_boost_lookup = {}
    if "freshness_score" in features.columns:
        for dataset_id in features.index:
            freshness_val = features.loc[dataset_id, "freshness_score"]
            # Boost range: [0.8, 1.0] based on freshness_score [0.2, 1.0]
            freshness_boost_lookup[int(dataset_id)] = 0.8 + 0.2 * float(freshness_val)

    for dataset_id, prob in zip(features.index.astype(int), probabilities.values):
        if dataset_id not in scores:
            continue
        prob = float(prob)

        # Apply freshness boost if available
        if dataset_id in freshness_boost_lookup:
            freshness_boost = freshness_boost_lookup[dataset_id]
            scores[dataset_id] += prob * freshness_boost
        else:
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
    target_dataset_id: Optional[int] = None,
) -> Dict[int, float]:
    """Apply LightGBM ranker to score candidates and filter low-quality items.

    LightGBM ranker outputs raw prediction scores in range (-∞, +∞). Negative scores
    indicate items predicted to be below average quality.

    Score filtering strategy (updated 2025-12-29):
    - Percentile-based: Remove bottom 30% of candidates by score
    - Adaptive: Works regardless of score distribution (handles negative-heavy distributions)
    - Minimum candidates: Only filter if >= 10 candidates available
    - Safety fallback: Keep top 50% if percentile filter is too aggressive
    - Logging: Record filtering stats with dynamic threshold value

    Args:
        scores: Mutable dict of candidate scores (updated in-place)
        reasons: Mutable dict of recall reasons (updated in-place)
        rank_model: LightGBM ranker model or Pipeline
        raw_features: Raw features DataFrame
        dataset_stats: Dataset statistics DataFrame
        slot_metrics_aggregated: Slot performance metrics
        feature_store: Optional Redis feature store
        endpoint: Request endpoint name
        variant: Experiment variant
        experiment_variant: Optional experiment variant override
        request_context: Optional request context dict
        channel_weights: Channel weight configuration
        user_features: Optional user-level features
        target_dataset_id: Target dataset ID for category relevance features

    Returns:
        Dict of ranking scores (before filtering) for display purposes

    Side Effects:
        - Modifies scores dict in-place (adds ranker predictions, removes low scores)
        - Removes low score items from scores and reasons dicts
        - Logs low score filtering statistics
    """
    if rank_model is None or not scores:
        return {}

    try:
        current_state = _get_app_state()
    except RuntimeError:
        current_state = None

    dataset_ids = list(scores.keys())
    # 行业类别索引（models/item_to_categories.json）：用于 category_overlap /
    # category_match 排序特征，必须与训练侧口径一致
    recall_indices = getattr(current_state, "recall_indices", None) or {}
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
        target_dataset_id=target_dataset_id,
        item_to_categories=recall_indices.get("item_to_categories"),
    )
    if features.empty:
        return {}

    try:
        _apply_ranking_with_circuit_breaker(scores, reasons, rank_model, features)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Ranking model failed (circuit breaker or error): %s", exc)
        # Continue without ranking - scores already populated from recall

    # 保存ranking分数副本（用于响应展示）
    ranking_scores = scores.copy()

    # 百分位过滤：过滤掉ranking分数最低的30%候选
    # 使用百分位数而非固定阈值，确保无论分数分布如何都能保留足够的候选
    FILTER_PERCENTILE = 30  # 过滤最差的30%，保留最好的70%
    MIN_CANDIDATES_FOR_FILTER = 10  # 至少10个候选才启用过滤

    total_items = len(scores)
    low_score_items = []

    if total_items >= MIN_CANDIDATES_FOR_FILTER:
        score_values = list(scores.values())
        # 计算30%分位点作为过滤阈值
        threshold = np.percentile(score_values, FILTER_PERCENTILE)
        low_score_items = [item_id for item_id, score in scores.items() if score < threshold]

        if low_score_items:
            low_score_ratio = len(low_score_items) / total_items * 100 if total_items > 0 else 0

            # 安全保护：确保至少保留50%候选
            if len(low_score_items) > total_items * 0.5:
                LOGGER.warning(
                    f"Percentile filter would remove {len(low_score_items)}/{total_items} items ({low_score_ratio:.1f}%), "
                    f"keeping top 50% as safety fallback"
                )
                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                keep_count = max(5, total_items // 2)
                keep_ids = {item_id for item_id, _ in sorted_items[:keep_count]}
                low_score_items = [item_id for item_id in low_score_items if item_id not in keep_ids]

            if low_score_items:
                LOGGER.info(
                    f"Percentile filter (p{FILTER_PERCENTILE}, threshold={threshold:.3f}): "
                    f"removing {len(low_score_items)}/{total_items} items ({low_score_ratio:.1f}%) "
                    f"with low scores (sample: {low_score_items[:5]})"
                )
                for item_id in low_score_items:
                    scores.pop(item_id, None)
                    reasons.pop(item_id, None)
    else:
        LOGGER.debug(f"Skipping score filter: only {total_items} candidates (minimum {MIN_CANDIDATES_FOR_FILTER} required)")

    return ranking_scores


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
    channel_weights: Optional[Dict[str, float]] = None,
) -> None:
    exposure_items = [
        {
            "dataset_id": item.dataset_id,
            "score": item.score,
            "reason": reasons.get(item.dataset_id, item.reason),
        }
        for item in items
    ]
    context: Dict[str, Any] = {"endpoint": event, "variant": variant}
    if request_context:
        for key, value in request_context.items():
            if value not in (None, ""):
                context[key] = value
    if degrade_reason:
        context["degrade_reason"] = degrade_reason
    if experiment_variant:
        context["experiment_variant"] = experiment_variant
    # 记录本次请求实际生效的渠道权重：下游 pipeline/evaluate_v2 会把它读成
    # channel_weights 列，build_training_labels 用它生成训练特征 channel_weight。
    # 之前这里从不写入，导致排序模型看到的 channel_weight 恒为固定常量表，
    # 与线上真实权重脱钩（改动权重时会产生训练/推理不一致）。
    if channel_weights:
        try:
            context["channel_weights"] = {
                str(key): float(value) for key, value in dict(channel_weights).items()
            }
        except (TypeError, ValueError):
            LOGGER.warning("channel_weights 无法序列化到曝光日志，已跳过: %r", channel_weights)
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

    # 不可售/已下架数据集过滤（业务硬约束）：从内存索引里彻底剔除，
    # 保证既不会被召回，也不会进入探索池
    excluded_ids = _load_excluded_dataset_ids()
    app.state.excluded_dataset_ids = excluded_ids
    if excluded_ids:
        pruned = _prune_unavailable_datasets(
            bundle=bundle,
            recall_indices=app.state.recall_indices,
            metadata=metadata,
            dataset_tags=dataset_tags,
            excluded=excluded_ids,
        )
        LOGGER.info(
            "Pruned unavailable datasets: %d ids, %d index references removed (sample=%s)",
            len(excluded_ids),
            pruned,
            sorted(excluded_ids)[:10],
        )
    else:
        LOGGER.info("No excluded dataset ids configured (RECO_EXCLUDED_DATASET_IDS / models/excluded_dataset_ids.json)")

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
                dataset_features=getattr(state, "raw_features_indexed", None),
            )
            _augment_with_multi_channel(
                state,
                target_id=dataset_id,
                scores=scores,
                reasons=reasons,
                limit=limit,
                weights=effective_weights,
            )
            _drop_excluded_from_scores(
                scores, reasons, getattr(state, "excluded_dataset_ids", None) or set()
            )
            user_feature_map: Dict[str, float] = {}
            ranking_scores = await _call_blocking(
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
                    target_dataset_id=dataset_id,
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
                ranking_scores=ranking_scores,  # 传递ranking分数用于展示
                target_dataset_id=dataset_id,
                item_to_categories=(getattr(state, "recall_indices", None) or {}).get("item_to_categories"),
            )
            return items, reasons, local_variant, local_bundle.run_id, effective_weights

        compute_started = time.perf_counter()
        try:
            items, reasons, variant, run_id, applied_channel_weights = await asyncio.wait_for(
                _compute(),
                timeout=TimeoutManager.get_timeout("recommendation_total"),
            )
            compute_duration = time.perf_counter() - compute_started
            # 注意：/similar 接口没有 user_id 上下文（历史上固定为 None），
            # 此处不能引用 user_id，否则抛 NameError 会被兜底逻辑接住 → 全部降级成 fallback:popular
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
            channel_weights=applied_channel_weights,
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
            # 缓存key加入时间桶，每小时刷新
            time_bucket = _get_time_bucket(bucket_hours=1)
            cache_key = f"recommend:{dataset_id}:{user_id}:{limit}:{time_bucket}"
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
                dataset_features=getattr(state, "raw_features_indexed", None),
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
                weights=effective_weights,
            )
            _drop_excluded_from_scores(
                scores, reasons, getattr(state, "excluded_dataset_ids", None) or set()
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
            ranking_scores = await _call_blocking(
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
                    target_dataset_id=dataset_id,
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
                exploration_epsilon=EXPLORATION_EPSILON,  # 默认 0.10，可用环境变量 EXPLORATION_EPSILON 调整
                all_dataset_ids=set(state.metadata.keys()),  # 全量dataset池
                ranking_scores=ranking_scores,  # 传递ranking分数用于展示
                target_dataset_id=dataset_id,
                item_to_categories=(getattr(state, "recall_indices", None) or {}).get("item_to_categories"),
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
            channel_weights=applied_channel_weights,
        )

        if cache and cache.enabled and user_id and degrade_reason is None:
            # 缓存key加入时间桶，每小时刷新
            time_bucket = _get_time_bucket(bucket_hours=1)
            cache_key = f"recommend:{dataset_id}:{user_id}:{limit}:{time_bucket}"
            try:
                await _call_blocking(
                    cache.set_json,
                    cache_key,
                    response.dict(),
                    ttl=3600,  # TTL从180秒改为3600秒（1小时），与时间桶对齐
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
