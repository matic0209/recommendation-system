"""Build feature datasets required for the recommendation models."""
from __future__ import annotations

import logging
import sqlite3
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from config.settings import BASE_DIR, DATA_DIR, FEATURE_STORE_PATH, DATASET_IMAGE_ROOT
from pipeline.data_cleaner import DataCleaner
from pipeline.build_features_v2 import FeatureEngineV2
from pipeline.feature_store_redis import RedisFeatureStore

# Optional: Visual image features (requires sentence-transformers)
try:
    from pipeline.image_features_visual import VisualImageFeatureExtractor, VisualEmbeddingConfig
    VISUAL_FEATURES_AVAILABLE = True
except ImportError:
    VISUAL_FEATURES_AVAILABLE = False
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Visual image features not available (sentence-transformers not installed)")

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_STORE_TABLES: Dict[str, tuple[str, ...]] = {
    "interactions": ("user_id", "dataset_id"),
    "dataset_features": ("dataset_id",),
    "user_profile": ("user_id",),
    "dataset_stats": ("dataset_id",),
}


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Missing input file: %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_interactions_from_frame(
    frame: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    price_col: str,
    time_col: str,
    source: str,
    column_mapping: Dict[str, str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["user_id", "dataset_id", "weight", "source", "last_event_time"])

    working = frame.copy()
    if column_mapping:
        working = working.rename(columns=column_mapping)

    required_cols = {user_col, item_col, price_col, time_col}
    missing = [col for col in required_cols if col not in working.columns]
    if missing:
        LOGGER.warning("%s table missing columns: %s", source, ", ".join(missing))
        return pd.DataFrame(columns=["user_id", "dataset_id", "weight", "source", "last_event_time"])

    working = working.dropna(subset=[user_col, item_col])
    working[price_col] = pd.to_numeric(working[price_col], errors="coerce").fillna(0)
    working[time_col] = pd.to_datetime(working[time_col], errors="coerce")
    working["weight"] = np.log1p(working[price_col].clip(lower=0))
    working["source"] = source
    working = working.rename(columns={user_col: "user_id", item_col: "dataset_id", time_col: "last_event_time"})

    aggregated = (
        working.groupby(["user_id", "dataset_id", "source"], as_index=False)
        .agg({"weight": "sum", "last_event_time": "max"})
    )
    return aggregated


def build_interactions(inputs: Dict[str, Path]) -> pd.DataFrame:
    task_path = inputs.get("task") or inputs.get("order_tab")
    api_order_path = inputs.get("api_order")

    interactions = []
    if task_path is not None:
        task_df = _load_parquet(task_path)
        interactions.append(
            _build_interactions_from_frame(
                task_df,
                user_col="create_user",
                item_col="dataset_id",
                price_col="price",
                time_col="create_time",
                source="task",
            )
        )
    if api_order_path is not None:
        api_df = _load_parquet(api_order_path)
        interactions.append(
            _build_interactions_from_frame(
                api_df,
                user_col="creator_id",
                item_col="dataset_id",
                price_col="price",
                time_col="create_time",
                source="api_order",
                column_mapping={"api_id": "dataset_id"},
            )
        )

    interactions = [df for df in interactions if not df.empty]
    if interactions:
        result = pd.concat(interactions, ignore_index=True)
        result = result.groupby(["user_id", "dataset_id"], as_index=False).agg(
            weight=("weight", "sum"),
            last_event_time=("last_event_time", "max"),
        )
        return result
    return pd.DataFrame(columns=["user_id", "dataset_id", "weight", "last_event_time"])


def build_dataset_stats(interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions.empty:
        return pd.DataFrame(columns=["dataset_id", "interaction_count", "total_weight", "last_event_time"])

    stats = interactions.copy()
    if "last_event_time" in stats.columns:
        stats["last_event_time"] = pd.to_datetime(stats["last_event_time"], errors="coerce")

    aggregated = stats.groupby("dataset_id", as_index=False).agg(
        interaction_count=("dataset_id", "count"),
        total_weight=("weight", "sum"),
        last_event_time=("last_event_time", "max"),
    )
    aggregated["total_weight"] = aggregated["total_weight"].fillna(0.0)
    return aggregated


def build_dataset_features(dataset_path: Path) -> pd.DataFrame:
    dataset_df = _load_parquet(dataset_path)
    if dataset_df.empty:
        return pd.DataFrame(columns=["dataset_id", "dataset_name", "description", "tag", "price", "create_company_name", "cover_id", "create_time"])

    if "is_delete" in dataset_df.columns:
        original_count = len(dataset_df)
        flags = pd.to_numeric(dataset_df["is_delete"], errors="coerce").fillna(0).astype(int)
        dataset_df = dataset_df[flags == 0].copy()
        removed = original_count - len(dataset_df)
        if removed > 0:
            LOGGER.info("Filtered %d deleted datasets (is_delete=1)", removed)

    dataset_df = dataset_df.rename(columns={"id": "dataset_id"})
    columns = [
        "dataset_id",
        "dataset_name",
        "description",
        "tag",
        "price",
        "create_company_name",
        "cover_id",
        "create_time",
    ]
    existing = [col for col in columns if col in dataset_df.columns]
    result = dataset_df[existing].copy()
    for col in ["description", "tag", "create_company_name"]:
        if col in result.columns:
            result[col] = result[col].fillna("")
    if "price" in result.columns:
        result["price"] = pd.to_numeric(result["price"], errors="coerce").fillna(0)
    return result


def build_user_profile(user_path: Path) -> pd.DataFrame:
    user_df = _load_parquet(user_path)
    if user_df.empty:
        return pd.DataFrame(columns=["user_id", "company_name", "province", "city", "is_consumption"])

    user_df = user_df.rename(columns={"id": "user_id"})
    columns = ["user_id", "company_name", "province", "city", "is_consumption"]
    existing = [col for col in columns if col in user_df.columns]
    profile = user_df[existing].copy()
    for col in ["company_name", "province", "city"]:
        if col in profile.columns:
            profile[col] = profile[col].fillna("unknown")
    if "is_consumption" in profile.columns:
        profile["is_consumption"] = pd.to_numeric(profile["is_consumption"], errors="coerce").fillna(0).astype(int)
    return profile


def _sync_feature_store(tables: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    FEATURE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(FEATURE_STORE_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_metadata (
                view_name TEXT PRIMARY KEY,
                refreshed_at TEXT NOT NULL,
                row_count INTEGER NOT NULL
            )
            """
        )

        for name, frame in tables.items():
            if frame.empty:
                LOGGER.warning("Skipping feature store sync for '%s' (empty frame)", name)
                continue

            sanitized = frame.copy()
            if name in {"interactions", "dataset_stats"} and "last_event_time" in sanitized.columns:
                timestamps = pd.to_datetime(sanitized["last_event_time"], errors="coerce", utc=True)
                sanitized["last_event_time"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            sanitized.to_sql(name, conn, if_exists="replace", index=False)
            counts[name] = len(sanitized)

            for idx, key in enumerate(FEATURE_STORE_TABLES[name]):
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_{idx} ON {name} ({key})")

            conn.execute(
                """
                INSERT INTO feature_metadata(view_name, refreshed_at, row_count)
                VALUES(?, ?, ?)
                ON CONFLICT(view_name) DO UPDATE SET
                    refreshed_at=excluded.refreshed_at,
                    row_count=excluded.row_count
                """,
                (name, datetime.now(timezone.utc).isoformat(), len(sanitized)),
            )

    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ensure_output_dir(PROCESSED_DIR)

    business_dir = DATA_DIR / "business"

    interaction_inputs: Dict[str, Path] = {"api_order": business_dir / "api_order.parquet"}
    task_parquet = business_dir / "task.parquet"
    legacy_order_parquet = business_dir / "order_tab.parquet"
    if task_parquet.exists() or not legacy_order_parquet.exists():
        interaction_inputs["task"] = task_parquet
    else:
        interaction_inputs["order_tab"] = legacy_order_parquet

    interactions = build_interactions(interaction_inputs)
    dataset_features = build_dataset_features(business_dir / "dataset.parquet")

    active_dataset_ids = set(dataset_features["dataset_id"].dropna().astype(int)) if not dataset_features.empty else set()
    if active_dataset_ids:
        before = len(interactions)
        interactions = interactions[interactions["dataset_id"].isin(active_dataset_ids)].reset_index(drop=True)
        removed = before - len(interactions)
        if removed > 0:
            LOGGER.info("Dropped %d interactions referencing deleted datasets", removed)

    interactions_path = PROCESSED_DIR / "interactions.parquet"
    interactions.to_parquet(interactions_path, index=False)
    LOGGER.info("Saved interactions to %s", interactions_path)

    dataset_path = PROCESSED_DIR / "dataset_features.parquet"
    dataset_features.to_parquet(dataset_path, index=False)
    LOGGER.info("Saved dataset features to %s", dataset_path)

    user_profile = build_user_profile(business_dir / "user.parquet")
    user_path = PROCESSED_DIR / "user_profile.parquet"
    user_profile.to_parquet(user_path, index=False)
    LOGGER.info("Saved user profile to %s", user_path)

    dataset_stats = build_dataset_stats(interactions)
    stats_path = PROCESSED_DIR / "dataset_stats.parquet"
    dataset_stats.to_parquet(stats_path, index=False)
    LOGGER.info("Saved dataset stats to %s", stats_path)

    sync_counts = _sync_feature_store(
        {
            "interactions": interactions,
            "dataset_features": dataset_features,
            "user_profile": user_profile,
            "dataset_stats": dataset_stats,
        }
    )
    if sync_counts:
        summary = ", ".join(f"{name}={count}" for name, count in sync_counts.items())
        LOGGER.info("Synced feature store (%s)", summary)
    else:
        LOGGER.warning("Feature store sync skipped because all views were empty")

    # Phase 2 enhancements: cleaning + advanced features
    LOGGER.info("Running data cleaning and enhanced feature engineering (v2)...")
    cleaner = DataCleaner()
    cleaned = cleaner.clean_all()

    # Build enhanced features using cleaned data
    engine = FeatureEngineV2(use_cleaned=True)
    cleaned_interactions = cleaned.get("interactions", interactions)
    cleaned_dataset = cleaned.get("dataset_features", dataset_features)
    cleaned_user_profile = cleaned.get("user_profile", user_profile)

    # Recompute dataset stats from cleaned interactions
    cleaned_dataset_stats = build_dataset_stats(cleaned_interactions)

    # Load image data for enhanced features
    dataset_image_path = business_dir / "dataset_image.parquet"
    dataset_image = None
    image_embeddings = pd.DataFrame()
    embeddings_path = PROCESSED_DIR / "dataset_image_embeddings.parquet"
    if dataset_image_path.exists():
        LOGGER.info("Loading dataset images from %s...", dataset_image_path)
        dataset_image = pd.read_parquet(dataset_image_path)
        LOGGER.info("Loaded %d image records for feature engineering", len(dataset_image))

        if active_dataset_ids:
            before_imgs = len(dataset_image)
            dataset_image = dataset_image[dataset_image["dataset_id"].isin(active_dataset_ids)].reset_index(drop=True)
            trimmed = before_imgs - len(dataset_image)
            if trimmed > 0:
                LOGGER.info("Removed %d image rows for deleted datasets", trimmed)

        if embeddings_path.exists():
            LOGGER.info("Using precomputed visual embeddings from %s", embeddings_path)
            image_embeddings = pd.read_parquet(embeddings_path)
        elif VISUAL_FEATURES_AVAILABLE:
            try:
                visual_cache = BASE_DIR / "cache" / "images"
                visual_config = VisualEmbeddingConfig(local_image_root=DATASET_IMAGE_ROOT)
                visual_extractor = VisualImageFeatureExtractor(visual_cache, visual_config)
                image_embeddings = visual_extractor.build_dataset_embeddings(dataset_image)
                try:
                    image_embeddings.to_parquet(embeddings_path, index=False)
                    LOGGER.info(
                        "Saved dataset image embeddings to %s (rows=%d, dim=%d)",
                        embeddings_path,
                        len(image_embeddings),
                        len([c for c in image_embeddings.columns if c.startswith('image_embed_mean_')]),
                    )
                except PermissionError:
                    fallback_path = BASE_DIR / "cache" / "dataset_image_embeddings.parquet"
                    fallback_path.parent.mkdir(parents=True, exist_ok=True)
                    image_embeddings.to_parquet(fallback_path, index=False)
                    LOGGER.warning(
                        "Unable to write embeddings to %s due to permissions; saved to %s instead",
                        embeddings_path,
                        fallback_path,
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Visual embedding extraction failed; continuing without embeddings: %s", exc)
                image_embeddings = pd.DataFrame()
        else:
            LOGGER.info("Visual embedding extraction skipped (dependencies not available)")

        if not image_embeddings.empty and active_dataset_ids:
            before_emb = len(image_embeddings)
            image_embeddings = image_embeddings[image_embeddings["dataset_id"].isin(active_dataset_ids)].reset_index(drop=True)
            trimmed_emb = before_emb - len(image_embeddings)
            if trimmed_emb > 0:
                LOGGER.info("Filtered %d visual embedding rows for deleted datasets", trimmed_emb)
    else:
        LOGGER.warning("Image data not found at %s, skipping image features", dataset_image_path)

    user_features_v2 = engine.build_user_features_v2(
        cleaned_interactions,
        cleaned_user_profile,
        cleaned_dataset,
    )
    dataset_features_v2 = engine.build_dataset_features_v2(
        cleaned_dataset,
        cleaned_dataset_stats,
        cleaned_interactions,
        dataset_image,
        image_embeddings,
    )

    engine.save_features(
        user_features_v2,
        dataset_features_v2,
        cleaned_interactions,
        cleaned_dataset_stats,
    )

    LOGGER.info(
        "Enhanced feature engineering completed (user_features=%d cols, dataset_features=%d cols)",
        len(user_features_v2.columns),
        len(dataset_features_v2.columns),
    )

    redis_url = os.getenv("FEATURE_REDIS_URL") or os.getenv("REDIS_FEATURE_URL")
    if redis_url:
        parsed = urllib.parse.urlparse(redis_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        password = parsed.password
        db = int(parsed.path.lstrip("/")) if parsed.path else 1

        try:
            redis_store = RedisFeatureStore(host=host, port=port, db=db, password=password)
            redis_store.sync_user_features(user_features_v2)
            redis_store.sync_dataset_features(dataset_features_v2)
            redis_store.sync_dataset_stats(cleaned_dataset_stats)
            redis_store.sync_user_history(cleaned_interactions)
            LOGGER.info(
                "Redis feature store synchronized successfully (host=%s, db=%s)",
                host,
                db,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Redis feature store sync skipped due to error: %s", exc)
    else:
        LOGGER.info("Redis feature store not configured; skipping sync step")


if __name__ == "__main__":
    main()
