"""Evaluate recommendation data against Matomo behaviour logs."""
from __future__ import annotations

import json
import logging
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config.settings import DATA_DIR, MODELS_DIR

LOGGER = logging.getLogger(__name__)
EVAL_DIR = DATA_DIR / "evaluation"
EXPOSURE_LOG_PATH = EVAL_DIR / "exposure_log.jsonl"

DATA_DETAIL_PATTERN = re.compile(r"dataDetail/(\d+)")
API_DETAIL_PATTERN = re.compile(r"dataAPIDetail/(\d+)")
PAYMENT_PATTERN = re.compile(r"payment/(\d+)")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_dataset_id(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    value = str(text)
    for pattern in (DATA_DETAIL_PATTERN, API_DETAIL_PATTERN, PAYMENT_PATTERN):
        match = pattern.search(value)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def _load_actions() -> Dict[int, int]:
    path = DATA_DIR / "matomo" / "matomo_log_action.parquet"
    if not path.exists():
        LOGGER.error("Matomo log_action parquet not found: %s", path)
        return {}
    actions = pd.read_parquet(path)
    mapping: Dict[int, int] = {}
    for _, row in actions.iterrows():
        dataset_id = _parse_dataset_id(row.get("name"))
        if dataset_id is not None:
            mapping[int(row["idaction"])] = dataset_id
    LOGGER.info("Loaded %d action mappings", len(mapping))
    return mapping


def _map_action_value(value: Optional[float], mapping: Dict[int, int]) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        key = int(value)
    except (ValueError, TypeError):
        return None
    return mapping.get(key)


def _extract_dataset_from_row(row: pd.Series, mapping: Dict[int, int]) -> Optional[int]:
    action_columns = [
        "idaction_url",
        "idaction_name",
        "idaction_event_action",
        "idaction_event_category",
        "idaction_product_name",
        "idaction_content_target",
    ]
    for col in action_columns:
        dataset_id = _map_action_value(row.get(col), mapping)
        if dataset_id is not None:
            return dataset_id
    text_columns = ["search_cat", "search_count", "url", "referer_url", "idpageview"]
    for col in text_columns:
        dataset_id = _parse_dataset_id(row.get(col))
        if dataset_id is not None:
            return dataset_id
    return None


def _load_views(mapping: Dict[int, int]) -> pd.DataFrame:
    path = DATA_DIR / "matomo" / "matomo_log_link_visit_action.parquet"
    if not path.exists():
        LOGGER.warning("Matomo log_link_visit_action parquet not found: %s", path)
        return pd.DataFrame(columns=["dataset_id", "views"])
    actions = pd.read_parquet(path)
    dataset_ids = []
    for _, row in actions.iterrows():
        dataset_id = _extract_dataset_from_row(row, mapping)
        if dataset_id is not None:
            dataset_ids.append(dataset_id)
    if not dataset_ids:
        return pd.DataFrame(columns=["dataset_id", "views"])
    views = pd.Series(dataset_ids, name="dataset_id").value_counts().rename_axis("dataset_id").reset_index(name="views")
    return views


def _load_action_events(mapping: Dict[int, int]) -> pd.DataFrame:
    path = DATA_DIR / "matomo" / "matomo_log_link_visit_action.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["dataset_id", "server_time"])
    actions = pd.read_parquet(path)
    records: List[Dict[str, object]] = []
    for _, row in actions.iterrows():
        dataset_id = _extract_dataset_from_row(row, mapping)
        if dataset_id is None:
            continue
        server_time = pd.to_datetime(row.get("server_time"), errors="coerce")
        if pd.isna(server_time):
            continue
        records.append({"dataset_id": int(dataset_id), "server_time": server_time})
    if not records:
        return pd.DataFrame(columns=["dataset_id", "server_time"])
    return pd.DataFrame.from_records(records)


def _load_conversions(mapping: Dict[int, int]) -> pd.DataFrame:
    path = DATA_DIR / "matomo" / "matomo_log_conversion.parquet"
    if not path.exists():
        LOGGER.warning("Matomo log_conversion parquet not found: %s", path)
        return pd.DataFrame(columns=["dataset_id", "conversions", "revenue"])
    df = pd.read_parquet(path)
    dataset_ids = []
    for _, row in df.iterrows():
        dataset_id = _map_action_value(row.get("idaction_url"), mapping)
        if dataset_id is None:
            dataset_id = _parse_dataset_id(row.get("url"))
        if dataset_id is not None:
            dataset_ids.append((dataset_id, row.get("revenue", 0.0)))
    if not dataset_ids:
        return pd.DataFrame(columns=["dataset_id", "conversions", "revenue"])
    conv_df = pd.DataFrame(dataset_ids, columns=["dataset_id", "revenue"])
    conv_df["revenue"] = pd.to_numeric(conv_df["revenue"], errors="coerce").fillna(0.0)
    agg = conv_df.groupby("dataset_id").agg(
        conversions=("dataset_id", "count"),
        revenue=("revenue", "sum"),
    ).reset_index()
    return agg


def _load_conversion_events(mapping: Dict[int, int]) -> pd.DataFrame:
    path = DATA_DIR / "matomo" / "matomo_log_conversion.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["dataset_id", "server_time"])
    df = pd.read_parquet(path)
    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        dataset_id = _map_action_value(row.get("idaction_url"), mapping)
        if dataset_id is None:
            dataset_id = _parse_dataset_id(row.get("url"))
        if dataset_id is None:
            continue
        server_time = pd.to_datetime(row.get("server_time"), errors="coerce")
        if pd.isna(server_time):
            continue
        records.append({"dataset_id": int(dataset_id), "server_time": server_time})
    if not records:
        return pd.DataFrame(columns=["dataset_id", "server_time"])
    return pd.DataFrame.from_records(records)


def _load_dataset_metadata_table() -> pd.DataFrame:
    path = DATA_DIR / "processed" / "dataset_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    # Ensure consistent column naming even when dataset_name absent
    columns = frame.columns.tolist()
    rename_map = {"dataset_name": "title"}
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in columns})
    if "title" not in frame.columns:
        frame["title"] = ""
    if "price" not in frame.columns:
        frame["price"] = 0.0
    if "create_company_name" not in frame.columns:
        frame["create_company_name"] = ""
    return frame


def _load_exposure_log() -> pd.DataFrame:
    if not EXPOSURE_LOG_PATH.exists():
        return pd.DataFrame(columns=["request_id", "algorithm_version", "user_id", "page_id", "dataset_id", "score", "reason", "timestamp"])
    records: List[Dict[str, object]] = []
    with EXPOSURE_LOG_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            items = payload.get("items") or []
            timestamp = payload.get("timestamp")
            for item in items:
                dataset_id = item.get("dataset_id")
                try:
                    dataset_id = int(dataset_id)
                except (TypeError, ValueError):
                    continue
                try:
                    score = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    score = float("nan")
                records.append(
                    {
                        "request_id": payload.get("request_id"),
                        "algorithm_version": payload.get("algorithm_version"),
                        "user_id": payload.get("user_id"),
                        "page_id": payload.get("page_id"),
                        "dataset_id": dataset_id,
                        "score": score,
                        "reason": item.get("reason"),
                        "timestamp": timestamp,
                    }
                )
    if not records:
        return pd.DataFrame(columns=["request_id", "algorithm_version", "user_id", "page_id", "dataset_id", "score", "reason", "timestamp"])
    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _normalize_timestamps(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_localize(None)


def _compute_exposure_metrics(
    exposures: pd.DataFrame,
    actions: pd.DataFrame,
    conversions: pd.DataFrame,
) -> Dict[str, object]:
    if exposures.empty:
        return {}

    exposures = exposures.dropna(subset=["dataset_id"])
    if exposures.empty:
        return {}

    exposures["algorithm_version"] = exposures["algorithm_version"].fillna("unknown")
    exposures["timestamp"] = _normalize_timestamps(exposures["timestamp"])
    start_time = exposures["timestamp"].min()
    if pd.isna(start_time):
        start_time = None
    else:
        exposures["timestamp"] = exposures["timestamp"].fillna(start_time)

    if not actions.empty:
        actions["server_time"] = _normalize_timestamps(actions["server_time"])
        actions = actions.dropna(subset=["server_time"])
        if start_time is not None:
            actions = actions[actions["server_time"] >= start_time]
    if not conversions.empty:
        conversions["server_time"] = _normalize_timestamps(conversions["server_time"])
        conversions = conversions.dropna(subset=["server_time"])
        if start_time is not None:
            conversions = conversions[conversions["server_time"] >= start_time]

    exposures_by_version = (
        exposures.groupby(["algorithm_version", "dataset_id"]).agg(
            exposures_count=("dataset_id", "count"),
            first_timestamp=("timestamp", "min"),
        )
    ).reset_index()

    total_by_dataset = exposures.groupby("dataset_id").size().rename("dataset_total_exposures")
    exposures_by_version = exposures_by_version.merge(total_by_dataset, on="dataset_id", how="left")
    exposures_by_version["dataset_total_exposures"] = exposures_by_version["dataset_total_exposures"].replace(0, np.nan)

    if not actions.empty:
        clicks_by_dataset = actions.groupby("dataset_id").size().rename("total_clicks")
    else:
        clicks_by_dataset = pd.Series(dtype=float)
    if not conversions.empty:
        conv_by_dataset = conversions.groupby("dataset_id").size().rename("total_conversions")
    else:
        conv_by_dataset = pd.Series(dtype=float)

    merged = exposures_by_version.merge(clicks_by_dataset, on="dataset_id", how="left")
    merged = merged.merge(conv_by_dataset, on="dataset_id", how="left")
    merged["total_clicks"] = merged["total_clicks"].fillna(0.0)
    merged["total_conversions"] = merged["total_conversions"].fillna(0.0)

    # Allocate clicks/conversions proportionally when多个版本共享同一 dataset
    ratio = merged["exposures_count"] / merged["dataset_total_exposures"].replace({0: np.nan})
    merged["clicks"] = merged["total_clicks"] * ratio.fillna(0.0)
    merged["conversions"] = merged["total_conversions"] * ratio.fillna(0.0)

    version_metrics = (
        merged.groupby("algorithm_version").agg(
            exposures=("exposures_count", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
    ).reset_index()
    version_metrics["ctr"] = np.where(
        version_metrics["exposures"] > 0,
        version_metrics["clicks"] / version_metrics["exposures"],
        0.0,
    )
    version_metrics["cvr"] = np.where(
        version_metrics["exposures"] > 0,
        version_metrics["conversions"] / version_metrics["exposures"],
        0.0,
    )

    totals = {
        "exposures_total": int(exposures.shape[0]),
        "exposures_unique_users": int(exposures["user_id"].dropna().nunique()),
        "exposures_ctr": float(version_metrics["clicks"].sum() / exposures.shape[0]) if exposures.shape[0] else 0.0,
        "exposures_cvr": float(version_metrics["conversions"].sum() / exposures.shape[0]) if exposures.shape[0] else 0.0,
    }

    return {
        "summary": totals,
        "versions": version_metrics.to_dict("records"),
    }


def _load_vector_recall() -> Dict[int, List[Dict[str, float]]]:
    path = MODELS_DIR / "item_recall_vector.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    parsed: Dict[int, List[Dict[str, float]]] = {}
    for key, value in raw.items():
        try:
            dataset_id = int(key)
        except ValueError:
            continue
        entries: List[Dict[str, float]] = []
        for entry in value:
            if isinstance(entry, dict):
                entries.append(
                    {
                        "dataset_id": int(entry.get("dataset_id", 0)),
                        "score": float(entry.get("score", 0.0)),
                    }
                )
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                entries.append({"dataset_id": int(entry[0]), "score": float(entry[1])})
        parsed[dataset_id] = entries
    return parsed


def _load_rank_model() -> Optional[Pipeline]:
    path = MODELS_DIR / "rank_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as stream:
        try:
            model = pickle.load(stream)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to load rank model: %s", exc)
            return None
    if not isinstance(model, Pipeline):
        LOGGER.warning("Rank model artifact is not a sklearn Pipeline")
        return None
    return model


def _load_raw_features() -> pd.DataFrame:
    path = DATA_DIR / "processed" / "dataset_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    base_columns = ["dataset_id", "price", "description", "tag"]
    optional_columns = [
        col
        for col in ["image_richness_score", "image_embed_norm", "has_images", "has_cover"]
        if col in frame.columns
    ]
    columns = [col for col in base_columns + optional_columns if col in frame.columns]
    subset = frame[columns].copy()
    subset["dataset_id"] = pd.to_numeric(subset.get("dataset_id"), errors="coerce").fillna(0).astype(int)
    subset["price"] = pd.to_numeric(subset.get("price"), errors="coerce").fillna(0.0)
    subset["description"] = subset.get("description", "").fillna("").astype(str)
    subset["tag"] = subset.get("tag", "").fillna("").astype(str)
    for col in optional_columns:
        subset[col] = pd.to_numeric(subset.get(col), errors="coerce").fillna(0.0)
    return subset


def _load_dataset_stats_frame() -> pd.DataFrame:
    path = DATA_DIR / "processed" / "dataset_stats.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["dataset_id", "interaction_count", "total_weight"])
    frame = pd.read_parquet(path)
    frame["dataset_id"] = pd.to_numeric(frame.get("dataset_id"), errors="coerce").fillna(0).astype(int)
    frame["interaction_count"] = pd.to_numeric(frame.get("interaction_count"), errors="coerce").fillna(0.0)
    frame["total_weight"] = pd.to_numeric(frame.get("total_weight"), errors="coerce").fillna(0.0)
    return frame


def _load_interactions_frame() -> pd.DataFrame:
    path = DATA_DIR / "processed" / "interactions.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["user_id", "dataset_id", "weight", "last_event_time"])
    frame = pd.read_parquet(path)
    frame["user_id"] = pd.to_numeric(frame.get("user_id"), errors="coerce").fillna(0).astype(int)
    frame["dataset_id"] = pd.to_numeric(frame.get("dataset_id"), errors="coerce").fillna(0).astype(int)
    return frame


def _compute_ranking_features(
    dataset_ids: List[int],
    raw_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
) -> pd.DataFrame:
    if not dataset_ids:
        return pd.DataFrame()

    if raw_features.empty:
        base = pd.DataFrame(index=dataset_ids)
        base["price"] = 0.0
        base["description"] = ""
        base["tag"] = ""
    else:
        base = raw_features.set_index("dataset_id").reindex(dataset_ids)
        base["price"] = pd.to_numeric(base.get("price"), errors="coerce").fillna(0.0)
        base["description"] = base.get("description", "").fillna("").astype(str)
        base["tag"] = base.get("tag", "").fillna("").astype(str)

    base["description_length"] = base.get("description", "").str.len().astype(float)
    base["tag_count"] = base.get("tag", "").apply(
        lambda text: float(len([t for t in text.split(';') if t.strip()])) if isinstance(text, str) else 0.0
    )

    optional_columns = [
        col
        for col in ["image_richness_score", "image_embed_norm", "has_images", "has_cover"]
        if col in base.columns
    ]

    if dataset_stats.empty:
        stats = pd.DataFrame(index=dataset_ids)
        stats["interaction_count"] = 0.0
        stats["total_weight"] = 0.0
    else:
        stats = dataset_stats.set_index("dataset_id").reindex(dataset_ids)
        stats["interaction_count"] = pd.to_numeric(stats.get("interaction_count"), errors="coerce").fillna(0.0)
        stats["total_weight"] = pd.to_numeric(stats.get("total_weight"), errors="coerce").fillna(0.0)

    features = pd.DataFrame(index=dataset_ids)
    features["price_log"] = np.log1p(base["price"].clip(lower=0.0))
    features["description_length"] = base["description_length"].fillna(0.0)
    features["tag_count"] = base["tag_count"].fillna(0.0)
    features["weight_log"] = np.log1p(stats["total_weight"].clip(lower=0.0))
    features["interaction_count"] = stats["interaction_count"].fillna(0.0)
    for col in optional_columns:
        features[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)
    return features


def _compute_vector_recall_metrics(
    vector_recall: Dict[int, List[Dict[str, float]]],
    interactions: pd.DataFrame,
    top_n: int = 50,
) -> Dict[str, float]:
    if not vector_recall or interactions.empty:
        return {"vector_recall_hit_rate_at_50": 0.0}

    co_occurrence: Dict[int, set[int]] = {}
    for _, group in interactions.groupby("user_id"):
        items = set(group["dataset_id"].tolist())
        for item in items:
            others = items - {item}
            if not others:
                continue
            co_occurrence.setdefault(item, set()).update(others)

    recalls: List[float] = []
    for dataset_id, neighbors in vector_recall.items():
        actual = co_occurrence.get(int(dataset_id))
        if not actual:
            continue
        recommended = [int(entry.get("dataset_id", 0)) for entry in neighbors[:top_n]]
        if not recommended:
            recalls.append(0.0)
            continue
        hits = len(set(recommended) & actual)
        recalls.append(hits / min(len(actual), top_n))

    return {
        "vector_recall_hit_rate_at_50": float(np.mean(recalls)) if recalls else 0.0,
    }


def compute_model_metrics() -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    vector_recall = _load_vector_recall()
    rank_model = _load_rank_model()
    raw_features = _load_raw_features()
    dataset_stats = _load_dataset_stats_frame()
    interactions = _load_interactions_frame()

    if rank_model is not None and not raw_features.empty:
        dataset_ids = raw_features["dataset_id"].tolist()
        features = _compute_ranking_features(dataset_ids, raw_features, dataset_stats)
        if not features.empty:
            try:
                probabilities = rank_model.predict_proba(features)[:, 1]
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Ranking model evaluation failed: %s", exc)
            else:
                scores = pd.Series(probabilities, index=features.index, dtype=float)
                stats_series = dataset_stats.set_index("dataset_id").reindex(scores.index)["total_weight"].fillna(0.0)
                if not stats_series.empty:
                    corr = scores.corr(stats_series, method="spearman")
                    metrics["ranking_spearman"] = float(corr) if not pd.isna(corr) else 0.0
                    top_k = 20
                    top_actual = set(stats_series.sort_values(ascending=False).head(top_k).index)
                    top_pred = set(scores.sort_values(ascending=False).head(top_k).index)
                    metrics["ranking_hit_rate_top20"] = float(len(top_actual & top_pred) / top_k) if top_actual else 0.0

    metrics.update(_compute_vector_recall_metrics(vector_recall, interactions))
    return metrics


def generate_report() -> Tuple[pd.DataFrame, Dict[str, float]]:
    mapping = _load_actions()
    views = _load_views(mapping)
    conversions = _load_conversions(mapping)
    metadata = _load_dataset_metadata_table()

    report = views.merge(conversions, on="dataset_id", how="outer")
    if metadata is not None:
        report = report.merge(metadata[["dataset_id", "title", "price", "create_company_name"]], on="dataset_id", how="left")
    report = report.fillna({"views": 0, "conversions": 0, "revenue": 0})
    report["conversion_rate"] = report.apply(
        lambda row: (row["conversions"] / row["views"]) if row["views"] else 0.0,
        axis=1,
    )
    report = report.sort_values(by=["conversions", "views"], ascending=False).reset_index(drop=True)

    summary = {
        "total_datasets_tracked": int(report["dataset_id"].nunique()),
        "total_views": float(report["views"].sum()),
        "total_conversions": float(report["conversions"].sum()),
        "total_revenue": float(report["revenue"].sum()),
        "average_conversion_rate": float(report["conversion_rate"].mean()),
    }
    return report, summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ensure_dir(EVAL_DIR)

    report, summary = generate_report()
    summary.update(compute_model_metrics())

    mapping = _load_actions()
    exposures = _load_exposure_log()
    exposure_metrics = _compute_exposure_metrics(exposures, _load_action_events(mapping), _load_conversion_events(mapping))
    if exposure_metrics:
        summary.update(exposure_metrics.get("summary", {}))
        exposure_path = EVAL_DIR / "exposure_metrics.json"
        exposure_path.write_text(json.dumps(exposure_metrics.get("versions", []), indent=2, ensure_ascii=False))
        LOGGER.info("Exposure metrics saved to %s", exposure_path)

    report_path = EVAL_DIR / "dataset_metrics.csv"
    report.to_csv(report_path, index=False)
    LOGGER.info("Saved dataset metrics to %s", report_path)

    summary_path = EVAL_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
