"""Build ranking training labels by joining exposures with Matomo clicks."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from config.settings import DATA_DIR
from pipeline.sentry_utils import init_pipeline_sentry, monitor_pipeline_step

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"
EXPOSURES_PATH = PROCESSED_DIR / "recommend_exposures.parquet"
CLICKS_PATH = PROCESSED_DIR / "recommend_clicks.parquet"
CONVERSIONS_PATH = PROCESSED_DIR / "recommend_conversions.parquet"
SLOT_METRICS_SOURCE_PATH = PROCESSED_DIR / "recommend_slot_metrics.parquet"
SAMPLES_PATH = PROCESSED_DIR / "ranking_training_samples.parquet"
DATASET_LABELS_PATH = PROCESSED_DIR / "ranking_labels_by_dataset.parquet"
SLOT_LABELS_PATH = PROCESSED_DIR / "ranking_slot_metrics.parquet"
DEFAULT_CHANNEL_WEIGHTS: Dict[str, float] = {
    "behavior": 1.5,
    "content": 0.8,
    "vector": 0.5,
    "popular": 0.05,
    "tag": 0.4,
    "category": 0.3,
    "price": 0.2,
    "usercf": 0.6,
    "image": 0.4,
}


def _load_optional_parquet(path: Path, expected_columns: Tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Expected file %s not found; returning empty frame.", path)
        return pd.DataFrame(columns=expected_columns)
    df = pd.read_parquet(path)
    missing = [col for col in expected_columns if col not in df.columns]
    for col in missing:
        df[col] = pd.NA
    return df


def _prepare_exposures() -> pd.DataFrame:
    exposures = _load_optional_parquet(
        EXPOSURES_PATH,
        (
            "request_id",
            "dataset_id",
            "algorithm_version",
            "user_id",
            "page_id",
            "endpoint",
            "variant",
            "experiment_variant",
            "degrade_reason",
            "source",
            "device_type",
            "locale",
            "client_app",
            "channel_weights",
            "score",
            "reason",
            "timestamp",
            "position",
        ),
    )
    if exposures.empty:
        return exposures
    exposures["timestamp"] = pd.to_datetime(exposures["timestamp"], errors="coerce")
    exposures = exposures.dropna(subset=["request_id", "dataset_id"])
    exposures["dataset_id"] = exposures["dataset_id"].astype(int)
    exposures = (
        exposures.sort_values("timestamp")
        .drop_duplicates(subset=["request_id", "dataset_id"], keep="first")
        .reset_index(drop=True)
    )
    return exposures


def _extract_channel(reason: object) -> str:
    if reason is None:
        return "unknown"
    text = str(reason).lower()
    if not text:
        return "unknown"
    parts = [token for token in text.split("+") if token]
    if not parts:
        return "unknown"
    channel = parts[0]
    if channel.startswith("fallback"):
        return "fallback"
    return channel


def _resolve_channel_weight(channel_weights: object, channel: str) -> float:
    if channel in DEFAULT_CHANNEL_WEIGHTS:
        default = DEFAULT_CHANNEL_WEIGHTS[channel]
    else:
        default = 0.1
    if not channel_weights:
        return default
    payload = channel_weights
    if isinstance(channel_weights, str):
        try:
            payload = json.loads(channel_weights)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict):
        try:
            value = float(payload.get(channel, default))
            return value
        except (TypeError, ValueError):
            return default
    return default


def _prepare_clicks() -> pd.DataFrame:
    clicks = _load_optional_parquet(
        CLICKS_PATH,
        ("request_id", "dataset_id", "server_time", "position"),
    )
    if clicks.empty:
        return clicks
    clicks = clicks.dropna(subset=["request_id", "dataset_id"])
    clicks["dataset_id"] = clicks["dataset_id"].astype(int)
    clicks["server_time"] = pd.to_datetime(clicks["server_time"], errors="coerce")
    clicks = (
        clicks.sort_values("server_time")
        .drop_duplicates(subset=["request_id", "dataset_id"], keep="first")
        .reset_index(drop=True)
    )
    return clicks


def _prepare_conversions() -> pd.DataFrame:
    conversions = _load_optional_parquet(
        CONVERSIONS_PATH,
        ("dataset_id", "revenue", "server_time"),
    )
    if conversions.empty:
        return conversions
    conversions = conversions.dropna(subset=["dataset_id"])
    conversions["dataset_id"] = conversions["dataset_id"].astype(int)
    conversions["revenue"] = pd.to_numeric(conversions.get("revenue", 0.0), errors="coerce").fillna(0.0)
    return conversions


@monitor_pipeline_step("build_training_labels", critical=True)
def build_training_labels() -> None:
    exposures = _prepare_exposures()
    if exposures.empty:
        LOGGER.warning("No exposures available; ranking labels will be empty.")
    clicks = _prepare_clicks()

    samples = exposures.copy()
    if not samples.empty:
        samples = samples.merge(
            clicks[["request_id", "dataset_id", "server_time"]],
            on=["request_id", "dataset_id"],
            how="left",
        )
        samples = samples.rename(columns={"server_time": "clicked_at"})
        samples["label"] = samples["clicked_at"].notna().astype(int)
    else:
        samples = pd.DataFrame(columns=[
            "request_id",
            "dataset_id",
            "algorithm_version",
            "user_id",
            "page_id",
            "timestamp",
            "position",
            "clicked_at",
            "label",
            "endpoint",
            "variant",
            "experiment_variant",
            "degrade_reason",
            "source",
            "device_type",
            "locale",
            "client_app",
            "channel_weights",
            "score",
            "reason",
        ])

    if not samples.empty:
        samples["reason"] = samples.get("reason").fillna("")
        samples["channel"] = samples["reason"].apply(_extract_channel)
        samples["channel_weight"] = samples.apply(
            lambda row: _resolve_channel_weight(row.get("channel_weights"), row.get("channel")),
            axis=1,
        )
        context_defaults = {
            "endpoint": "unknown",
            "variant": "primary",
            "experiment_variant": "control",
            "degrade_reason": "none",
            "source": "unknown",
            "device_type": "unknown",
            "locale": "unknown",
            "client_app": "",
        }
        for column, default in context_defaults.items():
            if column in samples.columns:
                samples[column] = samples[column].fillna(default).astype(str)
            else:
                samples[column] = default
        if "score" in samples.columns:
            samples["score"] = pd.to_numeric(samples["score"], errors="coerce").fillna(0.0)
        else:
            samples["score"] = 0.0
        if "position" in samples.columns:
            samples["position"] = pd.to_numeric(samples["position"], errors="coerce").fillna(-1).astype(int)
        else:
            samples["position"] = -1
    else:
        samples["channel"] = pd.Series(dtype="object")
        samples["channel_weight"] = pd.Series(dtype="float64")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(SAMPLES_PATH, index=False)
    LOGGER.info("Saved ranking training samples: %s (%d rows)", SAMPLES_PATH, len(samples))

    if not samples.empty:
        dataset_labels = (
            samples.groupby("dataset_id")["label"]
            .agg(["max", "sum", "count"])
            .rename(columns={"max": "label", "sum": "click_count", "count": "exposure_count"})
            .reset_index()
        )
        dataset_labels["label"] = dataset_labels["label"].astype(int)
    else:
        dataset_labels = pd.DataFrame(columns=["dataset_id", "label", "click_count", "exposure_count"])

    conversions = _prepare_conversions()
    if not conversions.empty:
        conversions_by_dataset = conversions.groupby("dataset_id").agg(
            conversion_count=("dataset_id", "count"),
            conversion_revenue=("revenue", "sum"),
        ).reset_index()
        conversions_by_dataset["conversion_count"] = conversions_by_dataset["conversion_count"].astype(int)
        dataset_labels = dataset_labels.merge(conversions_by_dataset, on="dataset_id", how="left")
    else:
        dataset_labels["conversion_count"] = 0
        dataset_labels["conversion_revenue"] = 0.0

    for column in ["conversion_count", "conversion_revenue"]:
        if column not in dataset_labels.columns:
            dataset_labels[column] = 0
    dataset_labels[["conversion_count", "conversion_revenue"]] = dataset_labels[["conversion_count", "conversion_revenue"]].fillna(0)
    exposures_safe = dataset_labels["exposure_count"].replace({0: pd.NA}).astype(float)
    dataset_labels["conversion_rate"] = (
        dataset_labels["conversion_count"] / exposures_safe
    ).fillna(0.0)

    dataset_labels.to_parquet(DATASET_LABELS_PATH, index=False)
    LOGGER.info(
        "Saved dataset-level labels: %s (%d rows; positives=%d)",
        DATASET_LABELS_PATH,
        len(dataset_labels),
        int(dataset_labels["label"].sum() if not dataset_labels.empty else 0),
    )

    slot_metrics = _load_optional_parquet(
        SLOT_METRICS_SOURCE_PATH,
        ("dataset_id", "position", "exposure_count", "click_count", "conversion_count", "conversion_revenue", "ctr", "cvr"),
    )
    if not slot_metrics.empty:
        slot_metrics["dataset_id"] = slot_metrics["dataset_id"].astype(int)
        slot_metrics["position"] = slot_metrics["position"].astype(int)
    slot_metrics.to_parquet(SLOT_LABELS_PATH, index=False)
    LOGGER.info("Saved slot-level metrics for ranking: %s (%d rows)", SLOT_LABELS_PATH, len(slot_metrics))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_pipeline_sentry("build_training_labels")
    build_training_labels()


if __name__ == "__main__":
    main()
