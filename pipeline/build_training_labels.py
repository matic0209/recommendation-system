"""Build ranking training labels by joining exposures with Matomo clicks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

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
        ("request_id", "dataset_id", "algorithm_version", "user_id", "page_id", "timestamp", "position"),
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
        samples = pd.DataFrame(columns=["request_id", "dataset_id", "algorithm_version", "user_id", "page_id", "timestamp", "position", "clicked_at", "label"])

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
