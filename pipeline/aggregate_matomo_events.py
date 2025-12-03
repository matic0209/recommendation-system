"""Aggregate Matomo recommendation events for downstream training."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config.settings import DATA_DIR
from pipeline import evaluate_v2

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"
EXPOSURES_PATH = PROCESSED_DIR / "recommend_exposures.parquet"
CLICKS_PATH = PROCESSED_DIR / "recommend_clicks.parquet"
CONVERSIONS_PATH = PROCESSED_DIR / "recommend_conversions.parquet"
SLOT_METRICS_PATH = PROCESSED_DIR / "recommend_slot_metrics.parquet"


def _ensure_processed_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)
    LOGGER.info("Saved %s (%d rows)", path, len(df))


def _normalize_positions(df: pd.DataFrame, exposures: pd.DataFrame, column_name: str = "position") -> pd.DataFrame:
    """
    Ensure every row has an integer position.

    If a row lacks position but shares (request_id, dataset_id) with exposure data, copy the exposure slot.
    """
    if df.empty:
        if column_name not in df.columns:
            df[column_name] = pd.Series(dtype="float64")
        return df

    result = df.copy()
    if column_name not in result.columns:
        result[column_name] = pd.NA

    if exposures is not None and not exposures.empty:
        missing_mask = result[column_name].isna()
        if missing_mask.any():
            lookup = exposures[["request_id", "dataset_id", "position"]].drop_duplicates()
            result = result.merge(
                lookup.rename(columns={"position": "_exposure_position"}),
                on=["request_id", "dataset_id"],
                how="left",
            )
            result.loc[missing_mask, column_name] = result.loc[missing_mask, "_exposure_position"]
            result = result.drop(columns=["_exposure_position"])

    result[column_name] = result[column_name].fillna(-1).astype(int)
    return result


def _group_slot_counts(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dataset_id", "position", value_name])
    grouped = (
        df.groupby(["dataset_id", "position"])
        .size()
        .reset_index(name=value_name)
        .sort_values(["dataset_id", "position"])
    )
    return grouped


def _group_conversion_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return (
            pd.DataFrame(columns=["dataset_id", "position", "conversion_count"]),
            pd.DataFrame(columns=["dataset_id", "position", "conversion_revenue"]),
        )
    conv_count = (
        df.groupby(["dataset_id", "position"])
        .size()
        .reset_index(name="conversion_count")
        .sort_values(["dataset_id", "position"])
    )
    conv_revenue = (
        df.groupby(["dataset_id", "position"])["revenue"]
        .sum()
        .reset_index(name="conversion_revenue")
        .sort_values(["dataset_id", "position"])
    )
    return conv_count, conv_revenue


def _compute_slot_metrics(exposures: pd.DataFrame, clicks: pd.DataFrame, conversions: pd.DataFrame) -> pd.DataFrame:
    exposures = _normalize_positions(exposures, exposures)
    clicks = _normalize_positions(clicks, exposures)
    conversions = _normalize_positions(conversions, exposures)

    exposure_counts = _group_slot_counts(exposures, "exposure_count")
    click_counts = _group_slot_counts(clicks, "click_count")
    conv_counts, conv_revenue = _group_conversion_metrics(conversions)

    metrics = exposure_counts
    for df in (click_counts, conv_counts, conv_revenue):
        metrics = metrics.merge(df, on=["dataset_id", "position"], how="left")

    for column in ("click_count", "conversion_count", "conversion_revenue"):
        if column not in metrics.columns:
            metrics[column] = 0
        metrics[column] = metrics[column].fillna(0)

    metrics["ctr"] = np.where(
        metrics["exposure_count"] > 0,
        metrics["click_count"] / metrics["exposure_count"],
        0.0,
    )
    metrics["cvr"] = np.where(
        metrics["exposure_count"] > 0,
        metrics["conversion_count"] / metrics["exposure_count"],
        0.0,
    )
    metrics = metrics.sort_values(["dataset_id", "position"]).reset_index(drop=True)
    return metrics


def aggregate_events() -> None:
    """Aggregate Matomo exposure/click/conversion events into parquet files."""
    _ensure_processed_dir()

    LOGGER.info("Loading exposure log...")
    exposures = evaluate_v2._load_exposure_log()
    if exposures.empty:
        LOGGER.warning("No exposure records found; downstream training labels may be empty.")
    else:
        exposures = (
            exposures.sort_values("timestamp")
            .drop_duplicates(subset=["request_id", "dataset_id"], keep="first")
            .reset_index(drop=True)
        )
    _save_dataframe(exposures, EXPOSURES_PATH)

    LOGGER.info("Loading Matomo action mappings...")
    mapping = evaluate_v2._load_actions()
    if not mapping:
        LOGGER.warning("Matomo action mapping is empty; skipping click/conversion aggregation.")
        clicks = pd.DataFrame(columns=["dataset_id", "request_id", "server_time", "position"])
        conversions = pd.DataFrame(columns=["dataset_id", "request_id", "server_time", "revenue", "position"])
    else:
        LOGGER.info("Aggregating recommendation clicks...")
        clicks = evaluate_v2._load_recommend_clicks(mapping)
        if clicks.empty:
            LOGGER.warning("No recommendation clicks with request_id found.")
        else:
            clicks = clicks.sort_values("server_time").drop_duplicates(
                subset=["request_id", "dataset_id"], keep="first"
            )

        LOGGER.info("Aggregating recommendation conversions...")
        conversions = evaluate_v2._load_recommend_conversions(mapping)
        if conversions.empty:
            LOGGER.warning("No recommendation conversions with request_id found.")
        else:
            conversions = conversions.sort_values("server_time").drop_duplicates(
                subset=["request_id", "dataset_id"], keep="first"
            )

    _save_dataframe(clicks, CLICKS_PATH)
    _save_dataframe(conversions, CONVERSIONS_PATH)

    LOGGER.info("Computing slot-level metrics...")
    slot_metrics = _compute_slot_metrics(exposures, clicks, conversions)
    _save_dataframe(slot_metrics, SLOT_METRICS_PATH)
    LOGGER.info("Matomo aggregation completed.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    aggregate_events()


if __name__ == "__main__":
    main()
