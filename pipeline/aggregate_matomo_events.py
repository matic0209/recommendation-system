"""Aggregate Matomo recommendation events for downstream training."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR
from pipeline import evaluate_v2

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"
EXPOSURES_PATH = PROCESSED_DIR / "recommend_exposures.parquet"
CLICKS_PATH = PROCESSED_DIR / "recommend_clicks.parquet"
CONVERSIONS_PATH = PROCESSED_DIR / "recommend_conversions.parquet"


def _ensure_processed_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)
    LOGGER.info("Saved %s (%d rows)", path, len(df))


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
    LOGGER.info("Matomo aggregation completed.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    aggregate_events()


if __name__ == "__main__":
    main()
