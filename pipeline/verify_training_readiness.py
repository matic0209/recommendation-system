"""Verify readiness of tracking data before training weekly models."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR

LOGGER = logging.getLogger(__name__)

PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

EXPOSURES_PATH = PROCESSED_DIR / "recommend_exposures.parquet"
CLICKS_PATH = PROCESSED_DIR / "recommend_clicks.parquet"
LABELS_PATH = PROCESSED_DIR / "ranking_labels_by_dataset.parquet"
SLOT_LABELS_PATH = PROCESSED_DIR / "ranking_slot_metrics.parquet"
REPORT_PATH = EVAL_DIR / "training_readiness.json"


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Expected file %s not found; treating as empty.", path)
        return pd.DataFrame()
    return pd.read_parquet(path)


def verify() -> dict:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    exposures = _load_parquet(EXPOSURES_PATH)
    clicks = _load_parquet(CLICKS_PATH)
    labels = _load_parquet(LABELS_PATH)
    slot_metrics = _load_parquet(SLOT_LABELS_PATH)

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "exposures_rows": int(len(exposures)),
        "exposures_request_ids": int(exposures["request_id"].nunique()) if "request_id" in exposures.columns else 0,
        "click_rows": int(len(clicks)),
        "click_request_ids": int(clicks["request_id"].nunique()) if "request_id" in clicks.columns else 0,
        "label_rows": int(len(labels)),
        "positive_datasets": int(labels["label"].sum()) if "label" in labels.columns else 0,
        "slot_metric_rows": int(len(slot_metrics)),
    }

    errors = []
    warnings = []

    if summary["exposures_rows"] == 0:
        errors.append("Exposure log is empty")

    if summary["click_rows"] == 0:
        warnings.append("No recommendation clicks captured")

    if summary["positive_datasets"] < 5:
        warnings.append("Too few positive datasets (labels) for robust ranking training")

    if summary["slot_metric_rows"] == 0:
        warnings.append("Slot-level metrics are empty; CTR/CVR features unavailable")

    summary["warnings"] = warnings
    summary["errors"] = errors

    REPORT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("Training readiness report written to %s", REPORT_PATH)

    if errors:
        raise RuntimeError("; ".join(errors))

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = verify()
    LOGGER.info(
        "Readiness summary: exposures=%d, clicks=%d, positive_datasets=%d",
        summary["exposures_rows"],
        summary["click_rows"],
        summary["positive_datasets"],
    )
    if summary["warnings"]:
        LOGGER.warning("Warnings: %s", "; ".join(summary["warnings"]))


if __name__ == "__main__":
    main()
