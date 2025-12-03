"""Quick sanity checks for processed Matomo and feature artifacts."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config.settings import DATA_DIR

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"

REQUIRED_FILES: List[Tuple[str, Tuple[str, ...]]] = [
    ("recommend_exposures.parquet", ("request_id", "dataset_id")),
    ("recommend_clicks.parquet", ("request_id", "dataset_id")),
    ("recommend_conversions.parquet", ("request_id", "dataset_id")),
    ("recommend_slot_metrics.parquet", ("dataset_id", "position", "exposure_count")),
    ("recommend_variant_metrics.parquet", ("dataset_id", "variant", "exposure_count")),
    ("ranking_slot_metrics.parquet", ("dataset_id", "slot_total_exposures")),
]


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required processed file missing: {path}")
    return pd.read_parquet(path)


def verify_processed_files() -> Dict[str, int]:
    """Ensure key processed artifacts exist and have rows."""
    summary: Dict[str, int] = {}
    for filename, required_columns in REQUIRED_FILES:
        path = PROCESSED_DIR / filename
        df = _load_frame(path)
        row_count = len(df)
        summary[filename] = row_count
        if row_count == 0:
            raise ValueError(f"Processed file {filename} is empty")
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Processed file {filename} missing columns: {', '.join(missing)}")
        LOGGER.info("Verified %s (rows=%d)", filename, row_count)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = verify_processed_files()
    LOGGER.info("Processed artifacts verified: %s", ", ".join(f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
