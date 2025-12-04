"""Quick sanity checks for processed Matomo and training artifacts."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config.settings import DATA_DIR
from pipeline.sentry_utils import init_pipeline_sentry, monitor_pipeline_step

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"

FILE_GROUPS: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {
    "matomo": [
        ("recommend_exposures.parquet", ("request_id", "dataset_id")),
        ("recommend_clicks.parquet", ("request_id", "dataset_id")),
        ("recommend_conversions.parquet", ("request_id", "dataset_id")),
        ("recommend_slot_metrics.parquet", ("dataset_id", "position", "exposure_count")),
        ("recommend_variant_metrics.parquet", ("dataset_id", "variant", "exposure_count")),
    ],
    "training": [
        ("ranking_slot_metrics.parquet", ("dataset_id", "position", "exposure_count")),
    ],
}


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required processed file missing: {path}")
    return pd.read_parquet(path)


def _verify_files(file_specs: List[Tuple[str, Tuple[str, ...]]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for filename, required_columns in file_specs:
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


def verify_processed_files(group: str) -> Dict[str, int]:
    if group == "all":
        summary: Dict[str, int] = {}
        for name, specs in FILE_GROUPS.items():
            LOGGER.info("Verifying %s files...", name)
            summary.update(_verify_files(specs))
        return summary

    if group not in FILE_GROUPS:
        raise ValueError(f"Unknown file group '{group}'. Available groups: {', '.join(FILE_GROUPS)}")
    LOGGER.info("Verifying %s files...", group)
    return _verify_files(FILE_GROUPS[group])


@monitor_pipeline_step("verify_processed_files", critical=False)
def main() -> None:
    parser = argparse.ArgumentParser(description="Verify processed parquet artifacts.")
    parser.add_argument("--group", choices=["matomo", "training", "all"], default="all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_pipeline_sentry(f"verify_{args.group}_files")
    summary = verify_processed_files(args.group)
    LOGGER.info("Processed artifacts verified: %s", ", ".join(f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
