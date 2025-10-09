"""Extract raw data from source databases into the local data directory."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine

from config.settings import DATA_DIR, DatabaseConfig, load_database_configs

LOGGER = logging.getLogger(__name__)
DEFAULT_TABLES = {
    "business": (
        "user",
        "dataset",
        "order_tab",
        "api_order",
        "dataset_image",
    ),
    "matomo": (
        "matomo_log_visit",
        "matomo_log_link_visit_action",
        "matomo_log_action",
        "matomo_log_conversion",
    ),
}


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _export_table(config: DatabaseConfig, table: str, output_dir: Path, dry_run: bool) -> None:
    output_path = output_dir / f"{table}.parquet"
    if dry_run:
        LOGGER.info("[dry-run] would export table '%s' to %s", table, output_path)
        return

    LOGGER.info("Exporting table '%s' from %s", table, config.name)
    engine = create_engine(config.sqlalchemy_url())
    try:
        frame = pd.read_sql_table(table, con=engine)
        frame.to_parquet(output_path, index=False)
        LOGGER.info("Saved %s (%d rows)", output_path.name, len(frame))
    finally:
        engine.dispose()


def extract_all(dry_run: bool = True) -> None:
    configs = load_database_configs()
    _ensure_output_dir(DATA_DIR)

    for source, tables in DEFAULT_TABLES.items():
        config = configs[source]
        output_dir = DATA_DIR / source
        _ensure_output_dir(output_dir)
        LOGGER.info("Processing source '%s' (database=%s)", source, config.name)
        for table in tables:
            _export_table(config, table, output_dir, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log planned actions without connecting to databases.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_all(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
