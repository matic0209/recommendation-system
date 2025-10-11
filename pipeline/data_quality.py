"""Run basic data quality checks on extracted parquet tables."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config.settings import DATA_DIR

LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = DATA_DIR / "evaluation"
BUSINESS_DIR = DATA_DIR / "business"
MATOMO_DIR = DATA_DIR / "matomo"


@dataclass
class TableRule:
    required_columns: List[str]
    min_rows: int = 0


BUSINESS_RULES: Dict[str, TableRule] = {
    "user": TableRule(["id", "user_name", "create_time"]),
    "dataset": TableRule(["id", "dataset_name", "price"], min_rows=1),
    "task": TableRule(["create_user", "dataset_id", "price", "create_time"]),
    "api_order": TableRule(["creator_id", "api_id", "price", "create_time"]),
}

MATOMO_RULES: Dict[str, TableRule] = {
    "matomo_log_visit": TableRule(["idvisit", "idsite", "visit_last_action_time"]),
    "matomo_log_link_visit_action": TableRule(["idlink_va", "idsite", "idvisit", "server_time"]),
    "matomo_log_action": TableRule(["idaction", "type", "name"]),
    "matomo_log_conversion": TableRule(["idvisit", "idsite", "server_time"]),
}


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Missing parquet file: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _check_table(base_dir: Path, table: str, rule: TableRule) -> Dict[str, object]:
    path = base_dir / f"{table}.parquet"
    df = _load_parquet(path)
    result = {
        "table": table,
        "path": str(path),
        "row_count": int(len(df)),
        "status": "pass",
        "missing_columns": [],
    }

    if df.empty:
        result["status"] = "missing"
        return result

    missing_columns = [col for col in rule.required_columns if col not in df.columns]
    result["missing_columns"] = missing_columns
    if missing_columns:
        result["status"] = "fail"

    if len(df) < rule.min_rows:
        result["status"] = "fail"

    return result


def run_quality_checks() -> Dict[str, object]:
    business_results = [_check_table(BUSINESS_DIR, table, rule) for table, rule in BUSINESS_RULES.items()]
    matomo_results = [_check_table(MATOMO_DIR, table, rule) for table, rule in MATOMO_RULES.items()]

    summary = {
        "business": business_results,
        "matomo": matomo_results,
    }
    issues = [res for group in summary.values() for res in group if res["status"] != "pass"]
    summary["overall_status"] = "pass" if not issues else "attention"
    summary["issues_count"] = len(issues)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = run_quality_checks()
    report_path = OUTPUT_DIR / "data_quality.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("Data quality report saved to %s", report_path)
    LOGGER.info("Overall status: %s (issues=%d)", summary.get("overall_status"), summary.get("issues_count", 0))


if __name__ == "__main__":
    main()
