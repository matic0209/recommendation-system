"""Evaluate recommendation data against Matomo behaviour logs."""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from config.settings import DATA_DIR

LOGGER = logging.getLogger(__name__)
EVAL_DIR = DATA_DIR / "evaluation"

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


def _load_dataset_metadata_table() -> pd.DataFrame:
    path = DATA_DIR / "processed" / "dataset_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    return frame.rename(columns={"dataset_id": "dataset_id", "dataset_name": "title"})


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

    report_path = EVAL_DIR / "dataset_metrics.csv"
    report.to_csv(report_path, index=False)
    LOGGER.info("Saved dataset metrics to %s", report_path)

    summary_path = EVAL_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
