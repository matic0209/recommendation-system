#!/usr/bin/env python3
"""
Daily reconciliation script for business orders and Matomo exposure logs.

Usage:
    python scripts/reconcile_business_metrics.py --start 2025-10-01 --end 2025-10-03
If no date is supplied the script reconciles yesterday's data.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.settings import DATA_DIR

LOGGER = logging.getLogger(__name__)
BUSINESS_DIR = DATA_DIR / "business"
MATOMO_DIR = DATA_DIR / "matomo"
OUTPUT_DIR = DATA_DIR / "evaluation"


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Missing parquet file: %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path)


def _normalize_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.date


def _load_business_orders(start: datetime, end: datetime) -> pd.DataFrame:
    """Aggregate business order metrics per day."""
    task_orders = _load_parquet(BUSINESS_DIR / "task.parquet")
    api_order = _load_parquet(BUSINESS_DIR / "api_order.parquet")

    frames = []
    if not task_orders.empty:
        task_orders = task_orders.assign(
            order_date=_normalize_date(task_orders.get("create_time")),
            amount=pd.to_numeric(task_orders.get("price"), errors="coerce").fillna(0.0),
        )
        frames.append(task_orders[["order_date", "dataset_id", "amount"]])
    if not api_order.empty:
        api_order = api_order.assign(
            order_date=_normalize_date(api_order.get("create_time")),
            amount=pd.to_numeric(api_order.get("price"), errors="coerce").fillna(0.0),
        )
        frames.append(api_order[["order_date", "api_id", "amount"]].rename(columns={"api_id": "dataset_id"}))

    if not frames:
        return pd.DataFrame(columns=["order_date", "order_count", "revenue"])

    merged = pd.concat(frames, ignore_index=True)
    mask = (merged["order_date"] >= start.date()) & (merged["order_date"] <= end.date())
    merged = merged.loc[mask]
    grouped = (
        merged.groupby("order_date", as_index=False)
        .agg(order_count=("dataset_id", "count"), revenue=("amount", "sum"))
        .sort_values("order_date")
    )
    return grouped


def _load_matomo_exposures(start: datetime, end: datetime) -> pd.DataFrame:
    """Aggregate exposures from Matomo visit actions per day."""
    visits = _load_parquet(MATOMO_DIR / "matomo_log_visit.parquet")
    link_actions = _load_parquet(MATOMO_DIR / "matomo_log_link_visit_action.parquet")

    if link_actions.empty:
        return pd.DataFrame(columns=["event_date", "exposures", "unique_visits"])

    link_actions = link_actions.assign(
        event_date=_normalize_date(link_actions.get("server_time")),
    )
    mask = (link_actions["event_date"] >= start.date()) & (link_actions["event_date"] <= end.date())
    link_actions = link_actions.loc[mask]

    exposures = (
        link_actions.groupby("event_date", as_index=False)
        .agg(exposures=("idlink_va", "count"), unique_visits=("idvisit", "nunique"))
        .sort_values("event_date")
    )

    if not visits.empty:
        visits = visits.assign(
            event_date=_normalize_date(visits.get("visit_last_action_time")),
        )
        visit_counts = (
            visits.groupby("event_date", as_index=False)
            .agg(total_visits=("idvisit", "nunique"))
        )
        exposures = exposures.merge(visit_counts, on="event_date", how="left")
    return exposures


def _build_reconciliation(
    start: datetime,
    end: datetime,
) -> Dict[str, Dict[str, float]]:
    business = _load_business_orders(start, end)
    matomo = _load_matomo_exposures(start, end)

    reconciliation: Dict[str, Dict[str, float]] = {}
    date_range = pd.date_range(start=start.date(), end=end.date(), freq="D")

    for day in date_range:
        date_key = day.strftime("%Y-%m-%d")
        business_row = business[business["order_date"] == day.date()]
        matomo_row = matomo[matomo["event_date"] == day.date()]

        business_count = int(business_row["order_count"].iloc[0]) if not business_row.empty else 0
        business_revenue = float(business_row["revenue"].iloc[0]) if not business_row.empty else 0.0
        exposures = int(matomo_row["exposures"].iloc[0]) if not matomo_row.empty else 0
        unique_visits = int(matomo_row["unique_visits"].iloc[0]) if not matomo_row.empty else 0
        if not matomo_row.empty and "total_visits" in matomo_row.columns:
            total_visits = int(matomo_row["total_visits"].fillna(0).iloc[0])
        else:
            total_visits = 0

        reconciliation[date_key] = {
            "business_order_count": business_count,
            "business_revenue": round(business_revenue, 2),
            "matomo_exposures": exposures,
            "matomo_unique_visits": unique_visits,
            "matomo_total_visits": total_visits,
            "exposure_to_order_ratio": round(exposures / business_count, 2) if business_count else None,
        }

    return reconciliation


def _default_date_range() -> tuple[datetime, datetime]:
    today = datetime.utcnow().date()
    start = datetime.combine(today - timedelta(days=1), datetime.min.time())
    end = datetime.combine(today - timedelta(days=1), datetime.max.time())
    return start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, help="Optional output path for reconciliation JSON")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start, _ = _default_date_range()

    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        _, end = _default_date_range()

    if end < start:
        raise ValueError("End date must be >= start date")

    LOGGER.info("Reconciling metrics from %s to %s", start.date(), end.date())
    reconciliation = _build_reconciliation(start, end)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"reconciliation_{start.date()}_{end.date()}.json"
    output_path.write_text(json.dumps(reconciliation, indent=2, ensure_ascii=False))

    LOGGER.info("Reconciliation saved to %s", output_path)
    print(json.dumps(reconciliation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
