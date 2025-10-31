"""Generate daily recommendation monitoring reports (JSON + HTML)."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from config.settings import BASE_DIR, DATA_DIR
from pipeline.evaluate_v2 import (
    _compute_request_id_metrics,
    _load_actions,
    _load_exposure_log,
    _load_recommend_clicks,
    _load_recommend_conversions,
)

LOGGER = logging.getLogger(__name__)

REPORT_DIR = DATA_DIR / "evaluation" / "daily_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_DIR = BASE_DIR / "templates"
TEMPLATE_FILE = "recommendation_daily_report.html.j2"


@dataclass
class HistoryPoint:
    day: date
    summary: Dict[str, Any]


def _filter_by_date(frame: pd.DataFrame, day: date, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.dropna(subset=[column])
    return frame[frame[column].dt.date == day]


def _build_merged_dataset(
    exposures: pd.DataFrame,
    clicks: pd.DataFrame,
    conversions: pd.DataFrame,
) -> pd.DataFrame:
    if exposures.empty:
        return pd.DataFrame(columns=[
            "request_id",
            "dataset_id",
            "algorithm_version",
            "endpoint",
            "variant",
            "experiment_variant",
            "degrade_reason",
            "model_run_id",
            "feature_snapshot_id",
            "exposure_count",
            "first_timestamp",
            "avg_score",
            "min_position",
            "click_count",
            "click_time",
            "conversion_count",
            "total_revenue",
        ])

    exp_cols = [
        "request_id",
        "dataset_id",
        "algorithm_version",
        "endpoint",
        "variant",
        "experiment_variant",
        "degrade_reason",
        "model_run_id",
        "feature_snapshot_id",
    ]
    for col in exp_cols:
        if col not in exposures.columns:
            exposures[col] = None

    exposures = exposures.dropna(subset=["request_id", "dataset_id"]).copy()
    exposures["timestamp"] = pd.to_datetime(exposures["timestamp"], errors="coerce")

    exposure_groups = exposures.groupby(exp_cols, dropna=False).agg(
        exposure_count=("dataset_id", "count"),
        first_timestamp=("timestamp", "min"),
        avg_score=("score", "mean"),
        min_position=("position", "min"),
    ).reset_index()

    if clicks.empty:
        click_groups = pd.DataFrame(columns=["request_id", "dataset_id", "click_count", "click_time"])
    else:
        clicks = clicks.dropna(subset=["request_id", "dataset_id"]).copy()
        clicks["server_time"] = pd.to_datetime(clicks["server_time"], errors="coerce")
        click_groups = clicks.groupby(["request_id", "dataset_id"]).agg(
            click_count=("dataset_id", "count"),
            click_time=("server_time", "min"),
        ).reset_index()

    if conversions.empty:
        conv_by_dataset = pd.DataFrame(columns=["dataset_id", "conversion_count", "total_revenue"])
    else:
        conversions = conversions.copy()
        conversions["server_time"] = pd.to_datetime(conversions["server_time"], errors="coerce")
        conv_by_dataset = conversions.groupby("dataset_id").agg(
            conversion_count=("dataset_id", "count"),
            total_revenue=("revenue", "sum"),
        ).reset_index()

    merged = exposure_groups.merge(
        click_groups,
        on=["request_id", "dataset_id"],
        how="left",
    )
    merged["click_count"] = merged.get("click_count", 0).fillna(0).astype(int)
    merged = merged.merge(
        conv_by_dataset,
        on="dataset_id",
        how="left",
    )
    merged["conversion_count"] = merged.get("conversion_count", 0).fillna(0).astype(int)
    merged["total_revenue"] = merged.get("total_revenue", 0.0).fillna(0.0)
    return merged


def _load_history(target_day: date, days: int = 7) -> List[HistoryPoint]:
    points: List[HistoryPoint] = []
    for offset in range(1, days + 1):
        day = target_day - timedelta(days=offset)
        path = REPORT_DIR / f"recommendation_report_{day.isoformat()}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            summary = data.get("summary") or {}
            points.append(HistoryPoint(day=day, summary=summary))
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse historical report %s", path)
    points.sort(key=lambda x: x.day)
    return points


def _compute_deltas(current: float, baseline: Optional[float]) -> Optional[float]:
    if baseline in (None, 0):
        return None
    return (current - baseline) / baseline


def _format_float(value: Optional[float], precision: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(value, precision)


def _aggregate_breakdowns(merged: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    if merged.empty:
        return {
            "by_version": [],
            "by_endpoint": [],
            "by_position": [],
            "top_datasets": [],
        }

    def _agg(df: pd.DataFrame, group_cols: List[str]) -> List[Dict[str, Any]]:
        grouped = df.groupby(group_cols, dropna=False).agg(
            exposures=("exposure_count", "sum"),
            clicks=("click_count", "sum"),
            conversions=("conversion_count", "sum"),
            revenue=("total_revenue", "sum"),
        ).reset_index()
        grouped["ctr"] = grouped.apply(
            lambda row: row["clicks"] / row["exposures"] if row["exposures"] > 0 else 0.0,
            axis=1,
        )
        grouped["cvr"] = grouped.apply(
            lambda row: row["conversions"] / row["exposures"] if row["exposures"] > 0 else 0.0,
            axis=1,
        )
        return grouped.to_dict("records")

    by_version = _agg(merged, ["algorithm_version"])
    by_endpoint = _agg(merged, ["endpoint"])
    by_position = _agg(merged, ["min_position"])

    dataset_group = merged.groupby("dataset_id", dropna=False).agg(
        exposures=("exposure_count", "sum"),
        clicks=("click_count", "sum"),
        conversions=("conversion_count", "sum"),
        revenue=("total_revenue", "sum"),
    ).reset_index()
    dataset_group["ctr"] = dataset_group.apply(
        lambda row: row["clicks"] / row["exposures"] if row["exposures"] > 0 else 0.0,
        axis=1,
    )
    dataset_group["cvr"] = dataset_group.apply(
        lambda row: row["conversions"] / row["exposures"] if row["exposures"] > 0 else 0.0,
        axis=1,
    )
    # Top datasets sorted by clicks then exposures
    dataset_group = dataset_group.sort_values(
        ["clicks", "exposures"], ascending=[False, False]
    ).head(10)
    top_datasets = dataset_group.to_dict("records")

    return {
        "by_version": by_version,
        "by_endpoint": by_endpoint,
        "by_position": by_position,
        "top_datasets": top_datasets,
    }


def generate_daily_report(target_day: date) -> Dict[str, Any]:
    LOGGER.info("Generating recommendation report for %s", target_day.isoformat())

    exposures = _load_exposure_log()
    exposures = _filter_by_date(exposures, target_day, "timestamp")

    mapping = _load_actions()
    clicks = _load_recommend_clicks(mapping)
    clicks = _filter_by_date(clicks, target_day, "server_time")

    conversions = _load_recommend_conversions(mapping)
    conversions = _filter_by_date(conversions, target_day, "server_time")

    metrics = _compute_request_id_metrics(exposures, clicks, conversions)
    summary = metrics.get("summary", {}) or {}
    summary["date"] = target_day.isoformat()
    metrics["summary"] = summary

    merged = _build_merged_dataset(exposures, clicks, conversions)
    breakdowns = _aggregate_breakdowns(merged)

    history = _load_history(target_day, days=7)
    prev_summary = history[-1].summary if history else None

    # 7-day average (excluding current day)
    if history:
        avg_fields = ["total_exposures", "total_clicks", "total_conversions", "overall_ctr", "overall_cvr"]
        avg_values = {}
        for field in avg_fields:
            values = [
                float(h.summary.get(field, 0) or 0)
                for h in history
                if h.summary.get(field) is not None
            ]
            avg_values[field] = sum(values) / len(values) if values else None
    else:
        avg_values = {field: None for field in ["total_exposures", "total_clicks", "total_conversions", "overall_ctr", "overall_cvr"]}

    deltas = {
        "vs_previous_day": {
            metric: _format_float(_compute_deltas(summary.get(metric) or 0, prev_summary.get(metric)))
            if prev_summary and prev_summary.get(metric) is not None
            else None
            for metric in ["total_exposures", "total_clicks", "total_conversions", "overall_ctr", "overall_cvr"]
        },
        "vs_7day_avg": {
            metric: _format_float(_compute_deltas(summary.get(metric) or 0, avg_values.get(metric)))
            for metric in ["total_exposures", "total_clicks", "total_conversions", "overall_ctr", "overall_cvr"]
        },
    }

    history_for_chart = history[-6:] if len(history) > 6 else history  # up to 6 previous days
    chart_data = [
        {"date": h.day.isoformat(), **{k: h.summary.get(k) for k in ("total_exposures", "overall_ctr", "overall_cvr")}}
        for h in history_for_chart
    ]
    chart_data.append({
        "date": summary["date"],
        "total_exposures": summary.get("total_exposures"),
        "overall_ctr": summary.get("overall_ctr"),
        "overall_cvr": summary.get("overall_cvr"),
    })

    return {
        "summary": summary,
        "deltas": deltas,
        "comparisons": {
            "previous_day": prev_summary,
            "seven_day_average": avg_values,
        },
        "breakdowns": breakdowns,
        "history_chart": chart_data,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_counts": {
            "exposures": len(exposures),
            "click_events": len(clicks),
            "conversions": len(conversions),
            "merged_rows": len(merged),
        },
    }


def _write_json(report: Dict[str, Any], target_day: date) -> Path:
    output_path = REPORT_DIR / f"recommendation_report_{target_day.isoformat()}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return output_path


def _render_html(report: Dict[str, Any], target_day: date) -> Path:
    template_path = TEMPLATE_DIR / TEMPLATE_FILE
    if not template_path.exists():
        LOGGER.warning("Template %s not found; skipping HTML generation", template_path)
        return template_path

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(TEMPLATE_FILE)
    chart_data_json = json.dumps(report.get("history_chart", []))
    html = template.render(report=report, chart_data_json=chart_data_json)

    output_path = REPORT_DIR / f"recommendation_report_{target_day.isoformat()}.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate daily recommendation monitoring report.")
    parser.add_argument(
        "--date",
        dest="date",
        help="Target date (YYYY-MM-DD). Defaults to yesterday (UTC).",
    )
    args = parser.parse_args()

    if args.date:
        target_day = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_day = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    report = generate_daily_report(target_day)
    json_path = _write_json(report, target_day)
    LOGGER.info("Daily recommendation JSON report saved to %s", json_path)

    html_path = _render_html(report, target_day)
    if html_path.exists():
        LOGGER.info("Daily recommendation HTML report saved to %s", html_path)


if __name__ == "__main__":
    main()
