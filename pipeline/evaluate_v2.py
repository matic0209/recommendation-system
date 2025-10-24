"""Evaluate recommendation with Request ID tracking for accurate CTR/CVR calculation."""
from __future__ import annotations

import json
import logging
import math
import re
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, MODELS_DIR

REQUEST_ID_DIMENSION = int(os.getenv("MATOMO_REQUEST_DIMENSION", "2"))
REQUEST_ID_COLUMN = f"custom_dimension_{REQUEST_ID_DIMENSION}"
REQUEST_ID_COLUMNS: List[str] = []
for candidate in [REQUEST_ID_COLUMN, "custom_dimension_3", "custom_dimension_1"]:
    if candidate and candidate not in REQUEST_ID_COLUMNS:
        REQUEST_ID_COLUMNS.append(candidate)
POSITION_DIMENSION = os.getenv("MATOMO_POSITION_DIMENSION")
POSITION_COLUMN = f"custom_dimension_{POSITION_DIMENSION}" if POSITION_DIMENSION else None

LOGGER = logging.getLogger(__name__)
EVAL_DIR = DATA_DIR / "evaluation"
EXPOSURE_LOG_PATH = EVAL_DIR / "exposure_log.jsonl"

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
    """Load Matomo action ID to dataset ID mapping."""
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


def _get_request_id_from_columns(row: pd.Series) -> Optional[str]:
    for column in REQUEST_ID_COLUMNS:
        value = row.get(column, "")
        if value and isinstance(value, str) and value.startswith("req_"):
            return value
    return None


def _is_recommend_click(row: pd.Series) -> bool:
    """Check if the click is from recommendation based on URL parameters or custom dimension."""
    # 方法1: 检查URL中是否包含from=recommend
    url = row.get('url', '') or row.get('idaction_url', '')
    if isinstance(url, str) and 'from=recommend' in url:
        return True

    # 方法2: 检查custom_dimension_1是否有值（表示有request_id）
    custom_dim = _get_request_id_from_columns(row)
    if custom_dim:
        return True

    return False


def _extract_request_id_from_row(row: pd.Series) -> Optional[str]:
    """Extract request_id from custom dimension or URL parameter."""
    # 优先从custom_dimension_1提取
    custom_dim = _get_request_id_from_columns(row)
    if custom_dim:
        return custom_dim

    # 尝试从URL参数提取
    url = row.get('url', '')
    if isinstance(url, str) and 'rid=' in url:
        match = re.search(r'rid=([^&]+)', url)
        if match:
            return match.group(1)

    return None


def _load_recommend_clicks(mapping: Dict[int, int]) -> pd.DataFrame:
    """Load clicks that came from recommendations (with request_id tracking)."""
    path = DATA_DIR / "matomo" / "matomo_log_link_visit_action.parquet"
    if not path.exists():
        LOGGER.warning("Matomo log_link_visit_action parquet not found: %s", path)
        return pd.DataFrame(columns=["dataset_id", "request_id", "server_time", "position"])

    actions = pd.read_parquet(path)
    records: List[Dict[str, object]] = []

    for _, row in actions.iterrows():
        # 只统计推荐点击
        if not _is_recommend_click(row):
            continue

        dataset_id = _extract_dataset_from_row(row, mapping)
        if dataset_id is None:
            continue

        request_id = _extract_request_id_from_row(row)
        if not request_id:
            # 没有request_id的推荐点击，可能是旧数据
            LOGGER.debug("Found recommend click without request_id for dataset %d", dataset_id)
            continue

        server_time = pd.to_datetime(row.get("server_time"), errors="coerce")
        if pd.isna(server_time):
            continue

        # 尝试提取position
        url = row.get('url', '')
        position = None
        if isinstance(url, str):
            pos_match = re.search(r'pos=(\d+)', url)
            if pos_match:
                position = int(pos_match.group(1))

        records.append({
            "dataset_id": int(dataset_id),
            "request_id": request_id,
            "server_time": server_time,
            "position": position
        })

    if not records:
        LOGGER.warning("No recommend clicks found with request_id")
        return pd.DataFrame(columns=["dataset_id", "request_id", "server_time", "position"])

    LOGGER.info("Found %d recommend clicks with request_id", len(records))
    return pd.DataFrame.from_records(records)


def _load_recommend_conversions(mapping: Dict[int, int]) -> pd.DataFrame:
    """
    Load conversions that came from recommendations.

    Try two methods to get request_id:
    1. Direct: from custom_dimension_1 in conversion table (if frontend sends it)
    2. Session-based: join with log_link_visit_action via idvisit
    """
    conv_path = DATA_DIR / "matomo" / "matomo_log_conversion.parquet"
    if not conv_path.exists():
        LOGGER.warning("Matomo log_conversion parquet not found: %s", conv_path)
        return pd.DataFrame(columns=["dataset_id", "request_id", "server_time", "revenue", "position"])

    conversions = pd.read_parquet(conv_path)

    # Load visit actions for session-based matching
    action_path = DATA_DIR / "matomo" / "matomo_log_link_visit_action.parquet"
    visit_actions = None
    if action_path.exists():
        visit_actions = pd.read_parquet(action_path)

    records: List[Dict[str, object]] = []

    for _, conv_row in conversions.iterrows():
        dataset_id = _map_action_value(conv_row.get("idaction_url"), mapping)
        if dataset_id is None:
            dataset_id = _parse_dataset_id(conv_row.get("url"))
        if dataset_id is None:
            continue

        server_time = pd.to_datetime(conv_row.get("server_time"), errors="coerce")
        if pd.isna(server_time):
            continue

        revenue = conv_row.get("revenue", 0.0)
        try:
            revenue = float(revenue)
        except (ValueError, TypeError):
            revenue = 0.0

        # Method 1: Try to get request_id from conversion's custom_dimension_1
        request_id = None
        position = None

        custom_dim = _get_request_id_from_columns(conv_row)
        if custom_dim:
            request_id = custom_dim
            LOGGER.debug("Found request_id in conversion custom_dimension_1: %s", request_id)

            # Try to get position from custom_dimension_2
            if POSITION_COLUMN and POSITION_COLUMN in conv_row:
                pos_dim = conv_row.get(POSITION_COLUMN, '')
            else:
                pos_dim = conv_row.get('custom_dimension_2', '')
            if pos_dim not in (None, ''):
                try:
                    position = int(pos_dim)
                except (ValueError, TypeError):
                    pass

        # Method 2: If no request_id, try session-based matching
        if not request_id and visit_actions is not None:
            idvisit = conv_row.get('idvisit')
            if idvisit and not pd.isna(idvisit):
                # Find actions in same visit with request_id
                same_visit = visit_actions[visit_actions['idvisit'] == idvisit]

                if not same_visit.empty:
                    # Look for actions with custom_dimension_1 (request_id)
                    same_visit = same_visit.copy()
                    same_visit["__req_id"] = same_visit.apply(_get_request_id_from_columns, axis=1)
                    with_rid = same_visit[same_visit["__req_id"].notna()]

                    if not with_rid.empty:
                        # Get the most recent action before conversion
                        with_rid = with_rid.copy()
                        with_rid['server_time'] = pd.to_datetime(with_rid['server_time'], errors='coerce')
                        before_purchase = with_rid[with_rid['server_time'] <= server_time]

                        if not before_purchase.empty:
                            # Get the latest one
                            latest = before_purchase.sort_values('server_time', ascending=False).iloc[0]
                            req_id = latest["__req_id"]

                            if req_id and isinstance(req_id, str) and req_id.startswith('req_'):
                                request_id = req_id
                                LOGGER.debug("Found request_id via session matching: %s", request_id)

                                # Try to extract position from URL
                                url = latest.get('url', '')
                                if isinstance(url, str) and 'pos=' in url:
                                    pos_match = re.search(r'pos=(\d+)', url)
                                    if pos_match:
                                        position = int(pos_match.group(1))

        records.append({
            "dataset_id": int(dataset_id),
            "request_id": request_id,  # Now we try to get it!
            "server_time": server_time,
            "revenue": revenue,
            "position": position
        })

    if not records:
        return pd.DataFrame(columns=["dataset_id", "request_id", "server_time", "revenue", "position"])

    result = pd.DataFrame.from_records(records)

    # Log statistics
    total = len(result)
    with_rid = result[result['request_id'].notna()]
    LOGGER.info("Loaded %d conversions, %d (%.1f%%) with request_id attribution",
                total, len(with_rid), len(with_rid) / total * 100 if total > 0 else 0)

    return result


def _load_exposure_log() -> pd.DataFrame:
    """Load exposure log with request_id."""
    if not EXPOSURE_LOG_PATH.exists():
        LOGGER.warning("Exposure log not found: %s", EXPOSURE_LOG_PATH)
        return pd.DataFrame(columns=[
            "request_id", "algorithm_version", "user_id", "page_id",
            "dataset_id", "score", "reason", "timestamp", "position"
        ])

    records: List[Dict[str, object]] = []
    with EXPOSURE_LOG_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            items = payload.get("items") or []
            timestamp = payload.get("timestamp")
            request_id = payload.get("request_id")

            if not request_id:
                LOGGER.warning("Exposure log entry missing request_id")
                continue

            for idx, item in enumerate(items):
                dataset_id = item.get("dataset_id")
                try:
                    dataset_id = int(dataset_id)
                except (TypeError, ValueError):
                    continue

                try:
                    score = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    score = float("nan")

                records.append({
                    "request_id": request_id,
                    "algorithm_version": payload.get("algorithm_version"),
                    "user_id": payload.get("user_id"),
                    "page_id": payload.get("page_id"),
                    "dataset_id": dataset_id,
                    "score": score,
                    "reason": item.get("reason"),
                    "timestamp": timestamp,
                    "position": idx  # 推荐位置
                })

    if not records:
        LOGGER.warning("No exposure records found")
        return pd.DataFrame(columns=[
            "request_id", "algorithm_version", "user_id", "page_id",
            "dataset_id", "score", "reason", "timestamp", "position"
        ])

    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    LOGGER.info("Loaded %d exposure records with %d unique request_ids",
                len(df), df["request_id"].nunique())
    return df


def _normalize_timestamps(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_localize(None)


def _compute_request_id_metrics(
    exposures: pd.DataFrame,
    clicks: pd.DataFrame,
    conversions: pd.DataFrame,
) -> Dict[str, object]:
    """
    Compute accurate CTR/CVR metrics using request_id tracking.

    This is the core improvement: we match exposures and clicks by request_id,
    ensuring we only count clicks that actually came from recommendations.
    """
    if exposures.empty:
        LOGGER.warning("No exposure data to compute metrics")
        return {"status": "no_data", "reason": "empty_exposures"}

    exposures = exposures.dropna(subset=["request_id", "dataset_id"])
    if exposures.empty:
        LOGGER.warning("No valid exposures after filtering")
        return {"status": "no_data", "reason": "invalid_exposures"}

    exposures["algorithm_version"] = exposures["algorithm_version"].fillna("unknown")
    exposures["timestamp"] = _normalize_timestamps(exposures["timestamp"])

    # 按request_id + dataset_id 分组统计曝光
    exposure_groups = exposures.groupby(["request_id", "dataset_id", "algorithm_version"]).agg(
        exposure_count=("dataset_id", "count"),
        first_timestamp=("timestamp", "min"),
        avg_score=("score", "mean"),
        min_position=("position", "min")
    ).reset_index()

    LOGGER.info("Grouped %d exposures into %d unique (request_id, dataset_id) pairs",
                len(exposures), len(exposure_groups))

    # 处理点击数据
    if not clicks.empty:
        clicks["server_time"] = _normalize_timestamps(clicks["server_time"])
        clicks = clicks.dropna(subset=["request_id", "dataset_id", "server_time"])

        # 按request_id + dataset_id 分组统计点击
        click_groups = clicks.groupby(["request_id", "dataset_id"]).agg(
            click_count=("dataset_id", "count"),
            click_time=("server_time", "min")
        ).reset_index()

        LOGGER.info("Grouped %d clicks into %d unique (request_id, dataset_id) pairs",
                    len(clicks), len(click_groups))
    else:
        click_groups = pd.DataFrame(columns=["request_id", "dataset_id", "click_count"])
        LOGGER.warning("No click data found")

    # 处理转化数据（简化版，暂不支持request_id关联）
    if not conversions.empty:
        conversions["server_time"] = _normalize_timestamps(conversions["server_time"])
        conv_by_dataset = conversions.groupby("dataset_id").agg(
            conversion_count=("dataset_id", "count"),
            total_revenue=("revenue", "sum")
        ).reset_index()
    else:
        conv_by_dataset = pd.DataFrame(columns=["dataset_id", "conversion_count", "total_revenue"])

    # 关联曝光和点击（通过request_id精确匹配）
    merged = exposure_groups.merge(
        click_groups,
        on=["request_id", "dataset_id"],
        how="left"
    )
    merged["click_count"] = merged["click_count"].fillna(0).astype(int)

    # 关联转化数据（按dataset_id，因为暂时没有request_id）
    merged = merged.merge(
        conv_by_dataset,
        on="dataset_id",
        how="left"
    )
    merged["conversion_count"] = merged["conversion_count"].fillna(0).astype(int)
    merged["total_revenue"] = merged["total_revenue"].fillna(0.0)

    # 计算整体指标
    total_exposures = merged["exposure_count"].sum()
    total_clicks = merged["click_count"].sum()
    total_conversions = merged["conversion_count"].sum()

    overall_ctr = total_clicks / total_exposures if total_exposures > 0 else 0.0
    overall_cvr = total_conversions / total_exposures if total_exposures > 0 else 0.0

    # 按算法版本分组
    version_metrics = merged.groupby("algorithm_version").agg(
        exposures=("exposure_count", "sum"),
        clicks=("click_count", "sum"),
        conversions=("conversion_count", "sum"),
        revenue=("total_revenue", "sum")
    ).reset_index()

    version_metrics["ctr"] = np.where(
        version_metrics["exposures"] > 0,
        version_metrics["clicks"] / version_metrics["exposures"],
        0.0
    )
    version_metrics["cvr"] = np.where(
        version_metrics["exposures"] > 0,
        version_metrics["conversions"] / version_metrics["exposures"],
        0.0
    )

    # 位置分析
    position_metrics = merged.groupby("min_position").agg(
        exposures=("exposure_count", "sum"),
        clicks=("click_count", "sum")
    ).reset_index()
    position_metrics["ctr"] = np.where(
        position_metrics["exposures"] > 0,
        position_metrics["clicks"] / position_metrics["exposures"],
        0.0
    )

    summary = {
        "status": "success",
        "total_exposures": int(total_exposures),
        "total_clicks": int(total_clicks),
        "total_conversions": int(total_conversions),
        "overall_ctr": float(overall_ctr),
        "overall_cvr": float(overall_cvr),
        "unique_request_ids": int(exposures["request_id"].nunique()),
        "unique_users": int(exposures["user_id"].dropna().nunique()),
        "unique_datasets_exposed": int(exposures["dataset_id"].nunique()),
        "unique_datasets_clicked": int(clicks["dataset_id"].nunique()) if not clicks.empty else 0,
    }

    # 转换Timestamp列为字符串，以便JSON序列化
    merged_copy = merged.copy()
    for col in merged_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(merged_copy[col]):
            merged_copy[col] = merged_copy[col].astype(str)

    return {
        "summary": summary,
        "by_version": version_metrics.to_dict("records"),
        "by_position": position_metrics.to_dict("records"),
        "detailed_data": merged_copy.to_dict("records")  # 供进一步分析
    }


def generate_tracking_report() -> Dict[str, object]:
    """Generate complete tracking report with request_id based metrics."""
    _ensure_dir(EVAL_DIR)

    LOGGER.info("Loading exposure log...")
    exposures = _load_exposure_log()

    LOGGER.info("Loading Matomo action mappings...")
    mapping = _load_actions()

    LOGGER.info("Loading recommend clicks...")
    clicks = _load_recommend_clicks(mapping)

    LOGGER.info("Loading conversions...")
    conversions = _load_recommend_conversions(mapping)

    LOGGER.info("Computing request_id based metrics...")
    metrics = _compute_request_id_metrics(exposures, clicks, conversions)

    # 保存报告
    report_path = EVAL_DIR / "tracking_report_v2.json"
    report_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    LOGGER.info("Tracking report saved to %s", report_path)

    # 打印关键指标
    if metrics.get("status") == "success":
        summary = metrics["summary"]
        LOGGER.info("=" * 60)
        LOGGER.info("Request ID Tracking Report")
        LOGGER.info("=" * 60)
        LOGGER.info("Total Exposures: %d", summary["total_exposures"])
        LOGGER.info("Total Clicks: %d", summary["total_clicks"])
        LOGGER.info("Total Conversions: %d", summary["total_conversions"])
        LOGGER.info("Overall CTR: %.4f", summary["overall_ctr"])
        LOGGER.info("Overall CVR: %.4f", summary["overall_cvr"])
        LOGGER.info("Unique Request IDs: %d", summary["unique_request_ids"])
        LOGGER.info("Unique Users: %d", summary["unique_users"])
        LOGGER.info("=" * 60)

        # 按版本打印
        if metrics.get("by_version"):
            LOGGER.info("\nBy Algorithm Version:")
            for v in metrics["by_version"]:
                LOGGER.info("  %s: CTR=%.4f, CVR=%.4f, Exposures=%d, Clicks=%d",
                           v["algorithm_version"], v["ctr"], v["cvr"],
                           v["exposures"], v["clicks"])

        # 按位置打印
        if metrics.get("by_position"):
            LOGGER.info("\nBy Position:")
            for p in sorted(metrics["by_position"], key=lambda x: x["min_position"]):
                LOGGER.info("  Position %d: CTR=%.4f, Exposures=%d, Clicks=%d",
                           p["min_position"], p["ctr"], p["exposures"], p["clicks"])

    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_tracking_report()


if __name__ == "__main__":
    main()
