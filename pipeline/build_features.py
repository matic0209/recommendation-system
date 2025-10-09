"""Build feature datasets required for the recommendation models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from config.settings import DATA_DIR

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Missing input file: %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_interactions_from_frame(
    frame: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    price_col: str,
    time_col: str,
    source: str,
    column_mapping: Dict[str, str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["user_id", "dataset_id", "weight", "source", "last_event_time"])

    working = frame.copy()
    if column_mapping:
        working = working.rename(columns=column_mapping)

    required_cols = {user_col, item_col, price_col, time_col}
    missing = [col for col in required_cols if col not in working.columns]
    if missing:
        LOGGER.warning("%s table missing columns: %s", source, ", ".join(missing))
        return pd.DataFrame(columns=["user_id", "dataset_id", "weight", "source", "last_event_time"])

    working = working.dropna(subset=[user_col, item_col])
    working[price_col] = pd.to_numeric(working[price_col], errors="coerce").fillna(0)
    working[time_col] = pd.to_datetime(working[time_col], errors="coerce")
    working["weight"] = np.log1p(working[price_col].clip(lower=0))
    working["source"] = source
    working = working.rename(columns={user_col: "user_id", item_col: "dataset_id", time_col: "last_event_time"})

    aggregated = (
        working.groupby(["user_id", "dataset_id", "source"], as_index=False)
        .agg({"weight": "sum", "last_event_time": "max"})
    )
    return aggregated


def build_interactions(inputs: Dict[str, Path]) -> pd.DataFrame:
    order_path = inputs.get("order_tab")
    api_order_path = inputs.get("api_order")

    interactions = []
    if order_path is not None:
        order_df = _load_parquet(order_path)
        interactions.append(
            _build_interactions_from_frame(
                order_df,
                user_col="create_user",
                item_col="dataset_id",
                price_col="price",
                time_col="create_time",
                source="order_tab",
            )
        )
    if api_order_path is not None:
        api_df = _load_parquet(api_order_path)
        interactions.append(
            _build_interactions_from_frame(
                api_df,
                user_col="creator_id",
                item_col="dataset_id",
                price_col="price",
                time_col="create_time",
                source="api_order",
                column_mapping={"api_id": "dataset_id"},
            )
        )

    interactions = [df for df in interactions if not df.empty]
    if interactions:
        result = pd.concat(interactions, ignore_index=True)
        result = result.groupby(["user_id", "dataset_id"], as_index=False).agg(
            weight=("weight", "sum"),
            last_event_time=("last_event_time", "max"),
        )
        return result
    return pd.DataFrame(columns=["user_id", "dataset_id", "weight", "last_event_time"])


def build_dataset_features(dataset_path: Path) -> pd.DataFrame:
    dataset_df = _load_parquet(dataset_path)
    if dataset_df.empty:
        return pd.DataFrame(columns=["dataset_id", "dataset_name", "description", "tag", "price", "create_company_name"])

    dataset_df = dataset_df.rename(columns={"id": "dataset_id"})
    columns = [
        "dataset_id",
        "dataset_name",
        "description",
        "tag",
        "price",
        "create_company_name",
    ]
    existing = [col for col in columns if col in dataset_df.columns]
    result = dataset_df[existing].copy()
    for col in ["description", "tag", "create_company_name"]:
        if col in result.columns:
            result[col] = result[col].fillna("")
    if "price" in result.columns:
        result["price"] = pd.to_numeric(result["price"], errors="coerce").fillna(0)
    return result


def build_user_profile(user_path: Path) -> pd.DataFrame:
    user_df = _load_parquet(user_path)
    if user_df.empty:
        return pd.DataFrame(columns=["user_id", "company_name", "province", "city", "is_consumption"])

    user_df = user_df.rename(columns={"id": "user_id"})
    columns = ["user_id", "company_name", "province", "city", "is_consumption"]
    existing = [col for col in columns if col in user_df.columns]
    profile = user_df[existing].copy()
    for col in ["company_name", "province", "city"]:
        if col in profile.columns:
            profile[col] = profile[col].fillna("unknown")
    if "is_consumption" in profile.columns:
        profile["is_consumption"] = pd.to_numeric(profile["is_consumption"], errors="coerce").fillna(0).astype(int)
    return profile


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ensure_output_dir(PROCESSED_DIR)

    business_dir = DATA_DIR / "business"

    interactions = build_interactions(
        {
            "order_tab": business_dir / "order_tab.parquet",
            "api_order": business_dir / "api_order.parquet",
        }
    )
    interactions_path = PROCESSED_DIR / "interactions.parquet"
    interactions.to_parquet(interactions_path, index=False)
    LOGGER.info("Saved interactions to %s", interactions_path)

    dataset_features = build_dataset_features(business_dir / "dataset.parquet")
    dataset_path = PROCESSED_DIR / "dataset_features.parquet"
    dataset_features.to_parquet(dataset_path, index=False)
    LOGGER.info("Saved dataset features to %s", dataset_path)

    user_profile = build_user_profile(business_dir / "user.parquet")
    user_path = PROCESSED_DIR / "user_profile.parquet"
    user_profile.to_parquet(user_path, index=False)
    LOGGER.info("Saved user profile to %s", user_path)


if __name__ == "__main__":
    main()
