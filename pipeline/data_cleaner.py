"""Data cleaning and anomaly handling for recommendation system."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DATA_DIR

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DIR = DATA_DIR / "cleaned"


class DataCleaner:
    """Clean and handle anomalies in recommendation data."""

    def __init__(self):
        """Initialize data cleaner."""
        self.processed_dir = PROCESSED_DIR
        self.cleaned_dir = CLEANED_DIR
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

        self.cleaning_stats = {
            "interactions": {},
            "dataset_features": {},
            "user_profile": {},
        }

    def clean_price_anomalies(
        self,
        df: pd.DataFrame,
        column: str = "price",
        method: str = "iqr",
        threshold: float = 3.0,
        handle: str = "clip",
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean price anomalies using statistical methods.

        Args:
            df: DataFrame with price column
            column: Column name
            method: 'zscore' or 'iqr'
            threshold: Threshold for outlier detection
            handle: 'remove', 'clip', or 'cap'

        Returns:
            (cleaned_df, stats)
        """
        original_count = len(df)
        df = df.copy()

        if column not in df.columns:
            return df, {"method": method, "anomalies_found": 0, "action": "none"}

        # Detect anomalies
        if method == "zscore":
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            anomaly_mask = z_scores > threshold

        elif method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            anomaly_mask = (df[column] < lower_bound) | (df[column] > upper_bound)

        else:
            raise ValueError(f"Unknown method: {method}")

        anomalies_found = anomaly_mask.sum()

        # Handle anomalies
        if handle == "remove":
            df = df[~anomaly_mask].copy()

        elif handle == "clip":
            if method == "iqr":
                df.loc[df[column] < lower_bound, column] = lower_bound
                df.loc[df[column] > upper_bound, column] = upper_bound

        elif handle == "cap":
            p95 = df[column].quantile(0.95)
            p05 = df[column].quantile(0.05)
            df.loc[df[column] > p95, column] = p95
            df.loc[df[column] < p05, column] = p05

        cleaned_count = len(df)

        stats_dict = {
            "method": method,
            "threshold": threshold,
            "handle": handle,
            "anomalies_found": int(anomalies_found),
            "original_count": original_count,
            "cleaned_count": cleaned_count,
            "removed_count": original_count - cleaned_count,
        }

        if method == "iqr":
            stats_dict.update({
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            })

        LOGGER.info(
            "Price cleaning (%s): found %d anomalies, %s → %d rows remain",
            method,
            anomalies_found,
            handle,
            cleaned_count,
        )

        return df, stats_dict

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: list = None,
        keep: str = "last",
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove duplicate rows.

        Args:
            df: DataFrame
            subset: Columns to check for duplicates
            keep: 'first', 'last', or False

        Returns:
            (cleaned_df, stats)
        """
        original_count = len(df)

        duplicate_count = df.duplicated(subset=subset, keep=False).sum()
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)

        cleaned_count = len(df_cleaned)
        removed_count = original_count - cleaned_count

        stats_dict = {
            "original_count": original_count,
            "duplicate_count": int(duplicate_count),
            "removed_count": removed_count,
            "cleaned_count": cleaned_count,
            "duplicate_ratio": duplicate_count / original_count if original_count > 0 else 0,
        }

        LOGGER.info(
            "Duplicate removal: found %d duplicates (%d total), removed %d rows",
            duplicate_count // 2 if keep else duplicate_count,
            duplicate_count,
            removed_count,
        )

        return df_cleaned, stats_dict

    def fix_timestamps(
        self,
        df: pd.DataFrame,
        column: str,
        max_future_days: int = 0,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Fix invalid timestamps (future dates, null values).

        Args:
            df: DataFrame with timestamp column
            column: Column name
            max_future_days: Maximum allowed days in future

        Returns:
            (cleaned_df, stats)
        """
        if column not in df.columns:
            return df, {"action": "none", "column_missing": True}

        df = df.copy()
        original_count = len(df)

        # Convert to datetime
        df[column] = pd.to_datetime(df[column], errors="coerce")

        # Count issues
        null_count = df[column].isnull().sum()

        now = datetime.now()
        max_allowed_date = now + pd.Timedelta(days=max_future_days)

        future_mask = df[column] > max_allowed_date
        future_count = future_mask.sum()

        # Fix future dates (set to now)
        df.loc[future_mask, column] = now

        # Remove null timestamps
        df_cleaned = df.dropna(subset=[column])

        cleaned_count = len(df_cleaned)

        stats_dict = {
            "column": column,
            "original_count": original_count,
            "null_count": int(null_count),
            "future_count": int(future_count),
            "cleaned_count": cleaned_count,
            "removed_count": original_count - cleaned_count,
        }

        LOGGER.info(
            "Timestamp cleaning: fixed %d future dates, removed %d null timestamps",
            future_count,
            null_count,
        )

        return df_cleaned, stats_dict

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategies: Dict[str, str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing values with different strategies per column.

        Args:
            df: DataFrame
            strategies: Dict mapping column -> strategy
                        Strategies: 'drop', 'mean', 'median', 'mode', 'ffill', 'bfill', 'constant'

        Returns:
            (cleaned_df, stats)
        """
        if strategies is None:
            strategies = {}

        df = df.copy()
        stats_dict = {}

        for column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count == 0:
                continue

            strategy = strategies.get(column, "drop")
            null_ratio = null_count / len(df)

            if strategy == "drop":
                df = df.dropna(subset=[column])

            elif strategy == "mean":
                df[column].fillna(df[column].mean(), inplace=True)

            elif strategy == "median":
                df[column].fillna(df[column].median(), inplace=True)

            elif strategy == "mode":
                df[column].fillna(df[column].mode()[0], inplace=True)

            elif strategy == "ffill":
                df[column].fillna(method="ffill", inplace=True)

            elif strategy == "bfill":
                df[column].fillna(method="bfill", inplace=True)

            elif strategy.startswith("constant:"):
                constant_value = strategy.split(":")[1]
                df[column].fillna(constant_value, inplace=True)

            stats_dict[column] = {
                "null_count": int(null_count),
                "null_ratio": float(null_ratio),
                "strategy": strategy,
            }

            LOGGER.info(
                "Missing values in %s: %d (%.1f%%) - strategy: %s",
                column,
                null_count,
                null_ratio * 100,
                strategy,
            )

        return df, stats_dict

    def clean_interactions(self) -> pd.DataFrame:
        """Clean interactions table."""
        LOGGER.info("Cleaning interactions table...")

        interactions_path = self.processed_dir / "interactions.parquet"
        if not interactions_path.exists():
            LOGGER.warning("Interactions file not found: %s", interactions_path)
            return pd.DataFrame()

        df = pd.read_parquet(interactions_path)
        LOGGER.info("Loaded %d interactions", len(df))

        # 1. Remove duplicates
        df, dup_stats = self.remove_duplicates(
            df,
            subset=["user_id", "dataset_id"],
            keep="last",
        )
        self.cleaning_stats["interactions"]["duplicates"] = dup_stats

        # 2. Fix timestamps
        if "last_event_time" in df.columns:
            df, time_stats = self.fix_timestamps(
                df,
                column="last_event_time",
                max_future_days=0,
            )
            self.cleaning_stats["interactions"]["timestamps"] = time_stats

        # 3. Clean weight anomalies
        if "weight" in df.columns:
            # Remove negative weights
            negative_mask = df["weight"] < 0
            negative_count = negative_mask.sum()
            if negative_count > 0:
                LOGGER.info("Removing %d negative weights", negative_count)
                df = df[~negative_mask]

            # Clip extreme values
            df, weight_stats = self.clean_price_anomalies(
                df,
                column="weight",
                method="iqr",
                threshold=3.0,
                handle="clip",
            )
            self.cleaning_stats["interactions"]["weights"] = weight_stats

        # 4. Ensure valid IDs
        df = df[df["user_id"].notna() & df["dataset_id"].notna()]
        df["user_id"] = df["user_id"].astype(int)
        df["dataset_id"] = df["dataset_id"].astype(int)

        # Save cleaned data
        output_path = self.cleaned_dir / "interactions.parquet"
        df.to_parquet(output_path, index=False)
        LOGGER.info("Saved cleaned interactions: %d rows → %s", len(df), output_path)

        return df

    def clean_dataset_features(self) -> pd.DataFrame:
        """Clean dataset features table."""
        LOGGER.info("Cleaning dataset_features table...")

        features_path = self.processed_dir / "dataset_features.parquet"
        if not features_path.exists():
            LOGGER.warning("Dataset features file not found: %s", features_path)
            return pd.DataFrame()

        df = pd.read_parquet(features_path)
        LOGGER.info("Loaded %d dataset features", len(df))

        # 1. Remove duplicates
        df, dup_stats = self.remove_duplicates(
            df,
            subset=["dataset_id"],
            keep="first",
        )
        self.cleaning_stats["dataset_features"]["duplicates"] = dup_stats

        # 2. Clean price anomalies
        if "price" in df.columns:
            df, price_stats = self.clean_price_anomalies(
                df,
                column="price",
                method="iqr",
                threshold=3.0,
                handle="clip",
            )
            self.cleaning_stats["dataset_features"]["prices"] = price_stats

        # 3. Handle missing text fields
        text_columns = ["description", "tag", "dataset_name"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna("")
                df[col] = df[col].astype(str).str.strip()

        # 4. Fix timestamps
        if "create_time" in df.columns:
            df, time_stats = self.fix_timestamps(
                df,
                column="create_time",
                max_future_days=0,
            )
            self.cleaning_stats["dataset_features"]["timestamps"] = time_stats

        # Save cleaned data
        output_path = self.cleaned_dir / "dataset_features.parquet"
        df.to_parquet(output_path, index=False)
        LOGGER.info("Saved cleaned dataset features: %d rows → %s", len(df), output_path)

        return df

    def clean_user_profile(self) -> pd.DataFrame:
        """Clean user profile table."""
        LOGGER.info("Cleaning user_profile table...")

        profile_path = self.processed_dir / "user_profile.parquet"
        if not profile_path.exists():
            LOGGER.warning("User profile file not found: %s", profile_path)
            return pd.DataFrame()

        df = pd.read_parquet(profile_path)
        LOGGER.info("Loaded %d user profiles", len(df))

        # 1. Remove duplicates
        df, dup_stats = self.remove_duplicates(
            df,
            subset=["user_id"],
            keep="first",
        )
        self.cleaning_stats["user_profile"]["duplicates"] = dup_stats

        # 2. Fill missing categorical values
        categorical_columns = ["province", "city", "company_name"]
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        # Save cleaned data
        output_path = self.cleaned_dir / "user_profile.parquet"
        df.to_parquet(output_path, index=False)
        LOGGER.info("Saved cleaned user profile: %d rows → %s", len(df), output_path)

        return df

    def clean_all(self) -> Dict[str, pd.DataFrame]:
        """Clean all tables and return cleaned DataFrames."""
        LOGGER.info("Starting data cleaning process...")

        interactions = self.clean_interactions()
        dataset_features = self.clean_dataset_features()
        user_profile = self.clean_user_profile()

        # Save cleaning statistics
        stats_path = self.cleaned_dir / "cleaning_stats.json"
        import json
        stats_path.write_text(json.dumps(self.cleaning_stats, indent=2))
        LOGGER.info("Cleaning statistics saved to %s", stats_path)

        # Print summary
        print("\n" + "=" * 80)
        print("DATA CLEANING SUMMARY")
        print("=" * 80)
        print(f"Interactions: {len(interactions):,} rows")
        print(f"Dataset Features: {len(dataset_features):,} rows")
        print(f"User Profile: {len(user_profile):,} rows")
        print(f"\nCleaned data saved to: {self.cleaned_dir}")
        print("=" * 80 + "\n")

        return {
            "interactions": interactions,
            "dataset_features": dataset_features,
            "user_profile": user_profile,
        }


def main() -> None:
    """Run data cleaning."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_all()

    LOGGER.info("Data cleaning completed successfully!")


if __name__ == "__main__":
    main()
