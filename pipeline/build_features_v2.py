"""Enhanced feature engineering with 75+ features for recommendation models (now with image features!)."""
from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from config.settings import BASE_DIR, DATA_DIR, FEATURE_STORE_PATH
from pipeline.image_features import ImageFeatureExtractor

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DIR = DATA_DIR / "cleaned"
BUSINESS_DIR = DATA_DIR / "business"


class FeatureEngineV2:
    """Advanced feature engineering for recommendation system."""

    def __init__(self, use_cleaned: bool = True):
        """
        Initialize feature engineer.

        Args:
            use_cleaned: Use cleaned data instead of processed data
        """
        self.data_dir = CLEANED_DIR if use_cleaned else PROCESSED_DIR
        self.output_dir = PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_user_features_v2(
        self,
        interactions: pd.DataFrame,
        user_profile: pd.DataFrame,
        dataset_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build enhanced user features (30+ features).

        Features:
        - Basic profile (5): company_name, province, city, is_consumption
        - Behavior statistics (10): purchase_count, total_spent, avg_price, etc.
        - Time features (8): days_since_first, days_since_last, purchase_frequency, etc.
        - Preference features (10+): preferred_tags, price_range, category_diversity, etc.
        """
        LOGGER.info("Building enhanced user features...")

        if interactions.empty:
            return pd.DataFrame()

        # Start with basic profile
        features = user_profile.copy()

        # === Behavior Statistics Features (10) ===
        user_stats = interactions.groupby("user_id").agg({
            "dataset_id": "count",  # purchase_count
            "weight": ["sum", "mean", "std", "min", "max"],  # spend statistics
            "last_event_time": ["min", "max"],  # time range
        })
        user_stats.columns = ["_".join(col).strip("_") for col in user_stats.columns]
        user_stats = user_stats.rename(columns={
            "dataset_id_count": "purchase_count",
            "weight_sum": "total_spent",
            "weight_mean": "avg_purchase_amount",
            "weight_std": "purchase_amount_std",
            "weight_min": "min_purchase",
            "weight_max": "max_purchase",
            "last_event_time_min": "first_purchase_time",
            "last_event_time_max": "last_purchase_time",
        })

        # === Time Features (8) ===
        now = datetime.now()
        user_stats["first_purchase_time"] = pd.to_datetime(user_stats["first_purchase_time"])
        user_stats["last_purchase_time"] = pd.to_datetime(user_stats["last_purchase_time"])

        user_stats["days_since_first_purchase"] = (
            now - user_stats["first_purchase_time"]
        ).dt.days

        user_stats["days_since_last_purchase"] = (
            now - user_stats["last_purchase_time"]
        ).dt.days

        user_stats["purchase_span_days"] = (
            user_stats["last_purchase_time"] - user_stats["first_purchase_time"]
        ).dt.days

        user_stats["purchase_frequency"] = (
            user_stats["purchase_count"] / (user_stats["purchase_span_days"] + 1)
        )

        # Recency score (higher = more recent)
        user_stats["recency_score"] = 1 / (user_stats["days_since_last_purchase"] + 1)

        # Activity level
        user_stats["is_active_user"] = (
            user_stats["days_since_last_purchase"] < 30
        ).astype(int)

        user_stats["is_power_user"] = (user_stats["purchase_count"] >= 10).astype(int)

        # === Preference Features (10+) ===
        # Price preference
        user_price_stats = self._compute_user_price_preferences(
            interactions, dataset_features
        )

        # Tag preferences
        user_tag_prefs = self._compute_user_tag_preferences(
            interactions, dataset_features
        )

        # Category diversity
        user_diversity = self._compute_user_diversity(interactions, dataset_features)

        # Merge all features
        features = features.merge(user_stats, on="user_id", how="left")
        features = features.merge(user_price_stats, on="user_id", how="left")
        features = features.merge(user_tag_prefs, on="user_id", how="left")
        features = features.merge(user_diversity, on="user_id", how="left")

        # Fill missing values
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features[numeric_columns] = features[numeric_columns].fillna(0)

        LOGGER.info("Built user features: %d users, %d features", len(features), len(features.columns))

        return features

    def _compute_user_price_preferences(
        self,
        interactions: pd.DataFrame,
        dataset_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute user price preferences."""
        # Merge to get prices
        merged = interactions.merge(
            dataset_features[["dataset_id", "price"]],
            on="dataset_id",
            how="left",
        )

        price_stats = merged.groupby("user_id")["price"].agg([
            ("preferred_price_min", "min"),
            ("preferred_price_max", "max"),
            ("preferred_price_mean", "mean"),
            ("preferred_price_median", "median"),
        ]).reset_index()

        # Price range (buckets)
        price_stats["preferred_price_range"] = pd.cut(
            price_stats["preferred_price_mean"],
            bins=[0, 100, 500, 1000, 5000, float("inf")],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

        return price_stats

    def _compute_user_tag_preferences(
        self,
        interactions: pd.DataFrame,
        dataset_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute user tag preferences."""
        # Merge to get tags
        merged = interactions.merge(
            dataset_features[["dataset_id", "tag"]],
            on="dataset_id",
            how="left",
        )

        # Extract tag distribution per user
        user_tags = []
        for user_id, group in merged.groupby("user_id"):
            all_tags = []
            for tags_str in group["tag"].dropna():
                if tags_str:
                    all_tags.extend([t.strip() for t in str(tags_str).split(";")])

            if all_tags:
                tag_counter = Counter(all_tags)
                most_common = tag_counter.most_common(3)

                user_tags.append({
                    "user_id": user_id,
                    "tag_diversity": len(tag_counter),
                    "top_tag_1": most_common[0][0] if len(most_common) > 0 else "",
                    "top_tag_1_count": most_common[0][1] if len(most_common) > 0 else 0,
                    "top_tag_2": most_common[1][0] if len(most_common) > 1 else "",
                    "top_tag_2_count": most_common[1][1] if len(most_common) > 1 else 0,
                    "top_tag_3": most_common[2][0] if len(most_common) > 2 else "",
                    "top_tag_3_count": most_common[2][1] if len(most_common) > 2 else 0,
                })

        return pd.DataFrame(user_tags) if user_tags else pd.DataFrame({"user_id": []})

    def _compute_user_diversity(
        self,
        interactions: pd.DataFrame,
        dataset_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute user's purchase diversity."""
        user_diversity = interactions.groupby("user_id").agg({
            "dataset_id": lambda x: x.nunique() / len(x),  # uniqueness ratio
        }).reset_index()

        user_diversity.columns = ["user_id", "purchase_uniqueness_ratio"]

        return user_diversity

    def build_dataset_features_v2(
        self,
        dataset: pd.DataFrame,
        dataset_stats: pd.DataFrame,
        interactions: pd.DataFrame,
        dataset_image: pd.DataFrame | None = None,
        image_embeddings: pd.DataFrame | None = None,
        text_embeddings: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build enhanced dataset features (30+ features including image features).

        Features:
        - Basic (5): id, name, description, tag, price, company
        - Text features (5): description_length, tag_count, word_count, etc.
        - Statistics (8): interaction_count, total_revenue, avg_price_per_interaction, etc.
        - Time features (5): days_since_created, days_since_last_purchase, etc.
        - Popularity (3): popularity_score, rank, percentile
        - Image features (10+): image_count, has_cover, image_freshness, etc.
        """
        LOGGER.info("Building enhanced dataset features (with images)...")

        features = dataset.copy()

        # === Image Features (10+) NEW! ===
        if dataset_image is not None and not dataset_image.empty:
            LOGGER.info("Adding image features...")
            image_extractor = ImageFeatureExtractor(self.data_dir)
            image_features = image_extractor.extract_image_features(dataset, dataset_image)

            # Merge image features
            features = features.merge(
                image_features[[
                    "dataset_id", "image_count", "has_images", "has_cover",
                    "avg_image_order", "image_freshness_days", "image_freshness_score",
                    "image_update_frequency", "image_richness_score", "cover_position"
                ]],
                on="dataset_id",
                how="left",
            )

            LOGGER.info("Added %d image features", len([c for c in features.columns if "image" in c or "cover" in c]))
        else:
            LOGGER.warning("No image data provided, skipping image features")

        if image_embeddings is not None and not image_embeddings.empty:
            embedding_dims = len([c for c in image_embeddings.columns if c.startswith("image_embed_mean_")])
            LOGGER.info(
                "Merging visual embeddings (%d datasets, %d dims)...",
                len(image_embeddings),
                embedding_dims,
            )
            features = features.merge(image_embeddings, on="dataset_id", how="left")
        else:
            LOGGER.warning("No visual embeddings available; downstream models will rely on text-only signals")

        if text_embeddings is not None and not text_embeddings.empty:
            text_dims = len([c for c in text_embeddings.columns if c.startswith("text_embed_")])
            LOGGER.info(
                "Merging text embeddings (%d datasets, %d dims)...",
                len(text_embeddings),
                text_dims,
            )
            features = features.merge(text_embeddings, on="dataset_id", how="left")
        else:
            LOGGER.warning("No text embeddings available; consider running pipeline.text_embeddings before build_features.")

        # === Text Features (5) ===
        features["description_length"] = (
            features["description"].fillna("").str.len()
        )

        features["tag_count"] = features["tag"].apply(
            lambda x: len([t for t in str(x).split(";") if t.strip()]) if pd.notna(x) else 0
        )

        features["word_count"] = features["description"].fillna("").str.split().str.len()

        features["has_description"] = (features["description_length"] > 0).astype(int)
        features["has_tags"] = (features["tag_count"] > 0).astype(int)

        # === Statistics Features (8) ===
        if not dataset_stats.empty:
            features = features.merge(dataset_stats, on="dataset_id", how="left")

            features["interaction_count"] = features["interaction_count"].fillna(0)
            features["total_weight"] = features["total_weight"].fillna(0)

            features["popularity_score"] = np.log1p(features["interaction_count"])

            features["avg_price_per_interaction"] = (
                features["total_weight"] / (features["interaction_count"] + 1)
            )

            # Revenue metrics
            features["estimated_revenue"] = features["total_weight"]  # Already weighted by price
            features["revenue_per_interaction"] = (
                features["estimated_revenue"] / (features["interaction_count"] + 1)
            )

        # === Time Features (5) ===
        now = datetime.now()
        if "create_time" in features.columns:
            features["create_time"] = pd.to_datetime(features["create_time"], errors="coerce")

            features["days_since_created"] = (
                now - features["create_time"]
            ).dt.days

            features["age_score"] = 1 / (features["days_since_created"] + 1)

        if "last_event_time" in features.columns:
            features["last_event_time"] = pd.to_datetime(features["last_event_time"], errors="coerce")

            features["days_since_last_purchase"] = (
                now - features["last_event_time"]
            ).dt.days

            features["freshness_score"] = 1 / (features["days_since_last_purchase"] + 1)

        # === Popularity Features (3) ===
        if "interaction_count" in features.columns:
            # Rank by popularity
            features["popularity_rank"] = features["interaction_count"].rank(
                method="dense", ascending=False
            )

            # Percentile
            features["popularity_percentile"] = features["interaction_count"].rank(
                pct=True
            )

        # === Price Features (3) ===
        if "price" in features.columns:
            # Price buckets
            features["price_bucket"] = pd.cut(
                features["price"],
                bins=[0, 100, 500, 1000, 5000, float("inf")],
                labels=[0, 1, 2, 3, 4],
            ).astype(float)

            # Normalized price
            price_mean = features["price"].mean()
            price_std = features["price"].std()
            features["price_normalized"] = (
                (features["price"] - price_mean) / (price_std + 1)
            )

        # Fill missing values
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features[numeric_columns] = features[numeric_columns].fillna(0)

        LOGGER.info(
            "Built dataset features: %d datasets, %d features",
            len(features),
            len(features.columns),
        )

        return features

    def build_cross_features(
        self,
        user_features: pd.DataFrame,
        dataset_features: pd.DataFrame,
        sample_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Build user-item cross features (for training data).

        Args:
            user_features: User features DataFrame
            dataset_features: Dataset features DataFrame
            sample_size: Number of samples to generate

        Returns:
            DataFrame with cross features
        """
        LOGGER.info("Building cross features for %d samples...", sample_size)

        # Sample user-item pairs (for demonstration)
        # In production, this would be based on actual interactions + negative samples

        users = user_features["user_id"].sample(n=min(sample_size, len(user_features)))
        items = dataset_features["dataset_id"].sample(n=min(sample_size, len(dataset_features)))

        # Create cross product
        cross = pd.DataFrame({
            "user_id": users.values,
            "dataset_id": items.values,
        })

        # Merge user and item features
        cross = cross.merge(user_features, on="user_id", how="left")
        cross = cross.merge(dataset_features, on="dataset_id", how="left", suffixes=("_user", "_item"))

        # === Cross Features (20+) ===

        # Price match
        if "preferred_price_mean" in cross.columns and "price" in cross.columns:
            cross["price_match_score"] = 1 / (
                1 + abs(cross["price"] - cross["preferred_price_mean"])
            )

            cross["price_within_range"] = (
                (cross["price"] >= cross["preferred_price_min"])
                & (cross["price"] <= cross["preferred_price_max"])
            ).astype(int)

        # Recency match
        if "recency_score" in cross.columns and "freshness_score" in cross.columns:
            cross["recency_freshness_product"] = (
                cross["recency_score"] * cross["freshness_score"]
            )

        # Activity match
        if "is_active_user" in cross.columns and "popularity_score" in cross.columns:
            cross["active_user_popular_item"] = (
                cross["is_active_user"] * cross["popularity_score"]
            )

        # Price affordability
        if "avg_purchase_amount" in cross.columns and "price" in cross.columns:
            cross["price_affordability_ratio"] = (
                cross["price"] / (cross["avg_purchase_amount"] + 1)
            )

        LOGGER.info("Built cross features: %d samples, %d total features", len(cross), len(cross.columns))

        return cross

    def save_features(
        self,
        user_features: pd.DataFrame,
        dataset_features: pd.DataFrame,
        interactions: pd.DataFrame,
        dataset_stats: pd.DataFrame,
    ) -> None:
        """Save all features to parquet and SQLite."""
        # Save enhanced versions
        user_features.to_parquet(self.output_dir / "user_features_v2.parquet", index=False)
        dataset_features.to_parquet(self.output_dir / "dataset_features_v2.parquet", index=False)
        interactions.to_parquet(self.output_dir / "interactions_v2.parquet", index=False)
        dataset_stats.to_parquet(self.output_dir / "dataset_stats_v2.parquet", index=False)

        # Overwrite legacy files for backward compatibility
        user_features.to_parquet(self.output_dir / "user_profile.parquet", index=False)
        dataset_features.to_parquet(self.output_dir / "dataset_features.parquet", index=False)
        interactions.to_parquet(self.output_dir / "interactions.parquet", index=False)
        dataset_stats.to_parquet(self.output_dir / "dataset_stats.parquet", index=False)

        LOGGER.info("Saved enhanced features to parquet files (v2 and legacy views)")

        # Sync to SQLite feature store (both legacy + v2 tables)
        self._sync_to_feature_store({
            "user_features_v2": user_features,
            "dataset_features_v2": dataset_features,
            "interactions_v2": interactions,
            "dataset_stats_v2": dataset_stats,
            "user_profile": user_features,
            "dataset_features": dataset_features,
            "interactions": interactions,
            "dataset_stats": dataset_stats,
        })

    def _sync_to_feature_store(self, tables: Dict[str, pd.DataFrame]) -> None:
        """Sync features to SQLite feature store."""
        FEATURE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(FEATURE_STORE_PATH, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            for name, df in tables.items():
                if df.empty:
                    continue

                # Convert timestamps to strings
                df_copy = df.copy()
                for col in df_copy.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                conn.execute(f'DROP TABLE IF EXISTS "{name}"')
                df_copy.to_sql(name, conn, if_exists="replace", index=False)

                # Create indexes
                if "user_id" in df_copy.columns:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_user ON {name} (user_id)")
                if "dataset_id" in df_copy.columns:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_dataset ON {name} (dataset_id)")

                LOGGER.info("Synced %s to feature store: %d rows", name, len(df))


def main() -> None:
    """Build enhanced features (with image features!)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = FeatureEngineV2(use_cleaned=True)

    # Load data
    LOGGER.info("Loading data...")
    interactions = pd.read_parquet(CLEANED_DIR / "interactions.parquet")
    dataset = pd.read_parquet(CLEANED_DIR / "dataset_features.parquet")
    user_profile = pd.read_parquet(CLEANED_DIR / "user_profile.parquet")

    # Load image data (NEW!)
    dataset_image_path = BUSINESS_DIR / "dataset_image.parquet"
    if dataset_image_path.exists():
        LOGGER.info("Loading dataset images...")
        dataset_image = pd.read_parquet(dataset_image_path)
        LOGGER.info("Loaded %d image records", len(dataset_image))
    else:
        LOGGER.warning("Image data not found at %s, skipping image features", dataset_image_path)
        dataset_image = None

    embeddings_path = PROCESSED_DIR / "dataset_image_embeddings.parquet"
    if embeddings_path.exists():
        LOGGER.info("Loading precomputed visual embeddings...")
        image_embeddings = pd.read_parquet(embeddings_path)
        LOGGER.info("Loaded %d visual embedding rows", len(image_embeddings))
    else:
        fallback_path = BASE_DIR / "cache" / "dataset_image_embeddings.parquet"
        if fallback_path.exists():
            LOGGER.info("Using fallback embeddings from %s", fallback_path)
            image_embeddings = pd.read_parquet(fallback_path)
        else:
            LOGGER.warning("Visual embeddings not found at %s, proceeding without them", embeddings_path)
            image_embeddings = None

    # Build dataset stats
    dataset_stats = interactions.groupby("dataset_id").agg({
        "dataset_id": "count",
        "weight": "sum",
        "last_event_time": "max",
    }).rename(columns={
        "dataset_id": "interaction_count",
        "weight": "total_weight",
    }).reset_index()

    # Build enhanced features (with images!)
    user_features = engine.build_user_features_v2(interactions, user_profile, dataset)
    dataset_features = engine.build_dataset_features_v2(
        dataset,
        dataset_stats,
        interactions,
        dataset_image,
        image_embeddings,
    )

    # Save features
    engine.save_features(user_features, dataset_features, interactions, dataset_stats)

    # Print summary
    print("\n" + "=" * 80)
    print("ENHANCED FEATURE ENGINEERING SUMMARY (WITH IMAGE FEATURES!)")
    print("=" * 80)
    print(f"User Features: {len(user_features):,} users × {len(user_features.columns)} features")
    print(f"Dataset Features: {len(dataset_features):,} items × {len(dataset_features.columns)} features")
    print(f"Interactions: {len(interactions):,} records")
    if dataset_image is not None:
        print(f"Image Records: {len(dataset_image):,}")
        print(f"Datasets with images: {dataset_features['has_images'].sum():,} ({100*dataset_features['has_images'].mean():.1f}%)")
    print(f"\nFeature files saved to: {engine.output_dir}")
    print("=" * 80 + "\n")

    LOGGER.info("Enhanced feature engineering completed!")


if __name__ == "__main__":
    main()
