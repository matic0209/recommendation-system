"""
Image feature extraction for recommendation system.

This module provides statistical image features without requiring image downloads.
For visual features using deep learning, see image_features_visual.py.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


class ImageFeatureExtractor:
    """Extract statistical features from dataset images."""

    def __init__(self, data_dir: Path):
        """
        Initialize image feature extractor.

        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir

    def extract_image_features(
        self,
        dataset: pd.DataFrame,
        dataset_image: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract statistical image features for datasets.

        Features extracted:
        - image_count: Number of images per dataset
        - has_images: Boolean indicator
        - has_cover: Whether dataset has a cover image
        - avg_image_order: Average image order (relevance ranking)
        - image_freshness: Days since most recent image update
        - image_update_frequency: Image updates over time span
        - cover_position: Position of cover image in gallery

        Args:
            dataset: Dataset DataFrame with dataset_id and cover_id
            dataset_image: Dataset images DataFrame

        Returns:
            DataFrame with image features per dataset_id
        """
        LOGGER.info("Extracting image features...")

        # Basic image count per dataset
        image_stats = dataset_image.groupby("dataset_id").agg({
            "id": "count",  # image_count
            "image_order": ["mean", "min", "max"],
            "update_time": ["max", "min"],
        })

        image_stats.columns = ["_".join(col).strip("_") for col in image_stats.columns]
        image_stats = image_stats.rename(columns={
            "id_count": "image_count",
            "image_order_mean": "avg_image_order",
            "image_order_min": "min_image_order",
            "image_order_max": "max_image_order",
            "update_time_max": "latest_image_time",
            "update_time_min": "first_image_time",
        })

        image_stats = image_stats.reset_index()

        # Image freshness (days since last image update)
        now = datetime.now()
        image_stats["latest_image_time"] = pd.to_datetime(
            image_stats["latest_image_time"], errors="coerce"
        )
        image_stats["first_image_time"] = pd.to_datetime(
            image_stats["first_image_time"], errors="coerce"
        )

        image_stats["image_freshness_days"] = (
            now - image_stats["latest_image_time"]
        ).dt.days

        # Image update span
        image_stats["image_update_span_days"] = (
            image_stats["latest_image_time"] - image_stats["first_image_time"]
        ).dt.days

        # Image update frequency
        image_stats["image_update_frequency"] = (
            image_stats["image_count"] / (image_stats["image_update_span_days"] + 1)
        )

        # Freshness score (higher = more recent)
        image_stats["image_freshness_score"] = 1 / (
            image_stats["image_freshness_days"] + 1
        )

        # Merge with dataset to get cover info
        # Note: dataset table uses 'id' not 'dataset_id'
        if "dataset_id" in dataset.columns:
            features = dataset[["dataset_id", "cover_id"]].copy()
        else:
            features = dataset[["id", "cover_id"]].copy()
            features = features.rename(columns={"id": "dataset_id"})

        # Add image stats
        features = features.merge(image_stats, on="dataset_id", how="left")

        # Boolean indicators
        features["has_images"] = (features["image_count"].fillna(0) > 0).astype(int)
        features["has_cover"] = (features["cover_id"].notna()).astype(int)

        # Cover position in gallery (if available)
        if "cover_id" in features.columns:
            # Convert cover_id to int for merging
            features["cover_id"] = pd.to_numeric(features["cover_id"], errors="coerce")

            # Merge to find cover image order
            cover_info = dataset_image[["id", "dataset_id", "image_order"]].rename(
                columns={"id": "cover_id"}
            )
            features = features.merge(
                cover_info[["cover_id", "image_order"]],
                on="cover_id",
                how="left",
                suffixes=("", "_cover"),
            )
            features = features.rename(columns={"image_order": "cover_position"})

        # Image richness score (综合评分)
        features["image_richness_score"] = (
            features["has_images"] * 0.3 +
            features["has_cover"] * 0.3 +
            (features["image_count"].fillna(0) / 10).clip(0, 1) * 0.2 +
            features["image_freshness_score"].fillna(0) * 0.2
        )

        # Fill missing values
        numeric_cols = features.select_dtypes(include=["number"]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)

        LOGGER.info(
            "Extracted image features: %d datasets, %d features",
            len(features),
            len(features.columns),
        )

        # Log statistics
        LOGGER.info("Image feature statistics:")
        LOGGER.info("  - Datasets with images: %d (%.1f%%)",
                    features["has_images"].sum(),
                    100 * features["has_images"].mean())
        LOGGER.info("  - Datasets with cover: %d (%.1f%%)",
                    features["has_cover"].sum(),
                    100 * features["has_cover"].mean())
        LOGGER.info("  - Avg images per dataset: %.2f",
                    features.loc[features["has_images"] == 1, "image_count"].mean())

        return features

    def create_image_similarity_features(
        self,
        dataset_image: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create image-based similarity features (without visual embeddings).

        This creates a simple similarity based on:
        - Datasets sharing similar image update patterns
        - Datasets with similar number of images

        Args:
            dataset_image: Dataset images DataFrame

        Returns:
            DataFrame with image similarity scores
        """
        LOGGER.info("Creating image-based similarity features...")

        # Group by dataset
        dataset_groups = dataset_image.groupby("dataset_id").agg({
            "id": "count",
            "image_order": "mean",
        }).rename(columns={
            "id": "image_count",
            "image_order": "avg_order",
        })

        # For now, return the aggregated stats
        # This can be extended to compute actual similarity matrix
        return dataset_groups.reset_index()


def main() -> None:
    """Test image feature extraction."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    from config.settings import DATA_DIR

    # Load data
    dataset = pd.read_parquet(DATA_DIR / "business" / "dataset.parquet")
    dataset_image = pd.read_parquet(DATA_DIR / "business" / "dataset_image.parquet")

    # Extract features
    extractor = ImageFeatureExtractor(DATA_DIR)
    image_features = extractor.extract_image_features(dataset, dataset_image)

    # Save for inspection
    output_path = DATA_DIR / "processed" / "image_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_features.to_parquet(output_path, index=False)

    print("\n" + "=" * 80)
    print("IMAGE FEATURE EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total datasets: {len(image_features):,}")
    print(f"Datasets with images: {image_features['has_images'].sum():,}")
    print(f"Datasets with cover: {image_features['has_cover'].sum():,}")
    print(f"\nFeatures extracted: {len(image_features.columns)}")
    print(f"Output saved to: {output_path}")
    print("=" * 80 + "\n")

    # Show sample
    print("Sample image features:")
    print(image_features[[
        "dataset_id", "image_count", "has_cover",
        "image_freshness_score", "image_richness_score"
    ]].head(10))


if __name__ == "__main__":
    main()
