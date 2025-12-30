"""
Tag Enhancement Script using Zero-Shot Classification.

Uses Erlangshen-Roberta-110M-NLI model to extract standardized category tags
from item descriptions, improving tag coverage and quality.

Usage:
    python -m pipeline.enhance_tags [--batch-size 32] [--threshold 0.5]

Environment:
    HF_ENDPOINT: HuggingFace mirror URL (default: https://hf-mirror.com)
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# Configure HuggingFace mirror before importing transformers
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = HF_ENDPOINT

from transformers import pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Standardized top-level categories
TOP_CATEGORIES = [
    "金融",
    "医疗健康",
    "政府政务",
    "交通运输",
    "教育培训",
    "能源环保",
    "农业",
    "科技",
    "商业零售",
    "文化娱乐",
    "社会民生",
]

# Model configuration
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-NLI"
DEFAULT_BATCH_SIZE = 32
DEFAULT_THRESHOLD = 0.5
MAX_TEXT_LENGTH = 512  # Max chars for classification input


def clean_html(text: str) -> str:
    """Remove HTML tags and clean text for NLP processing."""
    if not text or pd.isna(text):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", str(text))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to max length, preserving whole words."""
    if len(text) <= max_length:
        return text

    # Find last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        return truncated[:last_space]
    return truncated


def load_classifier():
    """Load zero-shot classification pipeline."""
    LOGGER.info("Loading zero-shot classifier: %s", MODEL_NAME)
    LOGGER.info("HuggingFace endpoint: %s", HF_ENDPOINT)

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=-1,  # CPU
    )

    LOGGER.info("Classifier loaded successfully")
    return classifier


def classify_description(
    classifier,
    text: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[str]:
    """
    Classify text into top categories using zero-shot classification.

    Args:
        classifier: HuggingFace zero-shot classification pipeline
        text: Cleaned description text
        threshold: Minimum score to include a category

    Returns:
        List of category labels with score >= threshold
    """
    if not text or len(text) < 10:
        return []

    try:
        result = classifier(
            text,
            TOP_CATEGORIES,
            multi_label=True,
            hypothesis_template="这段文本是关于{}的。",
        )

        # Extract categories above threshold
        categories = [
            label
            for label, score in zip(result["labels"], result["scores"])
            if score >= threshold
        ]

        return categories[:3]  # Max 3 categories per item

    except Exception as e:
        LOGGER.warning("Classification failed for text: %s... Error: %s", text[:50], e)
        return []


def merge_tags(original_tags: str, new_categories: List[str]) -> str:
    """
    Merge original user tags with new AI-classified categories.

    Args:
        original_tags: Semicolon-separated original tags
        new_categories: List of new category labels

    Returns:
        Semicolon-separated merged tags
    """
    # Parse original tags
    original = []
    if original_tags and pd.notna(original_tags):
        original = [t.strip() for t in str(original_tags).split(";") if t.strip()]

    # Merge and deduplicate (case-insensitive)
    seen = set()
    merged = []

    for tag in original + new_categories:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            seen.add(tag_lower)
            merged.append(tag)

    return ";".join(merged)


def build_tag_indices(
    enhanced_df: pd.DataFrame,
    tag_column: str = "tag_enhanced",
) -> Tuple[Dict[str, List[int]], Dict[int, List[str]]]:
    """
    Build tag-to-items and item-to-tags indices from enhanced tags.

    Args:
        enhanced_df: DataFrame with enhanced tags
        tag_column: Column name for enhanced tags

    Returns:
        (tag_to_items, item_to_tags)
    """
    tag_to_items: Dict[str, Set[int]] = {}
    item_to_tags: Dict[int, List[str]] = {}

    for _, row in enhanced_df.iterrows():
        dataset_id = int(row["dataset_id"])
        tags_str = row.get(tag_column, "")

        if not tags_str or pd.isna(tags_str):
            continue

        tags = [t.strip().lower() for t in str(tags_str).split(";") if t.strip()]
        item_to_tags[dataset_id] = tags

        for tag in tags:
            if tag not in tag_to_items:
                tag_to_items[tag] = set()
            tag_to_items[tag].add(dataset_id)

    # Convert sets to lists for JSON serialization
    tag_to_items_serializable = {k: list(v) for k, v in tag_to_items.items()}

    return tag_to_items_serializable, item_to_tags


def enhance_tags(
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Main function to enhance tags for all items.

    Args:
        batch_size: Number of items to process in each batch
        threshold: Minimum classification score to include a category
        dry_run: If True, only process first 100 items for testing

    Returns:
        DataFrame with enhanced tags
    """
    # Load data
    input_path = DATA_DIR / "cleaned" / "dataset_features.parquet"
    LOGGER.info("Loading data from %s", input_path)
    df = pd.read_parquet(input_path)
    LOGGER.info("Loaded %d items", len(df))

    if dry_run:
        df = df.head(100)
        LOGGER.info("Dry run: processing first 100 items only")

    # Load classifier
    classifier = load_classifier()

    # Process items
    enhanced_tags = []
    new_categories_list = []
    start_time = time.time()

    LOGGER.info("Starting tag enhancement with threshold=%.2f", threshold)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Enhancing tags"):
        # Clean and truncate description
        description = clean_html(row.get("description", ""))
        description = truncate_text(description)

        # Classify
        new_categories = classify_description(classifier, description, threshold)
        new_categories_list.append(new_categories)

        # Merge with original tags
        original_tags = row.get("tag", "")
        merged = merge_tags(original_tags, new_categories)
        enhanced_tags.append(merged)

        # Log progress every 1000 items
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            LOGGER.info(
                "Progress: %d/%d (%.1f items/sec)", i + 1, len(df), rate
            )

    # Add enhanced tags to DataFrame
    df["tag_enhanced"] = enhanced_tags
    df["new_categories"] = new_categories_list

    # Calculate statistics
    original_coverage = (df["tag"].notna() & (df["tag"] != "")).mean()
    enhanced_coverage = (df["tag_enhanced"].notna() & (df["tag_enhanced"] != "")).mean()

    LOGGER.info("=" * 60)
    LOGGER.info("Enhancement complete!")
    LOGGER.info("Original tag coverage: %.1f%%", original_coverage * 100)
    LOGGER.info("Enhanced tag coverage: %.1f%%", enhanced_coverage * 100)
    LOGGER.info("Total time: %.1f seconds", time.time() - start_time)

    # Count items with new categories added
    items_with_new_cats = sum(1 for cats in new_categories_list if cats)
    LOGGER.info(
        "Items with new categories: %d (%.1f%%)",
        items_with_new_cats,
        items_with_new_cats / len(df) * 100,
    )

    # Category distribution
    all_new_cats = [cat for cats in new_categories_list for cat in cats]
    if all_new_cats:
        from collections import Counter
        cat_counts = Counter(all_new_cats)
        LOGGER.info("Category distribution:")
        for cat, count in cat_counts.most_common():
            LOGGER.info("  %s: %d", cat, count)

    return df


def save_outputs(df: pd.DataFrame, dry_run: bool = False):
    """Save enhanced data and tag indices."""
    suffix = "_test" if dry_run else ""

    # Save enhanced parquet
    output_path = DATA_DIR / "processed" / f"dataset_features_enhanced{suffix}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Saved enhanced features to %s", output_path)

    # Build and save tag indices
    tag_to_items, item_to_tags = build_tag_indices(df)

    tag_to_items_path = MODELS_DIR / f"tag_to_items_enhanced{suffix}.json"
    item_to_tags_path = MODELS_DIR / f"item_to_tags_enhanced{suffix}.json"

    with open(tag_to_items_path, "w", encoding="utf-8") as f:
        json.dump(tag_to_items, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved tag_to_items index to %s (%d tags)", tag_to_items_path, len(tag_to_items))

    with open(item_to_tags_path, "w", encoding="utf-8") as f:
        json.dump(item_to_tags, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved item_to_tags index to %s (%d items)", item_to_tags_path, len(item_to_tags))

    return output_path, tag_to_items_path, item_to_tags_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhance tags using zero-shot classification")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for processing (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Classification threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only first 100 items for testing",
    )
    args = parser.parse_args()

    LOGGER.info("=" * 60)
    LOGGER.info("Tag Enhancement Script")
    LOGGER.info("=" * 60)
    LOGGER.info("Model: %s", MODEL_NAME)
    LOGGER.info("Categories: %s", TOP_CATEGORIES)
    LOGGER.info("Threshold: %.2f", args.threshold)
    LOGGER.info("Dry run: %s", args.dry_run)
    LOGGER.info("=" * 60)

    # Run enhancement
    df = enhance_tags(
        batch_size=args.batch_size,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )

    # Save outputs
    save_outputs(df, dry_run=args.dry_run)

    LOGGER.info("=" * 60)
    LOGGER.info("Done!")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
