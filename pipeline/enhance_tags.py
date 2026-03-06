"""
Tag Enhancement Script using Zero-Shot Classification.

Uses Erlangshen-Roberta-330M-NLI model to extract standardized category tags
from item descriptions, improving tag coverage and quality.

Usage:
    python -m pipeline.enhance_tags [--batch-size 8] [--threshold 0.5] [--missing-only]

Optimizations:
    - Batch processing for 5-10x speedup
    - --missing-only flag to only process items without tags

Environment:
    HF_ENDPOINT: HuggingFace mirror URL (default: https://hf-mirror.com)
"""

import html
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# Configure HuggingFace mirror before importing transformers
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = HF_ENDPOINT

# Offline mode configuration:
# - Set TRANSFORMERS_OFFLINE=1 to use cached models only (faster, no network)
# - Set TRANSFORMERS_OFFLINE=0 to allow downloading new models
# Default: allow online download for first-time model fetch
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "0")

from transformers import pipeline, AutoTokenizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Standardized top-level categories for data trading platform (12 categories)
# Covers major data asset domains based on mainstream data exchange platforms
TOP_CATEGORIES = [
    "政府政务",      # 政策法规、公共服务、行政管理
    "金融财经",      # 银行、证券、保险、支付
    "医疗健康",      # 医药、医院、健康管理
    "交通物流",      # 出行、物流、地理位置
    "教育科研",      # 学校、培训、科学研究
    "工业制造",      # 工厂、生产、设备制造
    "商业零售",      # 电商、消费、营销
    "能源环保",      # 电力、石油、环境保护
    "文化娱乐",      # 媒体、游戏、影视
    "农业农村",      # 农产品、土地、畜牧
    "互联网科技",    # IT、软件、AI、通信
    "社会民生",      # 人口、就业、社保、公共服务
    "数字产品",      # 虚拟商品、鼠标皮肤、壁纸、工具软件
]

# Model configuration
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-330M-NLI"
DEFAULT_BATCH_SIZE = 8  # Batch size for CPU processing
DEFAULT_THRESHOLD = 0.5
MAX_TEXT_LENGTH = 256  # Shorter for faster processing


def clean_html(text: str) -> str:
    """Remove HTML tags and clean text for NLP processing."""
    if not text or pd.isna(text):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", str(text))

    # Decode HTML entities (e.g., &ldquo; -> ", &mdash; -> —)
    text = html.unescape(text)

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
    """Load zero-shot classification pipeline with optimizations."""
    LOGGER.info("Loading zero-shot classifier: %s", MODEL_NAME)
    LOGGER.info("HuggingFace endpoint: %s", HF_ENDPOINT)

    # Check if we should use offline mode (skip remote version checks)
    use_offline = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
    if use_offline:
        LOGGER.info("Offline mode enabled - using cached model without remote checks")

    # Load tokenizer with max_length to avoid warnings
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=use_offline)

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        device=-1,  # CPU
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        model_kwargs={"local_files_only": use_offline} if use_offline else {},
    )

    LOGGER.info("Classifier loaded successfully")
    return classifier


def classify_batch(
    classifier,
    texts: List[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> List[List[str]]:
    """
    Classify a batch of texts into top categories.

    Args:
        classifier: HuggingFace zero-shot classification pipeline
        texts: List of cleaned description texts
        threshold: Minimum score to include a category

    Returns:
        List of category lists for each text
    """
    # Filter out empty texts
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if text and len(text) >= 10:
            valid_indices.append(i)
            valid_texts.append(text)

    # Initialize results with empty lists
    results = [[] for _ in texts]

    if not valid_texts:
        return results

    try:
        # Batch classification
        batch_results = classifier(
            valid_texts,
            TOP_CATEGORIES,
            multi_label=True,
            hypothesis_template="这段文本是关于{}的。",
            batch_size=len(valid_texts),
        )

        # Handle single result (not a list)
        if isinstance(batch_results, dict):
            batch_results = [batch_results]

        # Extract categories above threshold
        for idx, result in zip(valid_indices, batch_results):
            categories = [
                label
                for label, score in zip(result["labels"], result["scores"])
                if score >= threshold
            ][:3]  # Max 3 categories per item
            results[idx] = categories

    except Exception as e:
        LOGGER.warning("Batch classification failed: %s", e)

    return results


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


def build_category_indices(
    enhanced_df: pd.DataFrame,
    category_column: str = "new_categories",
) -> Tuple[Dict[str, List[int]], Dict[int, List[str]]]:
    """
    Build category-to-items and item-to-categories indices from 12 standard categories.

    Unlike build_tag_indices which uses merged tags, this function uses ONLY
    the AI-classified 12 standard categories for clean category matching.

    Args:
        enhanced_df: DataFrame with new_categories column
        category_column: Column name for categories (default: new_categories)

    Returns:
        (category_to_items, item_to_categories)
    """
    category_to_items: Dict[str, Set[int]] = {}
    item_to_categories: Dict[int, List[str]] = {}

    for _, row in enhanced_df.iterrows():
        dataset_id = int(row["dataset_id"])
        categories = row.get(category_column, [])

        # Handle various formats: list, string, or None
        if categories is None or (isinstance(categories, float) and pd.isna(categories)):
            continue

        if isinstance(categories, str):
            # If stored as string, try to parse as list
            if categories.startswith("["):
                try:
                    import ast
                    categories = ast.literal_eval(categories)
                except (ValueError, SyntaxError):
                    categories = [c.strip() for c in categories.split(";") if c.strip()]
            else:
                categories = [c.strip() for c in categories.split(";") if c.strip()]

        # Convert numpy array to list if needed
        if hasattr(categories, "tolist"):
            categories = categories.tolist()

        if not categories or len(categories) == 0:
            continue

        # Normalize to lowercase for consistent matching
        categories = [c.lower() for c in categories]
        item_to_categories[dataset_id] = categories

        for cat in categories:
            if cat not in category_to_items:
                category_to_items[cat] = set()
            category_to_items[cat].add(dataset_id)

    # Convert sets to lists for JSON serialization
    category_to_items_serializable = {k: list(v) for k, v in category_to_items.items()}

    LOGGER.info(
        "Built category indices: %d categories, %d items with categories",
        len(category_to_items_serializable),
        len(item_to_categories),
    )

    return category_to_items_serializable, item_to_categories


def enhance_tags(
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    dry_run: bool = False,
    missing_only: bool = False,
    since: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main function to enhance tags for all items.

    Args:
        batch_size: Number of items to process in each batch
        threshold: Minimum classification score to include a category
        dry_run: If True, only process first 100 items for testing
        missing_only: If True, only process items without existing tags
        since: Only process items created after this date (YYYY-MM-DD format)

    Returns:
        DataFrame with enhanced tags
    """
    # Load data
    input_path = DATA_DIR / "cleaned" / "dataset_features.parquet"
    LOGGER.info("Loading data from %s", input_path)
    df = pd.read_parquet(input_path)
    LOGGER.info("Loaded %d items", len(df))

    # Filter by date if specified
    if since:
        since_date = datetime.strptime(since, "%Y-%m-%d")
        if "create_time" in df.columns:
            df["create_time_dt"] = pd.to_datetime(df["create_time"], errors="coerce")
            date_mask = df["create_time_dt"] >= since_date
            df = df[date_mask].copy()
            LOGGER.info("Date filter: %d items since %s", len(df), since)
        else:
            LOGGER.warning("No create_time column found, skipping date filter")

    # Filter to items that need processing
    original_len = len(df)
    if missing_only:
        mask = df["tag"].isna() | (df["tag"] == "")
        df_to_process = df[mask].copy()
        df_with_tags = df[~mask].copy()
        LOGGER.info(
            "Missing-only mode: %d items without tags (%.1f%%), skipping %d items with tags",
            len(df_to_process),
            len(df_to_process) / original_len * 100,
            len(df_with_tags),
        )
    else:
        df_to_process = df.copy()
        df_with_tags = pd.DataFrame()

    if dry_run:
        df_to_process = df_to_process.head(100)
        LOGGER.info("Dry run: processing first 100 items only")

    if df_to_process.empty:
        LOGGER.info("No items to process!")
        df["tag_enhanced"] = df["tag"]
        df["new_categories"] = [[] for _ in range(len(df))]
        return df

    # Load classifier
    classifier = load_classifier()

    # Prepare texts
    LOGGER.info("Preparing texts...")
    texts = []
    for _, row in df_to_process.iterrows():
        description = clean_html(row.get("description", ""))
        description = truncate_text(description)
        texts.append(description)

    # Process in batches
    all_new_categories = []
    start_time = time.time()
    total_batches = (len(texts) + batch_size - 1) // batch_size

    LOGGER.info(
        "Starting batch processing: %d items, batch_size=%d, %d batches",
        len(texts),
        batch_size,
        total_batches,
    )

    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        # Classify batch
        batch_categories = classify_batch(classifier, batch_texts, threshold)
        all_new_categories.extend(batch_categories)

        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            items_done = end_idx
            rate = items_done / elapsed
            eta = (len(texts) - items_done) / rate if rate > 0 else 0
            LOGGER.info(
                "Progress: %d/%d items (%.1f items/sec, ETA: %.1f min)",
                items_done,
                len(texts),
                rate,
                eta / 60,
            )

    # Build results
    enhanced_tags = []
    for i, (_, row) in enumerate(df_to_process.iterrows()):
        original_tags = row.get("tag", "")
        new_cats = all_new_categories[i] if i < len(all_new_categories) else []
        merged = merge_tags(original_tags, new_cats)
        enhanced_tags.append(merged)

    df_to_process["tag_enhanced"] = enhanced_tags
    df_to_process["new_categories"] = all_new_categories

    # Merge back with items that already had tags (if missing_only mode)
    if missing_only and not df_with_tags.empty:
        # Try to preserve existing new_categories from enhanced parquet
        enhanced_path = DATA_DIR / "processed" / "dataset_features_enhanced.parquet"
        existing_categories = {}
        if enhanced_path.exists():
            try:
                existing_df = pd.read_parquet(enhanced_path, columns=["dataset_id", "new_categories"])
                for _, row in existing_df.iterrows():
                    cats = row.get("new_categories", [])
                    if cats is not None and (isinstance(cats, list) or (hasattr(cats, '__len__') and len(cats) > 0)):
                        existing_categories[row["dataset_id"]] = list(cats) if hasattr(cats, 'tolist') else cats
                LOGGER.info("Loaded %d existing categories from enhanced parquet", len(existing_categories))
            except Exception as e:
                LOGGER.warning("Could not load existing categories: %s", e)

        # For items with tags, preserve their existing new_categories if available
        df_with_tags["tag_enhanced"] = df_with_tags["tag"]
        df_with_tags["new_categories"] = df_with_tags["dataset_id"].apply(
            lambda x: existing_categories.get(x, [])
        )
        preserved_count = sum(1 for cats in df_with_tags["new_categories"] if cats)
        LOGGER.info("Preserved %d existing categories for items with tags", preserved_count)

        df = pd.concat([df_to_process, df_with_tags], ignore_index=True)
        df = df.sort_values("dataset_id").reset_index(drop=True)
    else:
        df = df_to_process

    # Calculate statistics
    original_coverage = (df["tag"].notna() & (df["tag"] != "")).mean()
    enhanced_coverage = (df["tag_enhanced"].notna() & (df["tag_enhanced"] != "")).mean()

    LOGGER.info("=" * 60)
    LOGGER.info("Enhancement complete!")
    LOGGER.info("Original tag coverage: %.1f%%", original_coverage * 100)
    LOGGER.info("Enhanced tag coverage: %.1f%%", enhanced_coverage * 100)
    LOGGER.info("Total time: %.1f seconds (%.1f minutes)", time.time() - start_time, (time.time() - start_time) / 60)

    # Count items with new categories added
    new_cats_list = df["new_categories"].tolist()
    items_with_new_cats = sum(1 for cats in new_cats_list if cats)
    LOGGER.info(
        "Items with new categories: %d (%.1f%%)",
        items_with_new_cats,
        items_with_new_cats / len(df) * 100,
    )

    # Category distribution
    all_new_cats = [cat for cats in new_cats_list for cat in cats]
    if all_new_cats:
        from collections import Counter

        cat_counts = Counter(all_new_cats)
        LOGGER.info("Category distribution:")
        for cat, count in cat_counts.most_common():
            LOGGER.info("  %s: %d", cat, count)

    return df


def save_outputs(df: pd.DataFrame, dry_run: bool = False, merge_mode: bool = False):
    """Save enhanced data and tag indices.

    Args:
        df: DataFrame with enhanced tags
        dry_run: If True, use _test suffix for output files
        merge_mode: If True, merge with existing indices instead of overwriting
    """
    suffix = "_test" if dry_run else ""

    # Save enhanced parquet
    output_path = DATA_DIR / "processed" / f"dataset_features_enhanced{suffix}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if merge_mode and output_path.exists():
        # Load existing parquet and merge
        existing_df = pd.read_parquet(output_path)
        processed_ids = set(df["dataset_id"].tolist())
        # Keep existing items that weren't reprocessed
        existing_df = existing_df[~existing_df["dataset_id"].isin(processed_ids)]
        df = pd.concat([existing_df, df], ignore_index=True)
        LOGGER.info("Merge mode: combined with existing data, total %d items", len(df))

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

    # Build and save category indices (clean categories only, no user tags)
    category_to_items, item_to_categories = build_category_indices(df)

    category_to_items_path = MODELS_DIR / f"category_to_items{suffix}.json"
    item_to_categories_path = MODELS_DIR / f"item_to_categories{suffix}.json"

    with open(category_to_items_path, "w", encoding="utf-8") as f:
        json.dump(category_to_items, f, ensure_ascii=False, indent=2)
    LOGGER.info(
        "Saved category_to_items index to %s (%d categories)",
        category_to_items_path,
        len(category_to_items),
    )

    with open(item_to_categories_path, "w", encoding="utf-8") as f:
        json.dump(item_to_categories, f, ensure_ascii=False, indent=2)
    LOGGER.info(
        "Saved item_to_categories index to %s (%d items)",
        item_to_categories_path,
        len(item_to_categories),
    )

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
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only process items without existing tags (faster)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only process items created after this date (YYYY-MM-DD format)",
    )
    args = parser.parse_args()

    LOGGER.info("=" * 60)
    LOGGER.info("Tag Enhancement Script (Optimized)")
    LOGGER.info("=" * 60)
    LOGGER.info("Model: %s", MODEL_NAME)
    LOGGER.info("Categories: %s", TOP_CATEGORIES)
    LOGGER.info("Batch size: %d", args.batch_size)
    LOGGER.info("Threshold: %.2f", args.threshold)
    LOGGER.info("Dry run: %s", args.dry_run)
    LOGGER.info("Missing only: %s", args.missing_only)
    LOGGER.info("Since: %s", args.since or "All dates")
    LOGGER.info("=" * 60)

    # Run enhancement
    df = enhance_tags(
        batch_size=args.batch_size,
        threshold=args.threshold,
        dry_run=args.dry_run,
        missing_only=args.missing_only,
        since=args.since,
    )

    # Save outputs (merge mode when using --since to preserve existing data)
    save_outputs(df, dry_run=args.dry_run, merge_mode=args.since is not None)

    LOGGER.info("=" * 60)
    LOGGER.info("Done!")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
