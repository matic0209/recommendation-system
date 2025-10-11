"""Enhanced multi-channel recall engine for recommendation system."""
from __future__ import annotations

import json
import logging
import pickle
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import DATA_DIR, MODELS_DIR
from pipeline.vector_recall_faiss import FAISS_LIBS_AVAILABLE, FaissVectorRecall

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"


class MultiChannelRecallEngine:
    """Multi-channel recall engine with 8+ recall paths."""

    def __init__(self, use_faiss: bool = True, faiss_embedding_type: str = "tfidf"):
        """Initialize recall engine.

        Args:
            use_faiss: Whether to use Faiss for vector recall
            faiss_embedding_type: "tfidf" or "sbert"
        """
        self.models = {}
        self.use_faiss = use_faiss
        self.faiss_vector_recall: Optional[FaissVectorRecall] = None

        if use_faiss and FAISS_LIBS_AVAILABLE:
            self.faiss_vector_recall = FaissVectorRecall(
                embedding_type=faiss_embedding_type,
                index_type="hnsw",  # Fast approximate search
                use_gpu=False,
            )
        elif use_faiss and not FAISS_LIBS_AVAILABLE:
            LOGGER.warning("Faiss/Sentence-Transformers not installed; disabling vector recall.")

    def train_usercf(
        self,
        interactions: pd.DataFrame,
        min_common_items: int = 2,
        top_k_users: int = 50,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Train User-based Collaborative Filtering.

        Args:
            interactions: User-item interactions
            min_common_items: Minimum common items for similarity
            top_k_users: Top K similar users to keep

        Returns:
            Dict mapping user_id -> List[(similar_user_id, similarity)]
        """
        LOGGER.info("Training UserCF...")

        if interactions.empty:
            return {}

        # Build user-item matrix
        user_items = interactions.groupby("user_id")["dataset_id"].apply(set).to_dict()

        # Compute user similarity
        user_similarity = {}

        for user_id, items in user_items.items():
            similarities = []

            for other_id, other_items in user_items.items():
                if user_id == other_id:
                    continue

                # Jaccard similarity
                common = len(items & other_items)

                if common < min_common_items:
                    continue

                union = len(items | other_items)
                similarity = common / union if union > 0 else 0

                if similarity > 0:
                    similarities.append((other_id, similarity))

            # Keep top K similar users
            similarities.sort(key=lambda x: x[1], reverse=True)
            user_similarity[user_id] = similarities[:top_k_users]

        LOGGER.info("UserCF trained: %d users", len(user_similarity))
        return user_similarity

    def train_tag_inverted_index(
        self,
        dataset_features: pd.DataFrame,
    ) -> Tuple[Dict[str, Set[int]], Dict[int, List[str]]]:
        """
        Build tag inverted index for tag-based recall.

        Args:
            dataset_features: Dataset features with tag column

        Returns:
            (tag_to_items, item_to_tags)
        """
        LOGGER.info("Building tag inverted index...")

        tag_to_items = defaultdict(set)
        item_to_tags = {}

        for row in dataset_features.to_dict(orient="records"):
            dataset_id = row.get("dataset_id")
            tags_str = row.get("tag", "")

            if not tags_str or not dataset_id:
                continue

            tags = [t.strip().lower() for t in str(tags_str).split(";") if t.strip()]
            item_to_tags[dataset_id] = tags

            for tag in tags:
                tag_to_items[tag].add(dataset_id)

        LOGGER.info(
            "Tag index built: %d unique tags, %d items",
            len(tag_to_items),
            len(item_to_tags),
        )

        return dict(tag_to_items), item_to_tags

    def train_category_index(
        self,
        dataset_features: pd.DataFrame,
    ) -> Dict[str, Set[int]]:
        """
        Build category/company index for same-type recall.

        Args:
            dataset_features: Dataset features

        Returns:
            Dict mapping category -> set of dataset_ids
        """
        LOGGER.info("Building category index...")

        category_index = defaultdict(set)

        for row in dataset_features.to_dict(orient="records"):
            dataset_id = row.get("dataset_id")
            company = row.get("create_company_name", "")

            if dataset_id and company:
                category_index[str(company).lower()].add(dataset_id)

        LOGGER.info(
            "Category index built: %d categories",
            len(category_index),
        )

        return dict(category_index)

    def train_price_bucket_index(
        self,
        dataset_features: pd.DataFrame,
    ) -> Dict[int, Set[int]]:
        """
        Build price bucket index for similar-price recall.

        Args:
            dataset_features: Dataset features with price

        Returns:
            Dict mapping price_bucket -> set of dataset_ids
        """
        LOGGER.info("Building price bucket index...")

        price_buckets = defaultdict(set)

        for row in dataset_features.to_dict(orient="records"):
            dataset_id = row.get("dataset_id")
            price = row.get("price", 0)

            if not dataset_id:
                continue

            # Price buckets: 0-100, 100-500, 500-1000, 1000-5000, 5000+
            if price < 100:
                bucket = 0
            elif price < 500:
                bucket = 1
            elif price < 1000:
                bucket = 2
            elif price < 5000:
                bucket = 3
            else:
                bucket = 4

            price_buckets[bucket].add(dataset_id)

        LOGGER.info("Price bucket index built: %d buckets", len(price_buckets))

        return dict(price_buckets)

    def usercf_recall(
        self,
        user_id: int,
        user_similarity: Dict[int, List[Tuple[int, float]]],
        user_history: Dict[int, Set[int]],
        limit: int = 50,
    ) -> List[Tuple[int, float]]:
        """
        UserCF recall: recommend items from similar users.

        Args:
            user_id: Target user
            user_similarity: User similarity dict
            user_history: User purchase history
            limit: Max items to return

        Returns:
            List[(dataset_id, score)]
        """
        if user_id not in user_similarity:
            return []

        similar_users = user_similarity.get(user_id, [])
        target_history = user_history.get(user_id, set())

        # Aggregate items from similar users
        item_scores = defaultdict(float)

        for similar_user, similarity in similar_users:
            similar_items = user_history.get(similar_user, set())

            for item in similar_items:
                if item not in target_history:
                    item_scores[item] += similarity

        # Sort by score
        results = sorted(
            item_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        return results

    def tag_recall(
        self,
        target_tags: List[str],
        tag_to_items: Dict[str, Set[int]],
        item_to_tags: Dict[int, List[str]],
        exclude_items: Set[int] = None,
        limit: int = 50,
    ) -> List[Tuple[int, float]]:
        """
        Tag-based recall.

        Args:
            target_tags: Tags to match
            tag_to_items: Tag inverted index
            item_to_tags: Item tags mapping
            exclude_items: Items to exclude
            limit: Max items

        Returns:
            List[(dataset_id, score)]
        """
        if not target_tags:
            return []

        exclude_items = exclude_items or set()
        item_scores = defaultdict(float)

        # Score items by tag overlap
        candidate_items = set()
        for tag in target_tags:
            candidate_items.update(tag_to_items.get(tag, set()))

        for item in candidate_items:
            if item in exclude_items:
                continue

            item_tags = set(item_to_tags.get(item, []))
            target_tags_set = set(tag.lower() for tag in target_tags)

            # Jaccard similarity
            intersection = len(item_tags & target_tags_set)
            union = len(item_tags | target_tags_set)

            if union > 0:
                item_scores[item] = intersection / union

        # Sort by score
        results = sorted(
            item_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        return results

    def category_recall(
        self,
        category: str,
        category_index: Dict[str, Set[int]],
        exclude_items: Set[int] = None,
        limit: int = 30,
    ) -> List[Tuple[int, float]]:
        """
        Same-category recall.

        Args:
            category: Target category (company name)
            category_index: Category index
            exclude_items: Items to exclude
            limit: Max items

        Returns:
            List[(dataset_id, score)]
        """
        exclude_items = exclude_items or set()
        category_lower = category.lower()

        items = category_index.get(category_lower, set())
        items = items - exclude_items

        # Equal score for same category
        results = [(item, 1.0) for item in list(items)[:limit]]

        return results

    def price_bucket_recall(
        self,
        price: float,
        price_bucket_index: Dict[int, Set[int]],
        exclude_items: Set[int] = None,
        limit: int = 30,
    ) -> List[Tuple[int, float]]:
        """
        Similar-price recall.

        Args:
            price: Target price
            price_bucket_index: Price bucket index
            exclude_items: Items to exclude
            limit: Max items

        Returns:
            List[(dataset_id, score)]
        """
        exclude_items = exclude_items or set()

        # Determine bucket
        if price < 100:
            bucket = 0
        elif price < 500:
            bucket = 1
        elif price < 1000:
            bucket = 2
        elif price < 5000:
            bucket = 3
        else:
            bucket = 4

        items = price_bucket_index.get(bucket, set())
        items = items - exclude_items

        results = [(item, 1.0) for item in list(items)[:limit]]

        return results

    def multi_channel_recall(
        self,
        target_dataset_id: int,
        user_id: int = None,
        limit: int = 200,
        **kwargs,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Multi-channel recall combining all strategies.

        Args:
            target_dataset_id: Context dataset
            user_id: Target user (optional)
            limit: Total limit
            **kwargs: Additional context (models, indices, etc.)

        Returns:
            Dict mapping channel -> List[(item_id, score)]
        """
        results = {}

        # Channel 1: ItemCF (behavior)
        behavior_model = kwargs.get("behavior_model", {})
        if target_dataset_id in behavior_model:
            behavior_items = behavior_model[target_dataset_id]
            results["behavior"] = [
                (item, score)
                for item, score in sorted(
                    behavior_items.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:limit // 7]
            ]

        # Channel 2: Content similarity
        content_model = kwargs.get("content_model", {})
        if target_dataset_id in content_model:
            content_items = content_model[target_dataset_id]
            results["content"] = [
                (item, score)
                for item, score in sorted(
                    content_items.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:limit // 7]
            ]

        # Channel 3: UserCF
        if user_id:
            user_similarity = kwargs.get("user_similarity", {})
            user_history = kwargs.get("user_history", {})

            if user_similarity and user_history:
                usercf_items = self.usercf_recall(
                    user_id,
                    user_similarity,
                    user_history,
                    limit=limit // 7,
                )
                if usercf_items:
                    results["usercf"] = usercf_items

        # Channel 4: Tag-based
        item_to_tags = kwargs.get("item_to_tags", {})
        tag_to_items = kwargs.get("tag_to_items", {})

        if target_dataset_id in item_to_tags:
            target_tags = item_to_tags[target_dataset_id]
            tag_items = self.tag_recall(
                target_tags,
                tag_to_items,
                item_to_tags,
                exclude_items={target_dataset_id},
                limit=limit // 7,
            )
            if tag_items:
                results["tag"] = tag_items

        # Channel 5: Same category
        dataset_features = kwargs.get("dataset_features")
        category_index = kwargs.get("category_index", {})

        if dataset_features is not None and not dataset_features.empty:
            target_row = dataset_features[
                dataset_features["dataset_id"] == target_dataset_id
            ]

            if not target_row.empty:
                category = target_row.iloc[0].get("create_company_name", "")

                if category and category_index:
                    category_items = self.category_recall(
                        category,
                        category_index,
                        exclude_items={target_dataset_id},
                        limit=limit // 7,
                    )
                    if category_items:
                        results["category"] = category_items

        # Channel 6: Price similarity
        price_bucket_index = kwargs.get("price_bucket_index", {})

        if dataset_features is not None and not dataset_features.empty:
            target_row = dataset_features[
                dataset_features["dataset_id"] == target_dataset_id
            ]

            if not target_row.empty:
                price = target_row.iloc[0].get("price", 0)

                if price_bucket_index:
                    price_items = self.price_bucket_recall(
                        price,
                        price_bucket_index,
                        exclude_items={target_dataset_id},
                        limit=limit // 7,
                    )
                    if price_items:
                        results["price"] = price_items

        # Channel 7: Faiss Vector Recall (high-performance)
        if self.faiss_vector_recall and self.faiss_vector_recall.index is not None:
            try:
                faiss_items = self.faiss_vector_recall.search(
                    target_dataset_id,
                    k=limit // 7,
                )
                if faiss_items:
                    results["vector_faiss"] = faiss_items
            except Exception as e:
                LOGGER.warning("Faiss vector recall failed: %s", e)

        # Channel 8: Popular (fallback)
        popular_items = kwargs.get("popular_items", [])
        if popular_items:
            results["popular"] = [
                (item, 0.01 - idx * 0.0001)
                for idx, item in enumerate(popular_items[:limit // 7])
                if item != target_dataset_id
            ]

        return results

    def merge_recall_results(
        self,
        recall_results: Dict[str, List[Tuple[int, float]]],
        weights: Dict[str, float] = None,
        limit: int = 200,
    ) -> List[Tuple[int, float, str]]:
        """
        Merge and deduplicate recall results from multiple channels.

        Args:
            recall_results: Dict mapping channel -> List[(item, score)]
            weights: Channel weights
            limit: Final limit

        Returns:
            List[(item_id, final_score, source_channel)]
        """
        if weights is None:
            weights = {
                "behavior": 1.0,
                "content": 0.5,
                "vector_faiss": 0.7,  # High weight for Faiss vector recall
                "usercf": 0.8,
                "tag": 0.6,
                "category": 0.4,
                "price": 0.3,
                "popular": 0.1,
            }

        # Merge scores
        item_scores = defaultdict(lambda: {"score": 0.0, "sources": []})

        for channel, items in recall_results.items():
            weight = weights.get(channel, 0.5)

            for item_id, score in items:
                item_scores[item_id]["score"] += score * weight
                item_scores[item_id]["sources"].append(channel)

        # Sort by final score
        merged = [
            (item_id, info["score"], "+".join(info["sources"]))
            for item_id, info in item_scores.items()
        ]

        merged.sort(key=lambda x: x[1], reverse=True)

        return merged[:limit]

    def save_models(self) -> None:
        """Save all recall models."""
        LOGGER.info("Saving recall models...")

        # Save UserCF
        if "user_similarity" in self.models:
            path = MODELS_DIR / "user_similarity.pkl"
            with open(path, "wb") as f:
                pickle.dump(self.models["user_similarity"], f)
            LOGGER.info("Saved user_similarity to %s", path)

        # Save tag index
        if "tag_to_items" in self.models:
            path = MODELS_DIR / "tag_to_items.json"
            # Convert sets to lists for JSON
            serializable = {
                tag: list(items)
                for tag, items in self.models["tag_to_items"].items()
            }
            with open(path, "w") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            LOGGER.info("Saved tag_to_items to %s", path)

        if "item_to_tags" in self.models:
            path = MODELS_DIR / "item_to_tags.json"
            with open(path, "w") as f:
                json.dump(self.models["item_to_tags"], f, ensure_ascii=False, indent=2)
            LOGGER.info("Saved item_to_tags to %s", path)

        # Save category index
        if "category_index" in self.models:
            path = MODELS_DIR / "category_index.json"
            serializable = {
                cat: list(items)
                for cat, items in self.models["category_index"].items()
            }
            with open(path, "w") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            LOGGER.info("Saved category_index to %s", path)

        # Save price bucket index
        if "price_bucket_index" in self.models:
            path = MODELS_DIR / "price_bucket_index.json"
            serializable = {
                str(bucket): list(items)
                for bucket, items in self.models["price_bucket_index"].items()
            }
            with open(path, "w") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            LOGGER.info("Saved price_bucket_index to %s", path)

        # Save Faiss vector recall
        if self.faiss_vector_recall and self.faiss_vector_recall.index is not None:
            saved_files = self.faiss_vector_recall.save(model_name="faiss_recall")
            LOGGER.info("Saved Faiss vector recall: %s", saved_files)


def main() -> None:
    """Train and save multi-channel recall models."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load data
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    dataset_features = pd.read_parquet(PROCESSED_DIR / "dataset_features.parquet")

    # Initialize engine
    use_faiss = os.getenv("USE_FAISS_RECALL", "1") not in {"0", "false", "False"}
    engine = MultiChannelRecallEngine(use_faiss=use_faiss)

    # Train all recall models
    LOGGER.info("Training multi-channel recall models...")

    # 1. UserCF
    user_similarity = engine.train_usercf(interactions)
    engine.models["user_similarity"] = user_similarity

    # 2. Tag index
    tag_to_items, item_to_tags = engine.train_tag_inverted_index(dataset_features)
    engine.models["tag_to_items"] = tag_to_items
    engine.models["item_to_tags"] = item_to_tags

    # 3. Category index
    category_index = engine.train_category_index(dataset_features)
    engine.models["category_index"] = category_index

    # 4. Price bucket index
    price_bucket_index = engine.train_price_bucket_index(dataset_features)
    engine.models["price_bucket_index"] = price_bucket_index

    # 5. Faiss vector recall
    if engine.faiss_vector_recall:
        LOGGER.info("Training Faiss vector recall...")
        faiss_stats = engine.faiss_vector_recall.train(dataset_features)
        LOGGER.info("Faiss stats: %s", faiss_stats)

    # Save models
    engine.save_models()

    # Print summary
    print("\n" + "=" * 80)
    print("MULTI-CHANNEL RECALL ENGINE SUMMARY")
    print("=" * 80)
    print(f"UserCF: {len(user_similarity):,} users with similarities")
    print(f"Tag Index: {len(tag_to_items):,} unique tags, {len(item_to_tags):,} items")
    print(f"Category Index: {len(category_index):,} categories")
    print(f"Price Bucket Index: {len(price_bucket_index):,} buckets")

    if engine.faiss_vector_recall and engine.faiss_vector_recall.index is not None:
        print(f"\nFaiss Vector Recall:")
        print(f"  Embedding Type: {engine.faiss_vector_recall.embedding_type}")
        print(f"  Index Type: {engine.faiss_vector_recall.index_type}")
        print(f"  Dimension: {engine.faiss_vector_recall.dimension:,}")
        print(f"  Total Vectors: {engine.faiss_vector_recall.index.ntotal:,}")

    print(f"\nModels saved to: {MODELS_DIR}")
    print("=" * 80 + "\n")

    LOGGER.info("Multi-channel recall training completed!")


if __name__ == "__main__":
    main()
