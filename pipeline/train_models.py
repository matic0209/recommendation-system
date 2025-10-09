"""Train lightweight recommendation models for dataset detail page suggestions."""
from __future__ import annotations

import json
import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import MODELS_DIR, DATA_DIR

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Missing input data: %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path)


def _ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_behavior_similarity(interactions: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    if interactions.empty:
        return {}

    by_user = interactions.groupby("user_id")["dataset_id"].apply(list)
    co_counts: Dict[int, Counter] = defaultdict(Counter)

    for datasets in by_user:
        unique_items = set(datasets)
        for item in unique_items:
            for other in unique_items:
                if item == other:
                    continue
                co_counts[item][other] += 1

    similarity = {}
    for item, counts in co_counts.items():
        total = sum(counts.values())
        if not total:
            continue
        similarity[item] = {other: count / total for other, count in counts.items()}
    return similarity



def train_content_similarity(dataset_features: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    if dataset_features.empty:
        return {}

    features = dataset_features.copy()
    features["text"] = (
        features.get("description", "")
        .fillna("")
        .astype(str)
        + " "
        + features.get("tag", "").fillna("")
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(features["text"])
    similarity_matrix = cosine_similarity(matrix)

    item_ids = features["dataset_id"].tolist()
    similarity: Dict[int, Dict[int, float]] = {}
    for idx, item_id in enumerate(item_ids):
        scores = {}
        for jdx, other_id in enumerate(item_ids):
            if item_id == other_id:
                continue
            score = float(similarity_matrix[idx, jdx])
            if score > 0:
                scores[other_id] = score
        similarity[item_id] = scores
    return similarity



def build_popular_items(interactions: pd.DataFrame, top_k: int = 50) -> List[int]:
    if interactions.empty:
        return []
    counts = interactions.groupby("dataset_id")["weight"].sum().sort_values(ascending=False)
    return counts.head(top_k).index.astype(int).tolist()



def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as stream:
        pickle.dump(obj, stream)
        LOGGER.info("Wrote %s", path)



def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote %s", path)



def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ensure_models_dir()

    interactions = _load_frame(PROCESSED_DIR / "interactions.parquet")
    dataset_features = _load_frame(PROCESSED_DIR / "dataset_features.parquet")

    behavior_model = train_behavior_similarity(interactions)
    content_model = train_content_similarity(dataset_features)
    popular_items = build_popular_items(interactions)

    save_pickle(behavior_model, MODELS_DIR / "item_sim_behavior.pkl")
    save_pickle(content_model, MODELS_DIR / "item_sim_content.pkl")
    save_json(popular_items, MODELS_DIR / "top_items.json")


if __name__ == "__main__":
    main()
