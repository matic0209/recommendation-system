import numpy as np
import pandas as pd

from pipeline.train_models import (
    VECTOR_RECALL_K,
    build_vector_recall,
    train_behavior_similarity,
    train_content_similarity,
    _prepare_ranking_dataset,
    _train_ranking_model,
)


def _sample_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": 1, "dataset_id": 101, "weight": 1.0, "last_event_time": "2024-01-01"},
            {"user_id": 1, "dataset_id": 102, "weight": 0.5, "last_event_time": "2024-01-02"},
            {"user_id": 2, "dataset_id": 101, "weight": 1.5, "last_event_time": "2024-01-03"},
            {"user_id": 2, "dataset_id": 103, "weight": 2.0, "last_event_time": "2024-01-02"},
            {"user_id": 3, "dataset_id": 104, "weight": 0.8, "last_event_time": "2024-01-04"},
            {"user_id": 3, "dataset_id": 102, "weight": 1.2, "last_event_time": "2024-01-05"},
        ]
    )


def _sample_dataset_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset_id": 101, "dataset_name": "A", "description": "finance risk", "tag": "finance;risk", "price": 100},
            {"dataset_id": 102, "dataset_name": "B", "description": "finance summary", "tag": "finance", "price": 80},
            {"dataset_id": 103, "dataset_name": "C", "description": "health record", "tag": "health", "price": 50},
            {"dataset_id": 104, "dataset_name": "D", "description": "travel review", "tag": "travel;review", "price": 120},
            {"dataset_id": 105, "dataset_name": "E", "description": "sports stats", "tag": "sports", "price": 60},
        ]
    )


def _sample_dataset_stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset_id": 101, "interaction_count": 5, "total_weight": 4.0},
            {"dataset_id": 102, "interaction_count": 3, "total_weight": 2.5},
            {"dataset_id": 103, "interaction_count": 1, "total_weight": 1.5},
            {"dataset_id": 104, "interaction_count": 0, "total_weight": 0.0},
            {"dataset_id": 105, "interaction_count": 0, "total_weight": 0.0},
        ]
    )


def test_behavior_similarity_generates_neighbors():
    interactions = _sample_interactions()
    behavior = train_behavior_similarity(interactions)
    assert behavior, "Expected non-empty behavior similarity map"
    # Make sure co-occurrence includes other items
    assert 102 in behavior[101]
    assert behavior[101][102] > 0


def test_content_similarity_matrix_shape_matches_items():
    dataset_features = _sample_dataset_features()
    content_map, matrix, item_ids, meta = train_content_similarity(dataset_features)
    assert content_map
    assert matrix.shape[0] == matrix.shape[1] == len(item_ids)
    # Ensure vector similarity bonds exist for finance items
    neighbours = content_map[101]
    assert 102 in neighbours and neighbours[102] > 0
    assert isinstance(meta, dict)


def test_vector_recall_top_k_limit():
    dataset_features = _sample_dataset_features()
    _, sim_matrix, item_ids, _ = train_content_similarity(dataset_features)
    top_k = min(2, VECTOR_RECALL_K)
    neighbors = build_vector_recall(sim_matrix, item_ids, top_k)
    assert neighbors
    for source, items in neighbors.items():
        assert len(items) <= top_k
        # Vector recall should not include the item itself
        assert all(entry["dataset_id"] != source for entry in items)


def test_ranking_model_trains_and_scores():
    dataset_features = _sample_dataset_features()
    dataset_stats = _sample_dataset_stats()
    meta, features, target = _prepare_ranking_dataset(dataset_features, dataset_stats)
    model, auc = _train_ranking_model(features, target)
    # We expect to train a model with AUC between 0 and 1
    assert model is not None
    assert 0.5 <= auc <= 1.0
    # Ranking scores should be available for each dataset
    scores = model.predict_proba(features)[:, 1]
    assert scores.shape[0] == meta.shape[0]
    assert np.all((scores >= 0) & (scores <= 1))
