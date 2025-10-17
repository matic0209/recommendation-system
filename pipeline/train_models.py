"""Train lightweight recommendation models for dataset detail page suggestions."""
from __future__ import annotations

import json
import logging
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when lightgbm missing
    LIGHTGBM_AVAILABLE = False

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover
    ONNX_AVAILABLE = False

from config.settings import (
    DATA_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_PATH,
    MODELS_DIR,
)

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"
REGISTRY_HISTORY_LIMIT = 20
VECTOR_RECALL_K = 200
IMAGE_SIMILARITY_WEIGHT = 0.4
VECTOR_RECALL_PATH = MODELS_DIR / "item_recall_vector.json"
RANK_MODEL_PATH = MODELS_DIR / "rank_model.pkl"


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


def train_content_similarity(
    dataset_features: pd.DataFrame,
) -> Tuple[Dict[int, Dict[int, float]], np.ndarray, List[int], Dict[str, float]]:
    if dataset_features.empty:
        return {}, np.empty((0, 0)), [], {"modalities": 0.0, "image_embedding_dim": 0.0, "image_similarity_weight": 0.0}

    features = dataset_features.copy()
    features["text"] = (
        features.get("description", "")
        .fillna("")
        .astype(str)
        + " "
        + features.get("tag", "").fillna("")
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    text_matrix = vectorizer.fit_transform(features["text"])
    text_similarity = cosine_similarity(text_matrix)

    embedding_cols = [col for col in features.columns if col.startswith("image_embed_mean_")]
    similarity_matrix = text_similarity
    modalities = 1
    embedding_dim = 0

    if embedding_cols:
        image_vectors = features[embedding_cols].fillna(0.0).to_numpy(dtype=np.float32)
        if image_vectors.size and image_vectors.shape[0] == text_similarity.shape[0]:
            image_vectors = normalize(image_vectors)
            image_similarity = cosine_similarity(image_vectors)
            similarity_matrix = (
                (1 - IMAGE_SIMILARITY_WEIGHT) * text_similarity +
                IMAGE_SIMILARITY_WEIGHT * image_similarity
            )
            modalities = 2
            embedding_dim = image_vectors.shape[1]
            LOGGER.info(
                "Combined text and image similarities (image_weight=%s, embedding_dim=%d)",
                IMAGE_SIMILARITY_WEIGHT,
                image_vectors.shape[1],
            )
        else:
            LOGGER.warning(
                "Image embeddings present but shape mismatch (vectors=%s); using text-only similarity",
                image_vectors.shape if image_vectors.size else 0,
            )

    item_ids = features["dataset_id"].tolist()
    similarity: Dict[int, Dict[int, float]] = {}
    for idx, item_id in enumerate(item_ids):
        scores = {}
        row = similarity_matrix[idx]
        for jdx, other_id in enumerate(item_ids):
            if item_id == other_id:
                continue
            score = float(row[jdx])
            if score > 0:
                scores[other_id] = score
        similarity[item_id] = scores
    meta = {
        "modalities": float(modalities),
        "image_embedding_dim": float(embedding_dim),
        "image_similarity_weight": float(IMAGE_SIMILARITY_WEIGHT if modalities > 1 else 0.0),
    }
    return similarity, similarity_matrix, item_ids, meta


def build_popular_items(interactions: pd.DataFrame, top_k: int = 50) -> List[int]:
    if interactions.empty:
        return []
    counts = interactions.groupby("dataset_id")["weight"].sum().sort_values(ascending=False)
    return counts.head(top_k).index.astype(int).tolist()


def build_vector_recall(similarity_matrix: np.ndarray, item_ids: List[int], top_k: int) -> Dict[int, List[Dict[str, float]]]:
    if similarity_matrix.size == 0 or not item_ids:
        return {}
    neighbors: Dict[int, List[Dict[str, float]]] = {}
    for idx, item_id in enumerate(item_ids):
        row = similarity_matrix[idx].copy()
        if idx < row.shape[0]:
            row[idx] = 0.0
        top_indices = np.argsort(row)[::-1][:top_k]
        entries: List[Dict[str, float]] = []
        for jdx in top_indices:
            score = float(row[jdx])
            if score <= 0:
                continue
            neighbor_id = int(item_ids[jdx])
            if neighbor_id == item_id:
                continue
            entries.append({"dataset_id": neighbor_id, "score": score})
        neighbors[int(item_id)] = entries
    return neighbors


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as stream:
        pickle.dump(obj, stream)
        LOGGER.info("Wrote %s", path)


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote %s", path)


def save_vector_recall(neighbors: Dict[int, List[Dict[str, float]]], path: Path) -> None:
    serializable = {str(key): value for key, value in neighbors.items()}
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote %s", path)


def _load_registry() -> Dict[str, object]:
    if not MODEL_REGISTRY_PATH.exists():
        return {"current": None, "history": []}
    try:
        return json.loads(MODEL_REGISTRY_PATH.read_text())
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse model registry at %s; starting fresh", MODEL_REGISTRY_PATH)
        return {"current": None, "history": []}


def _save_registry(registry: Dict[str, object]) -> None:
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False))


def _update_registry(entry: Dict[str, object]) -> None:
    registry = _load_registry()
    history = registry.get("history", []) or []
    current = registry.get("current")
    if current:
        history.insert(0, current)
    history = history[: REGISTRY_HISTORY_LIMIT - 1]
    _save_registry({"current": entry, "history": history})
    LOGGER.info("Model registry updated (run_id=%s)", entry.get("run_id"))


def _prepare_ranking_dataset(
    dataset_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if dataset_features.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int)

    base = dataset_features.copy()
    for col in ["interaction_count", "total_weight", "last_event_time"]:
        if col in base.columns:
            base = base.drop(columns=[col])
    base["price"] = pd.to_numeric(base.get("price"), errors="coerce").fillna(0.0)
    base["description"] = base.get("description", "").fillna("").astype(str)
    base["tag"] = base.get("tag", "").fillna("").astype(str)
    base["description_length"] = base["description"].str.len().astype(float)
    base["tag_count"] = base["tag"].apply(lambda text: float(len([t for t in text.split(';') if t.strip()])))

    stats = dataset_stats.copy() if not dataset_stats.empty else pd.DataFrame(columns=["dataset_id", "interaction_count", "total_weight", "last_event_time"])
    stats = stats[[col for col in ["dataset_id", "interaction_count", "total_weight"] if col in stats.columns]]
    stats["interaction_count"] = pd.to_numeric(stats.get("interaction_count"), errors="coerce").fillna(0.0)
    stats["total_weight"] = pd.to_numeric(stats.get("total_weight"), errors="coerce").fillna(0.0)

    merged = base.merge(stats, on="dataset_id", how="left")
    merged["interaction_count"] = merged["interaction_count"].fillna(0.0)
    merged["total_weight"] = merged["total_weight"].fillna(0.0)

    merged["price_log"] = np.log1p(merged["price"].clip(lower=0.0))
    merged["weight_log"] = np.log1p(merged["total_weight"].clip(lower=0.0))

    feature_columns = ["price_log", "description_length", "tag_count", "weight_log", "interaction_count"]
    optional_columns = [
        col
        for col in ["image_richness_score", "image_embed_norm", "has_images", "has_cover"]
        if col in merged.columns
    ]
    feature_columns.extend(optional_columns)
    features = merged[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target = (merged["interaction_count"] > 0).astype(int)
    meta = merged[["dataset_id"]].reset_index(drop=True)

    return meta, features, target


def _train_ranking_model(features: pd.DataFrame, target: pd.Series) -> Tuple[Pipeline | None, float]:
    if features.empty or target.nunique() <= 1:
        return None, 0.0

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    estimator: Pipeline
    if LIGHTGBM_AVAILABLE:
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                objective="binary",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
            )),
        ])
    else:
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ])

    estimator.fit(X_train, y_train)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    # Refit on full dataset for production use
    estimator.fit(features, target)
    return estimator, float(auc)


def _score_ranking_model(pipeline: Pipeline | None, features: pd.DataFrame) -> pd.Series:
    if pipeline is None or features.empty:
        return pd.Series(dtype=float)
    proba = pipeline.predict_proba(features)[:, 1]
    return pd.Series(proba, index=features.index, dtype=float)


def _log_to_mlflow(params: Dict[str, object], metrics: Dict[str, float], artifacts: List[Path]) -> Dict[str, str]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_name = f"train_models_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        for artifact in artifacts:
            if Path(artifact).exists():
                mlflow.log_artifact(str(artifact))
        LOGGER.info("Logged training run to MLflow (run_id=%s)", run.info.run_id)
        return {
            "run_id": run.info.run_id,
            "artifact_uri": mlflow.get_artifact_uri(),
        }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ensure_models_dir()

    interactions = _load_frame(PROCESSED_DIR / "interactions.parquet")
    dataset_features = _load_frame(PROCESSED_DIR / "dataset_features.parquet")
    dataset_stats = _load_frame(PROCESSED_DIR / "dataset_stats.parquet")

    # Prefer enhanced feature views when available
    interactions_v2 = _load_frame(PROCESSED_DIR / "interactions_v2.parquet")
    if not interactions_v2.empty:
        interactions = interactions_v2

    dataset_features_v2 = _load_frame(PROCESSED_DIR / "dataset_features_v2.parquet")
    if not dataset_features_v2.empty:
        dataset_features = dataset_features_v2

    dataset_stats_v2 = _load_frame(PROCESSED_DIR / "dataset_stats_v2.parquet")
    if not dataset_stats_v2.empty:
        dataset_stats = dataset_stats_v2

    behavior_model = train_behavior_similarity(interactions)
    content_model, content_matrix, content_ids, content_meta = train_content_similarity(dataset_features)
    vector_recall = build_vector_recall(content_matrix, content_ids, VECTOR_RECALL_K)
    popular_items = build_popular_items(interactions)

    ranking_meta, ranking_features, ranking_target = _prepare_ranking_dataset(dataset_features, dataset_stats)
    ranking_model, ranking_auc = _train_ranking_model(ranking_features, ranking_target)
    ranking_scores = _score_ranking_model(ranking_model, ranking_features)

    behavior_path = MODELS_DIR / "item_sim_behavior.pkl"
    content_path = MODELS_DIR / "item_sim_content.pkl"
    popular_path = MODELS_DIR / "top_items.json"
    vector_path = VECTOR_RECALL_PATH
    rank_model_path = RANK_MODEL_PATH

    save_pickle(behavior_model, behavior_path)
    save_pickle(content_model, content_path)
    save_json(popular_items, popular_path)
    save_vector_recall(vector_recall, vector_path)

    artifacts: List[Path] = [behavior_path, content_path, popular_path, vector_path]
    if ranking_model is not None:
        save_pickle(ranking_model, rank_model_path)
        artifacts.append(rank_model_path)
        if ONNX_AVAILABLE and not ranking_features.empty:
            try:
                initial_type = [
                    ("feature_input", FloatTensorType([None, ranking_features.shape[1]]))
                ]
                onnx_model = convert_sklearn(ranking_model, initial_types=initial_type)
                onnx_path = MODELS_DIR / "rank_model.onnx"
                onnx_path.write_bytes(onnx_model.SerializeToString())
                artifacts.append(onnx_path)
                LOGGER.info("Exported LightGBM ranking model to ONNX (%s)", onnx_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to export ONNX model: %s", exc)
        elif not ONNX_AVAILABLE:
            LOGGER.info("skl2onnx not installed; skipping ONNX export")
    else:
        LOGGER.warning("Ranking model training skipped due to insufficient label diversity")

    embeddings_available = content_meta.get("modalities", 0.0) > 1.0
    params = {
        "interactions_rows": len(interactions),
        "dataset_features_rows": len(dataset_features),
        "dataset_stats_rows": len(dataset_stats),
        "vector_recall_k": VECTOR_RECALL_K,
        "image_embeddings_available": embeddings_available,
        "image_similarity_weight": content_meta.get("image_similarity_weight", 0.0),
    }
    metrics = {
        "behavior_item_count": float(len(behavior_model)),
        "content_item_count": float(len(content_model)),
        "popular_item_count": float(len(popular_items)),
        "vector_recall_avg_neighbors": float(np.mean([len(v) for v in vector_recall.values()]) if vector_recall else 0.0),
        "ranking_auc": float(ranking_auc),
        "ranking_positive_rate": float(ranking_target.mean()) if not ranking_target.empty else 0.0,
    }
    metrics.update({
        "content_modalities": content_meta.get("modalities", 1.0),
        "content_image_embedding_dim": content_meta.get("image_embedding_dim", 0.0),
        "content_image_similarity_weight": content_meta.get("image_similarity_weight", 0.0),
    })

    embeddings_path = PROCESSED_DIR / "dataset_image_embeddings.parquet"
    if embeddings_path.exists():
        artifacts.append(embeddings_path)

    if not ranking_scores.empty:
        ranking_snapshot = ranking_meta.copy()
        ranking_snapshot["score"] = ranking_scores.values
        snapshot_path = MODELS_DIR / "ranking_scores_preview.json"
        snapshot_path.write_text(
            ranking_snapshot.sort_values("score", ascending=False)
            .head(50)
            .to_json(orient="records", force_ascii=False, indent=2)
        )
        artifacts.append(snapshot_path)
        metrics["ranking_top_score"] = float(ranking_snapshot["score"].max())

    run_info = _log_to_mlflow(params, metrics, artifacts)
    entry = {
        "run_id": run_info["run_id"],
        "artifact_uri": run_info["artifact_uri"],
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [str(p) for p in artifacts],
        "params": params,
        "metrics": metrics,
    }
    _update_registry(entry)


if __name__ == "__main__":
    main()
