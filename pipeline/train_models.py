"""Train lightweight recommendation models for dataset detail page suggestions."""
from __future__ import annotations

import json
import logging
import pickle
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# Load .env file if running outside Docker (for development/testing)
try:
    from dotenv import load_dotenv
    # Load .env from project root
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)  # Don't override existing env vars
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# IMPORTANT: Set HF_ENDPOINT before importing any HuggingFace libraries
# This ensures the mirror is used for all model downloads
if os.getenv("HF_ENDPOINT"):
    hf_endpoint = os.getenv("HF_ENDPOINT")
    # Set multiple env vars for compatibility with different HF versions
    os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = hf_endpoint  # Legacy support
    print(f"[INFO] HuggingFace endpoint set to: {hf_endpoint}")  # Use print since logging not configured yet

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from lightgbm import early_stopping

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when lightgbm missing
    LIGHTGBM_AVAILABLE = False

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover
    ONNX_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    SBERT_AVAILABLE = True
except ImportError:  # pragma: no cover
    SBERT_AVAILABLE = False
    LOGGER.warning("sentence-transformers not available, text embeddings will be skipped")

from config.settings import (
    DATA_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_PATH,
    MODELS_DIR,
)
from pipeline.sentry_utils import init_pipeline_sentry, monitor_pipeline_step
from pipeline.memory_optimizer import (
    optimize_dataframe_memory,
    build_sparse_similarity_matrix,
    reduce_memory_usage,
)

RANKING_CVR_WEIGHT = float(os.getenv("RANKING_CVR_WEIGHT", "0.5"))

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


def _load_ranking_labels() -> pd.DataFrame:
    path = DATA_DIR / "processed" / "ranking_labels_by_dataset.parquet"
    if not path.exists():
        LOGGER.info("Ranking labels file not found (%s); falling back to interaction-based labels", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        LOGGER.info("Ranking labels file %s is empty; falling back to interaction-based labels", path)
    return df


def _ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _build_user_profiles(interactions: pd.DataFrame, dataset_features: pd.DataFrame) -> pd.DataFrame:
    """
    Build comprehensive user profiles from interaction history.

    Args:
        interactions: User-item interactions
        dataset_features: Dataset metadata

    Returns:
        DataFrame with user_id and profile features
    """
    if interactions.empty:
        return pd.DataFrame(columns=["user_id"])

    # Merge with dataset features to get item attributes
    enriched = interactions.merge(
        dataset_features[["dataset_id", "price", "tag"]].copy() if not dataset_features.empty else pd.DataFrame(),
        on="dataset_id",
        how="left"
    )

    # Compute user statistics
    user_stats = enriched.groupby("user_id").agg(
        user_interaction_count=("dataset_id", "count"),
        user_unique_items=("dataset_id", "nunique"),
        user_avg_weight=("weight", "mean"),
        user_total_weight=("weight", "sum"),
        user_avg_price=("price", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        user_min_price=("price", lambda x: pd.to_numeric(x, errors="coerce").min()),
        user_max_price=("price", lambda x: pd.to_numeric(x, errors="coerce").max()),
    ).reset_index()

    # Add user activity recency
    if "last_event_time" in enriched.columns:
        user_stats["user_last_active"] = enriched.groupby("user_id")["last_event_time"].max().values
        user_stats["user_first_active"] = enriched.groupby("user_id")["last_event_time"].min().values
        user_stats["user_tenure_days"] = (
            pd.to_datetime(user_stats["user_last_active"], errors="coerce") -
            pd.to_datetime(user_stats["user_first_active"], errors="coerce")
        ).dt.days.fillna(0)
    else:
        user_stats["user_tenure_days"] = 0.0

    # Add user diversity (how many different items they interact with)
    user_stats["user_diversity"] = user_stats["user_unique_items"] / user_stats["user_interaction_count"]

    # Add price preference
    user_stats["user_price_range"] = user_stats["user_max_price"] - user_stats["user_min_price"]

    # Compute user category preferences (top category)
    if "tag" in enriched.columns:
        def get_top_tag(tags_series):
            all_tags = []
            for tags in tags_series.dropna():
                if isinstance(tags, str):
                    all_tags.extend([t.strip().lower() for t in tags.split(";") if t.strip()])
            if all_tags:
                from collections import Counter
                return Counter(all_tags).most_common(1)[0][0]
            return ""

        user_top_tags = enriched.groupby("user_id")["tag"].apply(get_top_tag).reset_index()
        user_top_tags.columns = ["user_id", "user_top_tag"]
        user_stats = user_stats.merge(user_top_tags, on="user_id", how="left")
    else:
        user_stats["user_top_tag"] = ""

    # Fill NaN values
    numeric_cols = user_stats.select_dtypes(include=[np.number]).columns
    user_stats[numeric_cols] = user_stats[numeric_cols].fillna(0)

    return user_stats


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
    """Train content-based similarity with memory optimization.

    OPTIMIZED: Uses batch processing to avoid creating full N x N similarity matrix in memory.
    """
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

    # Extract item IDs before processing
    item_ids = features["dataset_id"].tolist()

    # Build TF-IDF matrix (sparse, memory efficient)
    vectorizer = TfidfVectorizer(max_features=5000)
    text_matrix = vectorizer.fit_transform(features["text"])

    LOGGER.info("TF-IDF matrix shape: %s (sparse)", text_matrix.shape)

    # Check for image embeddings
    embedding_cols = [col for col in features.columns if col.startswith("image_embed_mean_")]
    image_vectors = None

    if embedding_cols:
        image_vectors = features[embedding_cols].fillna(0.0).to_numpy(dtype=np.float32)
        if image_vectors.size and image_vectors.shape[0] == text_matrix.shape[0]:
            LOGGER.info("Using multimodal similarity (text + image, dim=%d)", image_vectors.shape[1])
        else:
            LOGGER.warning(
                "Image embeddings shape mismatch (vectors=%s, expected=%d); using text-only",
                image_vectors.shape if image_vectors.size else 0,
                text_matrix.shape[0],
            )
            image_vectors = None

    # Use memory-efficient batch processing
    batch_size = int(os.getenv("SIMILARITY_BATCH_SIZE", "1000"))
    top_k = int(os.getenv("SIMILARITY_TOP_K", "200"))

    similarity_by_idx, meta = build_sparse_similarity_matrix(
        text_matrix,
        image_vectors=image_vectors,
        image_weight=IMAGE_SIMILARITY_WEIGHT,
        batch_size=batch_size,
        top_k=top_k,
    )

    # Convert index-based dict to item_id-based dict
    similarity: Dict[int, Dict[int, float]] = {}
    for idx, neighbors in similarity_by_idx.items():
        item_id = item_ids[idx]
        similarity[item_id] = {
            item_ids[neighbor_idx]: score
            for neighbor_idx, score in neighbors.items()
        }

    # Build a lightweight similarity matrix for vector recall (top K only)
    # Instead of full N x N, we only store top K neighbors
    n_items = len(item_ids)
    similarity_matrix = np.zeros((n_items, top_k), dtype=np.float32)

    for idx in range(n_items):
        if idx in similarity_by_idx:
            neighbors = similarity_by_idx[idx]
            # Sort by score and take top K
            sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for rank, (_, score) in enumerate(sorted_neighbors):
                if rank < top_k:
                    similarity_matrix[idx, rank] = score

    LOGGER.info(
        "Content similarity computed: %d items, avg %.1f neighbors per item",
        len(similarity),
        np.mean([len(v) for v in similarity.values()]) if similarity else 0,
    )

    # Free memory
    del text_matrix, features, similarity_by_idx
    if image_vectors is not None:
        del image_vectors
    reduce_memory_usage()

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


def generate_text_embeddings(
    dataset_features: pd.DataFrame,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    n_pca_components: int = 10,
) -> Tuple[pd.DataFrame, Optional[object]]:
    """Generate Sentence-BERT text embeddings and add them to dataset features.

    Args:
        dataset_features: DataFrame with dataset_id, description, tag columns
        model_name: Sentence-BERT model name
        n_pca_components: Number of PCA components for dimensionality reduction

    Returns:
        (updated_features, pca_model) - Features with embeddings, PCA model for inference
    """
    if not SBERT_AVAILABLE:
        LOGGER.warning("Sentence-transformers not available, skipping text embeddings")
        return dataset_features, None

    if dataset_features.empty:
        return dataset_features, None

    try:
        # HF_ENDPOINT is already set at module level, just log it
        hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
        LOGGER.info("Using HuggingFace endpoint: %s", hf_endpoint)

        # Check if model_name is a local path or model name
        from pathlib import Path
        model_path = Path(model_name)

        if model_path.exists() and model_path.is_dir():
            LOGGER.info("Loading Sentence-BERT model from local path: %s", model_name)
            model = SentenceTransformer(model_name, device='cpu')
        else:
            # Try to load model, handling both with and without prefix
            LOGGER.info("Loading Sentence-BERT model: %s", model_name)

            try:
                # First try with sentence-transformers/ prefix
                if not model_name.startswith("sentence-transformers/"):
                    full_model_name = f"sentence-transformers/{model_name}"
                else:
                    full_model_name = model_name

                LOGGER.info("Attempting to download from: %s", full_model_name)
                model = SentenceTransformer(full_model_name, device='cpu')
            except Exception as e:
                LOGGER.warning("Failed to download with prefix, trying without: %s", e)
                # Fallback: try without prefix
                model = SentenceTransformer(model_name, device='cpu')

        # Combine description and tags
        texts = (
            dataset_features.get("description", "").fillna("").astype(str)
            + " "
            + dataset_features.get("tag", "").fillna("").astype(str)
        ).tolist()

        LOGGER.info("Generating text embeddings for %d items...", len(texts))
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )

        LOGGER.info("Text embeddings generated: shape=%s", embeddings.shape)

        # Add statistical features
        dataset_features["text_embed_norm"] = np.linalg.norm(embeddings, axis=1)
        dataset_features["text_embed_mean"] = embeddings.mean(axis=1)
        dataset_features["text_embed_std"] = embeddings.std(axis=1)

        # Apply PCA for dimensionality reduction
        LOGGER.info("Applying PCA to reduce embeddings to %d dimensions", n_pca_components)
        pca = PCA(n_components=n_pca_components, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)

        explained_variance = pca.explained_variance_ratio_.sum()
        LOGGER.info(
            "PCA completed: %d components explain %.1f%% variance",
            n_pca_components,
            explained_variance * 100,
        )

        # Add PCA components as features
        for i in range(n_pca_components):
            dataset_features[f"text_pca_{i}"] = embeddings_pca[:, i]

        # Save full embeddings for user-item similarity computation
        # Store as columns for later use
        for i in range(embeddings.shape[1]):
            dataset_features[f"text_embed_{i}"] = embeddings[:, i]

        LOGGER.info(
            "Text embedding features added: 3 stats + %d PCA + %d full embeddings",
            n_pca_components,
            embeddings.shape[1],
        )

        return dataset_features, pca

    except Exception as e:  # noqa: BLE001
        LOGGER.error("Failed to generate text embeddings: %s", e)
        LOGGER.warning(
            "Text embeddings will be skipped. To fix this:\n"
            "  1. Check network access to HuggingFace mirror: %s\n"
            "  2. Or download model manually and set SBERT_MODEL to local path\n"
            "  3. Or disable embeddings by commenting out generate_text_embeddings() call",
            os.getenv("HF_ENDPOINT", "https://huggingface.co")
        )
        return dataset_features, None


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


def _aggregate_slot_metrics(slot_metrics: pd.DataFrame) -> pd.DataFrame:
    if slot_metrics.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "slot_total_exposures",
                "slot_total_clicks",
                "slot_total_conversions",
                "slot_total_revenue",
                "slot_mean_ctr",
                "slot_max_ctr",
                "slot_mean_cvr",
                "slot_position_coverage",
                "slot_ctr_top1",
                "slot_ctr_top3",
                "slot_cvr_top1",
                "slot_cvr_top3",
            ]
        )

    metrics = slot_metrics.copy()
    metrics["dataset_id"] = metrics["dataset_id"].astype(int)
    metrics["position"] = metrics["position"].astype(int)

    grouped = (
        metrics.groupby("dataset_id")
        .agg(
            slot_total_exposures=("exposure_count", "sum"),
            slot_total_clicks=("click_count", "sum"),
            slot_total_conversions=("conversion_count", "sum"),
            slot_total_revenue=("conversion_revenue", "sum"),
            slot_mean_ctr=("ctr", "mean"),
            slot_max_ctr=("ctr", "max"),
            slot_mean_cvr=("cvr", "mean"),
            slot_position_coverage=("position", "nunique"),
        )
        .reset_index()
    )

    top1 = metrics[metrics["position"] == 1].groupby("dataset_id").agg(
        slot_ctr_top1=("ctr", "mean"),
        slot_cvr_top1=("cvr", "mean"),
    )
    top3 = metrics[metrics["position"] <= 3].groupby("dataset_id").agg(
        slot_ctr_top3=("ctr", "mean"),
        slot_cvr_top3=("cvr", "mean"),
    )

    merged = grouped.merge(top1, on="dataset_id", how="left").merge(top3, on="dataset_id", how="left")
    for col in [
        "slot_ctr_top1",
        "slot_cvr_top1",
        "slot_ctr_top3",
        "slot_cvr_top3",
    ]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)
    merged["slot_total_exposures"] = merged["slot_total_exposures"].fillna(0.0)
    merged["slot_total_clicks"] = merged["slot_total_clicks"].fillna(0.0)
    merged["slot_total_conversions"] = merged["slot_total_conversions"].fillna(0.0)
    merged["slot_total_revenue"] = merged["slot_total_revenue"].fillna(0.0)
    merged["slot_mean_ctr"] = merged["slot_mean_ctr"].fillna(0.0)
    merged["slot_max_ctr"] = merged["slot_max_ctr"].fillna(0.0)
    merged["slot_mean_cvr"] = merged["slot_mean_cvr"].fillna(0.0)
    merged["slot_position_coverage"] = merged["slot_position_coverage"].fillna(0.0)
    return merged


def _prepare_ranking_dataset(
    dataset_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    slot_metrics: Optional[pd.DataFrame],
    interactions: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Optional[pd.Series], str]:
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

    # Add global popularity rank feature
    if "interaction_count" in merged.columns:
        merged["popularity_rank"] = merged["interaction_count"].rank(ascending=False, method="dense")
        merged["popularity_percentile"] = merged["interaction_count"].rank(pct=True)
    else:
        merged["popularity_rank"] = 0.0
        merged["popularity_percentile"] = 0.5

    # Add price bucket feature (categorical encoded as numeric)
    merged["price_bucket"] = pd.cut(
        merged["price"],
        bins=[-np.inf, 0.5, 1.0, 2.0, 5.0, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(0.0)

    # Add interaction density (interactions per day if timestamp available)
    if "last_event_time" in dataset_stats.columns and not dataset_stats.empty:
        try:
            merged_stats = merged.merge(
                dataset_stats[["dataset_id", "last_event_time"]],
                on="dataset_id",
                how="left"
            )
            now = pd.Timestamp.now()
            merged["days_since_last_interaction"] = (
                (now - pd.to_datetime(merged_stats["last_event_time"], errors="coerce")).dt.days
            ).fillna(365.0).clip(upper=365.0)
            merged["interaction_density"] = merged["interaction_count"] / (merged["days_since_last_interaction"] + 1)
        except Exception:  # noqa: BLE001
            merged["days_since_last_interaction"] = 30.0
            merged["interaction_density"] = merged["interaction_count"] / 30.0
    else:
        merged["days_since_last_interaction"] = 30.0
        merged["interaction_density"] = merged["interaction_count"] / 30.0

    # Add text richness features
    merged["has_description"] = (merged["description_length"] > 0).astype(float)
    merged["has_tags"] = (merged["tag_count"] > 0).astype(float)
    merged["content_richness"] = merged["description_length"] * merged["tag_count"]

    base_columns = [
        "price_log", "description_length", "tag_count", "weight_log", "interaction_count",
        "popularity_rank", "popularity_percentile", "price_bucket",
        "days_since_last_interaction", "interaction_density",
        "has_description", "has_tags", "content_richness"
    ]
    features = merged[base_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    optional_columns = ["image_richness_score", "image_embed_norm", "has_images", "has_cover"]
    for col in optional_columns:
        if col in merged.columns:
            features[col] = (
                pd.to_numeric(merged[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
        else:
            features[col] = 0.0

    slot_features = _aggregate_slot_metrics(slot_metrics if slot_metrics is not None else pd.DataFrame())
    if not slot_features.empty:
        slot_features = slot_features.rename(
            columns={
                "exposure_count": "slot_total_exposures",
                "click_count": "slot_total_clicks",
            }
        )
        merged = merged.merge(slot_features, on="dataset_id", how="left")
    slot_columns = [
        "slot_total_exposures",
        "slot_total_clicks",
        "slot_total_conversions",
        "slot_total_revenue",
        "slot_mean_ctr",
        "slot_max_ctr",
        "slot_mean_cvr",
        "slot_position_coverage",
        "slot_ctr_top1",
        "slot_ctr_top3",
        "slot_cvr_top1",
        "slot_cvr_top3",
    ]
    for col in slot_columns:
        if col not in merged.columns:
            merged[col] = 0.0
        features[col] = merged[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Add text embedding features
    text_embedding_columns = ["text_embed_norm", "text_embed_mean", "text_embed_std"]
    for col in text_embedding_columns:
        if col in merged.columns:
            features[col] = (
                pd.to_numeric(merged[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
        else:
            features[col] = 0.0

    # Add PCA text embedding features
    pca_columns = [col for col in merged.columns if col.startswith("text_pca_")]
    for col in pca_columns:
        if col in merged.columns:
            features[col] = (
                pd.to_numeric(merged[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    sample_weight: Optional[pd.Series]
    task_type = "classification"
    if labels is not None and not labels.empty and "label" in labels.columns:
        label_frame = labels.set_index("dataset_id")
        rename_mapping = {}
        if "click_count" in label_frame.columns:
            rename_mapping["click_count"] = "label_click_count"
        if "exposure_count" in label_frame.columns:
            rename_mapping["exposure_count"] = "label_exposure_count"
        if "conversion_count" in label_frame.columns:
            rename_mapping["conversion_count"] = "label_conversion_count"
        if "conversion_revenue" in label_frame.columns:
            rename_mapping["conversion_revenue"] = "label_conversion_revenue"
        if "conversion_rate" in label_frame.columns:
            rename_mapping["conversion_rate"] = "label_conversion_rate"
        if rename_mapping:
            label_frame = label_frame.rename(columns=rename_mapping)
        merged = merged.join(label_frame, on="dataset_id")
        merged["label"] = merged["label"].fillna(0).astype(int)
        for column in ["label_click_count", "label_exposure_count", "label_conversion_count", "label_conversion_rate"]:
            merged[column] = merged.get(column, 0.0)
            if hasattr(merged[column], "fillna"):
                merged[column] = merged[column].fillna(0.0)
        exposures_safe = merged["label_exposure_count"].replace({0: np.nan})
        merged["ctr_label"] = np.where(
            merged["label_exposure_count"] > 0,
            merged["label_click_count"] / merged["label_exposure_count"],
            0.0,
        )
        merged["cvr_label"] = np.where(
            merged["label_exposure_count"] > 0,
            merged["label_conversion_count"] / merged["label_exposure_count"],
            0.0,
        )
        target = (merged["ctr_label"] + RANKING_CVR_WEIGHT * merged["cvr_label"]).astype(float)
        # Use log-scaled weights to prevent very high exposure items from dominating
        sample_weight = np.log1p(merged["label_exposure_count"]).clip(lower=1.0).astype(float)
        task_type = "regression"
    else:
        target = (merged["interaction_count"] > 0).astype(int)
        # Add sample weights for classification based on interaction frequency
        # Give more weight to items with higher interaction counts
        sample_weight = np.log1p(merged["interaction_count"]).clip(lower=1.0).astype(float)
    meta = merged[["dataset_id"]].reset_index(drop=True)

    return meta, features, target, sample_weight, task_type


def _train_ranking_model(
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: Optional[pd.Series],
    task_type: str,
) -> Tuple[Pipeline | None, Dict[str, float]]:
    if features.empty or target.nunique() <= 1:
        return None, {"name": "auc" if task_type == "classification" else "r2", "value": 0.0}

    stratify = target if task_type == "classification" else None
    arrays = [features, target]
    if sample_weight is not None:
        arrays.append(sample_weight)

    # First split: train+val vs test (75% / 25%)
    split_result = train_test_split(
        *arrays,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    if sample_weight is not None:
        X_trainval, X_test, y_trainval, y_test, w_trainval, w_test = split_result
    else:
        X_trainval, X_test, y_trainval, y_test = split_result
        w_trainval = w_test = None

    # Second split: train vs val (80% / 20% of trainval, i.e., 60% / 15% of total)
    stratify_val = y_trainval if task_type == "classification" else None
    arrays_val = [X_trainval, y_trainval]
    if w_trainval is not None:
        arrays_val.append(w_trainval)

    split_result_val = train_test_split(
        *arrays_val,
        test_size=0.2,
        random_state=42,
        stratify=stratify_val,
    )

    if w_trainval is not None:
        X_train, X_val, y_train, y_val, w_train, w_val = split_result_val
    else:
        X_train, X_val, y_train, y_val = split_result_val
        w_train = w_val = None

    if task_type == "classification":
        class_counts = target.value_counts()
        if (class_counts < 2).any():
            LOGGER.warning(
                "Skipping ranking classification training; insufficient samples (counts=%s)",
                class_counts.to_dict(),
            )
            return None, {"name": "auc", "value": 0.0}
        final_step_name = "clf"
        if LIGHTGBM_AVAILABLE:
            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LGBMClassifier(
                    objective="binary",
                    n_estimators=500,  # Increased from 300
                    learning_rate=0.03,  # Reduced for better convergence
                    max_depth=8,  # Limited depth to prevent overfitting
                    num_leaves=64,  # More leaves for better expressiveness
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=0.1,  # L2 regularization
                    min_child_samples=20,  # Prevent overfitting on small nodes
                    min_split_gain=0.001,  # Minimum gain to make split
                    random_state=42,
                    verbose=-1,
                )),
            ])
        else:
            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
            ])
    else:
        final_step_name = "reg"
        if LIGHTGBM_AVAILABLE:
            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LGBMRegressor(
                    objective="regression",
                    n_estimators=600,  # Increased from 400
                    learning_rate=0.02,  # Reduced for better convergence
                    max_depth=10,  # Slightly deeper for regression
                    num_leaves=96,  # More leaves for complex patterns
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.05,  # L1 regularization
                    reg_lambda=0.05,  # L2 regularization
                    min_child_samples=20,
                    min_split_gain=0.001,
                    random_state=42,
                    verbose=-1,
                )),
            ])
        else:
            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ])

    # Prepare fit parameters with early stopping for LightGBM
    fit_params = {}
    if w_train is not None:
        fit_params[f"{final_step_name}__sample_weight"] = w_train

    # Add early stopping for LightGBM models
    if LIGHTGBM_AVAILABLE:
        # Scale validation data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Set up eval_set for early stopping
        if w_val is not None:
            fit_params[f"{final_step_name}__eval_set"] = [(X_val_scaled, y_val)]
            fit_params[f"{final_step_name}__eval_sample_weight"] = [w_val]
        else:
            fit_params[f"{final_step_name}__eval_set"] = [(X_val_scaled, y_val)]

        fit_params[f"{final_step_name}__callbacks"] = [
            early_stopping(stopping_rounds=50, verbose=False)
        ]

    estimator.fit(X_train, y_train, **fit_params)

    if task_type == "classification":
        y_score = estimator.predict_proba(X_test)[:, 1]
        metric_value = roc_auc_score(y_test, y_score)
        metric_name = "auc"
    else:
        y_pred = estimator.predict(X_test)
        metric_value = r2_score(y_test, y_pred)
        metric_name = "r2"

    # Refit on full dataset for production use
    if sample_weight is not None:
        fit_params_full = {f"{final_step_name}__sample_weight": sample_weight}
    else:
        fit_params_full = {}
    estimator.fit(features, target, **fit_params_full)
    return estimator, {"name": metric_name, "value": float(metric_value)}


def _score_ranking_model(pipeline: Pipeline | None, features: pd.DataFrame) -> pd.Series:
    if pipeline is None or features.empty:
        return pd.Series(dtype=float)
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(features)[:, 1]
    else:
        scores = pipeline.predict(features)
    return pd.Series(scores, index=features.index, dtype=float)


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


@monitor_pipeline_step("train_models", critical=True)
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_pipeline_sentry("train_models")
    _ensure_models_dir()

    LOGGER.info("=" * 80)
    LOGGER.info("MEMORY-OPTIMIZED MODEL TRAINING")
    LOGGER.info("=" * 80)

    # Load data with memory optimization
    interactions = _load_frame(PROCESSED_DIR / "interactions.parquet")
    dataset_features = _load_frame(PROCESSED_DIR / "dataset_features.parquet")
    dataset_stats = _load_frame(PROCESSED_DIR / "dataset_stats.parquet")
    ranking_labels = _load_ranking_labels()
    slot_metrics = _load_frame(PROCESSED_DIR / "ranking_slot_metrics.parquet")

    # Optimize memory usage for loaded DataFrames
    LOGGER.info("Optimizing DataFrame memory usage...")
    if not interactions.empty:
        interactions = optimize_dataframe_memory(interactions)
    if not dataset_features.empty:
        dataset_features = optimize_dataframe_memory(dataset_features)
    if not dataset_stats.empty:
        dataset_stats = optimize_dataframe_memory(dataset_stats)

    # Prefer enhanced feature views when available
    dataset_features_v3 = _load_frame(PROCESSED_DIR / "dataset_features_v3.parquet")
    interactions_v2 = _load_frame(PROCESSED_DIR / "interactions_v2.parquet")
    if not interactions_v2.empty:
        interactions = optimize_dataframe_memory(interactions_v2)
        del interactions_v2

    dataset_features_v2 = _load_frame(PROCESSED_DIR / "dataset_features_v2.parquet")
    if not dataset_features_v2.empty:
        dataset_features = optimize_dataframe_memory(dataset_features_v2)
        del dataset_features_v2
    if not dataset_features_v3.empty:
        dataset_features = optimize_dataframe_memory(dataset_features_v3)
        del dataset_features_v3

    dataset_stats_v2 = _load_frame(PROCESSED_DIR / "dataset_stats_v2.parquet")
    if not dataset_stats_v2.empty:
        dataset_stats = optimize_dataframe_memory(dataset_stats_v2)
        del dataset_stats_v2

    reduce_memory_usage()

    # Generate text embeddings for ranking features
    LOGGER.info("Generating text embeddings for ranking features...")
    dataset_features, text_pca_model = generate_text_embeddings(
        dataset_features,
        model_name=os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
        n_pca_components=int(os.getenv("TEXT_PCA_COMPONENTS", "10")),
    )
    reduce_memory_usage()

    # Train models with memory monitoring
    LOGGER.info("Training behavior similarity model...")
    behavior_model = train_behavior_similarity(interactions)
    reduce_memory_usage()

    LOGGER.info("Training content similarity model (memory-optimized)...")
    content_model, content_matrix, content_ids, content_meta = train_content_similarity(dataset_features)
    reduce_memory_usage()

    LOGGER.info("Building vector recall index...")
    vector_recall = build_vector_recall(content_matrix, content_ids, VECTOR_RECALL_K)

    # Free large matrices that are no longer needed
    del content_matrix
    reduce_memory_usage()

    LOGGER.info("Building popular items list...")
    popular_items = build_popular_items(interactions)

    ranking_meta, ranking_features, ranking_target, ranking_weights, ranking_task = _prepare_ranking_dataset(
        dataset_features,
        dataset_stats,
        ranking_labels,
        slot_metrics,
    )
    ranking_model, ranking_metric = _train_ranking_model(
        ranking_features,
        ranking_target,
        ranking_weights,
        ranking_task,
    )
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

    # Save text PCA model and enriched dataset features
    text_pca_path = MODELS_DIR / "text_pca_model.pkl"
    dataset_features_enriched_path = PROCESSED_DIR / "dataset_features_with_embeddings.parquet"

    artifacts: List[Path] = [behavior_path, content_path, popular_path, vector_path]

    if text_pca_model is not None:
        save_pickle(text_pca_model, text_pca_path)
        artifacts.append(text_pca_path)
        LOGGER.info("Saved text PCA model to %s", text_pca_path)

    # Save enriched dataset features with text embeddings
    if not dataset_features.empty:
        # Save only necessary columns (not full embeddings, they're too large)
        embedding_cols = [col for col in dataset_features.columns if col.startswith("text_embed_")]
        pca_cols = [col for col in dataset_features.columns if col.startswith("text_pca_")]
        stat_cols = ["text_embed_norm", "text_embed_mean", "text_embed_std"]

        # Keep original columns + embedding features
        base_cols = ["dataset_id", "description", "tag", "price"]
        keep_cols = [c for c in base_cols if c in dataset_features.columns]
        keep_cols.extend([c for c in stat_cols if c in dataset_features.columns])
        keep_cols.extend(pca_cols)
        keep_cols.extend(embedding_cols)

        dataset_features_to_save = dataset_features[keep_cols]
        dataset_features_to_save.to_parquet(dataset_features_enriched_path, index=False)
        LOGGER.info("Saved enriched dataset features to %s", dataset_features_enriched_path)
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
        "ranking_labels_source": "matomo" if not ranking_labels.empty else "interactions",
        "ranking_samples": int(len(ranking_target)),
        "ranking_positive_samples": int((ranking_target > 0).sum()) if not ranking_target.empty else 0,
        "ranking_task_type": ranking_task,
        "ranking_metric_name": ranking_metric["name"],
    }
    metrics = {
        "behavior_item_count": float(len(behavior_model)),
        "content_item_count": float(len(content_model)),
        "popular_item_count": float(len(popular_items)),
        "vector_recall_avg_neighbors": float(np.mean([len(v) for v in vector_recall.values()]) if vector_recall else 0.0),
        "ranking_metric_value": float(ranking_metric["value"]),
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
