"""Memory optimization utilities for recommendation pipeline.

This module provides memory-efficient alternatives for common operations
that previously caused OOM errors.
"""
import gc
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types.

    Args:
        df: Input DataFrame

    Returns:
        Memory-optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        # Skip object and datetime types
        if col_type == object or pd.api.types.is_datetime64_any_dtype(col_type):
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    LOGGER.info(
        "Memory optimization: %.2f MB -> %.2f MB (%.1f%% reduction)",
        start_mem,
        end_mem,
        100 * (start_mem - end_mem) / start_mem
    )

    return df


def compute_similarity_in_batches(
    matrix: csr_matrix | np.ndarray,
    batch_size: int = 1000,
    top_k: int = 200,
    min_score: float = 0.01,
) -> Dict[int, Dict[int, float]]:
    """Compute cosine similarity in batches to reduce memory usage.

    Instead of computing the full N x N similarity matrix at once,
    this function processes the matrix in batches and only keeps
    top K similar items for each item.

    Args:
        matrix: TF-IDF matrix or dense feature matrix (N x D)
        batch_size: Number of rows to process at once
        top_k: Number of top similar items to keep per item
        min_score: Minimum similarity score to keep

    Returns:
        Dict mapping item_idx -> {similar_item_idx: score}
    """
    n_items = matrix.shape[0]
    similarity_dict = {}

    LOGGER.info(
        "Computing similarity in batches (total=%d, batch_size=%d, top_k=%d)",
        n_items,
        batch_size,
        top_k,
    )

    for batch_start in range(0, n_items, batch_size):
        batch_end = min(batch_start + batch_size, n_items)

        # Extract batch
        if hasattr(matrix, 'toarray'):
            batch = matrix[batch_start:batch_end]
        else:
            batch = matrix[batch_start:batch_end, :]

        # Compute similarity for this batch against all items
        # This creates a (batch_size x n_items) matrix instead of (n_items x n_items)
        batch_similarities = cosine_similarity(batch, matrix)

        # Process each row in the batch
        for local_idx in range(batch_similarities.shape[0]):
            global_idx = batch_start + local_idx

            # Get similarity scores for this item
            scores = batch_similarities[local_idx]

            # Set self-similarity to 0
            scores[global_idx] = 0

            # Get top K items
            if top_k < len(scores):
                # Use argpartition for efficiency (faster than full sort)
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            else:
                top_indices = np.argsort(scores)[::-1]

            # Filter by minimum score and store
            similar_items = {}
            for idx in top_indices:
                score = float(scores[idx])
                if score >= min_score:
                    similar_items[int(idx)] = score
                else:
                    break  # Since sorted, we can stop

            if similar_items:
                similarity_dict[global_idx] = similar_items

        # Free memory
        del batch_similarities
        if (batch_start // batch_size) % 5 == 0:
            gc.collect()

        if (batch_start // batch_size + 1) % 10 == 0:
            LOGGER.info(
                "Progress: %d/%d batches (%.1f%%)",
                batch_start // batch_size + 1,
                (n_items + batch_size - 1) // batch_size,
                100 * batch_end / n_items,
            )

    LOGGER.info("Similarity computation completed: %d items with neighbors", len(similarity_dict))
    return similarity_dict


def build_sparse_similarity_matrix(
    text_matrix: csr_matrix,
    image_vectors: np.ndarray | None = None,
    image_weight: float = 0.4,
    batch_size: int = 1000,
    top_k: int = 200,
) -> Tuple[Dict[int, Dict[int, float]], Dict[str, float]]:
    """Build similarity matrix with optional multimodal fusion.

    This is a memory-efficient alternative to the original train_content_similarity.

    Args:
        text_matrix: TF-IDF text features (sparse)
        image_vectors: Image embedding vectors (dense), optional
        image_weight: Weight for image similarity when fusing
        batch_size: Batch size for similarity computation
        top_k: Top K neighbors to keep

    Returns:
        (similarity_dict, metadata)
    """
    modalities = 1
    embedding_dim = 0

    if image_vectors is not None and len(image_vectors) > 0:
        LOGGER.info("Computing multimodal similarity (text + image)...")

        # Normalize image vectors
        from sklearn.preprocessing import normalize
        image_vectors = normalize(image_vectors, norm='l2')

        # Compute similarities separately and combine in batches
        n_items = text_matrix.shape[0]
        combined_similarity = {}

        for batch_start in range(0, n_items, batch_size):
            batch_end = min(batch_start + batch_size, n_items)

            # Text similarity for batch
            text_batch = text_matrix[batch_start:batch_end]
            text_sim = cosine_similarity(text_batch, text_matrix)

            # Image similarity for batch
            image_batch = image_vectors[batch_start:batch_end]
            image_sim = cosine_similarity(image_batch, image_vectors)

            # Combine
            batch_combined = (1 - image_weight) * text_sim + image_weight * image_sim

            # Extract top K for each item in batch
            for local_idx in range(batch_combined.shape[0]):
                global_idx = batch_start + local_idx
                scores = batch_combined[local_idx]
                scores[global_idx] = 0  # Remove self-similarity

                # Get top K
                if top_k < len(scores):
                    top_indices = np.argpartition(scores, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
                else:
                    top_indices = np.argsort(scores)[::-1]

                # Store
                similar_items = {}
                for idx in top_indices:
                    score = float(scores[idx])
                    if score > 0.01:
                        similar_items[int(idx)] = score

                if similar_items:
                    combined_similarity[global_idx] = similar_items

            # Free memory
            del text_sim, image_sim, batch_combined
            if (batch_start // batch_size) % 5 == 0:
                gc.collect()

        modalities = 2
        embedding_dim = image_vectors.shape[1]
        similarity_dict = combined_similarity

        LOGGER.info(
            "Multimodal similarity computed (text + image, weight=%.2f, dim=%d)",
            image_weight,
            embedding_dim,
        )
    else:
        # Text-only similarity
        LOGGER.info("Computing text-only similarity...")
        similarity_dict = compute_similarity_in_batches(
            text_matrix,
            batch_size=batch_size,
            top_k=top_k,
        )

    meta = {
        "modalities": float(modalities),
        "image_embedding_dim": float(embedding_dim),
        "image_similarity_weight": float(image_weight if modalities > 1 else 0.0),
    }

    return similarity_dict, meta


def reduce_memory_usage() -> None:
    """Force garbage collection and log memory usage."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    gc.collect()

    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    LOGGER.info(
        "Memory usage: %.1f MB -> %.1f MB (freed %.1f MB)",
        mem_before,
        mem_after,
        mem_before - mem_after,
    )
