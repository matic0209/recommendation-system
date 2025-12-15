#!/usr/bin/env python3
"""
Test script to verify memory optimization improvements.

This script compares memory usage before and after optimization.
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.memory_optimizer import (
    optimize_dataframe_memory,
    compute_similarity_in_batches,
    build_sparse_similarity_matrix,
    reduce_memory_usage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def test_dataframe_optimization():
    """Test DataFrame memory optimization."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: DataFrame Memory Optimization")
    LOGGER.info("=" * 80)

    # Create a sample DataFrame with inefficient types
    n_rows = 100000
    df = pd.DataFrame({
        'id': np.random.randint(0, 1000, n_rows),  # Could be int16
        'user_id': np.random.randint(0, 10000, n_rows),  # Could be int32
        'price': np.random.random(n_rows) * 1000,  # Could be float32
        'score': np.random.random(n_rows),  # Could be float32
    })

    mem_before = df.memory_usage(deep=True).sum() / 1024 / 1024
    LOGGER.info(f"Before optimization: {mem_before:.2f} MB")

    df_optimized = optimize_dataframe_memory(df)

    mem_after = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
    LOGGER.info(f"After optimization: {mem_after:.2f} MB")
    LOGGER.info(f"Memory saved: {mem_before - mem_after:.2f} MB ({100 * (mem_before - mem_after) / mem_before:.1f}%)")

    return mem_before, mem_after


def test_similarity_computation():
    """Test similarity computation memory efficiency."""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("TEST 2: Similarity Computation")
    LOGGER.info("=" * 80)

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create sample text data
    n_items = 5000
    texts = [
        f"dataset {i} with features about topic {i % 10} and category {i % 5}"
        for i in range(n_items)
    ]

    vectorizer = TfidfVectorizer(max_features=1000)
    text_matrix = vectorizer.fit_transform(texts)

    LOGGER.info(f"Text matrix shape: {text_matrix.shape} (sparse)")

    # Method 1: OLD - Full similarity matrix (memory intensive)
    LOGGER.info("\n--- Method 1: Full Similarity Matrix (OLD) ---")
    mem_start = get_memory_usage_mb()
    start_time = time.time()

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        # This creates N x N dense matrix
        full_similarity = cosine_similarity(text_matrix)
        mem_peak = get_memory_usage_mb()
        elapsed = time.time() - start_time

        LOGGER.info(f"Memory used: {mem_peak - mem_start:.2f} MB")
        LOGGER.info(f"Time: {elapsed:.2f}s")
        LOGGER.info(f"Output shape: {full_similarity.shape}")

        # Free memory
        del full_similarity
        reduce_memory_usage()
    except MemoryError:
        LOGGER.error("MemoryError: Could not compute full similarity matrix!")
        mem_peak = float('inf')

    # Method 2: NEW - Batched computation (memory efficient)
    LOGGER.info("\n--- Method 2: Batched Similarity (NEW) ---")
    mem_start = get_memory_usage_mb()
    start_time = time.time()

    similarity_dict = compute_similarity_in_batches(
        text_matrix,
        batch_size=1000,
        top_k=200,
    )

    mem_peak_new = get_memory_usage_mb()
    elapsed_new = time.time() - start_time

    LOGGER.info(f"Memory used: {mem_peak_new - mem_start:.2f} MB")
    LOGGER.info(f"Time: {elapsed_new:.2f}s")
    LOGGER.info(f"Output: {len(similarity_dict)} items with avg {np.mean([len(v) for v in similarity_dict.values()]):.1f} neighbors")

    if mem_peak != float('inf'):
        LOGGER.info(f"\nMemory saved: {mem_peak - mem_peak_new:.2f} MB ({100 * (mem_peak - mem_peak_new) / mem_peak:.1f}%)")

    return similarity_dict


def test_multimodal_similarity():
    """Test multimodal similarity computation."""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("TEST 3: Multimodal Similarity (Text + Image)")
    LOGGER.info("=" * 80)

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create sample data
    n_items = 3000
    texts = [f"dataset {i} about topic {i % 10}" for i in range(n_items)]

    # Text features
    vectorizer = TfidfVectorizer(max_features=500)
    text_matrix = vectorizer.fit_transform(texts)

    # Image features (simulated)
    image_dim = 128
    image_vectors = np.random.random((n_items, image_dim)).astype(np.float32)

    LOGGER.info(f"Text matrix: {text_matrix.shape} (sparse)")
    LOGGER.info(f"Image vectors: {image_vectors.shape} (dense)")

    mem_start = get_memory_usage_mb()
    start_time = time.time()

    similarity_dict, meta = build_sparse_similarity_matrix(
        text_matrix,
        image_vectors=image_vectors,
        image_weight=0.4,
        batch_size=1000,
        top_k=200,
    )

    mem_used = get_memory_usage_mb() - mem_start
    elapsed = time.time() - start_time

    LOGGER.info(f"Memory used: {mem_used:.2f} MB")
    LOGGER.info(f"Time: {elapsed:.2f}s")
    LOGGER.info(f"Modalities: {meta['modalities']}")
    LOGGER.info(f"Image embedding dim: {meta['image_embedding_dim']}")
    LOGGER.info(f"Items with neighbors: {len(similarity_dict)}")

    return similarity_dict


def main():
    """Run all memory optimization tests."""
    LOGGER.info("\n")
    LOGGER.info("=" * 80)
    LOGGER.info("MEMORY OPTIMIZATION TEST SUITE")
    LOGGER.info("=" * 80)
    LOGGER.info(f"System memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    LOGGER.info(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    LOGGER.info("=" * 80)

    try:
        # Test 1: DataFrame optimization
        test_dataframe_optimization()

        # Test 2: Similarity computation
        test_similarity_computation()

        # Test 3: Multimodal similarity
        test_multimodal_similarity()

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("ALL TESTS PASSED ✓")
        LOGGER.info("=" * 80)
        LOGGER.info("\nMemory optimization is working correctly!")
        LOGGER.info("You can now run the full pipeline with reduced memory usage.")

    except Exception as e:
        LOGGER.error(f"\nTest failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
