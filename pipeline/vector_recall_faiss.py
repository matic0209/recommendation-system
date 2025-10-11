"""High-performance vector recall using Faiss indexing.

This module provides efficient similarity search using Faiss with support for:
- TF-IDF embeddings (traditional)
- Sentence-BERT embeddings (semantic)
- Multiple index types (Flat, IVF, HNSW)
- GPU acceleration support (optional)

Performance: 10-100x faster than sklearn for large datasets.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer

    FAISS_LIBS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    FAISS_LIBS_AVAILABLE = False

from config.settings import DATA_DIR

LOGGER = logging.getLogger(__name__)
MODEL_DIR = DATA_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"


if not FAISS_LIBS_AVAILABLE:  # pragma: no cover - graceful fallback

    class FaissVectorRecall:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Faiss/Sentence-Transformers not installed. Set USE_FAISS_RECALL=0 or install optional dependencies."
            )

else:

    class FaissVectorRecall:
        """High-performance vector recall using Faiss."""

        def __init__(
            self,
            embedding_type: str = "tfidf",
            index_type: str = "flat",
            use_gpu: bool = False,
            sentence_model: str = "all-MiniLM-L6-v2",
        ):
            """
            Initialize Faiss vector recall.

            Args:
                embedding_type: "tfidf" or "sbert"
                index_type: "flat" (exact), "ivf" (approximate), or "hnsw" (graph-based)
                use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
                sentence_model: Sentence-BERT model name for semantic embeddings
            """
            self.embedding_type = embedding_type
            self.index_type = index_type
            self.use_gpu = use_gpu

            # Embeddings
            self.vectorizer = None  # For TF-IDF
            self.sentence_model = None  # For Sentence-BERT
            self.embeddings = None  # Embedding matrix
            self.dataset_ids = None  # ID mapping

            # Faiss index
            self.index = None
            self.dimension = None

            # Load sentence model if needed
            if embedding_type == "sbert":
                LOGGER.info("Loading Sentence-BERT model: %s", sentence_model)
                self.sentence_model = SentenceTransformer(sentence_model)
                self.dimension = self.sentence_model.get_sentence_embedding_dimension()
                LOGGER.info("Sentence-BERT dimension: %d", self.dimension)

    def build_tfidf_embeddings(
        self,
        dataset_features: pd.DataFrame,
        text_columns: List[str] = None,
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 0.8,
    ) -> np.ndarray:
        """
        Build TF-IDF embeddings for datasets.

        Args:
            dataset_features: DataFrame with dataset information
            text_columns: Columns to use for TF-IDF (default: description, tag, dataset_name)
            max_features: Maximum number of TF-IDF features
            min_df: Minimum document frequency
            max_df: Maximum document frequency

        Returns:
            TF-IDF embedding matrix (n_datasets, n_features)
        """
        if text_columns is None:
            text_columns = ["description", "tag", "dataset_name"]

        LOGGER.info("Building TF-IDF embeddings from columns: %s", text_columns)

        # Combine text columns
        text_data = []
        for _, row in dataset_features.iterrows():
            texts = []
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    texts.append(str(row[col]))
            text_data.append(" ".join(texts))

        # Build TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
            ngram_range=(1, 2),
        )

        embeddings = self.vectorizer.fit_transform(text_data)
        embeddings_dense = embeddings.toarray().astype(np.float32)

        self.dimension = embeddings_dense.shape[1]
        LOGGER.info(
            "TF-IDF embeddings built: %d datasets × %d features",
            embeddings_dense.shape[0],
            self.dimension,
        )

        return embeddings_dense

    def build_sbert_embeddings(
        self,
        dataset_features: pd.DataFrame,
        text_columns: List[str] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Build Sentence-BERT embeddings for datasets.

        Args:
            dataset_features: DataFrame with dataset information
            text_columns: Columns to use (default: description, tag, dataset_name)
            batch_size: Batch size for encoding

        Returns:
            Sentence-BERT embedding matrix (n_datasets, embedding_dim)
        """
        if self.sentence_model is None:
            raise ValueError("Sentence-BERT model not loaded")

        if text_columns is None:
            text_columns = ["description", "tag", "dataset_name"]

        LOGGER.info("Building Sentence-BERT embeddings from columns: %s", text_columns)

        # Combine text columns
        text_data = []
        for _, row in dataset_features.iterrows():
            texts = []
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    texts.append(str(row[col]))
            combined_text = " ".join(texts)
            if not combined_text.strip():
                combined_text = "unknown dataset"
            text_data.append(combined_text)

        # Encode with Sentence-BERT
        LOGGER.info("Encoding %d texts with Sentence-BERT...", len(text_data))
        embeddings = self.sentence_model.encode(
            text_data,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float32)
        LOGGER.info(
            "Sentence-BERT embeddings built: %d datasets × %d dims",
            embeddings.shape[0],
            embeddings.shape[1],
        )

        return embeddings

    def build_faiss_index(
        self,
        embeddings: np.ndarray,
        dataset_ids: List[int],
        nlist: int = 100,
        nprobe: int = 10,
        hnsw_m: int = 32,
    ) -> faiss.Index:
        """
        Build Faiss index from embeddings.

        Args:
            embeddings: Embedding matrix (n_datasets, dimension)
            dataset_ids: List of dataset IDs
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search for IVF
            hnsw_m: Number of connections for HNSW

        Returns:
            Faiss index
        """
        n_datasets, dimension = embeddings.shape
        self.dataset_ids = np.array(dataset_ids, dtype=np.int64)
        self.embeddings = embeddings
        self.dimension = dimension

        LOGGER.info(
            "Building Faiss index: type=%s, n_datasets=%d, dimension=%d",
            self.index_type,
            n_datasets,
            dimension,
        )

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build index based on type
        if self.index_type == "flat":
            # Exact search using L2 distance
            index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine after normalize
            index.add(embeddings)
            LOGGER.info("Built Flat index (exact search)")

        elif self.index_type == "ivf":
            # Approximate search using inverted file index
            quantizer = faiss.IndexFlatIP(dimension)
            nlist = min(nlist, n_datasets // 10)  # Adjust for small datasets
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

            LOGGER.info("Training IVF index with %d clusters...", nlist)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = nprobe  # Search top nprobe clusters
            LOGGER.info("Built IVF index (nlist=%d, nprobe=%d)", nlist, nprobe)

        elif self.index_type == "hnsw":
            # Graph-based approximate search (no training required)
            index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.add(embeddings)
            LOGGER.info("Built HNSW index (M=%d)", hnsw_m)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # GPU support
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                LOGGER.info("Moved index to GPU")
            except Exception as e:
                LOGGER.warning("Failed to move index to GPU: %s", e)

        self.index = index
        LOGGER.info("Faiss index built successfully: %d vectors", index.ntotal)

        return index

    def train(
        self,
        dataset_features: pd.DataFrame,
        text_columns: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the vector recall system.

        Args:
            dataset_features: DataFrame with dataset information
            text_columns: Text columns to use for embeddings
            **kwargs: Additional arguments for embedding/index building

        Returns:
            Training statistics
        """
        LOGGER.info("Training vector recall: embedding_type=%s", self.embedding_type)

        # Build embeddings
        if self.embedding_type == "tfidf":
            embeddings = self.build_tfidf_embeddings(
                dataset_features,
                text_columns=text_columns,
                max_features=kwargs.get("max_features", 5000),
                min_df=kwargs.get("min_df", 2),
                max_df=kwargs.get("max_df", 0.8),
            )
        elif self.embedding_type == "sbert":
            embeddings = self.build_sbert_embeddings(
                dataset_features,
                text_columns=text_columns,
                batch_size=kwargs.get("batch_size", 32),
            )
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

        # Build Faiss index
        dataset_ids = dataset_features["dataset_id"].tolist()
        self.build_faiss_index(
            embeddings,
            dataset_ids,
            nlist=kwargs.get("nlist", 100),
            nprobe=kwargs.get("nprobe", 10),
            hnsw_m=kwargs.get("hnsw_m", 32),
        )

        stats = {
            "embedding_type": self.embedding_type,
            "index_type": self.index_type,
            "n_datasets": len(dataset_ids),
            "dimension": self.dimension,
            "index_ntotal": self.index.ntotal,
        }

        LOGGER.info("Vector recall training complete: %s", stats)
        return stats

    def search(
        self,
        query_dataset_id: int,
        k: int = 100,
    ) -> List[Tuple[int, float]]:
        """
        Search for similar datasets.

        Args:
            query_dataset_id: Query dataset ID
            k: Number of results to return

        Returns:
            List of (dataset_id, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call train() first.")

        # Get query embedding
        query_idx = np.where(self.dataset_ids == query_dataset_id)[0]
        if len(query_idx) == 0:
            LOGGER.warning("Query dataset %d not found in index", query_dataset_id)
            return []

        query_embedding = self.embeddings[query_idx[0]].reshape(1, -1)

        # Search
        scores, indices = self.index.search(query_embedding, k + 1)  # +1 to exclude self

        # Convert to results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            dataset_id = int(self.dataset_ids[idx])
            if dataset_id != query_dataset_id:  # Exclude self
                results.append((dataset_id, float(score)))

        return results[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int = 100,
    ) -> List[Tuple[int, float]]:
        """
        Search for datasets by text query.

        Args:
            query_text: Query text
            k: Number of results to return

        Returns:
            List of (dataset_id, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call train() first.")

        # Get query embedding
        if self.embedding_type == "tfidf":
            if self.vectorizer is None:
                raise ValueError("TF-IDF vectorizer not trained")
            query_embedding = self.vectorizer.transform([query_text]).toarray().astype(np.float32)
        elif self.embedding_type == "sbert":
            if self.sentence_model is None:
                raise ValueError("Sentence-BERT model not loaded")
            query_embedding = self.sentence_model.encode(
                [query_text], convert_to_numpy=True
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Convert to results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            dataset_id = int(self.dataset_ids[idx])
            results.append((dataset_id, float(score)))

        return results

    def batch_search(
        self,
        query_dataset_ids: List[int],
        k: int = 100,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Batch search for multiple queries.

        Args:
            query_dataset_ids: List of query dataset IDs
            k: Number of results per query

        Returns:
            Dict mapping query_id -> List of (dataset_id, score)
        """
        if self.index is None:
            raise ValueError("Index not built. Call train() first.")

        # Get query embeddings
        query_indices = []
        valid_query_ids = []
        for query_id in query_dataset_ids:
            query_idx = np.where(self.dataset_ids == query_id)[0]
            if len(query_idx) > 0:
                query_indices.append(query_idx[0])
                valid_query_ids.append(query_id)

        if len(query_indices) == 0:
            return {}

        query_embeddings = self.embeddings[query_indices]

        # Batch search
        scores, indices = self.index.search(query_embeddings, k + 1)

        # Convert to results
        results = {}
        for i, query_id in enumerate(valid_query_ids):
            query_results = []
            for idx, score in zip(indices[i], scores[i]):
                dataset_id = int(self.dataset_ids[idx])
                if dataset_id != query_id:  # Exclude self
                    query_results.append((dataset_id, float(score)))
            results[query_id] = query_results[:k]

        return results

    def get_embedding(self, dataset_id: int) -> Optional[np.ndarray]:
        """Get embedding for a specific dataset."""
        if self.embeddings is None:
            return None

        idx = np.where(self.dataset_ids == dataset_id)[0]
        if len(idx) == 0:
            return None

        return self.embeddings[idx[0]]

    def save(self, model_name: str = "faiss_vector_recall") -> Dict[str, Path]:
        """
        Save the model to disk.

        Args:
            model_name: Base name for saved files

        Returns:
            Dict of saved file paths
        """
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save Faiss index
        if self.index is not None:
            index_path = MODEL_DIR / f"{model_name}_index.faiss"
            # Move to CPU before saving if using GPU
            index_to_save = self.index
            if self.use_gpu:
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_to_save, str(index_path))
            saved_files["index"] = index_path
            LOGGER.info("Saved Faiss index: %s", index_path)

        # Save embeddings
        if self.embeddings is not None:
            embeddings_path = MODEL_DIR / f"{model_name}_embeddings.npy"
            np.save(embeddings_path, self.embeddings)
            saved_files["embeddings"] = embeddings_path

        # Save dataset IDs
        if self.dataset_ids is not None:
            ids_path = MODEL_DIR / f"{model_name}_ids.npy"
            np.save(ids_path, self.dataset_ids)
            saved_files["ids"] = ids_path

        # Save TF-IDF vectorizer
        if self.vectorizer is not None:
            vectorizer_path = MODEL_DIR / f"{model_name}_vectorizer.pkl"
            with open(vectorizer_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            saved_files["vectorizer"] = vectorizer_path

        # Save metadata
        metadata = {
            "embedding_type": self.embedding_type,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "n_datasets": len(self.dataset_ids) if self.dataset_ids is not None else 0,
        }
        metadata_path = MODEL_DIR / f"{model_name}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        saved_files["metadata"] = metadata_path

        LOGGER.info("Model saved: %s", saved_files)
        return saved_files

    def load(self, model_name: str = "faiss_vector_recall") -> bool:
        """
        Load the model from disk.

        Args:
            model_name: Base name for saved files

        Returns:
            True if successful
        """
        try:
            # Load metadata
            metadata_path = MODEL_DIR / f"{model_name}_metadata.json"
            if not metadata_path.exists():
                LOGGER.error("Metadata not found: %s", metadata_path)
                return False

            metadata = json.loads(metadata_path.read_text())
            self.embedding_type = metadata["embedding_type"]
            self.index_type = metadata["index_type"]
            self.dimension = metadata["dimension"]

            # Load Faiss index
            index_path = MODEL_DIR / f"{model_name}_index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                if self.use_gpu:
                    try:
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                        LOGGER.info("Moved loaded index to GPU")
                    except Exception as e:
                        LOGGER.warning("Failed to move index to GPU: %s", e)
                LOGGER.info("Loaded Faiss index: %s", index_path)

            # Load embeddings
            embeddings_path = MODEL_DIR / f"{model_name}_embeddings.npy"
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)

            # Load dataset IDs
            ids_path = MODEL_DIR / f"{model_name}_ids.npy"
            if ids_path.exists():
                self.dataset_ids = np.load(ids_path)

            # Load TF-IDF vectorizer if exists
            vectorizer_path = MODEL_DIR / f"{model_name}_vectorizer.pkl"
            if vectorizer_path.exists():
                with open(vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)

            LOGGER.info("Model loaded successfully: %s", model_name)
            return True

        except Exception as e:
            LOGGER.error("Failed to load model: %s", e)
            return False


def benchmark_search_performance(
    vector_recall: FaissVectorRecall,
    n_queries: int = 1000,
    k: int = 100,
) -> Dict[str, float]:
    """Benchmark search performance."""
    import time

    if vector_recall.dataset_ids is None or len(vector_recall.dataset_ids) == 0:
        return {"error": "No data loaded"}

    # Random queries
    np.random.seed(42)
    query_ids = np.random.choice(vector_recall.dataset_ids, size=min(n_queries, len(vector_recall.dataset_ids)), replace=False)

    # Single query benchmark
    start = time.time()
    for query_id in query_ids[:100]:
        _ = vector_recall.search(int(query_id), k=k)
    single_query_time = (time.time() - start) / 100

    # Batch query benchmark
    start = time.time()
    _ = vector_recall.batch_search([int(qid) for qid in query_ids[:100]], k=k)
    batch_time = time.time() - start
    batch_query_time = batch_time / 100

    results = {
        "single_query_ms": single_query_time * 1000,
        "batch_query_ms": batch_query_time * 1000,
        "batch_speedup": single_query_time / batch_query_time if batch_query_time > 0 else 0,
        "qps_single": 1.0 / single_query_time if single_query_time > 0 else 0,
        "qps_batch": 100.0 / batch_time if batch_time > 0 else 0,
    }

    LOGGER.info("Performance benchmark: %s", results)
    return results


def main() -> None:
    """Train and benchmark Faiss vector recall."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Faiss vector recall training")
    parser.add_argument("--embedding-type", type=str, default="tfidf", choices=["tfidf", "sbert"])
    parser.add_argument("--index-type", type=str, default="flat", choices=["flat", "ivf", "hnsw"])
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    args = parser.parse_args()

    # Load data
    dataset_features_path = PROCESSED_DIR / "dataset_features.parquet"
    if not dataset_features_path.exists():
        LOGGER.error("Dataset features not found: %s", dataset_features_path)
        return

    dataset_features = pd.read_parquet(dataset_features_path)
    LOGGER.info("Loaded %d dataset features", len(dataset_features))

    # Initialize and train
    vector_recall = FaissVectorRecall(
        embedding_type=args.embedding_type,
        index_type=args.index_type,
        use_gpu=args.use_gpu,
    )

    LOGGER.info("Training vector recall...")
    stats = vector_recall.train(dataset_features)

    # Save model
    model_name = f"faiss_{args.embedding_type}_{args.index_type}"
    saved_files = vector_recall.save(model_name=model_name)

    # Benchmark
    if args.benchmark:
        LOGGER.info("Running performance benchmark...")
        perf_stats = benchmark_search_performance(vector_recall, n_queries=1000, k=100)

    # Print summary
    print("\n" + "=" * 80)
    print("FAISS VECTOR RECALL TRAINING SUMMARY")
    print("=" * 80)
    print(f"Embedding Type: {args.embedding_type}")
    print(f"Index Type: {args.index_type}")
    print(f"Datasets: {stats['n_datasets']:,}")
    print(f"Dimension: {stats['dimension']:,}")
    print(f"Index Total: {stats['index_ntotal']:,}")
    print(f"\nSaved Files:")
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")

    if args.benchmark:
        print(f"\nPerformance Benchmark:")
        print(f"  Single Query: {perf_stats['single_query_ms']:.2f} ms")
        print(f"  Batch Query: {perf_stats['batch_query_ms']:.2f} ms")
        print(f"  Batch Speedup: {perf_stats['batch_speedup']:.1f}x")
        print(f"  QPS (Single): {perf_stats['qps_single']:.0f}")
        print(f"  QPS (Batch): {perf_stats['qps_batch']:.0f}")
    print("=" * 80 + "\n")

    LOGGER.info("Faiss vector recall training complete!")


if __name__ == "__main__":
    main()
