"""Visual image embedding extraction using CPU-friendly CLIP models."""
from __future__ import annotations

import hashlib
import logging
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image
from requests import Response
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualEmbeddingConfig:
    """Configuration for the visual embedding extractor."""

    model_name: str = field(default_factory=lambda: os.getenv("CLIP_MODEL_PATH", "clip-ViT-B-32"))
    device: str = "cpu"
    batch_size: int = 8
    max_images_per_item: int = 4
    request_timeout: int = 10
    retry_attempts: int = 2
    sleep_between_retries: float = 0.5
    local_image_root: Optional[Path] = None


class VisualImageFeatureExtractor:
    """Download images and create CLIP embeddings without requiring a GPU."""

    def __init__(
        self,
        cache_dir: Path,
        config: VisualEmbeddingConfig | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or VisualEmbeddingConfig()
        self.local_image_root = (
            Path(self.config.local_image_root).expanduser().resolve()
            if getattr(self.config, "local_image_root", None)
            else None
        )
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )
        LOGGER.info(
            "Initialized visual image extractor (model=%s, device=%s, cache=%s)",
            self.config.model_name,
            self.config.device,
            self.cache_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_dataset_embeddings(
        self,
        dataset_images: pd.DataFrame,
        dataset_ids: Optional[Iterable[int]] = None,
    ) -> pd.DataFrame:
        """
        Compute dataset-level visual embeddings.

        Args:
            dataset_images: DataFrame with dataset_id and image_url columns.
            dataset_ids: Optional iterable to restrict computation.

        Returns:
            DataFrame with flattened embedding columns and summary stats.
        """
        if dataset_images is None or dataset_images.empty:
            LOGGER.warning("Dataset image table is empty; skipping visual embeddings")
            return pd.DataFrame(columns=["dataset_id"])

        required_cols = {"dataset_id", "image_url"}
        if missing := required_cols.difference(dataset_images.columns):
            LOGGER.warning("Dataset images missing required columns: %s", ", ".join(sorted(missing)))
            return pd.DataFrame(columns=["dataset_id"])

        if dataset_ids is not None:
            dataset_ids = set(dataset_ids)
            dataset_images = dataset_images[dataset_images["dataset_id"].isin(dataset_ids)]

        dataset_images = dataset_images.dropna(subset=["dataset_id", "image_url"])
        if dataset_images.empty:
            LOGGER.warning("No valid dataset images after filtering")
            return pd.DataFrame(columns=["dataset_id"])

        grouped = dataset_images.groupby("dataset_id").head(self.config.max_images_per_item)
        downloads = self._download_images(grouped)
        if not downloads:
            LOGGER.warning("Failed to download any images; skipping visual embeddings")
            return pd.DataFrame(columns=["dataset_id"])

        embeddings = self._encode_images(downloads)
        if embeddings.size == 0:
            LOGGER.warning("Embedding matrix empty; nothing to persist")
            return pd.DataFrame(columns=["dataset_id"])

        dataset_vectors, summary = self._aggregate_embeddings(downloads, embeddings)
        features = self._build_feature_frame(dataset_vectors, summary)
        LOGGER.info(
            "Computed visual embeddings for %d datasets (embedding_dim=%d)",
            len(features),
            summary["embedding_dim"],
        )
        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _download_images(
        self,
        dataset_images: pd.DataFrame,
    ) -> List[Tuple[int, Path]]:
        """Download images to the cache directory."""
        if self.local_image_root and self.local_image_root.exists():
            records: List[Tuple[int, Path]] = []
            dataset_ids = (
                dataset_images.get("dataset_id")
                if isinstance(dataset_images, pd.DataFrame)
                else pd.Series(dtype=int)
            )
            if isinstance(dataset_ids, pd.Series) and not dataset_ids.empty:
                unique_ids = dataset_ids.dropna().astype(int).unique()
            else:
                unique_ids = []

            for dataset_id in unique_ids:
                dataset_dir = self.local_image_root / str(dataset_id)
                if not dataset_dir.exists():
                    LOGGER.debug("Local image directory missing for dataset %s", dataset_id)
                    continue
                image_paths = sorted(
                    path for path in dataset_dir.iterdir() if path.is_file()
                )
                if not image_paths:
                    LOGGER.debug("No images found under %s", dataset_dir)
                    continue
                for path in image_paths[: self.config.max_images_per_item]:
                    records.append((int(dataset_id), path))

            if records:
                LOGGER.info(
                    "Loaded %d pre-downloaded images from %s",
                    len(records),
                    self.local_image_root,
                )
                return records

        records: List[Tuple[int, Path]] = []
        for _, row in dataset_images.iterrows():
            dataset_id = int(row["dataset_id"])
            url = str(row["image_url"]).strip()
            if not url:
                continue

            cached_path = self._cached_path(dataset_id, url)
            if cached_path.exists():
                records.append((dataset_id, cached_path))
                continue

            response = self._fetch_with_retry(url)
            if response is None:
                continue

            cached_path.write_bytes(response.content)
            records.append((dataset_id, cached_path))

        return records

    def _fetch_with_retry(self, url: str) -> Response | None:
        """Fetch image bytes with retry logic."""
        attempt = 0
        while attempt <= self.config.retry_attempts:
            try:
                response = requests.get(url, timeout=self.config.request_timeout)
                response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                LOGGER.debug("Failed to fetch %s (attempt=%s): %s", url, attempt, exc)
                if attempt > self.config.retry_attempts:
                    LOGGER.warning("Giving up downloading %s after %d attempts", url, attempt)
                    return None
                time.sleep(self.config.sleep_between_retries)
        return None

    def _cached_path(self, dataset_id: int, url: str) -> Path:
        """Generate deterministic cache path for image."""
        digest = hashlib.md5(url.encode("utf8")).hexdigest()
        filename = f"{dataset_id}_{digest}.img"
        return self.cache_dir / filename

    def _encode_images(self, downloads: List[Tuple[int, Path]]) -> np.ndarray:
        """Encode downloaded images into embeddings."""
        if not downloads:
            return np.empty((0, 0))

        images: List[Image.Image] = []
        valid_indices: List[int] = []
        for idx, (_, path) in enumerate(downloads):
            try:
                image = Image.open(path).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Failed to load image %s: %s", path, exc)
                continue
            images.append(image)
            valid_indices.append(idx)

        if not images:
            return np.empty((0, 0))

        embeddings = self.model.encode(
            images,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Filter downloads to valid entries (to keep alignment)
        if len(valid_indices) != len(downloads):
            aligned = np.zeros((len(downloads), embeddings.shape[1]), dtype=np.float32)
            for new_idx, old_idx in enumerate(valid_indices):
                aligned[old_idx] = embeddings[new_idx]
            embeddings = aligned
        return embeddings

    def _aggregate_embeddings(
        self,
        downloads: List[Tuple[int, Path]],
        embeddings: np.ndarray,
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, int]]:
        """Aggregate embeddings per dataset."""
        if embeddings.size == 0:
            return {}, {"embedding_dim": 0}

        dataset_vectors: Dict[int, List[np.ndarray]] = {}
        for (dataset_id, _), vector in zip(downloads, embeddings):
            dataset_vectors.setdefault(dataset_id, []).append(vector)

        aggregated: Dict[int, np.ndarray] = {}
        for dataset_id, vectors in dataset_vectors.items():
            matrix = np.vstack(vectors)
            aggregated[dataset_id] = matrix.mean(axis=0)

        summary = {
            "embedding_dim": embeddings.shape[1],
            "datasets": len(aggregated),
            "total_images": len(downloads),
        }
        return aggregated, summary

    def _build_feature_frame(
        self,
        dataset_vectors: Dict[int, np.ndarray],
        summary: Dict[str, int],
    ) -> pd.DataFrame:
        """Convert aggregated embeddings into a feature frame."""
        if not dataset_vectors:
            return pd.DataFrame(columns=["dataset_id"])

        dim = summary["embedding_dim"]
        columns = [f"image_embed_mean_{i:03d}" for i in range(dim)]

        rows: List[Dict[str, float]] = []
        for dataset_id, vector in dataset_vectors.items():
            row = {"dataset_id": dataset_id}
            row.update({col: float(value) for col, value in zip(columns, vector)})
            row["image_embed_norm"] = float(np.linalg.norm(vector))
            rows.append(row)

        frame = pd.DataFrame(rows)
        numeric_cols = [col for col in frame.columns if col != "dataset_id"]
        frame[numeric_cols] = frame[numeric_cols].fillna(0.0)
        return frame


def main() -> None:
    """CLI helper to compute and persist visual embeddings."""
    import argparse
    from config.settings import BASE_DIR, DATA_DIR

    parser = argparse.ArgumentParser(description="Compute visual embeddings for dataset images.")
    parser.add_argument("--max-images", type=int, default=4, help="Max images per dataset")
    parser.add_argument("--model", type=str, default="clip-ViT-B-32", help="SentenceTransformer model name")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "processed" / "dataset_image_embeddings.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset_path = DATA_DIR / "business" / "dataset_image.parquet"
    if not dataset_path.exists():
        LOGGER.error("Dataset image parquet not found at %s", dataset_path)
        raise SystemExit(1)

    dataset_images = pd.read_parquet(dataset_path)
    config = VisualEmbeddingConfig(
        model_name=args.model,
        device=args.device,
        max_images_per_item=args.max_images,
    )
    extractor = VisualImageFeatureExtractor(BASE_DIR / "cache" / "images", config=config)
    features = extractor.build_dataset_embeddings(dataset_images)

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        LOGGER.warning("Unable to create output directory %s due to permissions", args.output.parent)
    try:
        features.to_parquet(args.output, index=False)
    except PermissionError:
        fallback = BASE_DIR / "cache" / args.output.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(fallback, index=False)
        LOGGER.warning("Permission denied writing %s; saved embeddings to %s", args.output, fallback)
        args.output = fallback
    embedding_dim = len([col for col in features.columns if col.startswith("image_embed_mean_")])
    LOGGER.info(
        "Saved visual embeddings to %s (rows=%d, embedding_dim=%d)",
        args.output,
        len(features),
        embedding_dim,
    )


if __name__ == "__main__":
    main()
