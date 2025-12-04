"""Generate sentence embeddings for dataset titles/descriptions using SentenceTransformers."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from config.settings import DATA_DIR
from pipeline.sentry_utils import init_pipeline_sentry, monitor_pipeline_step

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"
TEXT_EMBEDDINGS_PATH = PROCESSED_DIR / "dataset_text_embeddings.parquet"
DEFAULT_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _prepare_corpus() -> pd.DataFrame:
    dataset_path = PROCESSED_DIR / "dataset_features_v2.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset features not found: {dataset_path}")
    df = pd.read_parquet(dataset_path, columns=["dataset_id", "dataset_name", "description", "tag"])
    df["dataset_id"] = df["dataset_id"].astype(int)
    df["text_corpus"] = (
        df["dataset_name"].fillna("")
        + "\n"
        + df["description"].fillna("")
        + "\n"
        + df["tag"].fillna("")
    )
    df["text_corpus"] = df["text_corpus"].str.strip()
    return df[["dataset_id", "text_corpus"]]


def _load_model(model_name: str) -> SentenceTransformer:
    LOGGER.info("Loading sentence transformer model: %s", model_name)
    model = SentenceTransformer(model_name)
    return model


def _batch_iter(texts: list[str], batch_size: int = 512):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


@monitor_pipeline_step("text_embeddings", critical=True)
def build_text_embeddings(model_name: str = DEFAULT_MODEL) -> None:
    LOGGER.info("Preparing corpus for text embeddings...")
    corpus = _prepare_corpus()
    if corpus.empty:
        LOGGER.warning("No dataset entries found for text embeddings")
        return

    model = _load_model(model_name)
    texts = corpus["text_corpus"].tolist()
    embeddings_list = []
    LOGGER.info("Encoding %d dataset entries...", len(texts))
    for batch in _batch_iter(texts, batch_size=int(os.getenv("TEXT_EMBED_BATCH", "256"))):
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings_list.append(embeddings)
    import numpy as np

    all_embeddings = np.vstack(embeddings_list)
    dim = all_embeddings.shape[1]
    columns = [f"text_embed_{i:03d}" for i in range(dim)]
    embedding_df = pd.DataFrame(all_embeddings, columns=columns)
    embedding_df.insert(0, "dataset_id", corpus["dataset_id"].values)
    embedding_df.to_parquet(TEXT_EMBEDDINGS_PATH, index=False)
    LOGGER.info("Saved text embeddings to %s (dim=%d)", TEXT_EMBEDDINGS_PATH, dim)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_pipeline_sentry("text_embeddings")
    build_text_embeddings()


if __name__ == "__main__":
    main()
