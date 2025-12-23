"""Train channel fusion weights from exposure/click data."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config.settings import DATA_DIR, MODELS_DIR
from pipeline.sentry_utils import init_pipeline_sentry, monitor_pipeline_step

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PATH = MODELS_DIR / "channel_weights.json"

# Default weights provide priors when historical data is sparse.
DEFAULT_CHANNEL_WEIGHTS: Dict[str, float] = {
    "behavior": 1.5,
    "content": 0.8,
    "vector": 0.5,
    "popular": 0.05,
    "tag": 0.4,
    "category": 0.3,
    "price": 0.2,
    "usercf": 0.6,
    "image": 0.4,
}
KNOWN_CHANNELS = set(DEFAULT_CHANNEL_WEIGHTS.keys()) | {
    "fallback",
    "personalized",
    "rank",
}
MERGE_KEYS = ["request_id", "dataset_id"]


def _load_parquet(path: Path, required: List[str]) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Optional parquet %s missing; using empty dataframe.", path)
        return pd.DataFrame(columns=required)
    frame = pd.read_parquet(path)
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Parquet {path} missing required columns: {', '.join(missing)}")
    return frame


def _load_exposures() -> pd.DataFrame:
    required = MERGE_KEYS + ["reason", "score", "timestamp"]
    exposures = _load_parquet(PROCESSED_DIR / "recommend_exposures.parquet", required)
    if exposures.empty:
        LOGGER.warning("No exposures available for channel weight training.")
        return exposures
    exposures = exposures.dropna(subset=["dataset_id"])
    exposures["dataset_id"] = exposures["dataset_id"].astype(int)
    exposures["channel"] = exposures["reason"].apply(_extract_channel)
    exposures = exposures[exposures["channel"] != "unknown"]
    return exposures


def _load_event_flag(filename: str, flag_name: str) -> pd.DataFrame:
    frame = _load_parquet(PROCESSED_DIR / filename, MERGE_KEYS)
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(MERGE_KEYS)
    frame[flag_name] = 1.0
    return frame[MERGE_KEYS + [flag_name]]


def _extract_channel(reason: object) -> str:
    if reason is None:
        return "unknown"
    text = str(reason).strip().lower()
    if not text:
        return "unknown"
    tokens = [token for token in text.split("+") if token]
    for token in tokens:
        if token in KNOWN_CHANNELS:
            return token
        if token.startswith("fallback"):
            return "fallback"
    return tokens[0] if tokens else "unknown"


def _apply_event_flags(exposures: pd.DataFrame) -> pd.DataFrame:
    clicks = _load_event_flag("recommend_clicks.parquet", "clicked")
    conversions = _load_event_flag("recommend_conversions.parquet", "converted")

    merged = exposures
    for event_frame in (clicks, conversions):
        if event_frame.empty:
            continue
        merged = merged.merge(event_frame, on=MERGE_KEYS, how="left")

    if "clicked" not in merged.columns:
        merged["clicked"] = 0.0
    else:
        merged["clicked"] = merged["clicked"].fillna(0.0)

    if "converted" not in merged.columns:
        merged["converted"] = 0.0
    else:
        merged["converted"] = merged["converted"].fillna(0.0)

    return merged


def _normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    # Ensure positive weights and normalize to match default sum.
    filtered = {k: max(v, 0.0) for k, v in raw_weights.items() if v > 0}
    if not filtered:
        return DEFAULT_CHANNEL_WEIGHTS.copy()
    total = sum(filtered.values())
    base_total = sum(DEFAULT_CHANNEL_WEIGHTS.values())
    if total == 0:
        return DEFAULT_CHANNEL_WEIGHTS.copy()
    scale = base_total / total
    normalized = {k: round(v * scale, 6) for k, v in filtered.items()}

    # Add defaults for missing channels to keep API stable.
    for channel, default_value in DEFAULT_CHANNEL_WEIGHTS.items():
        normalized.setdefault(channel, round(default_value, 6))
    return normalized


def _compute_channel_stats(
    exposures: pd.DataFrame,
    alpha: float,
    beta: float,
    ctr_weight: float,
    cvr_weight: float,
    min_exposures: int,
) -> Dict[str, float]:
    stats = (
        exposures.groupby("channel")
        .agg(
            exposures=("dataset_id", "count"),
            clicks=("clicked", "sum"),
            conversions=("converted", "sum"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )

    stats["ctr"] = (stats["clicks"] + alpha) / (stats["exposures"] + beta)
    stats["cvr"] = (stats["conversions"] + alpha) / (stats["exposures"] + beta)
    stats["score"] = ctr_weight * stats["ctr"] + cvr_weight * stats["cvr"]

    weights: Dict[str, float] = {}
    for _, row in stats.iterrows():
        channel = str(row["channel"])
        if row["exposures"] < min_exposures:
            weights[channel] = DEFAULT_CHANNEL_WEIGHTS.get(channel, 0.1)
        else:
            weights[channel] = float(row["score"])
    LOGGER.info("Computed stats for %d channels.", len(stats))
    return weights


def _write_output(weights: Dict[str, float], metadata: Dict[str, object]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weights": weights,
        "metadata": metadata,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    LOGGER.info("Channel weights saved to %s", OUTPUT_PATH)


@monitor_pipeline_step("train_channel_weights", critical=False)
def main() -> None:
    parser = argparse.ArgumentParser(description="Train channel fusion weights from exposure logs.")
    parser.add_argument("--alpha", type=float, default=2.0, help="CTR Beta prior (clicks).")
    parser.add_argument("--beta", type=float, default=50.0, help="CTR Beta prior (exposures).")
    parser.add_argument("--ctr-weight", type=float, default=0.8, help="Importance of CTR in final score.")
    parser.add_argument("--cvr-weight", type=float, default=0.2, help="Importance of CVR in final score.")
    parser.add_argument("--min-exposures", type=int, default=500, help="Minimum exposures per channel.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_pipeline_sentry("train_channel_weights")

    exposures = _load_exposures()
    if exposures.empty:
        LOGGER.warning("Fallback to default channel weights due to missing exposure data.")
        _write_output(DEFAULT_CHANNEL_WEIGHTS.copy(), {"reason": "no_data"})
        return

    exposures = _apply_event_flags(exposures)
    raw_weights = _compute_channel_stats(
        exposures,
        alpha=args.alpha,
        beta=args.beta,
        ctr_weight=args.ctr_weight,
        cvr_weight=args.cvr_weight,
        min_exposures=args.min_exposures,
    )
    normalized = _normalize_weights(raw_weights)

    metadata = {
        "alpha": args.alpha,
        "beta": args.beta,
        "ctr_weight": args.ctr_weight,
        "cvr_weight": args.cvr_weight,
        "min_exposures": args.min_exposures,
        "channels": len(normalized),
    }
    _write_output(normalized, metadata)


if __name__ == "__main__":
    main()
