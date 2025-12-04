"""Automatically tune experiment allocations based on Matomo variant metrics."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from config.settings import BASE_DIR, DATA_DIR
from pipeline.sentry_utils import init_pipeline_sentry, monitor_pipeline_step

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_CONFIG = BASE_DIR / "config" / "experiments.yaml"


def _load_variant_metrics(path: Path, endpoint: str | None) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Variant metrics file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if endpoint:
        df = df[df.get("endpoint") == endpoint]
    if df.empty:
        raise ValueError("Variant metrics contain no rows for the requested endpoint")
    grouped = (
        df.groupby("variant")
        .agg(
            exposure_count=("exposure_count", "sum"),
            click_count=("click_count", "sum"),
        )
        .reset_index()
    )
    return grouped


def _load_experiments_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Experiment config {path} not found")
    return yaml.safe_load(path.read_text()) or {}


def _save_experiments_config(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def _compute_allocations(
    metrics: pd.DataFrame,
    variants: Dict[str, Dict],
    *,
    min_exposures: int,
    smoothing_prior: Tuple[float, float],
    exploration: float,
) -> Dict[str, float]:
    alpha, beta = smoothing_prior
    allocations: Dict[str, float] = {}
    total_score = 0.0
    scores: Dict[str, float] = {}

    for variant_name, spec in variants.items():
        row = metrics[metrics["variant"] == variant_name]
        exposures = float(row["exposure_count"].iloc[0]) if not row.empty else 0.0
        clicks = float(row["click_count"].iloc[0]) if not row.empty else 0.0

        if exposures < min_exposures:
            LOGGER.info(
                "Variant %s skipped due to insufficient exposures (%s < %s)",
                variant_name,
                exposures,
                min_exposures,
            )
            score = 0.0
        else:
            score = (clicks + alpha) / (exposures + beta)
        scores[variant_name] = score
        total_score += score

    if total_score == 0.0:
        LOGGER.warning("No variant has enough data; keeping original allocations.")
        return {name: float(spec.get("allocation", 0.0)) for name, spec in variants.items()}

    num_variants = len(variants)
    min_share = exploration / max(1, num_variants)
    remaining = 1.0 - exploration
    if remaining < 0:
        remaining = 0.0

    score_sum = sum(scores.values())
    for name, score in scores.items():
        share = (score / score_sum * remaining) + min_share if score_sum > 0 else (1.0 / num_variants)
        allocations[name] = share

    # Normalize to 1.0
    total = sum(allocations.values())
    if total > 0:
        allocations = {name: value / total for name, value in allocations.items()}
    return allocations


def optimize_experiment_allocations(
    *,
    experiment_name: str,
    metrics_path: Path,
    config_path: Path,
    endpoint: str,
    min_exposures: int,
    smoothing_prior: Tuple[float, float],
    exploration: float,
) -> Dict[str, float]:
    metrics = _load_variant_metrics(metrics_path, endpoint)
    if metrics.empty:
        LOGGER.warning("No variant metrics available; skipping experiment optimization.")
        return {}
    config = _load_experiments_config(config_path)

    experiment_spec = (config.get("experiments") or {}).get(experiment_name)
    if not experiment_spec:
        raise KeyError(f"Experiment '{experiment_name}' not found in {config_path}")

    variant_specs = experiment_spec.get("variants") or []
    if not variant_specs:
        raise ValueError(f"Experiment '{experiment_name}' has no variants defined")

    variant_lookup = {spec.get("name", "control"): spec for spec in variant_specs}
    allocations = _compute_allocations(
        metrics,
        variant_lookup,
        min_exposures=min_exposures,
        smoothing_prior=smoothing_prior,
        exploration=exploration,
    )

    for variant in variant_specs:
        name = variant.get("name", "control")
        variant["allocation"] = round(allocations.get(name, 0.0), 6)

    _save_experiments_config(config_path, config)
    LOGGER.info(
        "Updated experiment '%s' allocations: %s",
        experiment_name,
        ", ".join(f"{name}={allocations.get(name, 0.0):.3f}" for name in allocations),
    )
    return allocations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-adjust experiment allocations using Matomo variant metrics.")
    parser.add_argument("--experiment", default="recommendation_detail", help="Experiment name to optimize")
    parser.add_argument("--metrics", type=Path, default=PROCESSED_DIR / "recommend_variant_metrics.parquet", help="Variant metrics parquet path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiments YAML config path")
    parser.add_argument("--endpoint", default="recommend_detail", help="Endpoint to filter metrics by")
    parser.add_argument("--min-exposures", type=int, default=500, help="Minimum exposures required per variant")
    parser.add_argument("--alpha", type=float, default=1.0, help="Smoothing prior clicks (alpha)")
    parser.add_argument("--beta", type=float, default=20.0, help="Smoothing prior exposures (beta)")
    parser.add_argument("--exploration", type=float, default=0.1, help="Exploration share reserved for all variants (0-1)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


@monitor_pipeline_step("variant_optimizer", critical=False)
def main() -> None:
    args = parse_args()
    init_pipeline_sentry("variant_optimizer")
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    optimize_experiment_allocations(
        experiment_name=args.experiment,
        metrics_path=args.metrics,
        config_path=args.config,
        endpoint=args.endpoint,
        min_exposures=args.min_exposures,
        smoothing_prior=(args.alpha, args.beta),
        exploration=args.exploration,
    )


if __name__ == "__main__":
    main()
