from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from pipeline.variant_optimizer import optimize_experiment_allocations


def test_optimize_allocations(tmp_path: Path):
    metrics_path = tmp_path / "variant_metrics.parquet"
    config_path = tmp_path / "experiments.yaml"

    df = pd.DataFrame(
        [
            {"variant": "control", "exposure_count": 1000, "click_count": 50, "endpoint": "recommend_detail"},
            {"variant": "test", "exposure_count": 1000, "click_count": 80, "endpoint": "recommend_detail"},
        ]
    )
    df.to_parquet(metrics_path, index=False)

    config = {
        "experiments": {
            "recommendation_detail": {
                "salt": "test",
                "status": "active",
                "variants": [
                    {"name": "control", "allocation": 0.5, "parameters": {}},
                    {"name": "test", "allocation": 0.5, "parameters": {}},
                ],
            }
        }
    }
    config_path.write_text(yaml.safe_dump(config))

    allocations = optimize_experiment_allocations(
        experiment_name="recommendation_detail",
        metrics_path=metrics_path,
        config_path=config_path,
        endpoint="recommend_detail",
        min_exposures=10,
        smoothing_prior=(1.0, 20.0),
        exploration=0.1,
    )
    assert allocations["test"] > allocations["control"]

    saved = yaml.safe_load(config_path.read_text())
    saved_variants = saved["experiments"]["recommendation_detail"]["variants"]
    assert saved_variants[1]["allocation"] > saved_variants[0]["allocation"]


def test_optimize_allocations_missing_metrics(tmp_path: Path):
    config_path = tmp_path / "experiments.yaml"
    config = {
        "experiments": {
            "recommendation_detail": {
                "salt": "test",
                "status": "active",
                "variants": [
                    {"name": "control", "allocation": 0.5, "parameters": {}},
                    {"name": "test", "allocation": 0.5, "parameters": {}},
                ],
            }
        }
    }
    config_path.write_text(yaml.safe_dump(config))

    allocations = optimize_experiment_allocations(
        experiment_name="recommendation_detail",
        metrics_path=tmp_path / "missing.parquet",
        config_path=config_path,
        endpoint="recommend_detail",
        min_exposures=10,
        smoothing_prior=(1.0, 20.0),
        exploration=0.1,
    )
    assert allocations == {}
