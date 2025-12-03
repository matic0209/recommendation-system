from __future__ import annotations

import pandas as pd

from pipeline.aggregate_matomo_events import _compute_slot_metrics, _compute_variant_metrics


def test_slot_metrics_compute_counts():
    exposures = pd.DataFrame(
        [
            {"dataset_id": 1, "position": 1, "request_id": "req_a"},
            {"dataset_id": 1, "position": 2, "request_id": "req_b"},
            {"dataset_id": 2, "position": 1, "request_id": "req_c"},
        ]
    )
    clicks = pd.DataFrame(
        [
            {"dataset_id": 1, "position": 1, "request_id": "req_a"},
            {"dataset_id": 1, "position": 2, "request_id": "req_b"},
        ]
    )
    conversions = pd.DataFrame(
        [
            {"dataset_id": 1, "position": 1, "request_id": "req_a", "revenue": 10.0},
        ]
    )

    metrics = _compute_slot_metrics(exposures, clicks, conversions)
    row = metrics[(metrics["dataset_id"] == 1) & (metrics["position"] == 1)].iloc[0]
    assert row["exposure_count"] == 1
    assert row["click_count"] == 1
    assert row["conversion_count"] == 1
    assert row["conversion_revenue"] == 10.0


def test_variant_metrics_context_defaults():
    exposures = pd.DataFrame(
        [
            {"request_id": "req_a", "dataset_id": 1, "variant": "primary", "endpoint": "detail"},
            {"request_id": "req_b", "dataset_id": 1, "variant": "shadow", "endpoint": "detail"},
        ]
    )
    clicks = pd.DataFrame([{"request_id": "req_a", "dataset_id": 1}])
    conversions = pd.DataFrame(columns=["request_id", "dataset_id"])

    metrics = _compute_variant_metrics(exposures, clicks, conversions)
    primary = metrics[(metrics["dataset_id"] == 1) & (metrics["variant"] == "primary")].iloc[0]
    assert primary["exposure_count"] == 1
    assert primary["click_count"] == 1
    shadow = metrics[(metrics["dataset_id"] == 1) & (metrics["variant"] == "shadow")].iloc[0]
    assert shadow["click_count"] == 0
