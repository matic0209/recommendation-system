"""Lightweight experiment assignment helpers for online recommendation service."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass
class Variant:
    name: str
    allocation: float
    parameters: Dict[str, float]


@dataclass
class Experiment:
    name: str
    salt: str
    variants: List[Variant]
    status: str = "active"  # active/paused

    def assign(self, bucketing_key: str) -> Variant:
        """Return deterministic variant assignment for the provided key."""
        digest = hashlib.sha256(f"{self.salt}:{bucketing_key}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF

        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.allocation
            if bucket <= cumulative:
                return variant
        return self.variants[-1]


def _normalise_variants(raw_variants: List[dict]) -> List[Variant]:
    """Ensure allocations sum to 1.0 and convert to Variant objects."""
    total_allocation = sum(float(item.get("allocation", 0.0)) for item in raw_variants)
    if total_allocation <= 0:
        raise ValueError("Experiment variants must have positive allocation sum.")

    variants: List[Variant] = []
    cumulative = 0.0
    for item in raw_variants:
        allocation = float(item.get("allocation", 0.0)) / total_allocation
        cumulative += allocation
        params = item.get("parameters") or {}
        variants.append(
            Variant(
                name=item.get("name", "control"),
                allocation=allocation,
                parameters={k: float(v) for k, v in params.items()},
            )
        )

    # Adjust final variant to absorb rounding error so sum=1.0
    variants[-1].allocation += max(0.0, 1.0 - cumulative)
    return variants


def load_experiments(config_path: Path) -> Dict[str, Experiment]:
    """Load experiments configuration from YAML/JSON file."""
    if not config_path.exists():
        LOGGER.warning("Experiment config %s not found; experiments disabled.", config_path)
        return {}

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(config_path.read_text())
    else:
        raw = json.loads(config_path.read_text())

    experiments: Dict[str, Experiment] = {}
    for name, spec in (raw.get("experiments") or {}).items():
        try:
            experiment = Experiment(
                name=name,
                salt=str(spec.get("salt", name)),
                variants=_normalise_variants(spec.get("variants", [])),
                status=spec.get("status", "active"),
            )
            experiments[name] = experiment
            LOGGER.info("Loaded experiment '%s' with %d variants", name, len(experiment.variants))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load experiment '%s': %s", name, exc)
    return experiments


def assign_variant(
    experiments: Dict[str, Experiment],
    experiment_name: str,
    *,
    user_id: Optional[int],
    request_id: str,
) -> Tuple[str, Dict[str, float]]:
    """Assign experiment variant using deterministic hashing."""
    if not experiments or experiment_name not in experiments or user_id is None:
        return "control", {}

    experiment = experiments[experiment_name]
    if experiment.status != "active":
        return "control", {}

    bucketing_key = f"{user_id}"
    variant = experiment.assign(bucketing_key=bucketing_key)

    LOGGER.debug(
        "Experiment assignment: experiment=%s user=%s variant=%s request=%s",
        experiment_name,
        user_id,
        variant.name,
        request_id,
    )

    return variant.name, variant.parameters


__all__ = ["Experiment", "Variant", "load_experiments", "assign_variant"]
