"""Utilities for managing recommendation model lifecycle (hot reload, staging)."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.settings import MODEL_REGISTRY_PATH, MODELS_DIR

LOGGER = logging.getLogger(__name__)
STAGING_DIR = MODELS_DIR / "staging"


@dataclass
class ModelVersion:
    run_id: str
    artifact_uri: str


def load_registry() -> dict:
    if not MODEL_REGISTRY_PATH.exists():
        return {"current": None, "history": []}
    try:
        return json.loads(MODEL_REGISTRY_PATH.read_text())
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse model registry at %s", MODEL_REGISTRY_PATH)
        return {"current": None, "history": []}


def get_current_version() -> Optional[ModelVersion]:
    registry = load_registry()
    current = registry.get("current")
    if isinstance(current, dict) and current.get("run_id"):
        return ModelVersion(run_id=current["run_id"], artifact_uri=current.get("artifact_uri", ""))
    return None


def load_run_id_from_dir(base_dir: Path) -> Optional[str]:
    registry_path = base_dir / "model_registry.json"
    if not registry_path.exists():
        return None
    try:
        registry = json.loads(registry_path.read_text())
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse registry in %s", registry_path)
        return None
    current = registry.get("current")
    if isinstance(current, dict):
        return current.get("run_id")
    return None


def stage_new_model(source_dir: Path) -> Path:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    target = STAGING_DIR / source_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_dir, target)
    LOGGER.info("Staged model artifacts from %s to %s", source_dir, target)
    return target


def deploy_from_source(source_dir: Path) -> None:
    source_dir = source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        target = MODELS_DIR / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            if target.exists():
                target.unlink()
            shutil.copy2(item, target)
    LOGGER.info("Deployed model artifacts from %s to %s", source_dir, MODELS_DIR)


__all__ = [
    "ModelVersion",
    "load_registry",
    "get_current_version",
    "load_run_id_from_dir",
    "stage_new_model",
    "deploy_from_source",
]
