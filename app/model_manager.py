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

# 延迟导入 Sentry
def _get_sentry_funcs():
    try:
        from app.sentry_config import capture_exception_with_context, add_breadcrumb
        return capture_exception_with_context, add_breadcrumb
    except ImportError:
        return None, None


@dataclass
class ModelVersion:
    run_id: str
    artifact_uri: str


def load_registry() -> dict:
    if not MODEL_REGISTRY_PATH.exists():
        return {"current": None, "history": []}
    try:
        return json.loads(MODEL_REGISTRY_PATH.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse model registry at %s", MODEL_REGISTRY_PATH)

        # Sentry: 记录模型注册表解析失败
        capture_exc, _ = _get_sentry_funcs()
        if capture_exc:
            capture_exc(
                exc,
                level="warning",
                fingerprint=["model", "registry_parse_failed"],
                registry_path=str(MODEL_REGISTRY_PATH),
            )

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

    # Sentry: 添加面包屑
    _, add_bc = _get_sentry_funcs()
    if add_bc:
        add_bc(
            message=f"Starting model deployment from {source_dir}",
            category="model",
            level="info",
            source_dir=str(source_dir),
        )

    if not source_dir.exists():
        error = FileNotFoundError(f"Source directory {source_dir} does not exist")

        # Sentry: 记录模型源目录不存在
        capture_exc, _ = _get_sentry_funcs()
        if capture_exc:
            capture_exc(
                error,
                level="error",
                fingerprint=["model", "source_not_found"],
                source_dir=str(source_dir),
            )

        raise error

    try:
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

        # Sentry: 记录成功部署
        if add_bc:
            add_bc(
                message=f"Model deployment successful",
                category="model",
                level="info",
                source_dir=str(source_dir),
                target_dir=str(MODELS_DIR),
            )

    except Exception as exc:
        # Sentry: 记录模型部署失败
        capture_exc, _ = _get_sentry_funcs()
        if capture_exc:
            capture_exc(
                exc,
                level="error",
                fingerprint=["model", "deployment_failed"],
                source_dir=str(source_dir),
                target_dir=str(MODELS_DIR),
            )
        raise


__all__ = [
    "ModelVersion",
    "load_registry",
    "get_current_version",
    "load_run_id_from_dir",
    "stage_new_model",
    "deploy_from_source",
]
