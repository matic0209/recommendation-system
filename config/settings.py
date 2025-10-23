"""Global configuration helpers for the recommendation project."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
FEATURE_STORE_PATH: Path = DATA_DIR / "feature_store.db"
MLFLOW_DIR: Path = BASE_DIR / "mlruns"
_env_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if _env_tracking_uri and _env_tracking_uri.startswith("file://"):
    # Normalize file URIs so relative paths under repo don't resolve to root (/mlruns).
    relative_path = _env_tracking_uri[len("file://"):]
    if relative_path.startswith("./") or relative_path.startswith("../") or not relative_path.startswith("/"):
        resolved = (BASE_DIR / relative_path).resolve()
        MLFLOW_TRACKING_URI = f"file://{resolved}"
    else:
        MLFLOW_TRACKING_URI = _env_tracking_uri
else:
    MLFLOW_TRACKING_URI = _env_tracking_uri or f"file://{MLFLOW_DIR}"
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "dataset_recommendation")
MODEL_REGISTRY_PATH: Path = MODELS_DIR / "model_registry.json"

DATA_SOURCE: str = os.getenv("DATA_SOURCE", "json").lower()
DATA_JSON_DIR: Path = Path(os.getenv("DATA_JSON_DIR", DATA_DIR / "dianshu_data"))
BUSINESS_SOURCE_MODE: str = os.getenv("BUSINESS_DATA_SOURCE", "json").lower()
MATOMO_SOURCE_MODE: str = os.getenv("MATOMO_DATA_SOURCE", os.getenv("MATOMO_SOURCE", DATA_SOURCE)).lower()
SOURCE_DATA_MODES: Dict[str, str] = {
    "business": BUSINESS_SOURCE_MODE,
    "matomo": MATOMO_SOURCE_MODE,
}
_dataset_image_root_env = os.getenv("DATASET_IMAGE_ROOT")
if _dataset_image_root_env:
    DATASET_IMAGE_ROOT: Path = Path(_dataset_image_root_env)
else:
    DATASET_IMAGE_ROOT = DATA_JSON_DIR / "images"


@dataclass
class DatabaseConfig:
    """Simple container for database connection settings."""

    name: str
    host: str
    port: int
    user: str
    password: str

    def sqlalchemy_url(self) -> str:
        """Return a SQLAlchemy compatible URL string."""
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    def get_engine_kwargs(self) -> dict:
        """
        Get SQLAlchemy engine keyword arguments for connection pooling.

        Returns:
            Dict with pool_size, max_overflow, pool_recycle, pool_pre_ping, etc.
        """
        return {
            "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
            "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600")),
            "pool_pre_ping": os.getenv("DB_POOL_PRE_PING", "true").lower() == "true",
            "connect_args": {
                "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
            },
        }


def load_database_configs() -> Dict[str, DatabaseConfig]:
    """Load database configurations from environment variables."""
    return {
        "business": DatabaseConfig(
            name=os.getenv("BUSINESS_DB_NAME", "dianshu_backend"),
            host=os.getenv("BUSINESS_DB_HOST", "localhost"),
            port=int(os.getenv("BUSINESS_DB_PORT", "3306")),
            user=os.getenv("BUSINESS_DB_USER", "root"),
            password=os.getenv("BUSINESS_DB_PASSWORD", ""),
        ),
        "matomo": DatabaseConfig(
            name=os.getenv("MATOMO_DB_NAME", "matomo"),
            host=os.getenv("MATOMO_DB_HOST", "localhost"),
            port=int(os.getenv("MATOMO_DB_PORT", "3306")),
            user=os.getenv("MATOMO_DB_USER", "root"),
            password=os.getenv("MATOMO_DB_PASSWORD", ""),
        ),
    }


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "DATA_SOURCE",
    "DATA_JSON_DIR",
    "SOURCE_DATA_MODES",
    "BUSINESS_SOURCE_MODE",
    "MATOMO_SOURCE_MODE",
    "DATASET_IMAGE_ROOT",
    "FEATURE_STORE_PATH",
    "MLFLOW_DIR",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MODEL_REGISTRY_PATH",
    "DatabaseConfig",
    "load_database_configs",
]
