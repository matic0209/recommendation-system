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
    "DatabaseConfig",
    "load_database_configs",
]
