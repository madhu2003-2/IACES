"""Shared application configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def configured_path(variable: str, default: str) -> Path:
    """Resolve a configured path relative to the project root when needed."""
    value = Path(os.getenv(variable, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


MODEL_PATH = configured_path(
    "FITPREDICT_MODEL_PATH", "models/calorie_prediction_pipeline.joblib"
)
DATABASE_ENABLED = os.getenv("DATABASE_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
DATABASE_BACKEND = os.getenv("DATABASE_BACKEND", "mysql").strip().lower()
DATABASE_PATH = configured_path(
    "DATABASE_PATH", "database/fitpredict_history.sqlite3"
)
