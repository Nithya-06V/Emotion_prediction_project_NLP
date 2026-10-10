"""Local development configuration. Override values with environment variables."""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "emotion_svm_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "static" / "images" / "confusion_matrix.png"

SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "local-development-secret-change-me")
ADMIN_USERNAME = os.environ.get("EMOTION_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("EMOTION_ADMIN_PASSWORD", "admin123")

DB_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", "root"),
    "database": os.environ.get("MYSQL_DATABASE", "emotion_detection_db"),
}

# This is the conventional mapping for the attached Kaggle emotion dataset.
# Override with EMOTION_LABEL_MAP='{"0":"sadness", ...}' when using another source.
DEFAULT_LABEL_MAP = {
    "0": "Sadness",
    "1": "Joy",
    "2": "Love",
    "3": "Anger",
    "4": "Fear",
    "5": "Surprise",
}
try:
    LABEL_MAP = json.loads(os.environ.get("EMOTION_LABEL_MAP", "{}"))
    if not isinstance(LABEL_MAP, dict):
        raise ValueError("EMOTION_LABEL_MAP must be a JSON object")
except (json.JSONDecodeError, ValueError) as exc:
    raise ValueError(f"Invalid EMOTION_LABEL_MAP: {exc}") from exc
LABEL_MAP = {**DEFAULT_LABEL_MAP, **{str(key): str(value) for key, value in LABEL_MAP.items()}}

DATASET_FILES = {
    "training": "training.csv",
    "validation": "validation.csv",
    "testing": "testing.csv",
}