"""Experimental prediction of the body-performance dataset's class label."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd

from iaces.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "fitness_assessment.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "fitness_assessment_metrics.json"
FIELDS = {
    "age": (21, 64),
    "height_cm": (125, 194),
    "weight_kg": (26, 139),
    "body_fat_percent": (3, 60),
    "grip_force_kg": (1, 71),
    "flexibility_cm": (-25, 50),
    "situps": (0, 80),
    "broad_jump_cm": (20, 303),
}
FEATURES = ["gender", *FIELDS]


class AssessmentModelUnavailable(RuntimeError):
    pass


def validate(payload: object) -> tuple[dict | None, list[str]]:
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object."]
    errors = []
    cleaned = {}
    gender = payload.get("gender")
    if isinstance(gender, str) and gender.strip().lower() in ("male", "female"):
        cleaned["gender"] = "M" if gender.strip().lower() == "male" else "F"
    else:
        errors.append("gender must be Male or Female, as recorded in this dataset.")
    for key, (minimum, maximum) in FIELDS.items():
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            errors.append(f"{key} must be a number.")
        elif not minimum <= value <= maximum:
            errors.append(f"{key} must be between {minimum} and {maximum}.")
        elif key in ("age", "situps") and not float(value).is_integer():
            errors.append(f"{key} must be a whole number.")
        else:
            cleaned[key] = float(value)
    return (None, errors) if errors else (cleaned, [])


@lru_cache(maxsize=1)
def load_assessment():
    if not MODEL_PATH.is_file():
        raise AssessmentModelUnavailable("Fitness assessment model is unavailable; train it first.")
    return joblib.load(MODEL_PATH)


def predict_assessment(values: dict) -> dict:
    model = load_assessment()
    frame = pd.DataFrame([values], columns=FEATURES)
    prediction = str(model.predict(frame)[0])
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.is_file() else {}
    return {"category": prediction, "holdout_accuracy": metrics.get("accuracy"),
            "disclaimer": "Experimental dataset category, not a diagnosis, exercise prescription, or forecast. Test protocols and source labels require verification."}
