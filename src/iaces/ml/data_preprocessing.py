"""Normalize supported workout CSV schemas for model training."""

from __future__ import annotations

import re

import pandas as pd


FEATURE_COLUMNS = ["gender", "age", "weight", "duration", "heart_rate"]
TARGET_COLUMN = "calories"

ALIASES = {
    "gender": ("gender",),
    "age": ("age",),
    "weight": ("weight", "weight_kg", "actual_weight"),
    "duration": ("duration", "duration_min", "session_duration"),
    "duration_hours": ("session_duration_hours",),
    "heart_rate": ("heart_rate", "avg_bpm", "average_heart_rate"),
    "calories": ("calories", "calories_burn", "calories_burned"),
}


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_workout_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return canonical model columns and a small non-modeling data audit."""
    source_by_key = {_key(column): column for column in frame.columns}
    selected: dict[str, object] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in source_by_key:
                selected[canonical] = source_by_key[alias]
                break

    required = [*FEATURE_COLUMNS, TARGET_COLUMN]
    if "duration" not in selected and "duration_hours" in selected:
        selected["duration"] = selected["duration_hours"]
    missing = [column for column in required if column not in selected]
    if missing:
        available = ", ".join(str(column) for column in frame.columns)
        raise ValueError(
            "CSV is missing required workout columns: "
            f"{', '.join(missing)}. Available columns: {available}"
        )

    canonical_frame = pd.DataFrame(index=frame.index)
    for column in FEATURE_COLUMNS:
        canonical_frame[column] = frame[selected[column]]
    canonical_frame[TARGET_COLUMN] = frame[selected[TARGET_COLUMN]]

    if "duration_hours" in selected and selected["duration"] == selected["duration_hours"]:
        canonical_frame["duration"] = pd.to_numeric(
            canonical_frame["duration"], errors="coerce"
        ) * 60.0

    canonical_frame["gender"] = canonical_frame["gender"].astype("string").str.strip().str.lower()
    for column in ["age", "weight", "duration", "heart_rate", TARGET_COLUMN]:
        canonical_frame[column] = pd.to_numeric(canonical_frame[column], errors="coerce")

    before = len(canonical_frame)
    canonical_frame = canonical_frame.dropna(subset=required).copy()
    unexpected_genders = sorted(set(canonical_frame["gender"].dropna()) - {"male", "female"})
    if unexpected_genders:
        raise ValueError(
            "The API currently accepts only Male or Female, but the training CSV contains: "
            + ", ".join(unexpected_genders)
        )
    audit = {
        "source_rows": int(before),
        "usable_rows": int(len(canonical_frame)),
        "rows_dropped_for_missing_or_invalid_required_values": int(before - len(canonical_frame)),
        "source_columns": [str(column) for column in frame.columns],
        "model_features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
    }
    return canonical_frame.reset_index(drop=True), audit
