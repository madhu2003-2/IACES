"""Prediction and model-transparency services used by Flask routes."""

from __future__ import annotations

import json
from collections import defaultdict

from iaces.config import configured_path
from iaces.ml.feature_engineering import workout_inputs
from iaces.ml.predict import load_model, predict_calories


def estimate_calories(values: dict[str, object]) -> float:
    return predict_calories(values)


def calorie_model_inputs() -> list[str]:
    return workout_inputs(load_model())


def describe_model() -> dict[str, object]:
    """Return safe model metadata, evaluation results, and global feature importance."""
    pipeline = load_model()
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    manifest_path = configured_path("FITPREDICT_MANIFEST_PATH", "reports/training_manifest.json")
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    importance_by_input: dict[str, float] = defaultdict(float)
    if hasattr(estimator, "feature_importances_"):
        importance_method = "tree_impurity_fraction"
        for feature, importance in zip(
            preprocessor.get_feature_names_out(), estimator.feature_importances_, strict=True
        ):
            name = "gender" if str(feature).startswith("gender_") else str(feature)
            importance_by_input[name] += float(importance)
    else:
        importance_method = "permutation_training_sample_mae_increase_kcal"
        importance_by_input.update(manifest.get("permutation_importance_mae_kcal", {}))

    evaluation_model = manifest.get("evaluation_model", "random_forest")
    holdout_metrics = None
    for metric in manifest.get("holdout_metrics", []):
        if metric.get("model") == evaluation_model:
            holdout_metrics = {
                "rows": metric.get("rows"),
                "mae_kcal": metric.get("mae"),
                "rmse_kcal": metric.get("rmse"),
                "r2": metric.get("r2"),
            }
            break

    source_metrics = [
        metric
        for metric in manifest.get("source_metrics", [])
        if metric.get("model") == evaluation_model
    ]

    return {
        "model": type(estimator).__name__,
        "serving_candidate": manifest.get("serving_candidate", "random_forest"),
        "evaluation_model": evaluation_model,
        "target": "calories",
        "unit": "kcal",
        "model_version": manifest.get("artifact_sha256"),
        "features": manifest.get(
            "selected_features", ["gender", "age", "weight", "duration", "heart_rate"]
        ),
        "training_rows": manifest.get("usable_rows"),
        "trained_at_utc": manifest.get("trained_at_utc"),
        "best_parameters": manifest.get("best_parameters", manifest.get("best_random_forest_parameters", {})),
        "evaluation_protocol": manifest.get("evaluation_protocol", "random_holdout"),
        "holdout_metrics": holdout_metrics,
        "source_metrics": source_metrics,
        "feature_importance_method": importance_method,
        "feature_importance": [
            {"feature": name, "importance": value}
            for name, value in sorted(
                importance_by_input.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "disclaimer": (
            "Holdout scores describe this dataset split only. Feature importance is a global "
            "model summary, not a causal or per-workout explanation. Estimates are for "
            "demonstration and are not medical measurements."
        ),
    }
