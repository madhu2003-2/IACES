"""Train and evaluate an experimental FNDDS food-nutrient regressor."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge


LOGGER = logging.getLogger("fitpredict.nutrition_training")
SEED = 42
TARGET_COLUMNS = ("calories_kcal_per_100g", "protein_g_per_100g")
TABLES = ("food.csv", "food_nutrient.csv", "nutrient.csv")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _find_data_directory(data_dir: Path | None) -> Path:
    if data_dir is not None:
        candidate = data_dir.resolve()
        missing = [name for name in TABLES if not (candidate / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"The dataset folder is missing required tables: {', '.join(missing)}"
            )
        return candidate

    project_root = _project_root()
    for source_root in (project_root / "data" / "raw", project_root / "data" / "processed"):
        if source_root.exists():
            for food_file in source_root.rglob("food.csv"):
                candidate = food_file.parent
                if all((candidate / name).is_file() for name in TABLES):
                    return candidate
    raise FileNotFoundError(
        "No FNDDS folder found. Pass --data-dir with food.csv, food_nutrient.csv, "
        "and nutrient.csv."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training_data(data_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    foods = pd.read_csv(
        data_dir / "food.csv",
        usecols=["fdc_id", "data_type", "description", "food_category_id"],
        dtype={
            "fdc_id": "int64",
            "data_type": "string",
            "description": "string",
            "food_category_id": "string",
        },
    )
    foods = foods.loc[foods["data_type"].eq("survey_fndds_food")].copy()
    foods["description"] = foods["description"].fillna("").str.strip()
    foods["food_category_id"] = foods["food_category_id"].fillna("unknown").str.strip()
    foods = foods.loc[foods["description"].ne("")]

    nutrient_definitions = pd.read_csv(
        data_dir / "nutrient.csv", usecols=["nutrient_nbr"], dtype={"nutrient_nbr": "string"}
    )
    available = set(nutrient_definitions["nutrient_nbr"].dropna().str.strip())
    if not {"203", "208"}.issubset(available):
        raise ValueError("FNDDS nutrient.csv must define protein (203) and energy kcal (208).")

    nutrients = pd.read_csv(
        data_dir / "food_nutrient.csv",
        usecols=["fdc_id", "nutrient_id", "amount"],
        dtype={"fdc_id": "int64", "nutrient_id": "string"},
    )
    nutrients["amount"] = pd.to_numeric(nutrients["amount"], errors="coerce")
    nutrients["nutrient_id"] = nutrients["nutrient_id"].str.strip()
    nutrients = nutrients.loc[
        nutrients["nutrient_id"].isin({"203", "208"}) & nutrients["amount"].notna()
    ].copy()
    # In this FNDDS export, nutrient_id matches nutrient_nbr, not nutrient.csv.id.
    nutrients["target"] = nutrients["nutrient_id"].map(
        {"208": "calories_kcal_per_100g", "203": "protein_g_per_100g"}
    )
    nutrient_values = nutrients.pivot_table(
        index="fdc_id", columns="target", values="amount", aggfunc="mean"
    ).reset_index()
    for target in TARGET_COLUMNS:
        if target not in nutrient_values.columns:
            nutrient_values[target] = np.nan
    data = foods.merge(nutrient_values, on="fdc_id", how="inner")
    data = data.dropna(subset=list(TARGET_COLUMNS)).reset_index(drop=True)
    data = data.loc[(data[list(TARGET_COLUMNS)] >= 0).all(axis=1)].reset_index(drop=True)
    if len(data) < 50:
        raise ValueError(f"Only {len(data)} usable food rows were found; at least 50 are required.")

    counts = {
        "survey_food_rows": int(len(foods)),
        "rows_with_both_targets": int(len(data)),
        "unique_descriptions": int(data["description"].nunique()),
    }
    return data, counts


def _make_pipeline() -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            (
                "description",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
                "description",
            ),
            ("food_category", OneHotEncoder(handle_unknown="ignore"), ["food_category_id"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", features),
            ("regressor", MultiOutputRegressor(Ridge(alpha=10.0))),
        ]
    )


def _metrics(actual: pd.DataFrame | np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    actual_frame = pd.DataFrame(actual, columns=TARGET_COLUMNS)
    predicted_frame = pd.DataFrame(predicted, columns=TARGET_COLUMNS)
    result: dict[str, dict[str, float]] = {}
    for target in TARGET_COLUMNS:
        result[target] = {
            "mae": float(mean_absolute_error(actual_frame[target], predicted_frame[target])),
            "rmse": float(mean_squared_error(actual_frame[target], predicted_frame[target]) ** 0.5),
            "r2": float(r2_score(actual_frame[target], predicted_frame[target])),
        }
    return result


def _normalized_description(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def train(data_dir: Path, model_path: Path, reports_dir: Path) -> dict[str, object]:
    data, row_counts = _load_training_data(data_dir)
    groups = data["description"].map(_normalized_description)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_indices, test_indices = next(
        splitter.split(data, groups=groups)
    )
    train_rows = data.iloc[train_indices].reset_index(drop=True)
    test_rows = data.iloc[test_indices].reset_index(drop=True)
    feature_columns = ["description", "food_category_id"]
    X_train = train_rows[feature_columns]
    y_train = train_rows[list(TARGET_COLUMNS)]
    X_test = test_rows[feature_columns]
    y_test = test_rows[list(TARGET_COLUMNS)]

    evaluation_pipeline = _make_pipeline()
    evaluation_pipeline.fit(X_train, y_train)
    predicted = np.maximum(evaluation_pipeline.predict(X_test), 0.0)

    category_medians = train_rows.groupby("food_category_id")[list(TARGET_COLUMNS)].median()
    global_medians = train_rows[list(TARGET_COLUMNS)].median()
    category_lookup = category_medians.to_dict(orient="index")
    global_median_values = global_medians.to_dict()
    baseline_frame = pd.DataFrame(
        [
            category_lookup.get(category, global_median_values)
            for category in test_rows["food_category_id"]
        ],
        index=test_rows.index,
        columns=list(TARGET_COLUMNS),
    )
    baseline_predictions = np.maximum(baseline_frame.to_numpy(dtype=float), 0.0)

    metrics = {
        "model": _metrics(y_test, predicted),
        "category_median_baseline": _metrics(y_test, baseline_predictions),
    }
    predictions = test_rows[["fdc_id", "description", "food_category_id", *TARGET_COLUMNS]].copy()
    for index, target in enumerate(TARGET_COLUMNS):
        predictions[f"predicted_{target}"] = predicted[:, index]
        predictions[f"baseline_{target}"] = baseline_predictions[:, index]

    final_pipeline = _make_pipeline()
    final_pipeline.fit(data[feature_columns], data[list(TARGET_COLUMNS)])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path, compress=3)
    predictions.to_csv(reports_dir / "food_holdout_predictions.csv", index=False)

    source_hashes = {name: _sha256(data_dir / name) for name in TABLES}
    report = {
        "status": "trained and evaluated",
        "dataset": "FoodData Central FNDDS survey foods 2021-2023",
        "dataset_folder": data_dir.name,
        "target_definition": "energy (kcal) and protein (g) per 100 g edible food",
        "features": ["food description (TF-IDF unigrams and bigrams)", "food category ID"],
        "split": "80/20 grouped by normalized exact food description",
        "random_seed": SEED,
        "source_rows": row_counts["survey_food_rows"],
        "usable_rows": row_counts["rows_with_both_targets"],
        "unique_descriptions": row_counts["unique_descriptions"],
        "train_rows": int(len(train_rows)),
        "holdout_rows": int(len(test_rows)),
        "metrics": metrics,
        "data_sha256": source_hashes,
        "model_file": model_path.name,
        "artifact_sha256": _sha256(model_path),
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitation": (
            "Text/category predictions are an experiment. When a food catalog record exists, "
            "use its recorded nutrient values for logging rather than this model's estimates."
        ),
    }
    (reports_dir / "nutrition_model_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    LOGGER.info("Trained nutrition model on %s usable foods", len(data))
    LOGGER.info("Saved model to %s", model_path)
    LOGGER.info("Saved evaluation to %s", reports_dir / "nutrition_model_metrics.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder containing food.csv, food_nutrient.csv, and nutrient.csv; discovered under data/ otherwise.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Output model path; defaults to models/fndds_nutrition_estimator.joblib.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Output report directory; defaults to reports/fndds_nutrition.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_dir = _find_data_directory(args.data_dir)
    project_root = _project_root()
    model_path = args.model or project_root / "models" / "fndds_nutrition_estimator.joblib"
    reports_dir = args.reports_dir or project_root / "reports" / "fndds_nutrition"
    report = train(data_dir, model_path, reports_dir)
    for target in TARGET_COLUMNS:
        model_scores = report["metrics"]["model"][target]
        baseline_scores = report["metrics"]["category_median_baseline"][target]
        print(
            f"{target}: model MAE={model_scores['mae']:.3f}, RMSE={model_scores['rmse']:.3f}, "
            f"R2={model_scores['r2']:.4f}; category median MAE={baseline_scores['mae']:.3f}, "
            f"RMSE={baseline_scores['rmse']:.3f}, R2={baseline_scores['r2']:.4f}"
        )


if __name__ == "__main__":
    main()
