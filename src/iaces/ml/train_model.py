"""Train calorie-regression models from one or more workout CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

from iaces.config import MODEL_PATH, PROJECT_ROOT
from iaces.ml.data_preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, normalize_workout_frame
from iaces.ml.feature_engineering import (
    NonnegativePredictionLinearRegression,
    make_pipeline,
    make_rate_pipeline,
    workout_inputs,
)


SEED = 42
LOGGER = logging.getLogger("fitpredict.training")
DATASET_COLUMN = "dataset"


def _model_candidates() -> dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=SEED),
        "gradient_boosting": GradientBoostingRegressor(random_state=SEED),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Keep project-relative paths in the manifest without exposing external paths."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _read_sources(data_paths: Sequence[Path]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    source_records: list[dict[str, object]] = []
    source_names: set[str] = set()

    for path in data_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")
        dataset_name = path.stem
        if dataset_name in source_names:
            raise ValueError(
                f"Dataset names must be unique; more than one input is named {dataset_name!r}."
            )
        source_names.add(dataset_name)

        raw = pd.read_csv(path)
        clean, audit = normalize_workout_frame(raw)
        clean[DATASET_COLUMN] = dataset_name
        frames.append(clean)
        source_records.append(
            {
                "dataset": dataset_name,
                "file": _display_path(path),
                "sha256": _sha256(path),
                **audit,
            }
        )

    if not frames:
        raise ValueError("No input CSV files were selected.")
    data = pd.concat(frames, ignore_index=True)
    if len(data) < 10:
        raise ValueError("At least 10 usable rows are required for training.")
    return data, source_records


def _cv_splitter(groups: pd.Series, row_count: int):
    """Use source-held-out folds when possible, otherwise use reproducible row folds."""
    group_count = int(groups.nunique())
    if group_count >= 2:
        return GroupKFold(n_splits=min(5, group_count)), groups
    return KFold(n_splits=min(5, row_count), shuffle=True, random_state=SEED), None


def _random_forest_search(cv) -> GridSearchCV:
    return GridSearchCV(
        estimator=make_pipeline(RandomForestRegressor(random_state=SEED, n_jobs=1)),
        param_grid={
            "model__n_estimators": [150, 300],
            "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 2],
        },
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )


def _append_predictions(
    rows: list[dict[str, object]],
    model_name: str,
    dataset_name: str,
    actual: pd.Series,
    predicted,
) -> None:
    rows.extend(
        {
            "model": model_name,
            DATASET_COLUMN: dataset_name,
            "actual_calories": float(actual_value),
            "predicted_calories": float(predicted_value),
        }
        for actual_value, predicted_value in zip(actual, predicted, strict=True)
    )


def _fit_single_source(
    data: pd.DataFrame,
    prediction_rows: list[dict[str, object]],
    cross_validation_rows: list[dict[str, object]],
) -> tuple[object, dict[str, object], str]:
    """Keep the original random holdout workflow when a single CSV is selected."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    dataset_name = str(data[DATASET_COLUMN].iloc[0])
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, estimator in _model_candidates().items():
        pipeline = make_pipeline(estimator)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=-1,
            error_score="raise",
        )
        pipeline.fit(X_train, y_train)
        _append_predictions(
            prediction_rows, name, dataset_name, y_holdout, pipeline.predict(X_holdout)
        )
        cross_validation_rows.append(
            {
                "model": name,
                "validation_dataset": dataset_name,
                "cv_mae_mean": float(-scores["test_mae"].mean()),
                "cv_mae_std": float(scores["test_mae"].std()),
                "cv_r2_mean": float(scores["test_r2"].mean()),
                "best_parameters": "default estimator settings",
            }
        )

    search = _random_forest_search(cv)
    search.fit(X_train, y_train)
    forest = search.best_estimator_
    _append_predictions(
        prediction_rows,
        "random_forest",
        dataset_name,
        y_holdout,
        forest.predict(X_holdout),
    )
    cross_validation_rows.append(
        {
            "model": "random_forest",
            "validation_dataset": dataset_name,
            "cv_mae_mean": float(-search.best_score_),
            "cv_mae_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "cv_r2_mean": None,
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        }
    )
    forest.fit(X, y)
    return forest, search.best_params_, "random_holdout"


def _select_on_training_sources(
    training: pd.DataFrame,
    evaluation_name: str,
    cv_rows: list[dict[str, object]],
    predictions: list[dict[str, object]],
    held_out: pd.DataFrame | None = None,
):
    """Rank candidates by mean inner-fold MAE before predicting any excluded file."""
    X, y = training[FEATURE_COLUMNS], training[TARGET_COLUMN]
    cv, groups = _cv_splitter(training[DATASET_COLUMN], len(training))
    fit_options = {"groups": groups} if groups is not None else {}
    estimators = _model_candidates()
    estimators["linear_regression"] = NonnegativePredictionLinearRegression()
    candidates = {
        "training_median": make_pipeline(DummyRegressor(strategy="median")),
        "duration_only": make_rate_pipeline(use_heart_rate=False),
        "duration_heart_rate": make_rate_pipeline(use_heart_rate=True),
        **{name: make_pipeline(model) for name, model in estimators.items()},
    }
    # In a tie, the earlier, simpler candidate wins.
    scores = {}
    parameters = {}
    fold_rows = []
    for name, pipeline in candidates.items():
        result = cross_validate(
            pipeline, X, y, cv=cv, **fit_options, error_score="raise",
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        )
        scores[name] = float(-result["test_mae"].mean())
        parameters[name] = {}
        fold_rows.append({
            "model": name, "validation_dataset": evaluation_name,
            "cv_mae_mean": scores[name], "cv_mae_std": float(result["test_mae"].std()),
            "cv_r2_mean": float(result["test_r2"].mean()),
            "best_parameters": "fixed estimator settings",
        })

    search = _random_forest_search(cv)
    search.fit(X, y, **fit_options)
    candidates["random_forest"] = search.best_estimator_
    scores["random_forest"] = float(-search.best_score_)
    parameters["random_forest"] = search.best_params_
    fold_rows.append({
        "model": "random_forest", "validation_dataset": evaluation_name,
        "cv_mae_mean": scores["random_forest"],
        "cv_mae_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "cv_r2_mean": None, "best_parameters": json.dumps(search.best_params_, sort_keys=True),
    })
    winner = min(scores, key=scores.get)
    for name, pipeline in candidates.items():
        if held_out is None and name != winner:
            continue
        if name != "random_forest":  # GridSearchCV already refitted its winner.
            pipeline.fit(X, y)
        if held_out is not None:
            estimated = pipeline.predict(held_out[FEATURE_COLUMNS])
            _append_predictions(predictions, name, evaluation_name, held_out[TARGET_COLUMN], estimated)
            if name == winner:
                _append_predictions(predictions, "source_selected", evaluation_name, held_out[TARGET_COLUMN], estimated)

    for row in fold_rows:
        row["selected_for_prediction"] = row["model"] == winner
    cv_rows.extend(fold_rows)
    LOGGER.info("Selected %s using training folds for %s", winner, evaluation_name)
    return candidates[winner], {"candidate": winner, **parameters[winner]}


def _fit_multiple_sources(data, prediction_rows, cross_validation_rows):
    for held_out_name in sorted(data[DATASET_COLUMN].unique()):
        held_out_mask = data[DATASET_COLUMN] == held_out_name
        _select_on_training_sources(
            data.loc[~held_out_mask], held_out_name, cross_validation_rows,
            held_out=data.loc[held_out_mask], predictions=prediction_rows,
        )
    model, parameters = _select_on_training_sources(data, "all_sources", cross_validation_rows, prediction_rows)
    return model, parameters, "leave_one_dataset_out"


def train(
    data_path: Path | Sequence[Path], model_path: Path, reports_dir: Path
) -> None:
    """Train from one CSV or combine compatible workout CSVs with source-aware evaluation."""
    data_paths = [data_path] if isinstance(data_path, Path) else list(data_path)
    data, data_sources = _read_sources(data_paths)

    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, object]] = []
    cross_validation_rows: list[dict[str, object]] = []

    if len(data_sources) == 1:
        model, best_parameters, evaluation_protocol = _fit_single_source(
            data, prediction_rows, cross_validation_rows
        )
    else:
        model, best_parameters, evaluation_protocol = _fit_multiple_sources(
            data, prediction_rows, cross_validation_rows
        )

    pd.DataFrame(prediction_rows).to_csv(
        reports_dir / "model_holdout_predictions.csv", index=False
    )
    pd.DataFrame(cross_validation_rows).to_csv(
        reports_dir / "model_cross_validation.csv", index=False
    )
    joblib.dump(model, model_path, compress=3)

    combined_digest = hashlib.sha256()
    for source in data_sources:
        combined_digest.update(str(source["file"]).encode("utf-8"))
        combined_digest.update(str(source["sha256"]).encode("ascii"))
    candidate = best_parameters.get("candidate", "random_forest")
    manifest = {
        "status": "trained; holdout evaluation pending",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": data_sources[0]["file"] if len(data_sources) == 1 else None,
        "source_files": [source["file"] for source in data_sources],
        "data_sha256": data_sources[0]["sha256"] if len(data_sources) == 1 else combined_digest.hexdigest(),
        "data_sources": data_sources,
        "provenance_status": "Source ownership and dataset licenses have not been independently verified.",
        "model_file": _display_path(model_path),
        "random_seed": SEED,
        "primary_model": type(model.named_steps["model"]).__name__,
        "serving_candidate": candidate,
        "evaluation_model": "source_selected" if len(data_sources) > 1 else "random_forest",
        "selected_features": workout_inputs(model),
        "target": TARGET_COLUMN,
        "best_parameters": best_parameters,
        "evaluation_protocol": evaluation_protocol,
        "validation_protocol": {
            "evaluation": evaluation_protocol,
            "holdout_fraction": 0.2 if evaluation_protocol == "random_holdout" else None,
            "random_seed": SEED,
            "inner_cross_validation_folds": "up to 5; grouped by dataset when multiple sources are used",
        },
        "source_rows": int(sum(source["source_rows"] for source in data_sources)),
        "usable_rows": int(len(data)),
        "rows_dropped_for_missing_or_invalid_required_values": int(
            sum(source["rows_dropped_for_missing_or_invalid_required_values"] for source in data_sources)
        ),
        "training_dataset_count": len(data_sources),
        "source_columns": {source["dataset"]: source["source_columns"] for source in data_sources},
        "model_features": workout_inputs(model),
        "artifact_sha256": _sha256(model_path),
        "artifact_compression": "joblib zlib level 3",
    }
    if candidate == "random_forest":
        manifest["best_random_forest_parameters"] = {
            key: value for key, value in best_parameters.items() if key != "candidate"
        }
    if not hasattr(model.named_steps["model"], "feature_importances_"):
        sample = data.sample(n=min(1500, len(data)), random_state=SEED)
        importance = permutation_importance(
            model, sample[FEATURE_COLUMNS], sample[TARGET_COLUMN], scoring="neg_mean_absolute_error",
            n_repeats=5, random_state=SEED,
        )
        manifest["permutation_importance_mae_kcal"] = dict(zip(
            FEATURE_COLUMNS, importance.importances_mean.tolist(), strict=True
        ))
    (reports_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved trained pipeline to %s", model_path)
    LOGGER.info(
        "Next run: python -m iaces.ml.evaluate_model --predictions %s",
        reports_dir / "model_holdout_predictions.csv",
    )


def _resolve_data_paths(explicit_paths: Sequence[Path] | None) -> list[Path]:
    if explicit_paths:
        return [Path(path) for path in explicit_paths]
    data_dir = PROJECT_ROOT / "data" / "processed"
    csv_paths = sorted(path for path in data_dir.glob("*.csv") if path.is_file())
    paths: list[Path] = []
    for path in csv_paths:
        try:
            header = pd.read_csv(path, nrows=0)
            normalize_workout_frame(header)
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
            LOGGER.warning("Skipping %s: not a compatible workout-calorie CSV (%s)", path.name, exc)
            continue
        paths.append(path)
    if not paths:
        raise FileNotFoundError(
            f"No compatible workout-calorie CSV files found in {data_dir}. "
            "Pass one or more compatible files with --data."
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more compatible workout CSV files. By default, compatible workout-calorie "
            "files are selected from data/processed and unrelated CSVs are skipped."
        ),
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(_resolve_data_paths(args.data), args.model, args.reports_dir)


if __name__ == "__main__":
    main()
