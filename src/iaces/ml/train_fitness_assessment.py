"""Train a separate, reproducible body-performance class model."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from iaces.ml.fitness_assessment import FEATURES, FIELDS

RENAME = {"body fat_%": "body_fat_percent", "gripForce": "grip_force_kg",
          "sit and bend forward_cm": "flexibility_cm", "sit-ups counts": "situps",
          "broad jump_cm": "broad_jump_cm"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/bodyPerformance.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    raw = pd.read_csv(args.data)
    data = raw.rename(columns=RENAME)[FEATURES + ["class"]].copy()
    for key, (low, high) in FIELDS.items():
        data[key] = pd.to_numeric(data[key], errors="coerce")
        data = data[data[key].between(low, high)]
    data = data[data.gender.isin(["M", "F"]) & data["class"].isin(["A", "B", "C", "D"])].dropna().drop_duplicates()
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["class"])
    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, l2_regularization=1, random_state=42),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    cv_results = {}
    for name, candidate in candidates.items():
        pre = ColumnTransformer([("gender", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["gender"])], remainder="passthrough")
        pipeline = Pipeline([("preprocessor", pre), ("model", candidate)])
        scores = cross_validate(pipeline, train[FEATURES], train["class"], cv=cv,
                                scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"}, n_jobs=1)
        cv_results[name] = {"accuracy": float(scores["test_accuracy"].mean()),
                            "macro_f1": float(scores["test_macro_f1"].mean())}
    selected = max(cv_results, key=lambda name: cv_results[name]["macro_f1"])
    pre = ColumnTransformer([("gender", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["gender"])], remainder="passthrough")
    model = Pipeline([("preprocessor", pre), ("model", candidates[selected])])
    model.fit(train[FEATURES], train["class"])
    predicted = model.predict(test[FEATURES])
    metrics = {"dataset": "bodyPerformance.csv", "source_rows": len(raw), "usable_rows": len(data),
               "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(), "training_rows": len(train),
               "holdout_rows": len(test), "split": "stratified random row holdout, 20%, seed 42 (reused from prior model)",
               "selected_model": selected, "selection": "highest macro F1 in three-fold stratified CV on training data only",
               "cross_validation": cv_results,
               "accuracy": accuracy_score(test["class"], predicted),
               "macro_f1": f1_score(test["class"], predicted, average="macro"),
               "majority_class_baseline_accuracy": float((test["class"] == train["class"].mode()[0]).mean()),
               "classification_report": classification_report(test["class"], predicted, output_dict=True, zero_division=0),
               "confusion_matrix_labels": ["A", "B", "C", "D"],
               "confusion_matrix": confusion_matrix(test["class"], predicted, labels=["A", "B", "C", "D"]).tolist(),
               "features": FEATURES, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
               "limitations": "Previously used random row holdout; further unbiased evaluation needs a fresh independent dataset. Unseen people or other populations are not independently verified. Source, label rules, and physical test protocols require confirmation. This is not a clinical or goal-time model."}
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reports").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "models" / "fitness_assessment.joblib", compress=3)
    (args.output_dir / "reports" / "fitness_assessment_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("source_rows", "usable_rows", "training_rows", "holdout_rows", "selected_model", "cross_validation", "accuracy", "macro_f1", "majority_class_baseline_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
