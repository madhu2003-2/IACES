"""Calculate overall and per-dataset regression metrics from training predictions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from iaces.config import PROJECT_ROOT


def _metric_row(model_name: str, group: pd.DataFrame, dataset: str | None = None) -> dict[str, object]:
    actual = group["actual_calories"]
    predicted = group["predicted_calories"]
    mse = mean_squared_error(actual, predicted)
    row: dict[str, object] = {
        "model": model_name,
        "rows": int(len(group)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mse),
        "rmse": float(mse**0.5),
        "r2": float(r2_score(actual, predicted)),
    }
    if dataset is not None:
        row["dataset"] = dataset
    return row


def evaluate(predictions_path: Path, reports_dir: Path) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    required = {"model", "actual_calories", "predicted_calories"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {', '.join(sorted(missing))}")
    if predictions.empty:
        raise ValueError("Prediction file has no rows.")

    rows = [
        _metric_row(str(model_name), group)
        for model_name, group in predictions.groupby("model", sort=True)
    ]
    dataset_rows: list[dict[str, object]] = []
    if "dataset" in predictions.columns:
        dataset_rows = [
            _metric_row(str(model_name), group, str(dataset_name))
            for (model_name, dataset_name), group in predictions.groupby(
                ["model", "dataset"], sort=True
            )
        ]

    result = pd.DataFrame(rows)
    reports_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(reports_dir / "model_comparison.csv", index=False)
    if dataset_rows:
        pd.DataFrame(dataset_rows).to_csv(
            reports_dir / "model_source_comparison.csv", index=False
        )

    metrics_payload: dict[str, object] = {
        "status": "evaluated",
        "metrics": rows,
        "dataset_metrics": dataset_rows,
    }
    (reports_dir / "model_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    manifest_path = reports_dir / "training_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        manifest["status"] = "trained and evaluated"
        manifest["evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["holdout_metrics"] = rows
        manifest["source_metrics"] = dataset_rows
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path,
        default=PROJECT_ROOT / "reports" / "model_holdout_predictions.csv",
    )
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    result = evaluate(args.predictions, args.reports_dir)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
