"""Reusable preprocessing pipeline for training and serving."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from iaces.ml.data_preprocessing import FEATURE_COLUMNS


NUMERIC_COLUMNS = ["age", "weight", "duration", "heart_rate"]


class NonnegativePredictionLinearRegression(LinearRegression):
    """Apply the API's zero floor during validation as well as inference."""

    def predict(self, X):
        return np.maximum(super().predict(X), 0.0)


def _workload_features(frame, *, use_heart_rate):
    terms = frame[["duration"]].copy()
    if use_heart_rate:
        terms["duration_times_heart_rate"] = frame["duration"] * frame["heart_rate"] / 100.0
    return terms


def make_rate_pipeline(*, use_heart_rate: bool) -> Pipeline:
    """Fit total calories as duration times a nonnegative learned rate."""
    return Pipeline([
        ("preprocessor", FunctionTransformer(
            _workload_features, kw_args={"use_heart_rate": use_heart_rate}
        )),
        ("model", LinearRegression(fit_intercept=False, positive=True)),
    ])


def workout_inputs(pipeline: Pipeline) -> list[str]:
    """Describe the raw inputs used by this fitted pipeline, including older artifacts."""
    transform = pipeline.named_steps["preprocessor"]
    if isinstance(transform, FunctionTransformer) and transform.func is _workload_features:
        return ["duration", "heart_rate"] if transform.kw_args["use_heart_rate"] else ["duration"]
    if isinstance(pipeline.named_steps["model"], DummyRegressor):
        return []
    return list(FEATURE_COLUMNS)


def make_preprocessor() -> ColumnTransformer:
    """Encode gender and retain numeric inputs in a stable column order."""
    return ColumnTransformer(
        transformers=[
            ("gender", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["gender"]),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ]
    )
