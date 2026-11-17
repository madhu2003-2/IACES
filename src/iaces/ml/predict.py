"""Load the trained pipeline and make one calorie estimate."""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from iaces.config import MODEL_PATH
from iaces.ml.data_preprocessing import FEATURE_COLUMNS


class ModelUnavailableError(RuntimeError):
    """Raised when the user has not trained or saved the model yet."""


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None):
    path = MODEL_PATH if model_path is None else MODEL_PATH.__class__(model_path)
    if not path.is_file():
        raise ModelUnavailableError(
            f"No trained model found at {path}. Run the training command in README.md first."
        )
    return joblib.load(path)


def predict_calories(values: dict[str, object]) -> float:
    model = load_model()
    row = pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
    prediction = float(model.predict(row)[0])
    if not pd.notna(prediction):
        raise ValueError("The model returned a non-finite prediction.")
    return max(0.0, prediction)
