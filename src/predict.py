"""
src/predict.py
--------------
Reusable prediction module.

Accepts input as a plain Python dict (matching the API JSON schema),
validates and coerces fields, and returns:
  - prediction  : 0 (did not survive) or 1 (survived)
  - probability : probability of survival (float 0–1)
  - label       : human-readable string
"""

import os
import sys
import joblib
import pandas as pd
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessor import EXPECTED_FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "titanic_pipeline.pkl")

# ── type coercions ─────────────────────────────────────────────────────────────
FIELD_TYPES: dict[str, type] = {
    "Pclass":   int,
    "Sex":      str,
    "Age":      float,
    "Fare":     float,
    "Embarked": str,
}

VALID_VALUES: dict[str, list] = {
    "Pclass":   [1, 2, 3],
    "Sex":      ["male", "female"],
    "Embarked": ["C", "Q", "S"],
}


# ── singleton model loader ─────────────────────────────────────────────────────
_model_cache: Any = None


def load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python src/train.py` first."
            )
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


# ── validation ─────────────────────────────────────────────────────────────────
def validate_input(data: dict) -> dict:
    """
    Validate and coerce raw input dict.
    Returns cleaned dict or raises ValueError with a descriptive message.
    """
    cleaned = {}
    for field in EXPECTED_FEATURES:
        value = data.get(field)

        # Missing optional fields → leave as None (imputer handles it)
        if value is None or value == "":
            if field in ("Age", "Embarked"):   # known nullable fields
                cleaned[field] = None
                continue
            raise ValueError(f"Missing required field: '{field}'")

        # Type coercion
        try:
            coerced = FIELD_TYPES[field](value)
        except (ValueError, TypeError):
            raise ValueError(
                f"Field '{field}' must be {FIELD_TYPES[field].__name__}, "
                f"got {type(value).__name__} ({value!r})"
            )

        # Categorical value validation
        if field in VALID_VALUES and coerced not in VALID_VALUES[field]:
            raise ValueError(
                f"Field '{field}' must be one of {VALID_VALUES[field]}, "
                f"got {coerced!r}"
            )

        # Range validation
        if field == "Age" and coerced is not None and not (0 <= coerced <= 120):
            raise ValueError(f"'Age' must be between 0 and 120, got {coerced}")
        if field == "Fare" and coerced is not None and coerced < 0:
            raise ValueError(f"'Fare' must be non-negative, got {coerced}")

        cleaned[field] = coerced

    return cleaned


# ── main prediction function ───────────────────────────────────────────────────
def predict(data: dict) -> dict:
    """
    Parameters
    ----------
    data : dict
        Raw JSON-like input, e.g.:
        {"Pclass": 3, "Sex": "male", "Age": 22, "Fare": 7.25, "Embarked": "S"}

    Returns
    -------
    dict with keys:
        prediction  (int)   0 or 1
        probability (float) probability of survival
        label       (str)   "Survived" / "Did Not Survive"
    """
    cleaned = validate_input(data)

    # Build single-row DataFrame in the exact column order the pipeline expects
    df = pd.DataFrame([cleaned], columns=EXPECTED_FEATURES)

    model = load_model()
    prediction  = int(model.predict(df)[0])
    probability = round(float(model.predict_proba(df)[0][1]), 4)

    return {
        "prediction":  prediction,
        "probability": probability,
        "label":       "Survived" if prediction == 1 else "Did Not Survive",
    }


# ── CLI convenience ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_inputs = [
        {"Pclass": 1, "Sex": "female", "Age": 29, "Fare": 211.3375, "Embarked": "S"},
        {"Pclass": 3, "Sex": "male",   "Age": 22, "Fare": 7.25,     "Embarked": "S"},
        {"Pclass": 2, "Sex": "female", "Age": 14, "Fare": 30.07,    "Embarked": "C"},
    ]

    print("\n── Prediction Demo ──────────────────────────────────")
    for inp in sample_inputs:
        result = predict(inp)
        print(f"\nInput   : {inp}")
        print(f"Result  : {result['label']}  "
              f"(prob={result['probability']:.2%}, class={result['prediction']})")
