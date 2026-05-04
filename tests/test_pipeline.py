"""
tests/test_pipeline.py
-----------------------
Unit tests for the Titanic ML system.
Run with:  python -m pytest tests/ -v
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessor import build_preprocessor, EXPECTED_FEATURES
from src.predict import validate_input


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    """Verify the ColumnTransformer handles real-world edge cases."""

    def _make_df(self, **overrides) -> pd.DataFrame:
        base = {"Age": 30.0, "Fare": 15.0, "Pclass": 2, "Sex": "female", "Embarked": "C"}
        base.update(overrides)
        return pd.DataFrame([base], columns=EXPECTED_FEATURES)

    def test_transform_complete_row(self):
        prep = build_preprocessor()
        df = self._make_df()
        prep.fit(df)
        out = prep.transform(df)
        assert out.shape[0] == 1
        assert out.shape[1] > len(EXPECTED_FEATURES)   # OHE expands columns

    def test_missing_age_imputed(self):
        prep = build_preprocessor()
        train_df = pd.DataFrame([
            {"Age": 20.0, "Fare": 10.0, "Pclass": 1, "Sex": "male",   "Embarked": "S"},
            {"Age": 40.0, "Fare": 20.0, "Pclass": 2, "Sex": "female", "Embarked": "C"},
        ])
        prep.fit(train_df)

        df_missing = self._make_df(Age=None)
        out = prep.transform(df_missing)
        assert not np.isnan(out).any(), "Imputer should eliminate all NaNs"

    def test_missing_embarked_imputed(self):
        prep = build_preprocessor()
        train_df = pd.DataFrame([
            {"Age": 25.0, "Fare": 12.0, "Pclass": 3, "Sex": "male",   "Embarked": "S"},
            {"Age": 35.0, "Fare": 8.0,  "Pclass": 3, "Sex": "male",   "Embarked": "S"},
        ])
        prep.fit(train_df)
        df_missing = self._make_df(Embarked=None)
        out = prep.transform(df_missing)
        assert not np.isnan(out).any()

    def test_unseen_category_ignored(self):
        """handle_unknown='ignore' should not raise on novel categories."""
        prep = build_preprocessor()
        train_df = pd.DataFrame([
            {"Age": 25.0, "Fare": 12.0, "Pclass": 1, "Sex": "male",   "Embarked": "S"},
        ])
        prep.fit(train_df)
        df_novel = self._make_df(Embarked="X")   # 'X' was never seen
        out = prep.transform(df_novel)            # should not raise
        assert out is not None


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation:
    VALID = {"Pclass": 3, "Sex": "male", "Age": 22, "Fare": 7.25, "Embarked": "S"}

    def test_valid_input_passes(self):
        result = validate_input(self.VALID)
        assert result["Pclass"] == 3
        assert result["Sex"] == "male"

    def test_null_age_allowed(self):
        data = {**self.VALID, "Age": None}
        result = validate_input(data)
        assert result["Age"] is None

    def test_null_embarked_allowed(self):
        data = {**self.VALID, "Embarked": None}
        result = validate_input(data)
        assert result["Embarked"] is None

    def test_invalid_pclass_raises(self):
        with pytest.raises(ValueError, match="Pclass"):
            validate_input({**self.VALID, "Pclass": 5})

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="Sex"):
            validate_input({**self.VALID, "Sex": "unknown"})

    def test_invalid_embarked_raises(self):
        with pytest.raises(ValueError, match="Embarked"):
            validate_input({**self.VALID, "Embarked": "Z"})

    def test_negative_fare_raises(self):
        with pytest.raises(ValueError, match="Fare"):
            validate_input({**self.VALID, "Fare": -5.0})

    def test_missing_required_field_raises(self):
        data = {k: v for k, v in self.VALID.items() if k != "Pclass"}
        with pytest.raises(ValueError, match="Pclass"):
            validate_input(data)

    def test_wrong_type_coerced(self):
        """String numbers should be coerced successfully."""
        result = validate_input({**self.VALID, "Age": "22", "Fare": "7.25", "Pclass": "3"})
        assert result["Age"] == 22.0
        assert result["Fare"] == 7.25
        assert result["Pclass"] == 3

    def test_age_out_of_range(self):
        with pytest.raises(ValueError, match="Age"):
            validate_input({**self.VALID, "Age": 200})


# ══════════════════════════════════════════════════════════════════════════════
#  API TESTS  (requires a trained model)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """Flask test client — only runs if a model file exists."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "titanic_pipeline.pkl")
    if not os.path.exists(model_path):
        pytest.skip("No trained model found — run train.py first")

    from api.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestAPI:
    VALID = {"Pclass": 3, "Sex": "male", "Age": 22, "Fare": 7.25, "Embarked": "S"}

    def test_health_check(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["model_ready"] is True

    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=self.VALID)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        resp = client.post("/predict", json=self.VALID)
        data = resp.get_json()
        assert "prediction"  in data
        assert "probability" in data
        assert "label"       in data
        assert data["prediction"] in (0, 1)
        assert 0 <= data["probability"] <= 1

    def test_predict_first_class_female_higher_prob(self, client):
        """First-class woman should have higher survival prob than 3rd-class man."""
        resp_fc  = client.post("/predict", json={"Pclass": 1, "Sex": "female", "Age": 30, "Fare": 200, "Embarked": "C"})
        resp_3rd = client.post("/predict", json={"Pclass": 3, "Sex": "male",   "Age": 30, "Fare": 7,   "Embarked": "S"})
        assert resp_fc.get_json()["probability"] > resp_3rd.get_json()["probability"]

    def test_predict_invalid_input_422(self, client):
        resp = client.post("/predict", json={"Pclass": 9, "Sex": "alien", "Fare": 5})
        assert resp.status_code == 422

    def test_predict_non_json_415(self, client):
        resp = client.post("/predict", data="not json", content_type="text/plain")
        assert resp.status_code == 415

    def test_info_endpoint(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "features" in data
        assert "training_metrics" in data

    def test_unknown_endpoint_404(self, client):
        resp = client.get("/unknown")
        assert resp.status_code == 404
