"""
src/preprocessor.py
-------------------
Reusable preprocessing pipeline for the Titanic dataset.

Columns used:
  - Pclass   : ticket class (1st / 2nd / 3rd)
  - Sex      : male / female
  - Age      : passenger age in years  (has missing values)
  - Fare     : ticket fare
  - Embarked : port of embarkation C / Q / S  (has missing values)

Columns dropped (with justification):
  - PassengerId : arbitrary row identifier, no predictive signal
  - Name        : high-cardinality free text; titles could be engineered but
                  kept out for simplicity / generalisability
  - Ticket      : alphanumeric noise with no clear signal
  - Cabin       : ~77 % missing; keeping it would force imputation with mostly
                  noise, hurting generalisation
  - SibSp/Parch : correlated with family size; omitted to keep the API payload
                  simple.  Can be added back trivially.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ── feature lists ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = ["Age", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]

# All features the model expects (order matters for ColumnTransformer)
EXPECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── sub-pipelines ─────────────────────────────────────────────────────────────
def _numeric_pipeline() -> Pipeline:
    """Median imputation → z-score scaling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def _categorical_pipeline() -> Pipeline:
    """
    Most-frequent imputation → one-hot encoding.
    handle_unknown='ignore' makes the pipeline safe on unseen categories
    (e.g. a new embarkation port at inference time).
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


# ── public API ────────────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """
    Return an *unfitted* ColumnTransformer that handles the full
    preprocessing for both training and inference.
    """
    return ColumnTransformer(
        transformers=[
            ("num", _numeric_pipeline(),  NUMERIC_FEATURES),
            ("cat", _categorical_pipeline(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",   # silently drop any extra columns
        verbose_feature_names_out=False,
    )
