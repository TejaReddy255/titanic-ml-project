"""
src/train.py
------------
Train a Titanic survival classifier.

Steps
-----
1. Load / generate data
2. Split into train / test
3. Build an end-to-end sklearn Pipeline
   (ColumnTransformer preprocessing  +  RandomForestClassifier)
4. Hyperparameter search with RandomizedSearchCV
5. Evaluate on the held-out test set
6. Persist the best pipeline with joblib
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# allow running from project root or from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessor import build_preprocessor, EXPECTED_FEATURES
from data.generate_data import get_data

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_pipeline.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TARGET = "Survived"
RANDOM_STATE = 42


# ── helpers ────────────────────────────────────────────────────────────────────
def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = get_data()
    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGET]
    return X, y


def build_pipeline() -> Pipeline:
    """Combine preprocessor + classifier into a single sklearn Pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   RandomForestClassifier(random_state=RANDOM_STATE)),
    ])


def get_param_grid() -> dict:
    """
    Hyperparameter search space for RandomizedSearchCV.
    Prefix 'classifier__' routes parameters to the Pipeline's classifier step.
    """
    return {
        "classifier__n_estimators":      [100, 200, 300, 400],
        "classifier__max_depth":         [None, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf":  [1, 2, 4],
        "classifier__max_features":      ["sqrt", "log2"],
        "classifier__class_weight":      [None, "balanced"],
    }


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return a dict of evaluation metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
        "f1_score":  round(f1_score(y_test, y_pred),        4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),  4),
    }
    return metrics, classification_report(y_test, y_pred)


# ── main ───────────────────────────────────────────────────────────────────────
def train(test_size: float = 0.2, n_iter: int = 30, cv: int = 5) -> dict:
    print("=" * 60)
    print("  TITANIC SURVIVAL PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # 1. Data
    print("\n[1/5] Loading dataset...")
    X, y = load_dataset()
    print(f"      Total samples : {len(X)}")
    print(f"      Survival rate : {y.mean():.2%}")
    print(f"      Missing Age   : {X['Age'].isna().sum()}")
    print(f"      Missing Emb.  : {X['Embarked'].isna().sum()}")

    # 2. Split
    print("\n[2/5] Splitting train / test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train : {len(X_train)}  |  Test : {len(X_test)}")

    # 3. Pipeline
    print("\n[3/5] Building pipeline...")
    pipeline = build_pipeline()

    # 4. Hyperparameter search
    print(f"\n[4/5] Running RandomizedSearchCV  (n_iter={n_iter}, cv={cv})...")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_param_grid(),
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"\n      Best CV ROC-AUC : {search.best_score_:.4f}")
    print(f"      Best params     : {search.best_params_}")

    # 5. Evaluate
    print("\n[5/5] Evaluating on held-out test set...")
    metrics, report = evaluate(best_model, X_test, y_test)

    print("\n  ┌─────────────────────────────┐")
    print(f"  │  Accuracy  : {metrics['accuracy']:.4f}           │")
    print(f"  │  Precision : {metrics['precision']:.4f}           │")
    print(f"  │  Recall    : {metrics['recall']:.4f}           │")
    print(f"  │  F1-Score  : {metrics['f1_score']:.4f}           │")
    print(f"  │  ROC-AUC   : {metrics['roc_auc']:.4f}           │")
    print("  └─────────────────────────────┘")
    print("\nClassification Report:")
    print(report)

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n✅  Model saved  →  {MODEL_PATH}")

    full_metrics = {**metrics, "best_params": search.best_params_, "cv_roc_auc": round(search.best_score_, 4)}
    with open(METRICS_PATH, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"✅  Metrics saved →  {METRICS_PATH}")

    return full_metrics


if __name__ == "__main__":
    train()
