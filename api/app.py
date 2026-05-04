"""
api/app.py
----------
Flask REST API for Titanic survival prediction.

Endpoints
---------
GET  /           Health check
GET  /info       Model metadata + feature info
POST /predict    Return survival prediction for a passenger

Run locally
-----------
    python api/app.py

Or via gunicorn (production):
    gunicorn api.app:app --bind 0.0.0.0:5000 --workers 2
"""

import os
import sys
import json
import logging
from datetime import datetime

from flask import Flask, request, jsonify

# ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.predict import predict, load_model, validate_input
from src.preprocessor import EXPECTED_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── app ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")
START_TIME   = datetime.utcnow().isoformat() + "Z"


def _load_metrics() -> dict:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health_check():
    """
    Health check endpoint.

    Returns
    -------
    200  {"status": "ok", "service": "titanic-ml-api", ...}
    503  if model cannot be loaded
    """
    try:
        load_model()                          # verify model is loadable
        model_ready = True
    except FileNotFoundError:
        model_ready = False

    payload = {
        "status":      "ok" if model_ready else "degraded",
        "service":     "titanic-ml-api",
        "version":     "1.0.0",
        "model_ready": model_ready,
        "uptime_since": START_TIME,
    }
    code = 200 if model_ready else 503
    return jsonify(payload), code


@app.route("/info", methods=["GET"])
def model_info():
    """
    Return model metadata, feature schema, and latest evaluation metrics.
    """
    metrics = _load_metrics()
    payload = {
        "model":   "RandomForestClassifier (sklearn Pipeline)",
        "target":  "Survived (0 = Did Not Survive, 1 = Survived)",
        "features": {
            "numeric":     NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "all":         EXPECTED_FEATURES,
        },
        "valid_values": {
            "Pclass":   [1, 2, 3],
            "Sex":      ["male", "female"],
            "Embarked": ["C", "Q", "S"],
        },
        "training_metrics": metrics,
    }
    return jsonify(payload), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Predict survival for a Titanic passenger.

    Request body (JSON)
    -------------------
    {
        "Pclass":   3,          // int   1 | 2 | 3
        "Sex":      "male",     // str   "male" | "female"
        "Age":      22.0,       // float (optional, null allowed)
        "Fare":     7.25,       // float >= 0
        "Embarked": "S"         // str   "C" | "Q" | "S" (optional, null allowed)
    }

    Response (200)
    --------------
    {
        "prediction":  0,
        "probability": 0.1234,
        "label":       "Did Not Survive",
        "input":       { ...echoed cleaned input... }
    }

    Error (400 / 422 / 500)
    -----------------------
    { "error": "<message>" }
    """
    # ── parse JSON ──
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    logger.info("POST /predict  payload=%s", data)

    # ── validate ──
    try:
        cleaned = validate_input(data)
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return jsonify({"error": str(exc)}), 422

    # ── predict ──
    try:
        result = predict(cleaned)
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        return jsonify({"error": "Model not trained yet. Run train.py first."}), 503
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return jsonify({"error": "Internal prediction error. Check server logs."}), 500

    response = {**result, "input": cleaned}
    logger.info("Prediction: %s  (prob=%.4f)", result["label"], result["probability"])
    return jsonify(response), 200


# ── error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found. Available: GET /, GET /info, POST /predict"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": f"Method not allowed on this endpoint"}), 405


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    logger.info("Starting Titanic ML API on port %d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
