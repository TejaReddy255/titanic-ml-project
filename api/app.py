# api/app.py
import os, sys, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.predict import predict, load_model
from src.preprocessor import EXPECTED_FEATURES

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predicts whether a Titanic passenger survived using a Random Forest model.",
    version="1.0.0"
)

# ── Request schema (powers the /docs form) ────────────────────────────────────
class PassengerInput(BaseModel):
    Pclass:   int            = Field(..., ge=1, le=3,  example=3,      description="Ticket class: 1=1st, 2=2nd, 3=3rd")
    Sex:      str            = Field(...,               example="male", description="male or female")
    Age:      Optional[float]= Field(None, ge=0, le=120,example=22.0,  description="Age in years (optional)")
    Fare:     float          = Field(..., ge=0,         example=7.25,   description="Ticket fare")
    Embarked: Optional[str] = Field(None,               example="S",   description="Port: C=Cherbourg, Q=Queenstown, S=Southampton")

class PredictionOutput(BaseModel):
    prediction:  int
    probability: float
    label:       str
    input:       dict

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", summary="Health Check")
def health_check():
    try:
        load_model()
        return {"status": "ok", "model_ready": True, "service": "titanic-ml-api"}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet")

@app.get("/info", summary="Model Info")
def model_info():
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    return {
        "model": "RandomForestClassifier",
        "features": EXPECTED_FEATURES,
        "valid_values": {"Pclass": [1,2,3], "Sex": ["male","female"], "Embarked": ["C","Q","S"]},
        "training_metrics": metrics
    }

@app.post("/predict", response_model=PredictionOutput, summary="Predict Survival")
def predict_endpoint(passenger: PassengerInput):
    try:
        result = predict(passenger.model_dump())
        return {**result, "input": passenger.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run train.py first.")
