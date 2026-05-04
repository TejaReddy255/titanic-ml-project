# 🚢 Titanic Survival Prediction — Production ML System

A production-ready Machine Learning pipeline that predicts whether a Titanic passenger
survived, exposed via a REST API and fully containerised with Docker.

---

## 📁 Project Structure

```
titanic-ml-project/
│
├── data/
│   |__ titanic.csv          # Dataset 
│        
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py       # ColumnTransformer preprocessing pipeline
│   ├── train.py              # Model training + hyperparameter tuning
│   └── predict.py            # Prediction module (validation + inference)
│
├── api/
│   ├── __init__.py
│   └── app.py                # Flask REST API (GET /, GET /info, POST /predict)
│
├── model/
│   ├── titanic_pipeline.pkl  # Serialised sklearn Pipeline (created by train.py)
│   └── metrics.json          # Evaluation metrics from last training run
│
├── tests/
│   └── test_pipeline.py      # Unit + integration tests (pytest)
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 ML Architecture

```
Raw JSON Input
     │
     ▼
┌────────────────────────────────────────┐
│         Input Validation               │  ← predict.py::validate_input()
│  (type coercion, range checks, etc.)   │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│         sklearn Pipeline               │
│  ┌──────────────────────────────────┐  │
│  │  ColumnTransformer               │  │
│  │  ├── Numeric (Age, Fare)         │  │
│  │  │   └── MedianImputer → Scaler  │  │
│  │  └── Categorical (Pclass, Sex,   │  │
│  │      Embarked)                   │  │
│  │      └── ModeImputer → OHE       │  │
│  └──────────────────────────────────┘  │
│           ↓                            │
│  RandomForestClassifier                │
│  (tuned via RandomizedSearchCV)        │
└────────────────────┬───────────────────┘
                     │
                     ▼
         { prediction, probability, label }
```

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.11+
- pip
- Docker (for containerised deployment)

### Local Setup

```bash
# 1. Clone or extract the project
cd titanic-ml-project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python src/train.py

# 5. Start the API
python api/app.py
```

The API will be available at `http://localhost:5000`.

---

## 🌐 API Reference

### `GET /`  — Health Check

```bash
curl http://localhost:5000/
```

**Response 200**
```json
{
  "status": "ok",
  "service": "titanic-ml-api",
  "version": "1.0.0",
  "model_ready": true,
  "uptime_since": "2024-01-01T00:00:00Z"
}
```

---

### `GET /info`  — Model Metadata

```bash
curl http://localhost:5000/info
```

**Response 200**
```json
{
  "model": "RandomForestClassifier (sklearn Pipeline)",
  "target": "Survived (0 = Did Not Survive, 1 = Survived)",
  "features": {
    "numeric": ["Age", "Fare"],
    "categorical": ["Pclass", "Sex", "Embarked"],
    "all": ["Age", "Fare", "Pclass", "Sex", "Embarked"]
  },
  "valid_values": {
    "Pclass": [1, 2, 3],
    "Sex": ["male", "female"],
    "Embarked": ["C", "Q", "S"]
  },
  "training_metrics": {
    "accuracy": 0.782,
    "precision": 0.779,
    "recall": 0.688,
    "f1_score": 0.731,
    "roc_auc": 0.817
  }
}
```

---

### `POST /predict`  — Predict Survival

**Request**
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "Pclass": 3,
           "Sex": "male",
           "Age": 22,
           "Fare": 7.25,
           "Embarked": "S"
         }'
```

**Response 200 — Survivor**
```json
{
  "prediction": 1,
  "probability": 0.8499,
  "label": "Survived",
  "input": {
    "Age": 22.0,
    "Fare": 7.25,
    "Pclass": 3,
    "Sex": "male",
    "Embarked": "S"
  }
}
```

**Response 200 — Did Not Survive**
```json
{
  "prediction": 0,
  "probability": 0.236,
  "label": "Did Not Survive",
  "input": { "Pclass": 3, "Sex": "male", "Age": 22, "Fare": 7.25, "Embarked": "S" }
}
```

**Error Responses**

| Code | Reason                        |
|------|-------------------------------|
| 415  | Content-Type is not JSON      |
| 422  | Invalid / out-of-range fields |
| 503  | Model not trained yet         |

**422 example:**
```json
{ "error": "Field 'Pclass' must be one of [1, 2, 3], got 9" }
```

---

### Sample Predictions

| Pclass | Sex    | Age | Fare   | Embarked | Label             | Prob  |
|--------|--------|-----|--------|----------|-------------------|-------|
| 1      | female | 29  | 211.34 | S        | Survived          | 0.85  |
| 3      | male   | 22  | 7.25   | S        | Did Not Survive   | 0.24  |
| 2      | female | 14  | 30.07  | C        | Survived          | 0.80  |

---

## 🐳 Docker Deployment

### Build the image

```bash
docker build -t titanic-ml-api .
```

> The Dockerfile uses a **multi-stage build**: stage 1 installs all dependencies and
> trains the model; stage 2 copies only what's needed into a lean runtime image.

### Run the container

```bash
docker run -d \
  --name titanic-api \
  -p 5000:5000 \
  titanic-ml-api
```

### Test the container

```bash
curl http://localhost:5000/
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"Pclass":1,"Sex":"female","Age":29,"Fare":211.34,"Embarked":"S"}'
```

### Stop / remove

```bash
docker stop titanic-api && docker rm titanic-api
```

---

## 🧪 Running Tests

```bash
# With pytest installed
python -m pytest tests/ -v

# Without pytest (manual runner)
python -c "
import sys; sys.path.insert(0, '.')
from tests.test_pipeline import TestPreprocessor, TestValidation
# ... (see tests/test_pipeline.py)
"
```

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.782  |
| Precision | 0.779  |
| Recall    | 0.688  |
| F1-Score  | 0.731  |
| ROC-AUC   | 0.817  |

Tuned with `RandomizedSearchCV` (30 iterations, 5-fold CV, scoring=`roc_auc`).

---

## 🔍 Feature Engineering Decisions

| Column      | Action   | Justification                                          |
|-------------|----------|--------------------------------------------------------|
| PassengerId | Dropped  | Arbitrary row ID, zero predictive signal               |
| Name        | Dropped  | High-cardinality text; title extraction possible but omitted for robustness |
| Ticket      | Dropped  | Alphanumeric noise, no consistent structure            |
| Cabin       | Dropped  | ~77% missing; imputing would introduce noise           |
| SibSp/Parch | Dropped  | Correlated with family size; omitted to simplify API   |
| Age         | Imputed  | Median imputation (stable, handles outliers)           |
| Embarked    | Imputed  | Mode imputation (S is most common port)                |
| Sex, Pclass | OHE      | Nominal categoricals; no ordinal assumption for Sex    |

---

## ☁️ Cloud Deployment (Bonus)

### Render.com
```bash
# Push image to Docker Hub first
docker tag titanic-ml-api yourdockerhub/titanic-ml-api
docker push yourdockerhub/titanic-ml-api
# Then create a new Web Service on render.com pointing to your image
```

### Railway
```bash
railway login
railway init
railway up
```

### AWS ECS / Fargate
```bash
aws ecr create-repository --repository-name titanic-ml-api
docker tag titanic-ml-api <account>.dkr.ecr.<region>.amazonaws.com/titanic-ml-api
docker push <account>.dkr.ecr.<region>.amazonaws.com/titanic-ml-api
# Create ECS task definition + service via AWS console or CLI
```

---

## 📄 License
MIT
