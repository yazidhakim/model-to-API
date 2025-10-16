from __future__ import annotations
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np, joblib, json, os, sys


app = FastAPI(title="Sleep Productivity Model")

# === Load artefak ===
MODEL_PATH   = os.getenv("MODEL_PATH",   "models/rf_model.pkl")
SCALER_PATH  = os.getenv("SCALER_PATH",  "models/scaler.pkl")
RAWFEAT_PATH = os.getenv("RAWFEAT_PATH", "models/raw_feature_order.json")

try:
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    FEATURES: List[str] = json.load(open(RAWFEAT_PATH, "r", encoding="utf-8"))
    assert FEATURES == ["Sleep Duration", "Quality of Sleep", "Stress Level"], \
        f"FEATURES tidak sesuai ekspektasi: {FEATURES}"
except Exception as e:
    raise RuntimeError(f"Gagal load artefak: {e}")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production: batasi ke domain app kamu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Schemas ===
class InputData(BaseModel):
    sleep_duration: float = Field(..., ge=0, description="jam, contoh 7.5")
    sleep_quality:  float = Field(..., ge=0, le=10, description="skala 0..10")
    stress_level:   float = Field(..., ge=0, le=10, description="skala 0..10")

class BatchRequest(BaseModel):
    rows: List[InputData]

class PredictResponse(BaseModel):
    prediction: float

class PredictBatchResponse(BaseModel):
    predictions: List[float]

# === Helpers ===
def row_from_input(d: InputData) -> np.ndarray:
    # urutkan sesuai FEATURES (aman kalau urutan berubah)
    m = {
        "Sleep Duration":  d.sleep_duration,
        "Quality of Sleep":d.sleep_quality,
        "Stress Level":    d.stress_level,
    }
    return np.array([[m[f] for f in FEATURES]], dtype=np.float32)

# === Endpoints ===
@app.get("/")
def root():
    return {
        "app": app.title,
        "version": app.version,
        "features": FEATURES,
        "python": sys.version.split()[0]
    }

@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURES), "features": FEATURES}

@app.post("/predict", response_model=PredictResponse)
def predict(data: InputData):
    try:
        X = row_from_input(data)
        X_scaled = scaler.transform(X)
        y_pred = float(model.predict(X_scaled)[0])
        return PredictResponse(prediction=y_pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict error: {e}")

@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: BatchRequest):
    try:
        if not req.rows:
            return PredictBatchResponse(predictions=[])
        X = np.vstack([row_from_input(r) for r in req.rows])  # (N, 3)
        X_scaled = scaler.transform(X)
        ys = model.predict(X_scaled).astype(np.float64).tolist()
        return PredictBatchResponse(predictions=ys)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict error: {e}")