from __future__ import annotations
from typing import List
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib, json, os, sys

app = FastAPI(title="Sleep Productivity Model", version="1.0.0")

# === Paths aman (relatif ke file ini) ===
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH   = Path(os.getenv("MODEL_PATH")   or BASE_DIR / "models" / "rf_model.pkl")
SCALER_PATH  = Path(os.getenv("SCALER_PATH")  or BASE_DIR / "models" / "scaler.pkl")
RAWFEAT_PATH = Path(os.getenv("RAWFEAT_PATH") or BASE_DIR / "models" / "raw_feature_order.json")

# === Load artefak ===
try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with RAWFEAT_PATH.open("r", encoding="utf-8") as f:
        FEATURES: List[str] = json.load(f)
    assert FEATURES == ["Sleep Duration", "Quality of Sleep", "Stress Level"], \
        f"FEATURES tidak sesuai ekspektasi: {FEATURES}"
except Exception as e:
    raise RuntimeError(
        f"Gagal load artefak: {e} | "
        f"MODEL_PATH={MODEL_PATH}, SCALER_PATH={SCALER_PATH}, RAWFEAT_PATH={RAWFEAT_PATH}"
    )

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    m = {
        "Sleep Duration":   d.sleep_duration,
        "Quality of Sleep": d.sleep_quality,
        "Stress Level":     d.stress_level,
    }
    return np.array([[m[f] for f in FEATURES]], dtype=np.float32)

# === Endpoints ===
@app.get("/")
def root():
    return {
        "app": app.title,
        "version": app.version,
        "features": FEATURES,
        "python": sys.version.split()[0],
        "model_path": str(MODEL_PATH.name),
        "scaler_path": str(SCALER_PATH.name),
    }

@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURES), "features": FEATURES}

@app.post("/predict", response_model=PredictResponse)
def predict(data: InputData):
    try:
        X = row_from_input(data)
        Xs = scaler.transform(X)
        y = float(model.predict(Xs)[0])
        return PredictResponse(prediction=y)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict error: {e}")

@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: BatchRequest):
    try:
        if not req.rows:
            return PredictBatchResponse(predictions=[])
        X = np.vstack([row_from_input(r) for r in req.rows])
        Xs = scaler.transform(X)
        ys = model.predict(Xs).astype(float).tolist()
        return PredictBatchResponse(predictions=ys)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict error: {e}")
