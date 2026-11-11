import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from src.explain import score_transaction
from src.generator import DEVICE_TYPES, MERCHANT_CATEGORIES
from src.light_model import load_light_weights

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.pt"
WEIGHTS_PATH = ROOT / "models" / "weights.json"

app = FastAPI(title="Payment Risk Scorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
SCALER_MEAN = None
SCALER_STD = None
LIGHT_WEIGHTS = None


def load_saved_model():
    global MODEL, SCALER_MEAN, SCALER_STD, LIGHT_WEIGHTS
    if MODEL_PATH.exists():
        try:
            from src.model import load_model

            model, payload = load_model(str(MODEL_PATH))
            MODEL = model
            SCALER_MEAN = np.array(payload["scaler_mean"], dtype=np.float32)
            SCALER_STD = np.array(payload["scaler_std"], dtype=np.float32)
            return
        except Exception:
            MODEL = None
            SCALER_MEAN = None
            SCALER_STD = None
    if WEIGHTS_PATH.exists():
        try:
            LIGHT_WEIGHTS = load_light_weights(str(WEIGHTS_PATH))
        except Exception:
            LIGHT_WEIGHTS = None


load_saved_model()


class TransactionInput(BaseModel):
    amount: float = Field(gt=0, le=100000)
    merchant_category: str
    country: str
    hour: int = Field(ge=0, le=23)
    device_type: str
    velocity: int = Field(ge=0, le=100)

    @field_validator("merchant_category")
    @classmethod
    def check_merchant(cls, v):
        if v not in MERCHANT_CATEGORIES:
            raise ValueError("merchant_category must be one of the listed categories")
        return v

    @field_validator("country")
    @classmethod
    def check_country(cls, v):
        if len(v) != 2 or not v.isalpha():
            raise ValueError("country must be a 2 letter code")
        return v.upper()

    @field_validator("device_type")
    @classmethod
    def check_device(cls, v):
        if v not in DEVICE_TYPES:
            raise ValueError("device_type must be one of the listed types")
        return v


class ReasonOut(BaseModel):
    code: str
    text: str


class ScoreResponse(BaseModel):
    score: float
    risk_level: str
    reasons: list[ReasonOut]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def demo():
    return FileResponse(str(ROOT / "static" / "index.html"))


@app.post("/score", response_model=ScoreResponse)
def score(payload: TransactionInput):
    try:
        tx = {
            "amount": float(payload.amount),
            "merchant_category": payload.merchant_category,
            "country": payload.country,
            "hour": int(payload.hour),
            "device_type": payload.device_type,
            "velocity": int(payload.velocity),
        }
        result = score_transaction(tx, model=MODEL, scaler_mean=SCALER_MEAN, scaler_std=SCALER_STD, light_weights=LIGHT_WEIGHTS)
        return ScoreResponse(score=result["score"], risk_level=result["risk_level"], reasons=result["reasons"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="scoring failed") from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
