from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from src.explain import score_transaction
from src.generator import DEVICE_TYPES, MERCHANT_CATEGORIES
from src.model import load_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.pt"

app = FastAPI(title="Payment Risk Scorer")

MODEL = None
SCALER_MEAN = None
SCALER_STD = None


def load_saved_model():
    global MODEL, SCALER_MEAN, SCALER_STD
    if MODEL_PATH.exists():
        try:
            model, payload = load_model(str(MODEL_PATH))
            MODEL = model
            SCALER_MEAN = np.array(payload["scaler_mean"], dtype=np.float32)
            SCALER_STD = np.array(payload["scaler_std"], dtype=np.float32)
        except Exception:
            MODEL = None
            SCALER_MEAN = None
            SCALER_STD = None


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
        result = score_transaction(tx, model=MODEL, scaler_mean=SCALER_MEAN, scaler_std=SCALER_STD)
        return ScoreResponse(score=result["score"], risk_level=result["risk_level"], reasons=result["reasons"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="scoring failed") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
