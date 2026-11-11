import numpy as np

try:
    import torch
except Exception:
    torch = None

from src.generator import HIGH_RISK_COUNTRIES, HIGH_RISK_MERCHANTS, NIGHT_HOURS, encode_single
from src.light_model import light_predict

HIGH_AMOUNT = 500.0
VERY_HIGH_AMOUNT = 1000.0
HIGH_VELOCITY = 5


def reason_codes(transaction):
    amount = float(transaction["amount"])
    country = str(transaction["country"])
    velocity = int(transaction["velocity"])
    hour = int(transaction["hour"])
    merchant = str(transaction["merchant_category"])
    reasons = []
    if amount > VERY_HIGH_AMOUNT:
        reasons.append({"code": "very_high_amount", "text": "Amount is very high for this payment"})
    elif amount > HIGH_AMOUNT:
        reasons.append({"code": "high_amount", "text": "Amount is high for this payment"})
    if country in HIGH_RISK_COUNTRIES:
        reasons.append({"code": "unusual_country", "text": "Country is unusual for this account pattern"})
    if velocity >= HIGH_VELOCITY:
        reasons.append({"code": "high_velocity", "text": "Many transactions in a short time"})
    if hour in NIGHT_HOURS:
        reasons.append({"code": "night_hour", "text": "Transaction hour is in the night period"})
    if merchant in HIGH_RISK_MERCHANTS:
        reasons.append({"code": "high_risk_merchant", "text": "Merchant category is often linked to higher risk"})
    if not reasons:
        reasons.append({"code": "no_strong_signal", "text": "No strong risk signal in this transaction"})
    return reasons


def risk_level(score):
    s = float(score)
    if s < 0.30:
        return "low"
    if s < 0.70:
        return "medium"
    return "high"


def rule_score(transaction):
    reasons = reason_codes(transaction)
    codes = {r["code"] for r in reasons}
    score = 0.05
    if "very_high_amount" in codes:
        score += 0.35
    elif "high_amount" in codes:
        score += 0.20
    if "unusual_country" in codes:
        score += 0.25
    if "high_velocity" in codes:
        score += 0.20
    if "night_hour" in codes:
        score += 0.10
    if "high_risk_merchant" in codes:
        score += 0.10
    return float(max(0.0, min(0.95, score)))


def apply_scaler(x, scaler_mean, scaler_std):
    y = np.array(x, dtype=np.float32, copy=True)
    y[0] = (y[0] - float(scaler_mean[0])) / float(scaler_std[0]) if float(scaler_std[0]) != 0 else y[0]
    y[1] = (y[1] - float(scaler_mean[1])) / float(scaler_std[1]) if float(scaler_std[1]) != 0 else y[1]
    return y


def score_transaction(transaction, model=None, scaler_mean=None, scaler_std=None, light_weights=None):
    reasons = reason_codes(transaction)
    score = None
    if model is not None and scaler_mean is not None and scaler_std is not None and torch is not None:
        try:
            raw = encode_single(transaction)
            scaled = apply_scaler(raw, scaler_mean, scaler_std)
            model.eval()
            with torch.no_grad():
                tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
                logits = model(tensor)
                score = float(torch.sigmoid(logits).item())
        except Exception:
            score = None
    if score is None and light_weights is not None:
        try:
            raw = encode_single(transaction)
            scaled = apply_scaler(raw, light_weights["scaler_mean"], light_weights["scaler_std"])
            score = float(light_predict(light_weights, scaled))
        except Exception:
            score = None
    if score is None:
        score = rule_score(transaction)
    score = float(max(0.0, min(1.0, score)))
    return {"score": score, "risk_level": risk_level(score), "reasons": reasons}
