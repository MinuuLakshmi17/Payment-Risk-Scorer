import math

import numpy as np

SEED = 42

MERCHANT_CATEGORIES = ["grocery", "food", "fashion", "electronics", "travel", "luxury"]
MERCHANT_PROBS = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]

COUNTRIES = ["US", "GB", "IN", "CA", "BR", "NG", "RU"]
COUNTRY_PROBS = [0.60, 0.15, 0.10, 0.05, 0.04, 0.03, 0.03]
HIGH_RISK_COUNTRIES = {"NG", "RU"}

DEVICE_TYPES = ["mobile", "desktop", "tablet"]
DEVICE_PROBS = [0.50, 0.35, 0.15]

HIGH_RISK_MERCHANTS = {"electronics", "travel", "luxury"}
NIGHT_HOURS = {0, 1, 2, 3, 4}

FEATURE_NAMES = [
    "log_amount",
    "velocity",
    "hour_sin",
    "hour_cos",
    "unusual_country",
    "high_amount",
    "night_hour",
    "merchant_grocery",
    "merchant_food",
    "merchant_fashion",
    "merchant_electronics",
    "merchant_travel",
    "merchant_luxury",
    "device_mobile",
    "device_desktop",
    "device_tablet",
]


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def generate_transactions(n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    amounts = rng.lognormal(mean=math.log(80.0), sigma=1.0, size=n)
    amounts = np.clip(amounts, 1.0, 5000.0)
    amounts = np.round(amounts, 2)
    merchants = rng.choice(MERCHANT_CATEGORIES, size=n, p=MERCHANT_PROBS)
    countries = rng.choice(COUNTRIES, size=n, p=COUNTRY_PROBS)
    hours = rng.integers(0, 24, size=n)
    devices = rng.choice(DEVICE_TYPES, size=n, p=DEVICE_PROBS)
    velocity = 1 + rng.poisson(0.8, size=n)
    velocity = np.clip(velocity, 1, 10)
    noise = rng.normal(0.0, 0.2, size=n)
    out = []
    for i in range(n):
        amount = float(amounts[i])
        merchant = str(merchants[i])
        country = str(countries[i])
        hour = int(hours[i])
        device = str(devices[i])
        vel = int(velocity[i])
        unusual = 1.0 if country in HIGH_RISK_COUNTRIES else 0.0
        night = 1.0 if hour in NIGHT_HOURS else 0.0
        high_risk_merchant = 1.0 if merchant in HIGH_RISK_MERCHANTS else 0.0
        amount_factor = max(-2.0, min(2.0, math.log(amount / 100.0)))
        latent = 0.8 * amount_factor + 0.6 * (vel - 1) + 1.2 * unusual + 0.4 * night + 0.6 * high_risk_merchant + float(noise[i])
        prob = _sigmoid(latent - 2.5)
        fraud = 1 if rng.random() < prob else 0
        compliance = 0
        if amount > 500.0 and vel >= 3:
            compliance = 1
        if unusual == 1.0 and amount > 100.0:
            compliance = 1
        if vel >= 5:
            compliance = 1
        if amount > 1500.0:
            compliance = 1
        out.append(
            {
                "amount": amount,
                "merchant_category": merchant,
                "country": country,
                "hour": hour,
                "device_type": device,
                "velocity": vel,
                "fraud_label": int(fraud),
                "compliance_label": int(compliance),
            }
        )
    return out


def encode_transactions(transactions):
    n = len(transactions)
    x = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    y_fraud = np.zeros((n,), dtype=np.int64)
    y_comp = np.zeros((n,), dtype=np.int64)
    for i, t in enumerate(transactions):
        amount = float(t["amount"])
        vel = float(t["velocity"])
        hour = int(t["hour"])
        country = str(t["country"])
        merchant = str(t["merchant_category"])
        device = str(t["device_type"])
        x[i, 0] = math.log(amount)
        x[i, 1] = vel
        x[i, 2] = math.sin(2.0 * math.pi * hour / 24.0)
        x[i, 3] = math.cos(2.0 * math.pi * hour / 24.0)
        x[i, 4] = 1.0 if country in HIGH_RISK_COUNTRIES else 0.0
        x[i, 5] = 1.0 if amount > 500.0 else 0.0
        x[i, 6] = 1.0 if hour in NIGHT_HOURS else 0.0
        for j, cat in enumerate(MERCHANT_CATEGORIES):
            x[i, 7 + j] = 1.0 if merchant == cat else 0.0
        for j, dev in enumerate(DEVICE_TYPES):
            x[i, 13 + j] = 1.0 if device == dev else 0.0
        y_fraud[i] = int(t.get("fraud_label", 0))
        y_comp[i] = int(t.get("compliance_label", 0))
    return x, y_fraud, y_comp, list(FEATURE_NAMES)


def encode_single(transaction):
    x, _, _, _ = encode_transactions([transaction])
    return x[0]
