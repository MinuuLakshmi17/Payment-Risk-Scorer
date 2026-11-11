import json

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0.0)


def load_light_weights(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    weights = {
        "w1": np.array(raw["w1"], dtype=np.float32),
        "b1": np.array(raw["b1"], dtype=np.float32),
        "w2": np.array(raw["w2"], dtype=np.float32),
        "b2": np.array(raw["b2"], dtype=np.float32),
        "w3": np.array(raw["w3"], dtype=np.float32),
        "b3": np.array(raw["b3"], dtype=np.float32),
        "scaler_mean": np.array(raw["scaler_mean"], dtype=np.float32),
        "scaler_std": np.array(raw["scaler_std"], dtype=np.float32),
    }
    return weights


def light_predict(weights, x):
    arr = np.array(x, dtype=np.float32)
    single = False
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        single = True
    h1 = relu(arr @ weights["w1"].T + weights["b1"])
    h2 = relu(h1 @ weights["w2"].T + weights["b2"])
    logits = (h2 @ weights["w3"].T + weights["b3"]).reshape(-1)
    scores = sigmoid(logits)
    if single:
        return float(scores[0])
    return scores
