import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import encode_transactions, generate_transactions
from src.light_model import light_predict, load_light_weights
from src.model import load_model, predict_scores


def test_light_weights_match_torch_scores():
    weights_path = Path(__file__).resolve().parents[1] / "models" / "weights.json"
    model_path = Path(__file__).resolve().parents[1] / "models" / "model.pt"
    assert weights_path.exists()
    assert model_path.exists()
    weights = load_light_weights(str(weights_path))
    rows = generate_transactions(n=200, seed=123)
    x, _, _, _ = encode_transactions(rows)
    xs = x.copy()
    xs[:, 0] = (xs[:, 0] - weights["scaler_mean"][0]) / weights["scaler_std"][0]
    xs[:, 1] = (xs[:, 1] - weights["scaler_mean"][1]) / weights["scaler_std"][1]
    light = light_predict(weights, xs)
    model, _ = load_model(str(model_path))
    torch_scores = predict_scores(model, xs)
    assert np.max(np.abs(light - torch_scores)) < 1e-5


def test_light_predict_single_value_in_range():
    weights_path = Path(__file__).resolve().parents[1] / "models" / "weights.json"
    weights = load_light_weights(str(weights_path))
    rows = generate_transactions(n=10, seed=7)
    x, _, _, _ = encode_transactions(rows)
    xs = x.copy()
    xs[:, 0] = (xs[:, 0] - weights["scaler_mean"][0]) / weights["scaler_std"][0]
    xs[:, 1] = (xs[:, 1] - weights["scaler_mean"][1]) / weights["scaler_std"][1]
    score = light_predict(weights, xs[0])
    assert 0.0 <= score <= 1.0
