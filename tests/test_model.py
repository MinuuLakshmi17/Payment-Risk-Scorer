import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import encode_transactions, generate_transactions
from src.model import compute_metrics, predict_scores, train_model


def _small_data():
    rows = generate_transactions(n=400, seed=42)
    x, y_fraud, _, names = encode_transactions(rows)
    assert len(names) == 16
    return x, y_fraud


def test_encode_produces_expected_shape():
    x, y, = _small_data()
    assert x.shape == (400, 16)
    assert y.shape == (400,)


def test_model_forward_returns_score():
    x, y = _small_data()
    model, _ = train_model(x[:300], y[:300], x[300:], y[300:], epochs=2, seed=42)
    scores = predict_scores(model, x[300:])
    assert scores.shape == (100,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_model_trains_and_reports_metrics():
    x, y = _small_data()
    model, _ = train_model(x[:300], y[:300], x[300:], y[300:], epochs=5, seed=42)
    scores = predict_scores(model, x[300:])
    m = compute_metrics(y[300:], scores, threshold=0.5)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert m["roc_auc"] > 0.5
    assert m["n"] == 100
