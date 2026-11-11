import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_score_valid_transaction_returns_score():
    payload = {"amount": 120.0, "merchant_category": "grocery", "country": "US", "hour": 14, "device_type": "desktop", "velocity": 1}
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["risk_level"] in ("low", "medium", "high")
    assert len(body["reasons"]) >= 1


def test_score_high_risk_transaction_scores_higher():
    low = {"amount": 20.0, "merchant_category": "grocery", "country": "US", "hour": 14, "device_type": "desktop", "velocity": 1}
    high = {"amount": 2500.0, "merchant_category": "electronics", "country": "NG", "hour": 2, "device_type": "mobile", "velocity": 9}
    a = client.post("/score", json=low).json()
    b = client.post("/score", json=high).json()
    assert b["score"] > a["score"]


def test_score_invalid_amount_returns_error():
    payload = {"amount": -5.0, "merchant_category": "grocery", "country": "US", "hour": 14, "device_type": "desktop", "velocity": 1}
    r = client.post("/score", json=payload)
    assert r.status_code == 422


def test_score_invalid_hour_returns_error():
    payload = {"amount": 50.0, "merchant_category": "grocery", "country": "US", "hour": 25, "device_type": "desktop", "velocity": 1}
    r = client.post("/score", json=payload)
    assert r.status_code == 422


def test_demo_page_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "Payment Risk Scorer" in r.text
