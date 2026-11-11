import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain import reason_codes, risk_level, rule_score, score_transaction


def _tx(**over):
    base = {"amount": 50.0, "merchant_category": "grocery", "country": "US", "hour": 14, "device_type": "desktop", "velocity": 1}
    base.update(over)
    return base


def test_explain_high_amount_reason():
    reasons = reason_codes(_tx(amount=800.0))
    codes = [r["code"] for r in reasons]
    assert "high_amount" in codes


def test_explain_very_high_amount_reason():
    reasons = reason_codes(_tx(amount=2000.0))
    codes = [r["code"] for r in reasons]
    assert "very_high_amount" in codes


def test_explain_unusual_country_reason():
    reasons = reason_codes(_tx(country="NG"))
    codes = [r["code"] for r in reasons]
    assert "unusual_country" in codes


def test_explain_high_velocity_reason():
    reasons = reason_codes(_tx(velocity=8))
    codes = [r["code"] for r in reasons]
    assert "high_velocity" in codes


def test_explain_risk_level_thresholds():
    assert risk_level(0.1) == "low"
    assert risk_level(0.5) == "medium"
    assert risk_level(0.9) == "high"


def test_rule_score_increases_with_risk():
    low = rule_score(_tx())
    high = rule_score(_tx(amount=2000.0, country="NG", velocity=8, hour=2, merchant_category="electronics"))
    assert high > low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_score_transaction_returns_shape():
    out = score_transaction(_tx(amount=800.0, country="NG", velocity=6))
    assert 0.0 <= out["score"] <= 1.0
    assert out["risk_level"] in ("low", "medium", "high")
    assert len(out["reasons"]) >= 1
