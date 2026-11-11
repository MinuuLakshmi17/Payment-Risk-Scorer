import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import encode_transactions, generate_transactions


def test_generator_is_deterministic():
    a = generate_transactions(n=50, seed=42)
    b = generate_transactions(n=50, seed=42)
    assert a == b


def test_generator_returns_expected_count():
    rows = generate_transactions(n=100, seed=42)
    assert len(rows) == 100


def test_generator_amount_positive():
    rows = generate_transactions(n=100, seed=7)
    assert all(r["amount"] > 0 for r in rows)


def test_generator_labels_are_binary():
    rows = generate_transactions(n=200, seed=42)
    assert all(r["fraud_label"] in (0, 1) for r in rows)
    assert all(r["compliance_label"] in (0, 1) for r in rows)


def test_generator_velocity_range():
    rows = generate_transactions(n=200, seed=42)
    assert all(0 <= r["velocity"] <= 10 for r in rows)
    assert all(0 <= r["hour"] <= 23 for r in rows)
