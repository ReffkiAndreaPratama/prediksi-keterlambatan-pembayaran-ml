"""Pengujian modul prediksi (memerlukan model hasil training)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "best_model.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Model belum dilatih (jalankan python train.py)"
)


def test_predictor_returns_valid_output() -> None:
    from src.predict import LatePaymentPredictor

    predictor = LatePaymentPredictor(MODEL_PATH)
    result = predictor.predict(
        {
            "customer_type": "SME",
            "product_category": "Electronics",
            "payment_method": "Bank Transfer",
            "invoice_amount": 1_500_000,
            "discount": 5,
            "previous_late_payment": 1,
            "customer_transaction_count": 10,
            "customer_avg_payment_delay": 8.0,
        }
    )
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["label"], str)


def test_predictor_requires_fields() -> None:
    from src.predict import LatePaymentPredictor

    predictor = LatePaymentPredictor(MODEL_PATH)
    with pytest.raises(ValueError):
        predictor.predict({"customer_type": "SME"})
