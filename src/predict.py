"""Modul prediksi untuk data transaksi baru.

Menggunakan model terbaik hasil training (models/best_model.pkl).
Tidak ada hasil yang di-hardcode: probabilitas dihitung oleh model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"

PAYMENT_TERM_DAYS = {
    "Bank Transfer": 30,
    "Credit Card": 21,
    "E-Wallet": 7,
    "Cash on Delivery": 7,
    "Installment": 45,
}

INPUT_FIELDS = [
    "customer_type",
    "product_category",
    "payment_method",
    "invoice_amount",
    "discount",
    "previous_late_payment",
    "customer_transaction_count",
    "customer_avg_payment_delay",
]


class LatePaymentPredictor:
    """Wrapper untuk memuat model dan memprediksi transaksi baru."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di '{self.model_path}'. Jalankan dulu: python train.py"
            )
        self.model = joblib.load(self.model_path)

    @staticmethod
    def _build_feature_frame(record: dict) -> pd.DataFrame:
        data = {}
        for field in FEATURE_COLUMNS:
            if field in record and record[field] is not None:
                data[field] = record[field]

        if "payment_term_days" not in data:
            method = record.get("payment_method", "Bank Transfer")
            data["payment_term_days"] = PAYMENT_TERM_DAYS.get(method, 30)
        if "invoice_month" not in data:
            data["invoice_month"] = 1
        if "invoice_weekday" not in data:
            data["invoice_weekday"] = 0

        for field in FEATURE_COLUMNS:
            data.setdefault(field, 0)

        return pd.DataFrame([data])[FEATURE_COLUMNS]

    def predict(self, record: dict) -> dict:
        missing = [f for f in INPUT_FIELDS if f not in record]
        if missing:
            raise ValueError(f"Input prediksi tidak lengkap, kolom hilang: {missing}")

        frame = self._build_feature_frame(record)
        probability = float(self.model.predict_proba(frame)[0, 1])
        prediction = int(probability >= 0.5)

        if prediction == 1:
            label = "BERPOTENSI TERLAMBAT"
            interpretation = (
                "Transaksi ini memiliki risiko keterlambatan pembayaran yang tinggi. "
                "Disarankan melakukan konfirmasi/penagihan lebih awal."
            )
        else:
            label = "TIDAK TERLAMBAT"
            interpretation = (
                "Transaksi ini diprediksi akan dibayar tepat waktu. "
                "Tetap pantau jatuh tempo secara berkala."
            )

        return {
            "prediction": prediction,
            "label": label,
            "probability": probability,
            "interpretation": interpretation,
        }


def predict_transaction(record: dict, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Fungsi bantu untuk memprediksi satu transaksi."""
    predictor = LatePaymentPredictor(model_path)
    return predictor.predict(record)
