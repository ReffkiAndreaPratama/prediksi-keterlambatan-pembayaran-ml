"""Generator dataset sintetis untuk studi kasus prediksi keterlambatan pembayaran.

Dataset dibuat secara sintetis namun realistis: label target diturunkan dari
hubungan probabilistik terhadap beberapa fitur (bukan acak murni), sehingga
model machine learning benar-benar dapat mempelajarinya.

Catatan anti data-leakage:
    Kolom yang hanya diketahui setelah pembayaran terjadi (mis. payment_date,
    actual_payment_date, actual_overdue_days) TIDAK dibuat dan tidak boleh
    digunakan sebagai fitur prediksi.

Jalankan:
    python generate_dataset.py
    python generate_dataset.py --rows 5000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CUSTOMER_TYPES = ["Retail", "Corporate", "SME", "Government"]
PRODUCT_CATEGORIES = ["Electronics", "Fashion", "Groceries", "Furniture", "Services"]
PAYMENT_METHODS = ["Bank Transfer", "Credit Card", "E-Wallet", "Cash on Delivery", "Installment"]

PAYMENT_TERM_DAYS = {
    "Bank Transfer": 30,
    "Credit Card": 30,
    "E-Wallet": 7,
    "Cash on Delivery": 7,
    "Installment": 45,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n_rows: int = 3000, seed: int = 42) -> "pd.DataFrame":
    rng = np.random.default_rng(seed)
    n = int(n_rows)

    customer_type = rng.choice(
        ["Retail", "Corporate", "SME", "Government"],
        size=n,
        p=[0.45, 0.15, 0.30, 0.10],
    )
    product_category = rng.choice(
        ["Electronics", "Fashion", "Groceries", "Furniture", "Services"],
        size=n,
        p=[0.22, 0.26, 0.24, 0.14, 0.14],
    )
    payment_method = rng.choice(
        ["Bank Transfer", "Credit Card", "E-Wallet", "Cash on Delivery", "Installment"],
        size=n,
        p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )

    invoice_amount = np.round(
        np.clip(rng.lognormal(mean=13.5, sigma=0.9, size=n), 50_000, 500_000_000), 2
    )

    discount = np.where(
        rng.random(n) < 0.35, rng.uniform(0.0, 0.30, size=n) * 100, 0.0
    ).round(1)

    previous_late_payment = rng.binomial(1, 0.28, size=n)

    customer_transaction_count = rng.integers(1, 60, size=n)

    customer_avg_payment_delay = np.round(
        np.clip(rng.gamma(shape=2.0, scale=6.0, size=n), 0.0, None), 1
    )

    term_map = {
        "Bank Transfer": 30,
        "Credit Card": 21,
        "E-Wallet": 7,
        "Cash on Delivery": 7,
        "Installment": 45,
    }
    payment_term = np.array([term_map[m] for m in payment_method], dtype=float)

    invoice_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 2, size=n), unit="D"
    )
    due_date = invoice_date + pd.to_timedelta(payment_term.astype(int), unit="D")

    # Hubungan target dengan fitur (probabilistik, bukan deterministik).
    amount_tens_juta = invoice_amount / 1_000_000
    logit = (
        -2.60
        + 1.60 * previous_late_payment
        + 0.035 * np.clip(customer_avg_payment_delay, 0, 60)
        + 0.55 * (payment_method == "Installment")
        + 0.30 * (customer_type == "Government")
        + 0.20 * (payment_method == "Bank Transfer")
        - 0.45 * (product_category == "Groceries")
        - 0.30 * (customer_type == "Retail")
        - 0.004 * discount
        - 0.035 * np.clip(invoice_amount / 1_000_000, 0, None)
        + 0.18 * np.log1p(customer_transaction_count)
        + 0.25 * previous_late_payment * np.clip(invoice_amount / 1_000_000, 0, None)
        - 0.04 * (payment_method == "Credit Card") * 10
    )
    logit = logit + rng.normal(0, 0.9, size=len(logit))
    prob = 1.0 / (1.0 + np.exp(-logit))
    prob = np.clip(prob, 0.01, 0.99)
    late_payment = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "invoice_id": [f"INV-{i:06d}" for i in range(1, n + 1)],
            "invoice_date": invoice_date.strftime("%Y-%m-%d"),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "customer_id": [f"CUST-{i:05d}" for i in rng.integers(1, 800, size=n)],
            "customer_type": customer_type,
            "product_category": product_category,
            "payment_method": payment_method,
            "invoice_amount": np.round(invoice_amount, 2),
            "discount": discount,
            "previous_late_payment": previous_late_payment,
            "customer_transaction_count": customer_transaction_count,
            "customer_avg_payment_delay": customer_avg_payment_delay,
            "late_payment": late_payment,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator dataset sintetis keterlambatan pembayaran.")
    parser.add_argument("--rows", type=int, default=3000, help="Jumlah baris (default 3000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/transactions.csv",
        help="Path output relatif terhadap folder project",
    )
    args = parser.parse_args()

    df = generate_dataset(n_rows=args.rows, seed=args.seed)
    out_path = Path(__file__).resolve().parent / "dataset" / "transactions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rows:
        df.to_csv(out_path, index=False)
    print(f"[OK] Dataset dibuat: {out_path} ({len(df)} baris, {df.shape[1]} kolom)")
    dist = df["late_payment"].value_counts().sort_index().to_dict()
    print(f"     Distribusi target late_payment: {dist}")
    print(f"     Persentase terlambat: {df['late_payment'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
