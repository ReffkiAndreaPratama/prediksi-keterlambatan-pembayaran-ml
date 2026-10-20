"""Modul preprocessing dan pemeriksaan kualitas data.

Prinsip utama:
1. Tidak menggunakan fitur yang menyebabkan data leakage.
2. Semua langkah transformasi dibungkus scikit-learn Pipeline agar
   diterapkan konsisten pada data training dan data baru.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "late_payment"

IDENTIFIER_COLUMNS = ["invoice_id", "customer_id"]
DATE_COLUMNS = ["invoice_date", "due_date"]

LEAKAGE_COLUMNS = [
    "payment_date",
    "actual_payment_date",
    "final_payment_status",
    "overdue_days",
    "days_late",
    "settlement_date",
]

REQUIRED_COLUMNS = [
    "invoice_id",
    "customer_id",
    "customer_type",
    "product_category",
    "invoice_amount",
    "payment_method",
    "invoice_date",
    "due_date",
    "previous_late_payment",
    "customer_transaction_count",
    "customer_avg_payment_delay",
    "discount",
    TARGET_COLUMN,
]

NUMERIC_FEATURES = [
    "invoice_amount",
    "discount",
    "previous_late_payment",
    "customer_transaction_count",
    "customer_avg_payment_delay",
    "payment_term_days",
    "invoice_month",
    "invoice_weekday",
]

CATEGORICAL_FEATURES = [
    "customer_type",
    "product_category",
    "payment_method",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Membaca dataset dari CSV dan melakukan validasi kolom wajib."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di '{path}'. Jalankan dulu: python generate_dataset.py"
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset '{path}' kosong (0 baris).")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada dataset: {missing}")
    return df


def dataset_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Menghasilkan ringkasan pemeriksaan kualitas data."""
    report: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_values": df.isna().sum().to_dict(),
        "total_missing": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "descriptive_stats": df.describe(include="all").transpose().to_dict(),
    }
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts().sort_index()
        report["target_distribution"] = {int(k): int(v) for k, v in counts.items()}
        report["target_ratio"] = float(df[TARGET_COLUMN].mean())
    return report


def check_leakage_columns(df: pd.DataFrame) -> list[str]:
    """Mengembalikan daftar kolom leakage yang terdeteksi (harus kosong)."""
    return [c for c in df.columns if c in LEAKAGE_COLUMNS]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering sederhana tanpa data leakage."""
    df = df.copy()

    invoice_date = pd.to_datetime(df["invoice_date"], errors="coerce")
    due_date = pd.to_datetime(df["due_date"], errors="coerce")

    df["payment_term_days"] = (due_date - invoice_date).dt.days
    df["invoice_month"] = invoice_date.dt.month
    df["invoice_weekday"] = invoice_date.dt.weekday

    df["payment_term_days"] = df["payment_term_days"].clip(lower=0)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data: hapus duplikat, buang kolom leakage bila ada.

    Catatan: missing value tidak di-drop di sini, melainkan ditangani
    di dalam pipeline imputer agar tetap aman untuk data baru.
    """
    df = df.copy()

    leakage = check_leakage_columns(df)
    if leakage:
        df = df.drop(columns=leakage)

    df = df.drop_duplicates()

    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Memisahkan fitur (X) dan target (y)."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Fitur tidak lengkap: {missing}. Pastikan add_features() sudah dijalankan.")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Membangun ColumnTransformer untuk fitur numerik dan kategorikal."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def build_pipeline(model: Any) -> Pipeline:
    """Menggabungkan preprocessing dan model menjadi satu Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
