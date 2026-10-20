"""Pemeriksaan kualitas data dan deteksi anomali nilai transaksi.

Modul ini melengkapi fokus utama proyek (prediksi keterlambatan pembayaran)
dengan fitur pendukung pemeriksaan kualitas data:

- Ringkasan kualitas data (missing value, duplikat, tipe data).
- Deteksi nilai tidak wajar menggunakan Z-Score dan IQR.
- Deteksi duplikasi baris.

Semua nilai yang dilaporkan berasal dari data yang benar-benar dimuat,
tidak ada angka fiktif.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
    "invoice_amount",
    "discount",
    "customer_transaction_count",
    "customer_avg_payment_delay",
]


def quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Ringkasan kualitas data: jumlah baris, missing, duplikat, tipe data."""
    missing_series = df.isna().sum()
    missing_detail = {
        str(col): int(count) for col, count in missing_series.items() if count > 0
    }
    return {
        "jumlah_baris": int(df.shape[0]),
        "jumlah_kolom": int(df.shape[1]),
        "total_missing": int(missing_series.sum()),
        "kolom_missing": missing_detail,
        "baris_duplikat": int(df.duplicated().sum()),
        "tipe_data": {str(c): str(t) for c, t in df.dtypes.items()},
    }


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Mengembalikan baris yang terduplikasi (tanpa kemunculan pertama)."""
    if df.empty:
        return df.copy()
    return df[df.duplicated(keep="first")].copy()


def detect_zscore_outliers(
    df: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Mendeteksi nilai tidak wajar dengan Z-Score.

    Nilai dianggap tidak wajar jika |z| > threshold. Hanya kolom numerik
    yang tersedia pada DataFrame yang diperiksa.
    """
    features = features or DEFAULT_FEATURES
    available = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if not available or df.empty:
        return df.iloc[0:0].copy()

    numeric = df[available].astype(float)
    mean = numeric.mean()
    std = numeric.std(ddof=0).replace(0, np.nan)
    z = (numeric - mean) / std
    mask = (z.abs() > threshold).any(axis=1)
    result = df[mask.fillna(False)].copy()
    result["_zscore_outlier"] = True
    return result


def detect_iqr_outliers(
    df: pd.DataFrame,
    features: list[str] | None = None,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Mendeteksi nilai tidak wajar dengan rentang Interquartile (IQR)."""
    features = features or DEFAULT_FEATURES
    available = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if not available or df.empty:
        return df.iloc[0:0].copy()

    mask = pd.Series(False, index=df.index)
    detail = {}
    for col in available:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        col_mask = (df[col] < lower) | (df[col] > upper)
        mask = mask | col_mask
        detail[col] = {"lower_bound": float(lower), "upper_bound": float(upper)}
    result = df[mask].copy()
    result["_iqr_outlier"] = True
    result.attrs["iqr_bounds"] = detail
    return result


def anomaly_report(
    df: pd.DataFrame,
    features: list[str] | None = None,
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> dict[str, Any]:
    """Laporan lengkap anomali: kualitas data + outlier Z-Score & IQR."""
    z_out = detect_zscore_outliers(df, features, zscore_threshold)
    iqr_out = detect_iqr_outliers(df, features, iqr_multiplier)
    return {
        "kualitas": quality_summary(df),
        "jumlah_outlier_zscore": int(z_out.shape[0]),
        "jumlah_outlier_iqr": int(iqr_out.shape[0]),
        "contoh_outlier_zscore": z_out.head(20),
        "contoh_outlier_iqr": iqr_out.head(20),
    }
