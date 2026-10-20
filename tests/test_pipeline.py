"""Pengujian alur teknis TransactionGuard AI menggunakan data sintetis kecil."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anomaly import (  # noqa: E402
    anomaly_report,
    detect_duplicates,
    detect_iqr_outliers,
    detect_zscore_outliers,
    quality_summary,
)
from src.preprocessing import (  # noqa: E402
    add_features,
    clean_data,
    get_feature_target,
    load_dataset,
)
from src.train_model import (  # noqa: E402
    select_best_model,
    split_data,
    train_all_models,
)


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "invoice_id": [f"INV-{i:05d}" for i in range(n)],
            "invoice_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "due_date": pd.date_range("2024-02-01", periods=n, freq="D").astype(str),
            "customer_id": [f"CUST-{i % 10:03d}" for i in range(n)],
            "customer_type": rng.choice(["Retail", "Corporate", "SME", "Government"], n),
            "product_category": rng.choice(
                ["Electronics", "Fashion", "Groceries", "Furniture", "Services"], n
            ),
            "payment_method": rng.choice(
                ["Bank Transfer", "Credit Card", "E-Wallet"], n
            ),
            "invoice_amount": rng.uniform(100000, 5_000_000, n),
            "discount": rng.uniform(0, 20, n),
            "previous_late_payment": rng.integers(0, 2, n),
            "customer_transaction_count": rng.integers(1, 30, n),
            "customer_avg_payment_delay": rng.uniform(0, 20, n),
            "late_payment": rng.integers(0, 2, n),
        }
    )
    return df


def test_load_dataset_requires_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_dataset(bad)


def test_load_dataset_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset("tidak-ada-file.csv")


def test_quality_summary(synthetic_df: pd.DataFrame) -> None:
    summary = quality_summary(synthetic_df)
    assert summary["jumlah_baris"] == 60
    assert summary["total_missing"] == 0
    assert summary["baris_duplikat"] == 0


def test_detect_duplicates(synthetic_df: pd.DataFrame) -> None:
    dup = pd.concat([synthetic_df, synthetic_df.head(3)], ignore_index=True)
    assert detect_duplicates(dup).shape[0] == 3


def test_detect_outliers_flags_extreme_value(synthetic_df: pd.DataFrame) -> None:
    df = synthetic_df.copy()
    df.loc[0, "invoice_amount"] = 1e12
    z_out = detect_zscore_outliers(df, ["invoice_amount"], threshold=3.0)
    assert not z_out.empty
    iqr_out = detect_iqr_outliers(df, ["invoice_amount"], multiplier=1.5)
    assert not iqr_out.empty


def test_anomaly_report_keys(synthetic_df: pd.DataFrame) -> None:
    report = anomaly_report(synthetic_df)
    assert "kualitas" in report
    assert "jumlah_outlier_iqr" in report


def test_feature_engineering_and_clean(synthetic_df: pd.DataFrame) -> None:
    df = add_features(synthetic_df)
    assert "payment_term_days" in df.columns
    assert "invoice_month" in df.columns
    clean = clean_data(df)
    assert clean["late_payment"].dtype.kind in "iu"


def test_training_pipeline_end_to_end(synthetic_df: pd.DataFrame) -> None:
    df = clean_data(add_features(synthetic_df))
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models, results = train_all_models(X_train, y_train, X_test, y_test)
    assert set(results.keys()) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }
    for metrics in results.values():
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0
    best = select_best_model(results)
    assert best in models
    assert hasattr(models[best], "predict_proba")
