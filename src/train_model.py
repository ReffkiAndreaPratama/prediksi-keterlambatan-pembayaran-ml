"""Pelatihan dan perbandingan beberapa model klasifikasi.

Model yang dibandingkan:
- Logistic Regression
- Decision Tree
- Random Forest

Data imbalance ditangani dengan class_weight="balanced" (bukan oversampling
sembarangan). Semua metrik dihitung dari hasil training sebenarnya.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_pipeline

RANDOM_STATE = 42


def get_candidate_models() -> dict[str, Any]:
    """Mengembalikan dictionary model yang akan dilatih dan dibandingkan."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def split_data(X, y, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Train-test split dengan stratify agar proporsi kelas tetap seimbang."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Menghitung metrik evaluasi utama."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def train_all_models(X_train, y_train, X_test, y_test) -> tuple[dict, dict]:
    """Melatih semua model dan mengembalikan (pipeline terlatih, metrik)."""
    trained: dict[str, Any] = {}
    results: dict[str, dict[str, float]] = {}

    for name, model in get_candidate_models().items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["cv_roc_auc_mean"] = float(
            np.mean(cross_val_score(pipeline, X_train, y_train, cv=3, scoring="roc_auc"))
        )

        trained[name] = pipeline
        results[name] = metrics

    return trained, results


def select_best_model(results: dict[str, dict[str, float]], metric: str = "roc_auc") -> str:
    """Memilih model terbaik berdasarkan metrik (default ROC-AUC).

    ROC-AUC dipilih karena lebih tahan terhadap class imbalance dibanding accuracy.
    """
    if not results:
        raise ValueError("Tidak ada hasil model untuk dipilih.")
    return max(results, key=lambda name: results[name][metric])
