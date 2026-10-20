"""Evaluasi model: metrik, confusion matrix, ROC curve, dan perbandingan model.

Semua grafik disimpan ke folder results/ tanpa menampilkan angka palsu:
 nilai berasal langsung dari hasil prediksi model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def results_to_dataframe(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Mengubah dictionary hasil menjadi DataFrame terurut berdasar ROC-AUC."""
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update({k: metrics.get(k) for k in METRIC_ORDER})
        if "cv_roc_auc_mean" in metrics:
            row["cv_roc_auc_mean"] = metrics["cv_roc_auc_mean"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return df


def save_comparison_csv(results: dict[str, dict[str, float]], path: str | Path) -> pd.DataFrame:
    """Menyimpan tabel perbandingan model ke CSV."""
    df = results_to_dataframe(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def save_confusion_matrix(y_true, y_pred, path: str | Path, title: str) -> None:
    """Menyimpan gambar confusion matrix model terbaik."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Tidak Terlambat", "Terlambat"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc_curve(models: dict, X_test, y_test, path: str | Path) -> None:
    """Menyimpan gambar ROC curve untuk semua model terlatih."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipeline in models.items():
        try:
            RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=name, ax=ax)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Curve Perbandingan Model")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
