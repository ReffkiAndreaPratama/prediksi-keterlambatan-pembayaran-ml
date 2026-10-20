"""Entry point pelatihan dan evaluasi TransactionGuard AI (Sesi 1).

Menjalankan seluruh alur (CRISP-DM ringkas):
1. Muat dataset.
2. Periksa kualitas data dan anomali.
3. Feature engineering + cleaning tanpa data leakage.
4. Latih & bandingkan Logistic Regression, Decision Tree, Random Forest.
5. Simpan model terbaik, tabel perbandingan, confusion matrix, dan ROC curve.

Jalankan dari folder proyek:
    python train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.anomaly import anomaly_report  # noqa: E402
from src.evaluate import (  # noqa: E402
    save_comparison_csv,
    save_confusion_matrix,
    save_roc_curve,
)
from src.preprocessing import (  # noqa: E402
    add_features,
    clean_data,
    dataset_quality_report,
    get_feature_target,
    load_dataset,
)
from src.train_model import (  # noqa: E402
    compute_metrics,
    select_best_model,
    split_data,
    train_all_models,
)

DATA_PATH = ROOT / "data" / "transactions.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "outputs" / "results"


def main() -> int:
    print("=" * 70)
    print("TransactionGuard AI - Pelatihan & Evaluasi (Sesi 1)")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset tidak ditemukan: {DATA_PATH}")
        print("Jalankan dulu: python scripts/generate_dataset.py")
        return 1

    df = load_dataset(DATA_PATH)
    print(f"[1/6] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    quality = dataset_quality_report(df)
    anomali = anomaly_report(df)
    print(
        f"[2/6] Kualitas data: {quality['total_missing']} missing, "
        f"{quality['duplicated_rows']} duplikat, "
        f"{anomali['jumlah_outlier_iqr']} outlier IQR"
    )

    df = clean_data(add_features(df))
    print(f"[3/6] Setelah cleaning & feature engineering: {df.shape[0]} baris")

    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"[4/6] Split data -> train: {X_train.shape[0]}, test: {X_test.shape[0]}")

    models, results = train_all_models(X_train, y_train, X_test, y_test)
    print("[5/6] Perbandingan model:")
    for name, metrics in results.items():
        print(
            f"      - {name}: accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}"
        )

    best_name = select_best_model(results, metric="roc_auc")
    best_model = models[best_name]
    print(f"      Model terbaik (ROC-AUC): {best_name}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    comparison_df = save_comparison_csv(results, RESULTS_DIR / "model_comparison.csv")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    save_confusion_matrix(
        y_test, y_pred, RESULTS_DIR / "confusion_matrix.png", f"Confusion Matrix - {best_name}"
    )
    save_roc_curve(models, X_test, y_test, RESULTS_DIR / "roc_curve.png")

    summary = {
        "dataset": {
            "baris": int(df.shape[0]),
            "kolom": int(df.shape[1]),
            "total_missing": quality["total_missing"],
            "duplikat": quality["duplicated_rows"],
            "outlier_iqr": anomali["jumlah_outlier_iqr"],
        },
        "model_terbaik": best_name,
        "metrik_model_terbaik": compute_metrics(y_test, y_pred, y_proba),
        "perbandingan_semua_model": results,
    }
    with open(RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("[6/6] Artefak tersimpan:")
    print(f"      - model   : {MODELS_DIR / 'best_model.pkl'}")
    print(f"      - hasil   : {RESULTS_DIR}")
    print("\nSelesai. Jalankan aplikasi: streamlit run app.py")
    print("Catatan: dataset bersifat sintetis untuk demo teknis, bukan data penelitian nyata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
