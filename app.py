"""TransactionGuard AI - Antarmuka Streamlit (Sesi 1).

Aplikasi untuk:
- Memeriksa kualitas data transaksi (missing, duplikat, outlier).
- Melatih & membandingkan model keterlambatan pembayaran.
- Memprediksi transaksi baru.
- Mengekspor hasil prediksi dan laporan evaluasi.

Jalankan:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.anomaly import anomaly_report, detect_iqr_outliers, detect_zscore_outliers  # noqa: E402
from src.evaluate import results_to_dataframe  # noqa: E402
from src.preprocessing import (  # noqa: E402
    add_features,
    clean_data,
    dataset_quality_report,
    get_feature_target,
    load_dataset,
)
from src.train_model import (  # noqa: E402
    get_candidate_models,
    select_best_model,
    split_data,
    train_all_models,
)

DATA_PATH = ROOT / "data" / "transactions.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
RESULTS_DIR = ROOT / "outputs" / "results"

PAYMENT_TERM_DAYS = {
    "Bank Transfer": 30,
    "Credit Card": 21,
    "E-Wallet": 7,
    "Cash on Delivery": 7,
    "Installment": 45,
}

st.set_page_config(page_title="TransactionGuard AI", page_icon="🛡️", layout="wide")


def _load_dataframe(uploaded, use_demo: bool) -> pd.DataFrame | None:
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded)
        st.error("Format file tidak didukung. Gunakan CSV atau Excel.")
        return None
    if use_demo:
        if not DATA_PATH.exists():
            st.error("Dataset demo tidak tersedia. Jalankan: python train.py")
            return None
        return pd.read_csv(DATA_PATH)
    return None


def render_quality(df: pd.DataFrame) -> None:
    st.subheader("Pemeriksaan Kualitas Data")
    quality = dataset_quality_report(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Baris", f"{quality['n_rows']:,}")
    c2.metric("Jumlah Kolom", quality["n_columns"])
    c3.metric("Total Missing", quality["total_missing"])
    c4.metric("Baris Duplikat", quality["duplicated_rows"])

    missing = pd.Series(quality["missing_values"])
    missing = missing[missing > 0]
    if not missing.empty:
        st.markdown("**Missing value per kolom**")
        fig = px.bar(
            x=missing.index,
            y=missing.values,
            labels={"x": "Kolom", "y": "Jumlah Missing"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("Tidak ada missing value pada dataset ini.")

    with st.expander("Lihat statistik deskriptif"):
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


def render_anomaly(df: pd.DataFrame) -> None:
    st.subheader("Deteksi Anomali Nilai")
    fitur = [
        c
        for c in [
            "invoice_amount",
            "discount",
            "customer_transaction_count",
            "customer_avg_payment_delay",
        ]
        if c in df.columns
    ]
    col1, col2 = st.columns(2)
    with col1:
        z_thr = st.slider("Ambang Z-Score", 1.0, 5.0, 3.0, 0.1)
    with col2:
        iqr_mult = st.slider("Pengali IQR", 0.5, 3.0, 1.5, 0.1)

    report = anomaly_report(df, fitur, z_thr, iqr_mult)
    c1, c2 = st.columns(2)
    c1.metric("Outlier Z-Score", report["jumlah_outlier_zscore"])
    c2.metric("Outlier IQR", report["jumlah_outlier_iqr"])

    z_out = detect_zscore_outliers(df, fitur, z_thr)
    iqr_out = detect_iqr_outliers(df, fitur, iqr_mult)
    tab1, tab2 = st.tabs(["Contoh Outlier Z-Score", "Contoh Outlier IQR"])
    with tab1:
        st.dataframe(z_out.head(20), use_container_width=True)
    with tab2:
        st.dataframe(iqr_out.head(20), use_container_width=True)

    if fitur:
        fig = px.box(df, y=fitur[0], points="outliers", title=f"Distribusi {fitur[0]}")
        st.plotly_chart(fig, use_container_width=True)


def train_models(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame, pd.Series]:
    df = clean_data(add_features(df))
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models, results = train_all_models(X_train, y_train, X_test, y_test)
    return models, results, X_test, y_test


def render_training(df: pd.DataFrame) -> None:
    st.subheader("Pelatihan & Perbandingan Model")
    st.caption(
        "Logistic Regression, Decision Tree, dan Random Forest dilatih pada data yang sama "
        "dengan stratifikasi. Model terbaik dipilih berdasarkan ROC-AUC."
    )
    if "tg_models" not in st.session_state:
        st.session_state.tg_models = None

    if st.button("Latih & Bandingkan Model", type="primary"):
        with st.spinner("Melatih model... (Random Forest bisa beberapa detik)"):
            models, results, X_test, y_test = train_models(df)
        st.session_state.tg_models = models
        st.session_state.tg_results = results
        st.session_state.tg_test = (X_test, y_test)
        st.success("Pelatihan selesai.")

    if st.session_state.get("tg_models"):
        results = st.session_state.tg_results
        comparison = results_to_dataframe(results)
        st.dataframe(comparison, use_container_width=True)

        fig = go.Figure()
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if metric in comparison.columns:
                fig.add_trace(
                    go.Bar(name=metric, x=comparison["model"], y=comparison[metric])
                )
        fig.update_layout(barmode="group", title="Perbandingan Metrik per Model")
        st.plotly_chart(fig, use_container_width=True)

        best = select_best_model(results, metric="roc_auc")
        st.info(f"Model terbaik berdasarkan ROC-AUC: **{best}**")

        buf = io.StringIO()
        comparison.to_csv(buf, index=False)
        st.download_button(
            "Unduh Laporan Evaluasi (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name="laporan_evaluasi_model.csv",
            mime="text/csv",
        )


def render_prediction() -> None:
    st.subheader("Prediksi Transaksi Baru")
    if not MODEL_PATH.exists():
        st.warning(
            "Model belum tersedia. Latih model terlebih dahulu pada tab **Pelatihan Model**, "
            "atau jalankan `python train.py`."
        )
        return

    import joblib

    model = joblib.load(MODEL_PATH)

    with st.form("prediksi"):
        c1, c2 = st.columns(2)
        with c1:
            customer_type = st.selectbox(
                "Tipe Pelanggan", ["Retail", "Corporate", "SME", "Government"]
            )
            product_category = st.selectbox(
                "Kategori Produk",
                ["Electronics", "Fashion", "Groceries", "Furniture", "Services"],
            )
            payment_method = st.selectbox(
                "Metode Pembayaran",
                ["Bank Transfer", "Credit Card", "E-Wallet", "Cash on Delivery", "Installment"],
            )
            invoice_amount = st.number_input("Nilai Invoice", 0.0, value=1000000.0, step=100000.0)
        with c2:
            discount = st.number_input("Diskon (persen)", 0.0, 100.0, value=0.0, step=1.0)
            previous_late = st.selectbox("Pernah Terlambat Sebelumnya", [0, 1])
            tx_count = st.number_input("Jumlah Transaksi Pelanggan", 0, value=5, step=1)
            avg_delay = st.number_input(
                "Rata-rata Keterlambatan (hari)", 0.0, value=0.0, step=0.5
            )
            submitted = st.form_submit_button("Prediksi", type="primary")

    if submitted:
        record = {
            "customer_type": customer_type,
            "product_category": product_category,
            "payment_method": payment_method,
            "invoice_amount": invoice_amount,
            "discount": discount,
            "previous_late_payment": previous_late,
            "customer_transaction_count": tx_count,
            "customer_avg_payment_delay": avg_delay,
            "payment_term_days": PAYMENT_TERM_DAYS.get(payment_method, 30),
            "invoice_month": 1,
            "invoice_weekday": 0,
        }
        frame = pd.DataFrame([record])
        proba = float(model.predict_proba(frame)[0, 1])
        pred = int(proba >= 0.5)
        if pred == 1:
            st.error(f"BERPOTENSI TERLAMBAT - probabilitas {proba:.1%}")
        else:
            st.success(f"TIDAK TERLAMBAT - probabilitas keterlambatan {proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

        output = pd.DataFrame([{**record, "probabilitas_terlambat": proba, "prediksi": pred}])
        st.download_button(
            "Unduh Hasil Prediksi (CSV)",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi.csv",
            mime="text/csv",
        )


def render_trained_summary() -> None:
    path = RESULTS_DIR / "training_summary.json"
    if not path.exists():
        return
    with st.expander("Ringkasan pelatihan terakhir (tersimpan)"):
        with open(path, encoding="utf-8") as fh:
            st.json(json.load(fh))


def main() -> None:
    st.title("🛡️ TransactionGuard AI")
    st.caption(
        "Deteksi Risiko Keterlambatan Pembayaran & Kualitas Data Transaksi - "
        "Sesi 1 (Logistic Regression, Decision Tree, Random Forest)"
    )

    with st.sidebar:
        st.header("Sumber Data")
        uploaded = st.file_uploader("Unggah CSV / Excel", type=["csv", "xlsx", "xls"])
        use_demo = st.checkbox("Gunakan dataset demo bawaan", value=uploaded is None)
        st.divider()
        st.caption("Dataset demo bersifat sintetis untuk keperluan teknis.")

    df = _load_dataframe(uploaded, use_demo)
    if df is None:
        st.info("Unggah dataset atau centang 'Gunakan dataset demo bawaan' untuk mulai.")
        return

    st.markdown("**Pratinjau Data**")
    st.dataframe(df.head(10), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(
        ["Kualitas & Anomali Data", "Pelatihan Model", "Prediksi Baru"]
    )
    with tab1:
        render_quality(df)
        st.divider()
        render_anomaly(df)
    with tab2:
        render_training(df)
        render_trained_summary()
    with tab3:
        render_prediction()


if __name__ == "__main__":
    main()
