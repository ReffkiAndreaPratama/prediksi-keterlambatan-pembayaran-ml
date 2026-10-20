# TransactionGuard AI — Deteksi Anomali & Prediksi Keterlambatan Pembayaran (Sesi 1)

## 1. Identitas Proyek

| Item | Keterangan |
|---|---|
| Nama Proyek | TransactionGuard AI |
| Sesi | 1 dari 6 |
| Judul Penelitian | Perbandingan Algoritma Logistic Regression, Decision Tree, dan Random Forest untuk Prediksi Keterlambatan Pembayaran Tagihan |
| Jenis | Klasifikasi biner + pemeriksaan kualitas data |

## 2. Latar Belakang

Keterlambatan pembayaran tagihan berdampak langsung pada arus kas dan risiko piutang.
Dengan memanfaatkan data transaksi historis, potensi keterlambatan dapat diprediksi lebih
awal. Proyek ini juga menekankan pentingnya **kualitas data**: missing value, duplikat, dan
nilai tidak wajar (outlier) diperiksa sebelum pelatihan model.

## 3. Tujuan

1. Memeriksa kualitas data transaksi (missing value, duplikat, tipe data).
2. Mendeteksi nilai tidak wajar (anomali) dengan Z-Score dan IQR.
3. Melatih dan membandingkan Logistic Regression, Decision Tree, dan Random Forest.
4. Mengevaluasi model dengan Accuracy, Precision, Recall, F1, dan ROC-AUC.
5. Menyediakan antarmuka web untuk prediksi transaksi baru dan ekspor hasil.

## 4. Teknologi

Python, Pandas, NumPy, scikit-learn, Streamlit, Plotly, Joblib, pytest.

## 5. Sumber Dataset

Dataset yang digunakan **bersifat sintetis** (`scripts/generate_dataset.py`, 3.000 baris),
dibuat secara probabilistik sehingga label benar-benar dapat dipelajari model, bukan acak
murni. Distribusi target tidak seimbang (~27,6% terlambat) menyerupai kasus nyata.

> **Catatan penting:** hasil pada dataset sintetis ini adalah **demo teknis**, bukan hasil
> penelitian pada data nyata. Ganti dengan dataset transaksi nyata berlabel valid untuk
> penggunaan sesungguhnya.

Kolom utama: `invoice_date`, `due_date`, `customer_type`, `product_category`,
`payment_method`, `invoice_amount`, `discount`, `previous_late_payment`,
`customer_transaction_count`, `customer_avg_payment_delay`, target `late_payment`.

## 6. Struktur Folder

```
Sesi-1-Transaction-Anomaly-Detection/
├── app.py                     # Antarmuka Streamlit
├── train.py                   # Entry point pelatihan & evaluasi
├── requirements.txt
├── config/config.yaml
├── src/
│   ├── preprocessing.py       # Kualitas data, cleaning, pipeline transformasi
│   ├── anomaly.py             # Deteksi outlier Z-Score & IQR
│   ├── train_model.py         # 3 model klasifikasi & metrik
│   ├── evaluate.py            # Tabel perbandingan, confusion matrix, ROC
│   └── predict.py             # Prediksi transaksi baru
├── scripts/generate_dataset.py
├── tests/                     # Pengujian otomatis (pytest)
├── data/                      # Dataset (transactions.csv)
├── models/                    # best_model.pkl
└── outputs/results/           # Laporan evaluasi & grafik
```

## 7. Instalasi

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 8. Cara Menjalankan

```bash
python train.py               # latih & evaluasi model (menyimpan model + laporan)
streamlit run app.py          # jalankan antarmuka web
```

## 9. Cara Melatih Model

`python train.py` menjalankan alur lengkap: muat data → kualitas & anomali → cleaning →
feature engineering → split → pelatihan 3 model → pemilihan model terbaik (ROC-AUC) →
penyimpanan artefak ke `models/` dan `outputs/results/`.

## 10. Cara Evaluasi

Metrik (Accuracy, Precision, Recall, F1, ROC-AUC) dihitung dari hasil prediksi model pada
data uji. Hasil tersimpan di `outputs/results/model_comparison.csv`,
`confusion_matrix.png`, `roc_curve.png`, dan `training_summary.json`. Tabel perbandingan
juga dapat diunduh dari antarmuka.

## 11. Contoh Hasil

Contoh keluaran pelatihan pada dataset demo (angka aktual dapat berbeda tiap dataset):

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.698 | 0.539 | 0.736 |
| Random Forest | 0.710 | 0.545 | 0.722 |
| Decision Tree | 0.648 | 0.487 | 0.667 |

## 12. Keterbatasan

- Dataset bersifat sintetis; performa pada data nyata dapat berbeda.
- Kelas tidak seimbang sehingga akurasi bukan metrik utama (ROC-AUC lebih relevan).
- Model tidak menyimpulkan penyebab keterlambatan, hanya probabilitas risiko.

## 13. Pengujian

```bash
python -m pytest tests -q
```

## 14. Referensi

- CRISP-DM (Cross Industry Standard Process for Data Mining).
- scikit-learn: https://scikit-learn.org
- Streamlit: https://streamlit.io
