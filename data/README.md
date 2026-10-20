# Data

Folder ini menyimpan dataset transaksi.

- `transactions.csv` — dataset demo sintetis (3.000 baris) yang dihasilkan oleh
  `scripts/generate_dataset.py`.

Untuk data nyata, letakkan file CSV/Excel transaksi berlabel di sini dan sesuaikan kolom
wajib pada `src/preprocessing.py` (lihat `REQUIRED_COLUMNS`).

Jangan mengunggah dataset besar atau data sensitif ke GitHub (lihat `.gitignore` root).
