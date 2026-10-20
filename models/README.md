# Models

Folder ini menyimpan artefak model hasil pelatihan.

- `best_model.pkl` — pipeline (preprocessing + model terbaik) yang dihasilkan `train.py`.

Model dimuat kembali oleh `src/predict.py` dan `app.py`. File `.pkl` besar tidak
disarankan diunggah ke GitHub; latih ulang dengan `python train.py` bila diperlukan.
