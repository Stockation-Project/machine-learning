# 🤖 Stockation — Machine Learning Service

Microservice AI berbasis **FastAPI** yang menyediakan fitur analitik cerdas untuk platform simulasi investasi Stockation. Semua endpoint dikonsumsi oleh Express API Gateway dan tidak diakses langsung oleh Frontend React.

---

## 🗂️ Struktur Folder

```text
machine-learning/
├── main.py               # Entry point FastAPI
├── requirements.txt      # Python dependencies
│
├── config/               # Konfigurasi terpusat
│   └── settings.py       # Semua konstanta: anomali, risiko, clustering, paths
│
├── core/                 # Shared utilities
│   └── data_fetcher.py   # Fetch data OHLCV dari yfinance
│
├── models/               # Class definitions model ML (bukan trained artifacts)
│   ├── autoencoder.py    # CNNAutoencoder
│   ├── clustering.py     # StockClusterModel (KMeans)
│   ├── preprocessor.py   # SequencePreprocessor
│   └── data_updater.py   # IDX80DataUpdater
│
├── services/             # Business logic per fitur AI
│   ├── deteksi_anomali.py      # Deteksi anomali harga & volume saham
│   ├── optimasi_portofolio.py  # Optimasi alokasi portofolio
│   ├── clustering_service.py   # Klasifikasi risiko saham (Low/Medium/High)
│   ├── ai_explanation.py       # Penjelasan istilah via Groq LLM
│   └── volatility/             # Prediksi volatilitas saham
│       ├── config.py
│       ├── features.py
│       ├── loader.py
│       ├── predictor.py
│       └── schemas.py
│
├── artifacts/            # Trained model files (TIDAK ADA kode di sini)
│   ├── clustering/       # autoencoder.keras, encoder.keras, kmeans.pkl, dll
│   └── volatility/       # multi_horizon_model.keras, stock_scalers.pkl, dll
│
├── data/
│   └── idx80_raw.parquet # Dataset historis IDX80
│
├── dashboard/
│   └── dashboard.py      # Streamlit companion app
│
├── notebooks/            # Jupyter notebooks eksperimen & training
├── tests/                # Test files
├── scripts/              # Utility scripts ad-hoc
└── docs/                 # Dokumentasi internal tim Data Science
```

---

## ⚙️ Setup & Instalasi

### 1. Buat Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Environment Variables

Salin `.env.example` menjadi `.env` dan isi nilainya:

```bash
copy .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Menjalankan Service

### FastAPI Server

```bash
uvicorn main:app --reload --port 8000
```

API akan tersedia di `http://127.0.0.1:8000`
Dokumentasi interaktif: `http://127.0.0.1:8000/docs`

### Streamlit Dashboard

```bash
streamlit run dashboard/dashboard.py
```

Dashboard akan tersedia di `http://localhost:8501`

---

## 📡 API Endpoints

Semua endpoint diawali dengan `/api/ai/`.

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/api/ai/anomalies/{ticker}` | Deteksi anomali harga & volume saham |
| `POST` | `/api/ai/optimize` | Hitung bobot optimal portofolio |
| `POST` | `/api/ai/recommend` | Rekomendasikan saham terbaik dari universe |
| `GET` | `/api/ai/clustering` | Klasifikasi risiko semua saham IDX80 |
| `POST` | `/api/ai/explain` | Penjelasan istilah investasi via AI |
| `POST` | `/api/ai/volatility` | Prediksi volatilitas 10D / 1M / 3M |

### Contoh Request

**Deteksi Anomali:**
```bash
GET /api/ai/anomalies/BBCA.JK?period_key=1_tahun
```

**Optimasi Portofolio:**
```bash
POST /api/ai/optimize
{
  "tickers": ["BBCA.JK", "TLKM.JK", "ASII.JK"],
  "risk_profile": "moderat",
  "period_key": "1_tahun"
}
```

**Prediksi Volatilitas:**
```bash
POST /api/ai/volatility
{
  "ticker": "BBCA.JK",
  "sector": "Financials",
  "industry": "Banks"
}
```

**Penjelasan Istilah:**
```bash
POST /api/ai/explain
{
  "term": "Sharpe Ratio",
  "context": "Hasil optimasi portofolio BBCA dan TLKM"
}
```

---

## 🧠 Fitur AI

### 1. Deteksi Anomali Saham
Mendeteksi perilaku tidak wajar pada harga dan volume saham menggunakan:
- **Z-Score rolling** — deviasi dari rata-rata rolling window
- **IQR (Interquartile Range)** — nilai di luar fence 1.5×IQR
- **Gap Detector** — selisih Open hari ini vs Close kemarin

Output berupa list `AnomalyEvent` dengan severity: `low | medium | high | critical`.

### 2. Optimasi Portofolio
Menghitung alokasi bobot optimal berdasarkan profil risiko dengan metode:
- `max_sharpe` — Maksimalkan Sharpe Ratio
- `min_volatility` — Minimasi volatilitas
- `max_return` — Maksimalkan return
- `risk_parity` — Distribusi risiko merata
- `equal_weight` — Bobot sama rata (baseline)

### 3. Clustering Risiko Saham
Mengklasifikasikan saham IDX80 ke dalam tiga kategori risiko:
- **Low** — saham stabil (volatilitas rendah)
- **Medium** — saham moderat
- **High** — saham sangat fluktuatif

Menggunakan **CNN Autoencoder** untuk ekstraksi embeddings + **KMeans** untuk clustering.

### 4. Prediksi Volatilitas
Model deep learning multi-horizon yang memprediksi volatilitas:
- **10 hari ke depan** — jangka sangat pendek
- **1 bulan ke depan** — jangka pendek
- **3 bulan ke depan** — jangka menengah

Output dilengkapi dengan **price range** (batas bawah & atas harga prediksi).

### 5. AI Explanation (Groq LLM)
Penjelasan istilah investasi dalam bahasa Indonesia yang sederhana menggunakan model `llama-3.3-70b-versatile` via Groq API.

---

## 🧪 Testing

```bash
# Jalankan dari root machine-learning/
python tests/test_fastapi.py        # Test endpoint via HTTP
python tests/test_clustering.py     # Test clustering service
python tests/test_ai_explanation.py # Test Groq AI service
```

---

## 📦 Dependencies Utama

| Package | Kegunaan |
|---------|----------|
| `fastapi` | Web framework API |
| `uvicorn` | ASGI server |
| `tensorflow` | Model CNN Autoencoder & LSTM |
| `scikit-learn` | KMeans clustering |
| `yfinance` | Fetch data saham |
| `pandas` / `numpy` | Data processing |
| `scipy` | Optimasi portofolio |
| `groq` | Groq LLM API client |
| `streamlit` | Dashboard visualisasi |
| `pyarrow` | Baca/tulis parquet |

---

## 🔗 Arsitektur Komunikasi

```
React Frontend
     ↕
Express API Gateway  ←→  Redis Cache
     ↕
FastAPI ML Service   ←→  yfinance (Yahoo Finance)
     ↕
Groq LLM API
```

> **Catatan:** Endpoint FastAPI **tidak diakses langsung** dari Frontend. Semua request melewati Express Gateway terlebih dahulu.

---

## 📁 Artifacts & Data

File model terlatih disimpan di folder `artifacts/` dan **tidak boleh dimodifikasi secara manual**. Dataset historis IDX80 disimpan di `data/idx80_raw.parquet` dan akan diperbarui otomatis saat endpoint clustering dipanggil.

```
artifacts/
├── clustering/      → autoencoder.keras, encoder.keras, kmeans.pkl, ...
└── volatility/      → multi_horizon_model.keras, stock_scalers.pkl, ...
```