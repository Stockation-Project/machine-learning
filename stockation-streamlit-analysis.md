# 📊 STRATEGIC ANALYSIS REPORT: STOCKATION STREAMLIT COMPANION APP
**Penyelarasan Dashboard Data Sains dengan Misi Edukasi, Pengendalian FOMO, dan Profil Risiko Gen Z**

---

## 1. Executive Summary (Ringkasan Eksekutif)

### Latar Belakang & Masalah
Dashboard Streamlit (`dashboard.py`) yang dibuat oleh tim data sains saat ini sangat kaya akan visualisasi data historis dan statistik. Namun, dashboard tersebut masih bersifat **umum (generic)** dan belum mencerminkan nilai inti serta misi dari **Stockation**:
* **Terlalu Teknis bagi Pemula**: Penggunaan rumus statistik kompleks seperti koefisien korelasi Pearson ($\rho$), deviasi standar ($\sigma$), dan heatmap korelasi silang cenderung mengintimidasi target pengguna utama Stockation (mahasiswa dan pemula dari Gen Z).
* **Tidak Ada Filter Psikologis (FOMO Guard)**: Belum ada indikator yang secara langsung mendidik user untuk menahan bias psikologis (FOMO - *Fear of Missing Out*), yang merupakan salah satu *problem statement* terbesar dalam proyek.
* **Bersifat Statis (Satu Ukuran untuk Semua)**: Analisis saham yang ditampilkan bernilai sama untuk investor konservatif tipe Kura-kura maupun investor agresif tipe Singa.

### Solusi Strategis: Deep-Linking & Personalization
Kita akan mengubah dashboard ini menjadi **Stockation Companion App** yang terintegrasi secara mulus (*seamless*). Ketika pengguna menjelajahi saham di platform utama React, mereka dapat menekan tombol **"Analisis Mendalam AI"** yang akan membuka tab Streamlit baru dengan parameter khusus di URL:

```text
http://localhost:8501/?ticker=BBCA&risk_profile=capybara
```

Streamlit akan secara dinamis membaca parameter tersebut, lalu menyesuaikan tingkat bahasa, peringatan risiko, dan indikator FOMO secara instan sesuai dengan profil unik pengguna tersebut!

---

## 2. Analisis Kesenjangan (Gap Analysis)

| Fitur Saat Ini (Data Science Dashboard) | Kebutuhan Stockation (Edukasi & Anti-FOMO) | Dampak & Solusi Pengembangan |
| :--- | :--- | :--- |
| **Bubble Chart Risk vs Return** umum. | Penilaian kesesuaian saham dengan profil risiko pengguna (`Kura-kura`, `Capybara`, `Singa`). | **Personalized Risk Checker**: Menampilkan cap kelayakan saham berdasarkan tipe risiko user. |
| **Harga & Return Harian** mentah. | Alarm penunjuk tingkat kegilaan pasar (*hype bubble*) untuk mencegah pembelian di harga pucuk. | **FOMO & Impulse Controller**: Indikator visual sederhana (Aman, Waspada Hype, Bahaya FOMO). |
| **Metrik Teknis Tradisional** (Std Dev, Beta). | Edukasi prediksi masa depan berbasis AI dengan batas ketidakpastian yang mudah dipahami. | **LSTM AI Forecasting Integration**: Visualisasi rentang prediksi harga 30 hari ke depan dengan *Confidence Interval*. |
| **Heatmap Korelasi** 15 saham besar. | Simulasi alokasi mandiri yang interaktif tanpa jargon matematika berat. | **Interactive Sandbox**: Simulasi kalkulator *"Bagaimana jika saya berinvestasi..."* yang disederhanakan. |

---

## 3. Desain Modul Kustom Baru untuk Stockation

Berikut adalah 4 Modul Khusus yang direkomendasikan untuk ditambahkan oleh tim data sains Anda ke dalam `dashboard.py`:

### 🛡️ Modul A: FOMO & Impulse Controller (Penahan Trading Impulsif)
Modul ini bertindak sebagai "rem darurat" bagi Gen Z yang rentan membeli saham karena ramai dibicarakan di media sosial (impulsive trading).
* **Logika Indikator**: Menggunakan kombinasi **RSI (Relative Strength Index)** dan deviasi harga saat ini dari **MA20 (Moving Average 20 hari)**.
* **Tiga Level Status di Dashboard**:
  1. 🟢 **ZONA STABIL (Kondisi Aman)**: Harga bergerak wajar.
     * *Pesan Edukasi*: *"Harga bergerak stabil. Keputusan pembelian saat ini didukung oleh fondasi harga yang sehat, terbebas dari gelembung FOMO."*
  2. 🟡 **ZONA HYPE (Waspada Pucuk)**: RSI berkisar 60-70 atau harga > 8% di atas MA20.
     * *Pesan Edukasi*: *"Perhatian! Saham ini mulai memanas karena antusiasme pasar yang tinggi. Disarankan melakukan cicil beli secara bertahap daripada langsung masuk dengan modal besar."*
  3. 🔴 **ZONA DANGER FOMO (Sangat Overbought)**: RSI > 70 atau harga > 15% di atas MA20.
     * *Pesan Edukasi*: *"⚠️ AWAS PUCUK! Saham ini sedang mengalami demam FOMO ekstrem. Secara statistik, membeli di zona ini sangat berisiko membuat Anda tersangkut saat pasar melakukan koreksi mendadak. Ambil napas dalam-dalam sebelum membeli!"*

---

### 🐾 Modul B: Personalized Risk Profile Checker (Kesesuaian Tipe Investor)
Streamlit membaca query parameter `?risk_profile=...` untuk merender widget kecocokan investasi:
* **Tipe Kura-kura (Turtle) & Kuda Nil (Hippo) - [Low Risk]**:
  * *Evaluasi Saham Volatilitas Tinggi (e.g. ADRO, BUMI)*: *"⚠️ Peringatan Risiko: Tipe Kura-kura Anda lebih menyukai stabilitas. Saham ini memiliki fluktuasi harga yang sangat tajam, kurang cocok untuk ketenangan pikiran Anda."*
  * *Evaluasi Saham Blue Chip (e.g. BBCA, TLKM)*: *"✅ Sangat Cocok! Tingkat fluktuasi saham ini sangat rendah dan stabil, bagaikan cangkang pelindung Kura-kura Anda."*
* **Tipe Capybara - [Medium Risk]**:
  * *Evaluasi Saham*: *"⚖️ Cocok dalam Batas Wajar. Saham ini menawarkan keseimbangan risiko dan profit yang pas untuk ketenangan jiwa Capybara Anda."*
* **Tipe Serigala (Wolf) & Singa (Lion) - [High Risk]**:
  * *Evaluasi Saham Volatilitas Tinggi*: *"🔥 Zona Berburu Anda! Tingkat fluktuasi yang ekstrem ini menyajikan peluang profit kilat yang sangat disukai jiwa pemberani Serigala/Singa Anda. Kelola modal dengan bijak!"*

---

### 🔮 Modul C: LSTM AI Forecasting (Prediksi Masa Depan Interaktif)
Menampilkan visualisasi hasil model **LSTM TensorFlow** yang sudah direncanakan pada arsitektur utama Stockation.
* **Grafik Rentang Keyakinan (*Confidence Interval*)**:
  * AI tidak hanya meramalan satu garis harga mati, melainkan menampilkan **lorong prediksi** (batas atas dan batas bawah).
  * *Teks Edukasi*: *"Grafik di atas menunjukkan ramalan AI untuk 30 hari ke depan. Garis tengah adalah prediksi utama, dan lorong berwarna biru muda adalah Rentang Keyakinan. Ingat! AI pun memiliki batas ketidakpastian karena pasar saham sangat dinamis."*

---

### 🎮 Modul D: Gamified Strategy Sandbox (Simulator Portofolio Masa Lalu)
Sebagai pengganti rumus korelasi statistik yang membosankan bagi pemula, kita menyajikan simulator sederhana:
* User menggeser slider alokasi modal (misal: Rp10 Juta).
* Sistem menghitung kinerja keuntungan historis 1 tahun terakhir secara interaktif.
* Menampilkan perbandingan performa investasi vs bunga tabungan bank biasa untuk memicu antusiasme literasi keuangan mereka.

---

## 4. Arsitektur Integrasi Teknis (React $\leftrightarrow$ Streamlit)

Integrasi ini memanfaatkan fitur bawaan Streamlit query parameters yang sangat ringan dan tanpa memerlukan basis data tambahan untuk sinkronisasi sesi.

### A. Sisi React Frontend (`StockDetailPage.tsx` atau `SimulationBuyPage.tsx`)
Tombol analisis mendalam memicu fungsi pembukaan URL dengan parameter terenkripsi atau teks bersih:

```typescript
// frontend/src/features/stock/hooks/useStockDetail.ts
export const handleOpenDeepAnalysis = (ticker: string, riskProfile: string) => {
  const STREAMLIT_BASE_URL = "http://localhost:8501";
  // Memformat URL: http://localhost:8501/?ticker=${ticker}&risk_profile=${riskProfile}
  const cleanTicker = ticker.replace(".JK", "");
  const targetUrl = `${STREAMLIT_BASE_URL}/?ticker=${cleanTicker}&risk_profile=${riskProfile.toLowerCase()}`;
  
  // Membuka di tab baru
  window.open(targetUrl, "_blank");
};
```

### B. Sisi Python Streamlit (`dashboard.py` / `main.py`)
Membaca parameter URL secara aman di baris paling atas fungsi utama Streamlit:

```python
# machine-learning/dashboard/dashboard.py
import streamlit as st

# 1. Ambil query parameter dinamis dari React
params = st.query_params

# Default ke BBCA dan Capybara jika tidak ada parameter yang dikirim
selected_ticker = params.get("ticker", "BBCA").upper()
user_risk_profile = params.get("risk_profile", "capybara").lower()

st.session_state["active_ticker"] = selected_ticker
st.session_state["user_risk_profile"] = user_risk_profile
```

---

## 5. Rencana Jalan Pelaksanaan (Implementation Roadmap)

Untuk merealisasikan penyelarasan ini tanpa merusak fondasi dashboard yang sudah dibangun dengan indah oleh tim data sains, berikut adalah langkah bertahap yang disarankan:

### 📍 Tahap 1: Dynamic Parameter Setup (Minggu Pertama)
* Modifikasi baris awal `dashboard.py` untuk membaca parameter `ticker` dan `risk_profile`.
* Pastikan dropdown pilihan saham otomatis berpindah ke `ticker` yang dikirim dari React saat halaman pertama kali dimuat.

### 📍 Tahap 2: Implementasi Modul Anti-FOMO & Risk Checker (Minggu Kedua)
* Tambahkan fungsi perhitungan RSI sederhana di sisi Python.
* Gambar card indikator visual **Impulse Controller** dengan warna dinamis (merah/kuning/hijau) di bagian atas halaman **Stock Exploration**.
* Masukkan narasi personalisasi kecocokan saham berdasarkan tipe risiko hewan pengguna.

### 📍 Tahap 3: Penyederhanaan Visualisasi & Glosarium Edukasi (Minggu Ketiga)
* Tambahkan tombol *toggle tooltip* tanda tanya (`?`) di setiap judul grafik teknis. Ketika ditekan, muncul penjelasan sederhana mengenai arti grafik tersebut dalam bahasa Indonesia yang ramah pemula.
* Sambungkan lorong prediksi hasil LSTM TensorFlow ke halaman visualisasi utama.

---
*Laporan Analisis ini disusun dengan cermat untuk memastikan Stockation tidak hanya menjadi sekadar simulator jual-beli biasa, melainkan platform edukasi komprehensif penahan gelembung FOMO bagi generasi muda Indonesia.*
