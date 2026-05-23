# 📋 DOKUMEN PENUGASAN (DATA SCIENCE DIRECTIVE)
**Subjek**: Penyelarasan Streamlit Dashboard dengan Misi Edukasi Anti-FOMO Platform Stockation  
**Kepada**: Tim Data Science Stockation  
**Dari**: Product Manager & Lead Systems Architect  
**Status**: Prioritas Tinggi (High Priority) — Cabang Kerja: `dev-dashboard`  

---

## 1. LATAR BELAKANG & TUJUAN UTAMA

Rekan-rekan Tim Data Science,

Terima kasih atas kerja kerasnya dalam membangun prototipe `dashboard.py` yang kaya akan data analisis IDX80. Struktur visualisasi dan analisis data historis yang Anda bangun sudah sangat baik secara statistik. 

Namun, untuk peluncuran produksi **Stockation**, kita perlu menyelaraskan dashboard ini agar sesuai dengan visi utama platform: **mendidik generasi muda (Gen Z) tentang literasi keuangan, menekan bias psikologis (FOMO), dan memberikan ruang aman untuk belajar investasi tanpa risiko finansial.**

Platform utama kita (React Web App) akan menyediakan tombol **"Analisis Mendalam AI"** di setiap halaman detail saham. Ketika ditekan, tombol tersebut akan mengarahkan pengguna ke URL Streamlit Anda dengan parameter khusus:
```text
http://localhost:8501/?ticker=BBCA&risk_profile=capybara
```

Tugas Anda adalah **memodifikasi dashboard Streamlit** agar dapat menerima parameter tersebut dan menyajikan visualisasi serta narasi yang dipersonalisasi khusus untuk user tersebut.

---

## 2. DETAIL TUGAS & SPESIFIKASI FITUR YANG WAJIB DIBUAT

Berikut adalah 5 tugas utama pengembangan yang harus diimplementasikan ke dalam kode Streamlit Anda:

### 🏷️ TUGAS 1: Integrasi Dynamic Query Parameter & Sinkronisasi Dropdown
* **Detail Tugas**:
  Gunakan modul `st.query_params` bawaan Streamlit untuk membaca nilai `ticker` dan `risk_profile` saat aplikasi dimuat pertama kali. Nilai dropdown pilihan saham di sidebar harus otomatis berpindah ke saham yang dikirim dari React (default jika kosong: `BBCA`).
* **Alasan Mengapa Harus Dibuat**:
  **User Experience (UX) Integration**. Pengguna tidak ingin memilih ulang kode saham di dashboard setelah mereka mengkliknya di platform utama React. Sistem harus terasa menyatu (*seamless integration*).

---

### 🛡️ TUGAS 2: Pembangunan Modul "FOMO & Impulse Controller" (Indikator Anti-Hype)
* **Detail Tugas**:
  Tambahkan kalkulasi teknis **RSI (Relative Strength Index)** dan deviasi harga dari **MA20** di sisi Python. Render panel status visual dengan warna dinamis (merah/kuning/hijau) di bagian paling atas halaman eksplorasi saham:
  1. 🟢 **Zona Stabil (Aman)**: Jika RSI < 60 dan harga bergerak wajar di sekitar MA20.
  2. 🟡 **Zona Hype (Waspada Pucuk)**: Jika RSI 60-70.
  3. 🔴 **Zona Danger FOMO (Sangat Overbought)**: Jika RSI > 70. Tampilkan peringatan tegas: *"Saham ini sedang mengalami demam FOMO ekstrem. Membeli di pucuk sangat berisiko membuat modal Anda tersangkut!"*
* **Alasan Mengapa Harus Dibuat**:
  **Mitigasi FOMO & Impulsive Trading (Misi Inti Stockation)**. Gen Z sangat rentan terhadap *herd behavior* (membeli saham hanya karena sedang ramai di media sosial). Peringatan ini melatih kedisiplinan mereka untuk tidak membeli saham di harga puncak.

---

### 🐾 TUGAS 3: Widget Kesesuaian Profil Risiko Investor (Animal Risk Matcher)
* **Detail Tugas**:
  Buat logika evaluasi yang memetakan tingkat volatilitas historis saham (menggunakan metrik Beta atau Standar Deviasi) terhadap tipe profil risiko pengguna yang dikirim via URL (`turtle`, `hippo`, `capybara`, `wolf`, `lion`):
  * **Tipe Konservatif (`turtle` / `hippo`)** yang membuka saham sangat bergejolak (misal: BUMI, ADRO) wajib diberikan peringatan berwarna merah: *"⚠️ Peringatan: Fluktuasi saham ini sangat tinggi, kurang ramah untuk profil Kura-kura Anda."*
  * **Tipe Agresif (`wolf` / `lion`)** pada saham volatil diberikan dukungan: *"🔥 Cocok dengan jiwa Singa Anda! Fluktuasi ini menghadirkan peluang keuntungan cepat bagi jiwa pemberani."*
* **Alasan Mengapa Harus Dibuat**:
  **Edukasi Manajemen Portofolio secara Praktis**. Pemula sering kali tidak paham arti volatilitas secara matematis. Dengan memetakannya ke dalam "hewan profil risiko", mereka belajar secara naluriah untuk memilih saham yang sesuai dengan toleransi psikologis mereka.

---

### 🔮 TUGAS 4: Integrasi Model Prediksi LSTM TensorFlow dengan Lorong Keyakinan
* **Detail Tugas**:
  Sambungkan visualisasi hasil prediksi 30 hari ke depan menggunakan model **LSTM** yang sudah Anda latih. Jangan hanya menampilkan satu garis prediksi lurus, melainkan tampilkan juga area bayangan **Rentang Keyakinan (*Confidence Interval*)** di sekeliling garis prediksi tersebut.
* **Alasan Mengapa Harus Dibuat**:
  **Edukasi Ekspektasi AI**. Kita wajib mendidik pengguna bahwa model kecerdasan buatan (AI) pun memiliki batas ketidakpastian dalam meramal pasar keuangan. Menampilkan lorong keyakinan mengajarkan mereka untuk skeptis secara sehat terhadap prediksi teknologi.

---

### 📚 TUGAS 5: Sederhanisasi Jargon Statistik & Penambahan Tooltip Edukatif
* **Detail Tugas**:
  1. Ubah istilah statistik yang rumit ke bahasa sehari-hari:
     * *Expected Return* $\rightarrow$ **"Potensi Profit"**
     * *Volatility* $\rightarrow$ **"Risiko Fluktuasi"**
     * *Sharpe Ratio* $\rightarrow$ **"Skor Efisiensi Investasi"**
  2. Tambahkan tombol info bantuan interaktif `st.help()` atau tanda tanya kecil (`?`) di setiap judul bagan teknis (seperti Heatmap Korelasi Pearson) untuk menjelaskan maknanya dengan sangat sederhana bagi orang awam.
* **Alasan Mengapa Harus Dibuat**:
  **Demokratisasi Literasi Keuangan**. Jargon kuantitatif yang terlalu berat akan membuat mahasiswa atau pemula merasa bodoh dan meninggalkan platform. Tugas kita adalah menerjemahkan matematika yang rumit menjadi wawasan yang memberdayakan.

---

## 3. CONTOH IMPLEMENTASI KODE PYTHON (BOILERPLATE UNTUK ACUAN)

Untuk membantu Anda memulai, berikut adalah contoh potongan kode Python untuk diintegrasikan di bagian atas berkas `dashboard.py`:

```python
# dashboard.py - Tim Data Science Stockation
import streamlit as st
import numpy as np
import pandas as pd

# ─── 1. MEMBACA PARAMETER DINAMIS DARI REACT WEB APP ───
params = st.query_params

# Membaca parameter dan menyiapkan nilai default
url_ticker = params.get("ticker", "BBCA").upper()
url_risk = params.get("risk_profile", "capybara").lower()

# Menyimpan ke session state agar konsisten
if "active_ticker" not in st.session_state:
    st.session_state["active_ticker"] = url_ticker
if "user_risk_profile" not in st.session_state:
    st.session_state["user_risk_profile"] = url_risk

# ─── 2. LOGIKA MATCHER PROFIL RISIKO (TUGAS 3) ───
def evaluate_risk_suitability(stock_beta, profile):
    if profile in ["turtle", "hippo"]: # Profil Risiko Rendah
        if stock_beta > 1.2:
            return "🔴 Kurang Cocok (Fluktuasi Terlalu Ekstrem untuk Kura-kura/Kuda Nil)", "#FF3D57"
        else:
            return "✅ Sangat Cocok (Stabil & Aman)", "#00E676"
            
    elif profile in ["wolf", "lion"]: # Profil Risiko Tinggi
        if stock_beta > 1.2:
            return "🔥 Sangat Cocok (Volatilitas Tinggi Cocok untuk Jiwa Singa/Serigala Anda)", "#7B61FF"
        else:
            return "⚖️ Cocok (Namun profitnya mungkin kurang menantang bagi Anda)", "#FFD600"
            
    else: # Moderat (Capybara)
        return "⚖️ Cocok dalam Batas Toleransi Capybara", "#00D4FF"

# ─── 3. LOGIKA FOMO GUARD (TUGAS 2) ───
def get_fomo_status(rsi_value, price_deviation):
    if rsi_value > 70 or price_deviation > 15:
        return "🔴 DANGER FOMO (Sangat Panas / Overbought)", "Awas! Saham ini sedang digoreng antusiasme pasar..."
    elif rsi_value > 60 or price_deviation > 8:
        return "🟡 WASPADA HYPE (Mulai Memanas)", "Kondisi pasar sedang ramai. Disarankan cicil beli perlahan..."
    else:
        return "🟢 ZONA STABIL (Kondisi Sehat)", "Harga bergerak wajar. Aman dari gelembung FOMO."
```

Silakan lakukan pengembangan langsung di cabang kerja Git **`dev-dashboard`** agar tidak mengganggu sistem *deployment* Railway di cabang utama (`main`). 

Selamat bekerja! Kontribusi Anda sangat krusial dalam menciptakan generasi muda yang cerdas berinvestasi tanpa terjebak gelembung FOMO. Jika ada kendala teknis atau arsitektur data, silakan koordinasikan langsung.

**Salam hangat,**  
*Lead Systems Architect & Product Manager Stockation*
