# config/settings.py

# Konfigurasi batas anomali (berdasarkan variabel yang dipanggil tim AI-mu)
ANOMALY_CONFIG = {
    "min_data_points": 30,
    "detect_price_spike": True,
    "detect_volume_spike": True,
    "detect_return_anomaly": True,
    "detect_gap": True,
    "rolling_window": 20,
    "zscore_threshold": 2.5,
    "iqr_multiplier": 1.5,
    "gap_threshold_pct": 5.0,
}

# Daftar metode optimasi
OPTIMIZATION_METHODS = {
    "max_sharpe": "Max Sharpe Ratio",
    "min_volatility": "Minimasi Volatilitas",
    "max_return": "Maksimalkan Return",
    "risk_parity": "Risk Parity (Risiko Merata)",
    "equal_weight": "Bobot Sama Rata",
}

# Konfigurasi Profil Risiko
RISK_PROFILES = {
    "konservatif": {
        "max_volatility": 0.10,
        "target_return": 0.06,
        "risk_free_rate": 0.05,
        "min_weight": 0.05,
        "max_weight": 0.40,
        "method": "min_volatility"
    },
    "moderat": {
        "max_volatility": 0.15,
        "target_return": 0.10,
        "risk_free_rate": 0.05,
        "min_weight": 0.05,
        "max_weight": 0.50,
        "method": "max_sharpe"
    },
    "agresif": {
        "max_volatility": 0.25,
        "target_return": 0.15,
        "risk_free_rate": 0.05,
        "min_weight": 0.05,
        "max_weight": 0.80,
        "method": "max_return"
    }
}

# Limit slider toleransi risiko
SLIDER_CONFIG = {
    "min": 0.0,
    "max": 1.0
}