# services/volatility/config.py
# ============================================================
# CONFIG — Volatility Forecasting
# ============================================================

import os
from pathlib import Path

# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "volatility"

MODEL_PATH           = str(ARTIFACTS_DIR / "multi_horizon_model.keras")
STOCK_SCALERS_PATH   = str(ARTIFACTS_DIR / "stock_scalers.pkl")
TICKER_ENCODER_PATH  = str(ARTIFACTS_DIR / "ticker_encoder.pkl")
SECTOR_ENCODER_PATH  = str(ARTIFACTS_DIR / "sector_encoder.pkl")
INDUSTRY_ENCODER_PATH = str(ARTIFACTS_DIR / "industry_encoder.pkl")
FEATURE_INFO_PATH    = str(ARTIFACTS_DIR / "feature_info.json")

# ============================================================
# FEATURE COLS — same order as training
# ============================================================

FEATURE_COLS = [
    "log_return",
    "volume_change",
    "volatility_10d",
    "volatility_20d",
    "hl_spread",
    "rolling_mean_5",
    "rolling_std_5",
    "rolling_mean_20",
    "rolling_std_20",
    "rsi_14",
    "ihsg_return",
    "macd_normalized",
]

LOOKBACK = 90

# ============================================================
# HORIZONS
# Trading days per horizon — used for price range calculation
# Formula from notebook: close * exp(±vol * sqrt(horizon_days))
# ============================================================

HORIZONS = {
    "10d": 10,
    "1m":  22,
    "3m":  66,
}

# ============================================================
# IHSG TICKER
# ============================================================

IHSG_TICKER = "^JKSE"
