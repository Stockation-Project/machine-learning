# ============================================================
# CONFIG
# ============================================================

import os

# ============================================================
# PATHS
# ============================================================

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")

TICKER_METADATA_PATH = os.path.join(MODELS_DIR, "ticker_metadata.json")
MODEL_PATH          = os.path.join(MODELS_DIR, "multi_horizon_model.keras")
STOCK_SCALERS_PATH  = os.path.join(MODELS_DIR, "stock_scalers.pkl")
TICKER_ENCODER_PATH = os.path.join(MODELS_DIR, "ticker_encoder.pkl")
SECTOR_ENCODER_PATH = os.path.join(MODELS_DIR, "sector_encoder.pkl")
INDUSTRY_ENCODER_PATH = os.path.join(MODELS_DIR, "industry_encoder.pkl")
FEATURE_INFO_PATH   = os.path.join(MODELS_DIR, "feature_info.json")

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
