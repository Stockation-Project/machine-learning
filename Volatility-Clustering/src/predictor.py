# ============================================================
# PREDICTOR
# Sequence builder + inference + price range
# ============================================================

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS, LOOKBACK, HORIZONS
from src.features import build_features


# ============================================================
# PRICE RANGE
# Formula from notebook:
#   upper = close * exp(+vol * sqrt(horizon_days))
#   lower = close * exp(-vol * sqrt(horizon_days))
# ============================================================

def vol_to_price_range(
    current_price: float,
    vol: float,
    horizon: str,
) -> tuple[float, float]:

    days       = HORIZONS[horizon]
    multiplier = vol * np.sqrt(days)
    upper      = current_price * np.exp(+multiplier)
    lower      = current_price * np.exp(-multiplier)
    return round(lower, 2), round(upper, 2)


# ============================================================
# PREDICT ONE TICKER
# ============================================================

def predict_one(
    ticker: str,
    artifacts: dict,
    sector: str = None,
    industry: str = None,
) -> dict:
    """
    Run full prediction pipeline for a single ticker.

    Parameters
    ----------
    ticker    : e.g. "BBCA.JK"
    sector    : sector string exactly as in training label encoder
    industry  : industry string exactly as in training label encoder
    artifacts : output of loader.load_artifacts()

    Returns
    -------
    dict with keys: ticker, current_price,
        pred_vol_10d, range_10d_low, range_10d_high,
        pred_vol_1m,  range_1m_low,  range_1m_high,
        pred_vol_3m,  range_3m_low,  range_3m_high
    """
    if sector is None or industry is None:
        ticker_metadata = artifacts["ticker_metadata"]
        if ticker not in ticker_metadata:
            raise ValueError(
                f"Ticker '{ticker}' tidak ada di ticker_metadata. "
                f"Isi sector dan industry secara manual."
            )
        meta     = ticker_metadata[ticker]
        sector   = sector   or meta["Sector"]
        industry = industry or meta["Industry"]

    model            = artifacts["model"]
    stock_scalers    = artifacts["stock_scalers"]
    ticker_encoder   = artifacts["ticker_encoder"]
    sector_encoder   = artifacts["sector_encoder"]
    industry_encoder = artifacts["industry_encoder"]

    # ----------------------------------------------------------
    # VALIDATE TICKER EXISTS IN TRAINING
    # Notebook stores ticker with .JK suffix as dict key
    # ----------------------------------------------------------
    if ticker not in stock_scalers:
        raise ValueError(f"Ticker '{ticker}' not found in stock_scalers.")

    # ticker_encoder was fit on ticker column which includes .JK
    if ticker not in ticker_encoder.classes_:
        raise ValueError(f"Ticker '{ticker}' not found in ticker_encoder.")

    if sector not in sector_encoder.classes_:
        raise ValueError(
            f"Sector '{sector}' not found. "
            f"Valid values: {list(sector_encoder.classes_)}"
        )

    if industry not in industry_encoder.classes_:
        raise ValueError(
            f"Industry '{industry}' not found. "
            f"Valid values: {list(industry_encoder.classes_)}"
        )

    # ----------------------------------------------------------
    # FETCH & ENGINEER FEATURES
    # ----------------------------------------------------------
    df = build_features(ticker, lookback=LOOKBACK)

    current_price = float(df["close"].iloc[-1])

    # ----------------------------------------------------------
    # SCALE — use the per-stock scaler from training
    # ----------------------------------------------------------
    scaler          = stock_scalers[ticker]
    features        = df[FEATURE_COLS].copy()
    features_scaled = scaler.transform(features)

    # Last LOOKBACK rows → shape (1, LOOKBACK, n_features)
    seq = features_scaled[-LOOKBACK:]
    X_seq = seq[np.newaxis, :, :].astype(np.float32)

    # ----------------------------------------------------------
    # ENCODE CATEGORICALS
    # ----------------------------------------------------------
    X_ticker   = np.array([[ticker_encoder.transform([ticker])[0]]],   dtype=np.int32)
    X_sector   = np.array([[sector_encoder.transform([sector])[0]]],   dtype=np.int32)
    X_industry = np.array([[industry_encoder.transform([industry])[0]]], dtype=np.int32)

    # ----------------------------------------------------------
    # INFERENCE
    # ----------------------------------------------------------
    pred_10d, pred_1m, pred_3m = model.predict(
        [X_seq, X_ticker, X_sector, X_industry],
        verbose=0,
    )

    vol_10d = float(pred_10d.flatten()[0])
    vol_1m  = float(pred_1m.flatten()[0])
    vol_3m  = float(pred_3m.flatten()[0])

    # ----------------------------------------------------------
    # PRICE RANGES
    # ----------------------------------------------------------
    low_10d, high_10d = vol_to_price_range(current_price, vol_10d, "10d")
    low_1m,  high_1m  = vol_to_price_range(current_price, vol_1m,  "1m")
    low_3m,  high_3m  = vol_to_price_range(current_price, vol_3m,  "3m")

    return {
        "ticker":          ticker,
        "current_price":   current_price,

        "pred_vol_10d":    round(vol_10d, 6),
        "range_10d_low":   low_10d,
        "range_10d_high":  high_10d,

        "pred_vol_1m":     round(vol_1m, 6),
        "range_1m_low":    low_1m,
        "range_1m_high":   high_1m,

        "pred_vol_3m":     round(vol_3m, 6),
        "range_3m_low":    low_3m,
        "range_3m_high":   high_3m,
    }
