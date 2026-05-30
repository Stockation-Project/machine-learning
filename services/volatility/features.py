# ============================================================
# FEATURES
# Mirrors notebook feature engineering exactly
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf

from services.volatility.config import IHSG_TICKER, LOOKBACK


# ============================================================
# RSI — same as notebook
# ============================================================

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:

    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ============================================================
# MACD — same as notebook
# ============================================================

def compute_macd(series: pd.Series) -> pd.Series:

    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26


# ============================================================
# FETCH IHSG RETURN
# ============================================================

def _fetch_ihsg(period: str) -> pd.DataFrame:

    raw = yf.download(IHSG_TICKER, period=period, auto_adjust=False, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
    raw["ihsg_return"] = np.log(raw["close"] / raw["close"].shift(1))

    return raw[["date", "ihsg_return"]]


# ============================================================
# MAIN FEATURE BUILDER
# ============================================================

def build_features(ticker: str, lookback: int = LOOKBACK) -> pd.DataFrame:
    """
    Download price data for `ticker` and compute the same
    FEATURE_COLS used during training.

    Returns a clean DataFrame sorted by date with no NaNs.
    Raises ValueError if there aren't enough rows after cleaning.
    """

    # Extra rows needed for rolling windows (max=66 for volatility_20d
    # note: notebook uses rolling(10) for both 10d and 20d — replicated below)
    # plus lookback itself
    download_period = f"{lookback + 120}d"

    # ----------------------------------------------------------
    # DOWNLOAD STOCK
    # ----------------------------------------------------------
    raw = yf.download(ticker, period=download_period, auto_adjust=False, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    # ----------------------------------------------------------
    # BASIC RETURN FEATURES
    # ----------------------------------------------------------
    raw["log_return"]    = np.log(raw["close"] / raw["close"].shift(1))
    raw["log_volume"]    = np.log1p(raw["volume"])
    raw["volume_change"] = raw["log_volume"].diff()

    # ----------------------------------------------------------
    # PRICE ACTION
    # ----------------------------------------------------------
    raw["hl_spread"] = (raw["high"] - raw["low"]) / raw["close"]

    # ----------------------------------------------------------
    # ROLLING TREND
    # ----------------------------------------------------------
    for w in [5, 20]:
        raw[f"rolling_mean_{w}"] = raw["log_return"].rolling(w).mean()
        raw[f"rolling_std_{w}"]  = raw["log_return"].rolling(w).std()

    # ----------------------------------------------------------
    # ROLLING VOLATILITY
    # Note: notebook uses rolling(10).std() for BOTH 10d and 20d
    # Replicated exactly here
    # ----------------------------------------------------------
    raw["volatility_10d"] = raw["log_return"].rolling(10).std()
    raw["volatility_20d"] = raw["log_return"].rolling(10).std()

    # ----------------------------------------------------------
    # RSI — normalised to [0,1] as in notebook
    # ----------------------------------------------------------
    raw["rsi_14"] = compute_rsi(raw["close"], window=14) / 100

    # ----------------------------------------------------------
    # MACD NORMALISED
    # ----------------------------------------------------------
    raw["macd"]            = compute_macd(raw["close"])
    raw["macd_normalized"] = raw["macd"] / raw["close"]

    # ----------------------------------------------------------
    # IHSG RETURN
    # ----------------------------------------------------------
    ihsg = _fetch_ihsg(download_period)
    raw["date"] = pd.to_datetime(raw["date"])
    ihsg["date"] = pd.to_datetime(ihsg["date"])
    raw = raw.merge(ihsg, on="date", how="left")

    # ----------------------------------------------------------
    # CLEAN
    # ----------------------------------------------------------
    raw = (
        raw.replace([np.inf, -np.inf], np.nan)
           .dropna()
           .reset_index(drop=True)
    )

    if len(raw) < lookback:
        raise ValueError(
            f"Not enough data for {ticker}: got {len(raw)} rows, need {lookback}."
        )

    return raw
