"""
volatility_risk.py
===================
Jembatan antara model FORECASTING VOLATILITAS (LSTM multi-horizon) dengan
optimizer portofolio.

Ide inti (diferensiator capstone):
    Alih-alih memakai volatilitas historis (cerminan MASA LALU), optimizer
    memakai volatilitas HASIL PREDIKSI model LSTM (cerminan RISIKO KE DEPAN).
    Korelasi antar-saham tetap diambil dari data historis (struktur korelasi
    relatif stabil), sedangkan besaran risiko per-saham (diagonal kovarians)
    digantikan oleh prediksi:

        Cov_forecast = D_f · Corr_historis · D_f
        dengan D_f = diag(volatilitas_tahunan_prediksi)

Robust: bila prediksi gagal untuk suatu ticker (tidak ada di data latih,
fetch gagal, dll.), otomatis fallback ke volatilitas historis ticker tsb.
"""

import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# Satu sesi optimasi dapat memanggil model berulang saat slider digeser.
# Simpan prediksi per ticker agar setiap perbandingan slider memakai asumsi
# volatilitas yang sama; tanpa ini angka CVaR bisa berubah walau bobotnya sama.
_forecast_vol_cache: dict[str, float] = {}

# Horizon prediksi → bobot kepercayaan untuk membentuk satu angka volatilitas.
# Lebih menekankan horizon menengah (1 bulan) sebagai estimasi risiko utama.
_HORIZON_WEIGHTS = {
    "pred_vol_10d": 0.25,
    "pred_vol_1m": 0.50,
    "pred_vol_3m": 0.25,
}

# Fallback statis sektor/industri untuk ticker IDX umum, dipakai bila
# yfinance.info gagal/lambat. Nilai harus cocok dengan kelas encoder model.
_SECTOR_FALLBACK: dict[str, tuple[str, str]] = {
    "BBCA.JK": ("Financial Services", "Banks - Regional"),
    "BBRI.JK": ("Financial Services", "Banks - Regional"),
    "BMRI.JK": ("Financial Services", "Banks - Regional"),
    "BBNI.JK": ("Financial Services", "Banks - Regional"),
    "TLKM.JK": ("Communication Services", "Telecom Services"),
    "ASII.JK": ("Industrials", "Conglomerates"),
    "ANTM.JK": ("Basic Materials", "Other Industrial Metals & Mining"),
    "ADRO.JK": ("Energy", "Thermal Coal"),
    "ADHI.JK": ("Industrials", "Engineering & Construction"),
    "EMTK.JK": ("Communication Services", "Broadcasting"),
    "AALI.JK": ("Consumer Defensive", "Farm Products"),
    "SRTG.JK": ("Financial Services", "Asset Management"),
}


@lru_cache(maxsize=256)
def _resolve_sector_industry(ticker: str) -> Optional[tuple[str, str]]:
    """Ambil (sector, industry) untuk ticker. Coba yfinance, lalu fallback."""
    if ticker in _SECTOR_FALLBACK:
        return _SECTOR_FALLBACK[ticker]
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        industry = info.get("industry")
        if sector and industry:
            return sector, industry
    except Exception as e:
        logger.debug(f"Gagal ambil sector/industry {ticker}: {e}")
    return None


def _blended_annual_vol(pred: dict) -> float:
    """Gabungkan prediksi volatilitas multi-horizon → satu volatilitas tahunan."""
    daily = 0.0
    total_w = 0.0
    for key, w in _HORIZON_WEIGHTS.items():
        v = pred.get(key)
        if v is not None and np.isfinite(v) and v > 0:
            daily += w * v
            total_w += w
    if total_w == 0:
        return float("nan")
    daily /= total_w
    # Volatilitas harian → tahunan (scaling akar waktu)
    return float(daily * np.sqrt(TRADING_DAYS))


def forecast_annual_volatilities(
    tickers: list[str],
    artifacts: Optional[dict],
    hist_vol: pd.Series,
) -> tuple[pd.Series, dict[str, str]]:
    """
    Hasilkan volatilitas tahunan per ticker memakai model forecasting.

    Args:
        tickers   : daftar ticker (format .JK)
        artifacts : artefak model volatility (load_artifacts()); None → skip
        hist_vol  : volatilitas tahunan historis per ticker (fallback)

    Returns:
        (vols, sources)
        vols    : pd.Series volatilitas tahunan final per ticker
        sources : dict ticker → "forecast" | "historical" (untuk transparansi)
    """
    vols = {}
    sources: dict[str, str] = {}

    # Tanpa artefak model → semua pakai historis
    if not artifacts:
        for t in tickers:
            vols[t] = float(hist_vol.get(t, np.nan))
            sources[t] = "historical"
        return pd.Series(vols), sources

    # Import lokal agar modul ini tidak wajib punya TF saat hanya fallback
    from services.volatility.predictor import predict_one

    for t in tickers:
        fallback = float(hist_vol.get(t, np.nan))
        cached = _forecast_vol_cache.get(t)
        if cached is not None and np.isfinite(cached) and cached > 0:
            vols[t] = cached
            sources[t] = "forecast"
            continue
        si = _resolve_sector_industry(t)
        if si is None:
            vols[t] = fallback
            sources[t] = "historical"
            continue
        sector, industry = si
        try:
            pred = predict_one(t, sector, industry, artifacts)
            v = _blended_annual_vol(pred)
            if np.isfinite(v) and v > 0:
                vols[t] = v
                sources[t] = "forecast"
                _forecast_vol_cache[t] = v
            else:
                vols[t] = fallback
                sources[t] = "historical"
        except Exception as e:
            logger.info(f"Forecast volatilitas {t} gagal, pakai historis: {e}")
            vols[t] = fallback
            sources[t] = "historical"

    return pd.Series(vols), sources


def blend_covariance_with_forecast(
    cov_hist: pd.DataFrame,
    forecast_vol: pd.Series,
) -> pd.DataFrame:
    """
    Bentuk ulang matriks kovarians: pertahankan korelasi historis, ganti
    volatilitas (diagonal) dengan prediksi.

        Cov = D · Corr · D,  D = diag(forecast_vol)

    Ticker tanpa volatilitas prediksi valid memakai volatilitas historisnya
    sendiri (sudah di-resolve di forecast_annual_volatilities).
    """
    tickers = list(cov_hist.columns)
    hist_std = np.sqrt(np.diag(cov_hist.values))
    hist_std_safe = np.where(hist_std > 1e-12, hist_std, 1e-12)

    # Korelasi historis dari kovarians historis
    corr = cov_hist.values / np.outer(hist_std_safe, hist_std_safe)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # Vektor volatilitas final (fallback ke historis bila NaN)
    new_std = np.array([
        forecast_vol.get(t) if (t in forecast_vol.index
                                and np.isfinite(forecast_vol.get(t))
                                and forecast_vol.get(t) > 0)
        else hist_std[i]
        for i, t in enumerate(tickers)
    ], dtype=float)

    new_cov = np.outer(new_std, new_std) * corr
    return pd.DataFrame(new_cov, index=tickers, columns=tickers)
