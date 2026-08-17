# config/settings.py
# Konfigurasi terpusat untuk seluruh layanan Machine Learning

from pathlib import Path

# =============================================================================
# BASE PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

# =============================================================================
# ANOMALY DETECTION CONFIG
# =============================================================================

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

# =============================================================================
# PORTFOLIO OPTIMIZATION CONFIG
# =============================================================================

OPTIMIZATION_METHODS = {
    "mean_cvar": "Mean-CVaR (Conditional Value at Risk)",
    "max_sharpe": "Max Sharpe Ratio",
    "min_volatility": "Minimasi Volatilitas",
    "max_return": "Maksimalkan Return",
    "risk_parity": "Risk Parity (Risiko Merata)",
    "equal_weight": "Bobot Sama Rata",
}

# Profil risiko menentukan:
#   - batas konsentrasi bobot (min_weight / max_weight)
#   - posisi default slider risk_tolerance (default_risk_tolerance)
#   - tingkat kepercayaan CVaR (cvar_confidence)
# Slider risk_tolerance (0..1) menggeser target return pada efficient
# frontier CVaR: 0.0 = minimasi CVaR (paling aman), 1.0 = kejar return
# maksimum (paling agresif). Default mengikuti profil risiko user.
RISK_PROFILES = {
    "konservatif": {
        "max_volatility": 0.10,
        "target_return": 0.06,
        "risk_free_rate": 0.05,
        "min_weight": 0.00,
        "max_weight": 0.40,
        "method": "mean_cvar",
        "default_risk_tolerance": 0.20,
        "cvar_confidence": 0.95,
    },
    "moderat": {
        "max_volatility": 0.15,
        "target_return": 0.10,
        "risk_free_rate": 0.05,
        "min_weight": 0.00,
        "max_weight": 0.50,
        "method": "mean_cvar",
        "default_risk_tolerance": 0.50,
        "cvar_confidence": 0.95,
    },
    "agresif": {
        "max_volatility": 0.25,
        "target_return": 0.15,
        "risk_free_rate": 0.05,
        "min_weight": 0.00,
        "max_weight": 0.80,
        "method": "mean_cvar",
        "default_risk_tolerance": 0.80,
        "cvar_confidence": 0.90,
    }
}

SLIDER_CONFIG = {
    "min": 0.0,
    "max": 1.0
}

# =============================================================================
# CLUSTERING CONFIG
# =============================================================================

CLUSTERING_ARTIFACTS_DIR = ARTIFACTS_DIR / "clustering"

IDX80_TICKERS = [
    'AADI.JK', 'ACES.JK', 'ADMR.JK', 'ADRO.JK', 'AKRA.JK', 'AMMN.JK',
    'AMRT.JK', 'ANTM.JK', 'ARTO.JK', 'ASII.JK', 'BBCA.JK', 'BBNI.JK',
    'BBRI.JK', 'BBTN.JK', 'BMRI.JK', 'BREN.JK', 'BRMS.JK', 'BRPT.JK',
    'BSDE.JK', 'BTPS.JK', 'BUKA.JK', 'BUMI.JK', 'CMRY.JK', 'CPIN.JK',
    'CTRA.JK', 'CUAN.JK', 'DSNG.JK', 'DSSA.JK', 'ELSA.JK', 'EMTK.JK',
    'ENRG.JK', 'ERAA.JK', 'ESSA.JK', 'EXCL.JK', 'GOTO.JK', 'HEAL.JK',
    'HRTA.JK', 'HRUM.JK', 'ICBP.JK', 'INCO.JK', 'INDF.JK', 'INDY.JK',
    'INKP.JK', 'INTP.JK', 'ISAT.JK', 'ITMG.JK', 'JPFA.JK', 'JSMR.JK',
    'KIJA.JK', 'KLBF.JK', 'KPIG.JK', 'MAPA.JK', 'MAPI.JK', 'MBMA.JK',
    'MDKA.JK', 'MEDC.JK', 'MIKA.JK', 'MTEL.JK', 'MYOR.JK', 'NCKL.JK',
    'PANI.JK', 'PGAS.JK', 'PGEO.JK', 'PNLF.JK', 'PTBA.JK', 'PTRO.JK',
    'PWON.JK', 'RAJA.JK', 'RATU.JK', 'SCMA.JK', 'SIDO.JK', 'SMGR.JK',
    'SMRA.JK', 'SSIA.JK', 'TAPG.JK', 'TLKM.JK', 'TOWR.JK', 'UNTR.JK',
    'UNVR.JK', 'WIFI.JK'
]

RAW_DATA_PATH = DATA_DIR / "idx80_raw.parquet"

CLUSTERING_START_DATE = "2026-03-27"

# Model settings
CLUSTERING_SEQ_LENGTH = 30
CLUSTERING_LATENT_DIM = 32
CLUSTERING_N_CLUSTERS = 3
CLUSTERING_RETRAIN_FREQUENCY = "daily"

# =============================================================================
# VOLATILITY FORECASTING CONFIG
# =============================================================================

VOLATILITY_ARTIFACTS_DIR = ARTIFACTS_DIR / "volatility"
