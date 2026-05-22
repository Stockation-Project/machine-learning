from pathlib import Path

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

# RAW_DATA_PATH menunjuk langsung ke file parquet di dalam folder clustering/
RAW_DATA_PATH = Path(__file__).resolve().parent / "idx80_raw.parquet"

# =========================
# STOCK SETTINGS
# =========================
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

START_DATE = "2026-03-27"

# =========================
# MODEL SETTINGS
# =========================
SEQ_LENGTH = 30
LATENT_DIM = 32
N_CLUSTERS = 3

# =========================
# RETRAIN SETTINGS
# =========================
RETRAIN_FREQUENCY = "daily"