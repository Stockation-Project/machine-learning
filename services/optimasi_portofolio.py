"""
optimasi_portofolio.py
======================
Modul utama optimasi alokasi portofolio saham.

Mendukung dua alur penggunaan:
  1. Pengguna pilih saham → sistem rekomendasikan bobot optimal
  2. Pengguna minta sistem pilihkan saham + bobot sekaligus

Metode optimasi yang tersedia:
  - mean_cvar       : Mean-CVaR (minimasi Conditional Value at Risk) — DEFAULT
  - max_sharpe      : Maksimalkan Sharpe Ratio
  - min_volatility  : Minimasi volatilitas
  - max_return      : Maksimalkan return
  - risk_parity     : Distribusi risiko merata
  - equal_weight    : Bobot sama rata (baseline)

Profil risiko (3 level) menentukan parameter default termasuk posisi
default slider (default_risk_tolerance). Slider risk_tolerance (0.0–1.0)
menggeser target return sepanjang efficient frontier Mean-CVaR:
  - 0.0 → minimasi CVaR murni (paling aman)
  - 1.0 → kejar return maksimum (paling agresif)
Jika slider tidak diberikan, dipakai default sesuai profil risiko user.

Cara pakai:
    from optimasi_portofolio import PortfolioOptimizer

    optimizer = PortfolioOptimizer(risk_profile="moderat")

    # Alur 1: user pilih saham
    result = optimizer.optimize(
        tickers=["BBCA.JK", "TLKM.JK", "ASII.JK"],
        period_key="1_tahun",
    )

    # Alur 2: sistem rekomendasikan dari universe saham
    result = optimizer.recommend(
        universe=["BBCA.JK", "TLKM.JK", "ASII.JK", "BMRI.JK"],
        top_n=5,
        period_key="1_tahun",
    )

    print(result.summary())
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog

from config.settings import OPTIMIZATION_METHODS, RISK_PROFILES, SLIDER_CONFIG
from core.data_fetcher import DataFetcher

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

TRADING_DAYS = 252  # hari trading per tahun (standar bursa)


# ─────────────────────────────────────────────────────────
# DATA CLASS — HASIL OPTIMASI
# ─────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """
    Container hasil optimasi portofolio.

    Attributes:
        weights         : dict ticker → bobot (0–1), total = 1.0
        expected_return : estimasi return tahunan portofolio
        volatility      : estimasi volatilitas tahunan (risiko)
        sharpe_ratio    : return per unit risiko (lebih tinggi = lebih baik)
        method          : metode optimasi yang dipakai
        risk_profile    : profil risiko pengguna
        period_key      : periode data yang dipakai
        tickers_used    : ticker yang masuk hasil akhir
        tickers_dropped : ticker yang di-drop karena data tidak cukup
        metadata        : info tambahan (risk_tolerance, dll)
    """
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str
    risk_profile: str
    period_key: str
    tickers_used: list[str]
    tickers_dropped: list[str] = field(default_factory=list)
    cvar: float = 0.0
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Tampilkan ringkasan hasil optimasi dalam format teks."""
        lines = [
            "=" * 55,
            f"  HASIL OPTIMASI PORTOFOLIO",
            "=" * 55,
            f"  Profil Risiko  : {self.risk_profile.replace('_', ' ').title()}",
            f"  Metode         : {OPTIMIZATION_METHODS.get(self.method, self.method)}",
            f"  Periode Data   : {self.period_key.replace('_', ' ')}",
            "-" * 55,
            f"  Expected Return: {self.expected_return * 100:.2f}% / tahun",
            f"  Volatilitas    : {self.volatility * 100:.2f}% / tahun",
            f"  CVaR (95%)     : {self.cvar * 100:.2f}% / tahun",
            f"  Sharpe Ratio   : {self.sharpe_ratio:.4f}",
            "-" * 55,
            "  ALOKASI BOBOT:",
        ]
        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        for ticker, w in sorted_weights:
            bar = "█" * int(w * 40)
            lines.append(f"    {ticker:<12} {w * 100:5.1f}%  {bar}")
        if self.tickers_dropped:
            lines.append(f"\n  ⚠ Ticker di-drop: {', '.join(self.tickers_dropped)}")
        if self.metadata.get("risk_tolerance") is not None:
            lines.append(f"\n  Risk Tolerance Slider: {self.metadata['risk_tolerance']:.2f}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Konversi ke dict untuk serialisasi JSON/API response."""
        return {
            "weights": self.weights,
            "metrics": {
                "expected_return": round(self.expected_return, 6),
                "volatility": round(self.volatility, 6),
                "cvar": round(self.cvar, 6),
                "sharpe_ratio": round(self.sharpe_ratio, 6),
            },
            "method": self.method,
            "risk_profile": self.risk_profile,
            "period_key": self.period_key,
            "tickers_used": self.tickers_used,
            "tickers_dropped": self.tickers_dropped,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────────────────
# KELAS UTAMA
# ─────────────────────────────────────────────────────────

class PortfolioOptimizer:
    """
    Optimasi alokasi portofolio saham berbasis profil risiko.

    Args:
        risk_profile      : salah satu dari RISK_PROFILES di settings.py
        risk_tolerance    : slider 0.0 (sangat konservatif) – 1.0 (sangat agresif).
                            Jika None, pakai default dari profil.
        fetcher           : instance DataFetcher (inject untuk testing)
        vol_artifacts     : artefak model volatility forecasting (load_artifacts()).
                            Jika diberikan, risiko (diagonal kovarians) memakai
                            volatilitas PREDIKSI LSTM, bukan historis.
        use_forecast_vol  : aktifkan integrasi forecast volatility (default True
                            bila vol_artifacts tersedia).
    """

    def __init__(
        self,
        risk_profile: str = "moderat",
        risk_tolerance: Optional[float] = None,
        fetcher: Optional[DataFetcher] = None,
        vol_artifacts: Optional[dict] = None,
        use_forecast_vol: bool = True,
    ):
        if risk_profile not in RISK_PROFILES:
            raise ValueError(
                f"risk_profile '{risk_profile}' tidak dikenal. "
                f"Pilihan: {list(RISK_PROFILES.keys())}"
            )

        self.risk_profile = risk_profile
        self.profile_cfg = RISK_PROFILES[risk_profile].copy()
        self.fetcher = fetcher or DataFetcher()
        self.vol_artifacts = vol_artifacts
        self.use_forecast_vol = use_forecast_vol and (vol_artifacts is not None)

        # Tentukan toleransi risiko efektif untuk slider Mean-CVaR:
        #   - jika user menggeser slider  → pakai nilai yang dikirim
        #   - jika tidak (None)           → pakai default sesuai profil risiko
        # Nilai di-clamp ke rentang [SLIDER_CONFIG.min, SLIDER_CONFIG.max].
        self.slider_overridden = risk_tolerance is not None
        if risk_tolerance is None:
            risk_tolerance = self.profile_cfg.get("default_risk_tolerance", 0.5)
        self.risk_tolerance = float(
            max(SLIDER_CONFIG["min"], min(SLIDER_CONFIG["max"], risk_tolerance))
        )

    # ─────────────────────────────────────────────
    # PUBLIC — ALUR 1: USER PILIH SAHAM
    # ─────────────────────────────────────────────

    def optimize(
        self,
        tickers: list[str],
        period_key: str = "1_tahun",
        method: Optional[str] = None,
    ) -> OptimizationResult:
        """
        Hitung bobot optimal untuk saham yang sudah dipilih pengguna.

        Args:
            tickers    : list ticker saham pilihan pengguna
            period_key : periode historis untuk kalkulasi
            method     : override metode optimasi (opsional)

        Returns:
            OptimizationResult dengan bobot dan metrik portofolio.
        """
        if len(tickers) < 2:
            raise ValueError("Minimal 2 saham dibutuhkan untuk optimasi portofolio.")

        method = method or self.profile_cfg["method"]
        if method not in OPTIMIZATION_METHODS:
            raise ValueError(f"Metode '{method}' tidak dikenal: {list(OPTIMIZATION_METHODS.keys())}")

        # Ambil data harga
        prices = self.fetcher.get_prices(tickers, period_key)
        # Buang baris yang seluruhnya kosong (mis. hari libur / data belum ada)
        prices = prices.dropna(how="all")
        tickers_used = list(prices.columns)
        tickers_dropped = [t for t in tickers if t not in tickers_used]

        if len(tickers_used) < 2:
            raise ValueError(
                f"Tidak cukup saham valid ({len(tickers_used)}). "
                "Periksa ticker dan coba lagi."
            )

        # Validasi ketersediaan data historis: tanpa baris harga, seluruh
        # statistik (return/kovarians) akan NaN dan hasil tidak bermakna.
        if prices.shape[0] < 2:
            raise ValueError(
                "Data harga historis tidak tersedia untuk saham terpilih. "
                "Periksa koneksi ke sumber data dan coba lagi."
            )

        # Hitung return dan matriks kovarians
        returns, cov_matrix, mean_returns = self._compute_stats(prices)

        # Pengaman: buang ticker yang statistiknya NaN (data tidak memadai)
        valid = mean_returns.dropna().index.tolist()
        if len(valid) < len(tickers_used):
            dropped_nan = [t for t in tickers_used if t not in valid]
            tickers_dropped.extend(dropped_nan)
            tickers_used = valid
            if len(tickers_used) < 2:
                raise ValueError(
                    "Data historis tidak memadai untuk menghitung optimasi. "
                    "Coba pilih saham lain atau periode lain."
                )
            returns = returns[tickers_used]
            mean_returns = mean_returns[tickers_used]
            cov_matrix = cov_matrix.loc[tickers_used, tickers_used]

        # ── INTEGRASI VOLATILITY FORECASTING ──
        # Ganti volatilitas historis (diagonal kovarians) dengan volatilitas
        # PREDIKSI model LSTM, sambil mempertahankan korelasi historis.
        # Risiko jadi cerminan proyeksi ke depan, bukan sekadar masa lalu.
        vol_sources: dict[str, str] = {t: "historical" for t in tickers_used}
        vol_forecast_used = False
        vol_scale = None
        if self.use_forecast_vol:
            try:
                from services.volatility_risk import (
                    forecast_annual_volatilities,
                    blend_covariance_with_forecast,
                )
                hist_vol = pd.Series(
                    np.sqrt(np.diag(cov_matrix.values)), index=tickers_used
                )
                fc_vol, vol_sources = forecast_annual_volatilities(
                    tickers_used, self.vol_artifacts, hist_vol
                )
                cov_matrix = blend_covariance_with_forecast(cov_matrix, fc_vol)
                vol_forecast_used = any(
                    s == "forecast" for s in vol_sources.values()
                )
                # Faktor skala untuk CVaR: forecast_vol / historical_vol per saham
                hv = hist_vol.values
                hv_safe = np.where(hv > 1e-12, hv, 1e-12)
                fv = np.array([
                    fc_vol.get(t) if (t in fc_vol.index
                                      and np.isfinite(fc_vol.get(t))
                                      and fc_vol.get(t) > 0)
                    else hv[i]
                    for i, t in enumerate(tickers_used)
                ], dtype=float)
                vol_scale = fv / hv_safe
                logger.info(f"Volatility forecast diterapkan: {vol_sources}")
            except Exception as e:
                logger.warning(
                    f"Integrasi forecast volatility gagal, pakai historis: {e}"
                )

        # Skenario return yang dipakai untuk OPTIMASI & PELAPORAN CVaR harus
        # SAMA. Bila forecast aktif, skala skenario tiap saham di sekitar
        # rata-ratanya (volatilitas menyesuaikan prediksi, ekspektasi return
        # tetap). Ini menjamin optimizer meminimalkan CVaR yang sama dengan
        # yang ditampilkan — menghindari "aman malah lebih berisiko".
        returns_opt = returns
        if vol_scale is not None:
            mean_vec = returns.values.mean(axis=0)
            scaled = mean_vec + (returns.values - mean_vec) * vol_scale
            returns_opt = pd.DataFrame(
                scaled, index=returns.index, columns=returns.columns
            )

        # Jalankan optimasi sesuai metode (memakai skenario forecast bila ada)
        weights = self._run_optimization(method, mean_returns, cov_matrix, returns_opt)

        # Hitung metrik hasil. CVaR memakai skenario yang sama (returns_opt)
        # agar konsisten dengan yang dioptimasi.
        port_return, port_vol, sharpe = self._portfolio_metrics(
            weights, mean_returns, cov_matrix
        )
        port_cvar = self._portfolio_cvar(weights, returns_opt)

        result = OptimizationResult(
            weights=dict(zip(tickers_used, weights)),
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            cvar=port_cvar,
            method=method,
            risk_profile=self.risk_profile,
            period_key=period_key,
            tickers_used=tickers_used,
            tickers_dropped=tickers_dropped,
            metadata={
                "risk_tolerance": self.risk_tolerance,
                "slider_overridden": self.slider_overridden,
                "cvar_confidence": self.profile_cfg.get("cvar_confidence", 0.95),
                "volatility_source": "forecast" if vol_forecast_used else "historical",
                "volatility_per_ticker": vol_sources,
            },
        )

        logger.info(
            f"Optimasi selesai | {method} | "
            f"Return={port_return:.2%} Vol={port_vol:.2%} "
            f"CVaR={port_cvar:.2%} Sharpe={sharpe:.3f}"
        )
        return result

    # ─────────────────────────────────────────────
    # PUBLIC — ALUR 2: SISTEM REKOMENDASIKAN SAHAM
    # ─────────────────────────────────────────────

    def recommend(
        self,
        universe: list[str],
        top_n: int = 5,
        period_key: str = "1_tahun",
        method: Optional[str] = None,
    ) -> OptimizationResult:
        """
        Pilihkan saham terbaik dari universe dan hitung bobot optimalnya.

        Pemilihan saham berdasarkan:
        - Sharpe Ratio individual (return/risiko terbaik)
        - Korelasi antar saham (pilih yang lebih beragam)

        Args:
            universe   : pool saham yang bisa dipilih sistem
            top_n      : jumlah saham yang akan dipilih
            period_key : periode historis
            method     : override metode optimasi

        Returns:
            OptimizationResult dengan saham dan bobot pilihan sistem.
        """
        if len(universe) < top_n:
            raise ValueError(f"Universe ({len(universe)}) harus >= top_n ({top_n}).")

        prices = self.fetcher.get_prices(universe, period_key)
        _, cov_matrix, mean_returns = self._compute_stats(prices)

        # Scoring individual: Sharpe ratio per saham
        rf = self.profile_cfg["risk_free_rate"] / TRADING_DAYS
        individual_sharpe = {}
        for ticker in prices.columns:
            ret = mean_returns[ticker]
            vol = np.sqrt(cov_matrix.loc[ticker, ticker])
            individual_sharpe[ticker] = (ret - rf) / vol if vol > 0 else 0

        # Urutkan berdasarkan Sharpe, pilih top_n dengan memperhatikan diversifikasi
        ranked = sorted(individual_sharpe, key=individual_sharpe.get, reverse=True)
        selected = self._diversified_selection(ranked, cov_matrix, top_n)

        logger.info(f"Saham terpilih dari universe: {selected}")

        # Optimasi dengan saham terpilih
        return self.optimize(selected, period_key, method)

    # ─────────────────────────────────────────────
    # PRIVATE — OPTIMASI INTERNALS
    # ─────────────────────────────────────────────

    def _compute_stats(
        self, prices: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Hitung daily returns, matriks kovarians tahunan, dan mean return tahunan.
        Kovarians di-annualize dengan mengalikan TRADING_DAYS.
        """
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * TRADING_DAYS          # annualized
        cov_matrix = returns.cov() * TRADING_DAYS             # annualized
        return returns, cov_matrix, mean_returns

    def _run_optimization(
        self,
        method: str,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> np.ndarray:
        """
        Dispatch ke fungsi optimasi yang sesuai.
        Semua mengembalikan array numpy bobot (sum = 1).
        """
        n = len(mean_returns)
        if method == "mean_cvar":
            return self._mean_cvar(mean_returns, cov_matrix, returns, n)

        dispatch = {
            "max_sharpe":     self._max_sharpe,
            "min_volatility": self._min_volatility,
            "max_return":     self._max_return,
            "risk_parity":    self._risk_parity,
            "equal_weight":   self._equal_weight,
        }
        return dispatch[method](mean_returns, cov_matrix, n)

    def _constraints_and_bounds(self, n: int) -> tuple:
        """
        Buat constraints (sum=1) dan bounds (min/max per ticker)
        berdasarkan profil risiko aktif.
        """
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        min_w = self.profile_cfg["min_weight"]
        max_w = self.profile_cfg["max_weight"]
        bounds = tuple((min_w, max_w) for _ in range(n))
        return constraints, bounds

    # ─────────────────────────────────────────────
    # MEAN-CVaR (metode utama / default)
    # ─────────────────────────────────────────────

    def _mean_cvar(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        returns: pd.DataFrame,
        n: int,
    ) -> np.ndarray:
        """
        Optimasi Mean-CVaR memakai formulasi linear Rockafellar–Uryasev.

        CVaR (Conditional Value at Risk) pada tingkat kepercayaan β mengukur
        rata-rata kerugian pada skenario terburuk (ekor distribusi), sehingga
        lebih sensitif terhadap risiko ekstrem dibanding volatilitas biasa.

        Slider risk_tolerance (0..1) menggeser TARGET RETURN MINIMUM sepanjang
        efficient frontier Mean-CVaR:
            tol = 0.0 → minimasi CVaR murni (risiko terburuk PALING RENDAH)
            tol = 1.0 → target = return maksimum (paling agresif, CVaR naik)

        Penting: keamanan = CVaR rendah, BUKAN sekadar diversifikasi. Memaksa
        bobot merata ke saham volatil justru menaikkan CVaR, sehingga slider
        hanya mengatur target return dan membiarkan optimizer memilih bobot
        (boleh terkonsentrasi pada saham defensif) untuk meminimalkan CVaR.

        Variabel keputusan LP: [w_1..w_n, alpha (VaR), u_1..u_S]
            minimize  alpha + 1/((1-β)·S) · Σ u_s
            s.t.      u_s >= -(scenario_s · w) - alpha,  u_s >= 0
                      Σ w = 1,   w·mean >= R_target,   min_w <= w <= max_w

        Jika solver gagal, fallback ke min_volatility.
        """
        beta = self.profile_cfg.get("cvar_confidence", 0.95)
        min_w = self.profile_cfg["min_weight"]
        max_w = self.profile_cfg["max_weight"]

        # Matriks skenario return harian historis (S baris × n aset)
        scenarios = returns.values
        S = scenarios.shape[0]
        if S < 2:
            logger.warning("Skenario CVaR tidak cukup, fallback ke min_volatility.")
            return self._min_volatility(mean_returns, cov_matrix, n)

        # Target return floor: interpolasi antara return MIN & MAX yang feasible
        # di bawah cap profil. tol rendah → target rendah → optimizer bebas
        # meminimalkan CVaR (paling aman). tol tinggi → target tinggi → kejar
        # return (CVaR naik sebagai konsekuensi).
        ann_mean = mean_returns.values
        r_lo_annual, r_hi_annual = self._feasible_return_bounds(ann_mean, min_w, max_w)
        target_return_annual = r_lo_annual + self.risk_tolerance * (r_hi_annual - r_lo_annual)
        target_return_daily = target_return_annual / TRADING_DAYS

        # Susun variabel: x = [w(n), alpha(1), u(S)]
        num_vars = n + 1 + S
        inv = 1.0 / ((1.0 - beta) * S)

        # Fungsi objektif: minimize alpha + inv·Σ u
        c = np.zeros(num_vars)
        c[n] = 1.0                      # alpha
        c[n + 1:] = inv                 # u_s

        # Kendala ketidaksamaan A_ub · x <= b_ub
        A_ub, b_ub = [], []

        # CVaR linearization: -(scenario · w) - alpha - u_s <= 0
        for s in range(S):
            row = np.zeros(num_vars)
            row[:n] = -scenarios[s]
            row[n] = -1.0               # -alpha
            row[n + 1 + s] = -1.0       # -u_s
            A_ub.append(row)
            b_ub.append(0.0)

        # Target return: -(w · mean_daily) <= -target_daily
        mean_daily = ann_mean / TRADING_DAYS
        row = np.zeros(num_vars)
        row[:n] = -mean_daily
        A_ub.append(row)
        b_ub.append(-target_return_daily)

        # Kendala persamaan: Σ w = 1
        A_eq = np.zeros((1, num_vars))
        A_eq[0, :n] = 1.0
        b_eq = [1.0]

        # Bounds: w ∈ [min_w, max_w], alpha bebas, u >= 0
        var_bounds = (
            [(min_w, max_w)] * n
            + [(None, None)]
            + [(0, None)] * S
        )

        res = linprog(
            c,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            A_eq=A_eq,
            b_eq=np.array(b_eq),
            bounds=var_bounds,
            method="highs",
        )

        if not res.success:
            # Target return mungkin terlalu tinggi → longgarkan dengan
            # menghapus kendala target dan minimasi CVaR murni.
            logger.warning(
                f"LP Mean-CVaR gagal ({res.message}). Coba tanpa target return."
            )
            res = linprog(
                c,
                A_ub=np.array(A_ub[:-1]),
                b_ub=np.array(b_ub[:-1]),
                A_eq=A_eq,
                b_eq=np.array(b_eq),
                bounds=var_bounds,
                method="highs",
            )

        if not res.success:
            logger.warning("Mean-CVaR tetap gagal, fallback ke min_volatility.")
            return self._min_volatility(mean_returns, cov_matrix, n)

        weights = np.array(res.x[:n])
        return self._clean_weights(weights, n)

    def _feasible_return_bounds(
        self,
        ann_mean: np.ndarray,
        min_w: float,
        max_w: float,
    ) -> tuple[float, float]:
        """
        Hitung return portofolio MINIMUM dan MAKSIMUM yang masih feasible
        di bawah kendala Σw=1 dan min_w <= w_i <= max_w.

        Dipakai untuk menskalakan target return slider agar LP tidak pernah
        infeasible. Menggunakan greedy water-filling:
          - Return maksimum: alokasikan kelebihan budget ke aset return tertinggi
          - Return minimum : alokasikan ke aset return terendah
        """
        n = len(ann_mean)
        base = np.full(n, min_w)
        budget = 1.0 - base.sum()              # sisa bobot yang harus dialokasi
        cap = max_w - min_w                    # kapasitas tambahan per aset

        if budget <= 1e-12 or cap <= 1e-12:
            # Bobot praktis terkunci di min_w (mis. n*min_w ≈ 1)
            val = float(np.dot(base, ann_mean) / base.sum()) if base.sum() > 0 else 0.0
            return val, val

        def _greedy(order: np.ndarray) -> float:
            w = base.copy()
            remaining = budget
            for idx in order:
                add = min(cap, remaining)
                w[idx] += add
                remaining -= add
                if remaining <= 1e-12:
                    break
            return float(np.dot(w, ann_mean))

        asc = np.argsort(ann_mean)             # return terendah dulu
        desc = asc[::-1]                       # return tertinggi dulu
        r_min = _greedy(asc)
        r_max = _greedy(desc)
        return r_min, r_max

    def _max_sharpe(self, mean_returns, cov_matrix, n) -> np.ndarray:
        """Minimasi negatif Sharpe Ratio."""
        rf = self.profile_cfg["risk_free_rate"]
        constraints, bounds = self._constraints_and_bounds(n)

        def neg_sharpe(w):
            ret = np.dot(w, mean_returns)
            vol = np.sqrt(w @ cov_matrix.values @ w)
            return -(ret - rf) / vol if vol > 1e-9 else 0

        result = minimize(
            neg_sharpe,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        return self._clean_weights(result.x, n)

    def _min_volatility(self, mean_returns, cov_matrix, n) -> np.ndarray:
        """Minimasi volatilitas portofolio."""
        constraints, bounds = self._constraints_and_bounds(n)

        def portfolio_vol(w):
            return np.sqrt(w @ cov_matrix.values @ w)

        result = minimize(
            portfolio_vol,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        return self._clean_weights(result.x, n)

    def _max_return(self, mean_returns, cov_matrix, n) -> np.ndarray:
        """Maksimalkan return — linear programming via minimize."""
        constraints, bounds = self._constraints_and_bounds(n)

        def neg_return(w):
            return -np.dot(w, mean_returns)

        result = minimize(
            neg_return,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return self._clean_weights(result.x, n)

    def _risk_parity(self, mean_returns, cov_matrix, n) -> np.ndarray:
        """
        Risk Parity: setiap saham berkontribusi risiko yang sama.
        Menggunakan pendekatan iteratif (equal risk contribution).
        """
        constraints, bounds = self._constraints_and_bounds(n)
        target_risk = np.ones(n) / n  # setiap saham kontribusi risiko 1/n

        def risk_budget_objective(w):
            cov = cov_matrix.values
            port_var = w @ cov @ w
            marginal_risk = cov @ w
            risk_contrib = w * marginal_risk / port_var if port_var > 1e-9 else w
            return np.sum((risk_contrib - target_risk) ** 2)

        result = minimize(
            risk_budget_objective,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000},
        )
        return self._clean_weights(result.x, n)

    def _equal_weight(self, mean_returns, cov_matrix, n) -> np.ndarray:
        """Equal weight: bobot sama rata, tidak ada optimasi matematik."""
        return np.ones(n) / n

    def _portfolio_metrics(
        self,
        weights: np.ndarray,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """
        Hitung return, volatilitas, dan Sharpe Ratio portofolio.

        Returns:
            (expected_return, volatility, sharpe_ratio)
        """
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        rf = self.profile_cfg["risk_free_rate"]
        sharpe = (port_return - rf) / port_vol if port_vol > 1e-9 else 0.0
        return float(port_return), float(port_vol), float(sharpe)

    def _portfolio_cvar(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
    ) -> float:
        """
        Hitung CVaR (Conditional Value at Risk) portofolio dalam skala tahunan,
        dinyatakan sebagai nilai positif (besar kerugian rata-rata di ekor).

        CVaR_β = rata-rata kerugian pada (1-β) skenario terburuk.

        Catatan: bila integrasi forecast aktif, `returns` yang dikirim sudah
        diskalakan ke volatilitas prediksi, sehingga CVaR konsisten dengan
        skenario yang dioptimasi.
        """
        beta = self.profile_cfg.get("cvar_confidence", 0.95)
        port_daily = returns.values @ weights        # return harian portofolio
        if port_daily.size == 0:
            return 0.0

        # VaR = kuantil (1-β) dari distribusi return harian
        var_threshold = np.quantile(port_daily, 1.0 - beta)
        tail = port_daily[port_daily <= var_threshold]
        if tail.size == 0:
            cvar_daily = -var_threshold
        else:
            cvar_daily = -tail.mean()

        # Annualize (skala akar waktu) dan jadikan non-negatif
        cvar_annual = cvar_daily * np.sqrt(TRADING_DAYS)
        return float(max(cvar_annual, 0.0))

    def _clean_weights(self, weights: np.ndarray, n: int) -> np.ndarray:
        """
        Bersihkan bobot hasil optimasi:
        - Clip ke [min_weight, max_weight]
        - Normalisasi ulang agar total = 1
        - Handle NaN dengan fallback equal weight
        """
        if np.any(np.isnan(weights)):
            logger.warning("Optimasi menghasilkan NaN, fallback ke equal weight.")
            return np.ones(n) / n

        min_w = self.profile_cfg["min_weight"]
        max_w = self.profile_cfg["max_weight"]
        weights = np.clip(weights, min_w, max_w)
        total = weights.sum()
        return weights / total if total > 0 else np.ones(n) / n

    def _diversified_selection(
        self,
        ranked: list[str],
        cov_matrix: pd.DataFrame,
        top_n: int,
    ) -> list[str]:
        """
        Pilih top_n saham dari ranked list dengan memperhatikan diversifikasi.
        Saham yang sangat berkorelasi tinggi (>0.85) tidak dipilih bersamaan.
        """
        selected = [ranked[0]]

        # Hitung matriks korelasi
        corr = pd.DataFrame(
            np.array(cov_matrix) / np.outer(
                np.sqrt(np.diag(cov_matrix)),
                np.sqrt(np.diag(cov_matrix))
            ),
            index=cov_matrix.index,
            columns=cov_matrix.columns,
        )

        for candidate in ranked[1:]:
            if len(selected) >= top_n:
                break
            # Cek korelasi dengan saham yang sudah terpilih
            max_corr = max(abs(corr.loc[candidate, s]) for s in selected)
            if max_corr < 0.85:  # threshold diversifikasi
                selected.append(candidate)

        # Jika masih kurang, tambah sisanya tanpa cek korelasi
        for candidate in ranked:
            if len(selected) >= top_n:
                break
            if candidate not in selected:
                selected.append(candidate)

        return selected[:top_n]
