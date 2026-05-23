"""
optimasi_portofolio.py
======================
Modul utama optimasi alokasi portofolio saham.

Mendukung dua alur penggunaan:
  1. Pengguna pilih saham → sistem rekomendasikan bobot optimal
  2. Pengguna minta sistem pilihkan saham + bobot sekaligus

Metode optimasi yang tersedia:
  - max_sharpe      : Maksimalkan Sharpe Ratio
  - min_volatility  : Minimasi volatilitas
  - max_return      : Maksimalkan return
  - risk_parity     : Distribusi risiko merata
  - equal_weight    : Bobot sama rata (baseline)

Profil risiko (5 level) menentukan parameter default,
slider risk_tolerance (0.0–1.0) bisa override secara fleksibel.

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
from scipy.optimize import minimize

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
        risk_profile    : salah satu dari RISK_PROFILES di settings.py
        risk_tolerance  : slider 0.0 (sangat konservatif) – 1.0 (sangat agresif).
                          Jika None, pakai default dari profil.
        fetcher         : instance DataFetcher (inject untuk testing)
    """

    def __init__(
        self,
        risk_profile: str = "moderat",
        risk_tolerance: Optional[float] = None,
        fetcher: Optional[DataFetcher] = None,
    ):
        if risk_profile not in RISK_PROFILES:
            raise ValueError(
                f"risk_profile '{risk_profile}' tidak dikenal. "
                f"Pilihan: {list(RISK_PROFILES.keys())}"
            )

        self.risk_profile = risk_profile
        self.profile_cfg = RISK_PROFILES[risk_profile].copy()
        self.fetcher = fetcher or DataFetcher()

        # Terapkan slider jika diberikan
        self.risk_tolerance = risk_tolerance
        if risk_tolerance is not None:
            self._apply_slider(risk_tolerance)

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
        tickers_used = list(prices.columns)
        tickers_dropped = [t for t in tickers if t not in tickers_used]

        if len(tickers_used) < 2:
            raise ValueError(
                f"Tidak cukup saham valid ({len(tickers_used)}). "
                "Periksa ticker dan coba lagi."
            )

        # Hitung return dan matriks kovarians
        returns, cov_matrix, mean_returns = self._compute_stats(prices)

        # Jalankan optimasi sesuai metode
        weights = self._run_optimization(method, mean_returns, cov_matrix)

        # Hitung metrik hasil
        port_return, port_vol, sharpe = self._portfolio_metrics(
            weights, mean_returns, cov_matrix
        )

        result = OptimizationResult(
            weights=dict(zip(tickers_used, weights)),
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            method=method,
            risk_profile=self.risk_profile,
            period_key=period_key,
            tickers_used=tickers_used,
            tickers_dropped=tickers_dropped,
            metadata={"risk_tolerance": self.risk_tolerance},
        )

        logger.info(
            f"Optimasi selesai | {method} | "
            f"Return={port_return:.2%} Vol={port_vol:.2%} Sharpe={sharpe:.3f}"
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
    ) -> np.ndarray:
        """
        Dispatch ke fungsi optimasi yang sesuai.
        Semua mengembalikan array numpy bobot (sum = 1).
        """
        n = len(mean_returns)
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

    def _apply_slider(self, risk_tolerance: float):
        """
        Sesuaikan parameter profil berdasarkan nilai slider (0.0–1.0).

        Slider menginterpolasi antara parameter profil yang paling konservatif
        dan yang paling agresif secara proporsional.

        Contoh: slider=0.7 → max_volatility lebih tinggi, min_weight lebih kecil.
        """
        slider = max(SLIDER_CONFIG["min"], min(SLIDER_CONFIG["max"], risk_tolerance))

        # Batas parameter dari seluruh profil
        all_profiles = list(RISK_PROFILES.values())
        min_vol = all_profiles[0]["max_volatility"]   # sangat konservatif
        max_vol = all_profiles[-1]["max_volatility"]  # sangat agresif
        min_ret = all_profiles[0]["target_return"]
        max_ret = all_profiles[-1]["target_return"]

        # Interpolasi linear berdasarkan slider
        self.profile_cfg["max_volatility"] = min_vol + slider * (max_vol - min_vol)
        self.profile_cfg["target_return"] = min_ret + slider * (max_ret - min_ret)

        # Sesuaikan metode jika slider ekstrem
        if slider <= 0.2:
            self.profile_cfg["method"] = "min_volatility"
        elif slider >= 0.8:
            self.profile_cfg["method"] = "max_return"
        else:
            self.profile_cfg["method"] = "max_sharpe"

        logger.info(
            f"Slider diterapkan: {slider:.2f} → "
            f"max_vol={self.profile_cfg['max_volatility']:.2%} "
            f"method={self.profile_cfg['method']}"
        )

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
