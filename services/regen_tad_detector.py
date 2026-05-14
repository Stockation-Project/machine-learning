"""
ReGEN-TAD Anomaly Detector (Production)
=========================================
Production-ready anomaly detection engine using the ReGEN-TAD framework.
Replaces the old z-score/IQR methods with a deep generative ensemble approach.

Key Components:
  1. Data purification (Stage I)
  2. Generative backbone training (Stage II)
  3. Ensemble scoring + calibration (Stage III)
  4. Factor-level attribution
  5. REST API-ready interface

Architecture: Modular, async-compatible, logging-instrumented.
"""

import json
import logging
import math
import os
import pickle
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, TensorDataset

from regen_tad_model import ReGENTADBackbone

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# CONFIGURATION DEFAULTS
# ─────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    # Window & horizon
    "window_size": 36,
    "forecast_horizon": 5,
    "stride": 1,

    # Model architecture
    "d_model": 128,
    "n_heads": 6,
    "dropout": 0.1,
    "lstm_hidden": 32,
    "latent_dim": 128,

    # Training
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 15,
    "train_split": 0.6,
    "val_split": 0.2,
    "loss_weights": (0.2, 0.8, 0.5),  # (f1, f2, recon)
    "latent_reg": 0.0,

    # Purification (Stage I)
    "purify_quantile": 0.97,
    "max_remove_frac": 0.30,
    "purify_max_iter": 5,

    # Ensemble scoring
    "k_neighbors": 20,
    "ewma_span": 5,
    "alpha": 0.05,          # top fraction flagged as anomaly
    "use_rank_based": True,
    "score_weights": None,  # auto-normalized if None
    "epsilon": 1e-6,

    # Thresholds (fallback if not rank-based)
    "zscore_threshold": 3.0,

    # Runtime
    "device": "auto",
    "num_workers": 0,
    "seed": 42,
}


# ─────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    """Representasi satu kejadian anomali (production-grade)."""

    ticker: str
    date: Union[date, str]
    anomaly_type: str
    anomaly_score: float
    severity: str
    description: str
    method: str
    attribution: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    _SEVERITY_LABELS = {
        "critical": "🔴 Critical",
        "high": "🟠 High",
        "medium": "🟡 Medium",
        "low": "🟢 Low",
    }

    def summary(self) -> str:
        label = self._SEVERITY_LABELS.get(self.severity, self.severity)
        lines = [
            f"[{label}] {self.ticker} | {self.date} | ReGEN-TAD",
            f"  Anomaly Score : {self.anomaly_score:.4f}",
            f"  Severity      : {self.severity.upper()}",
            f"  Description   : {self.description}",
        ]
        if self.attribution:
            top3 = sorted(self.attribution.items(), key=lambda kv: -kv[1])[:3]
            lines.append(f"  Top Attribution: {', '.join(f'{k}({v:.3f})' for k, v in top3)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": str(self.date),
            "anomaly_type": self.anomaly_type,
            "anomaly_score": round(self.anomaly_score, 6),
            "severity": self.severity,
            "description": self.description,
            "method": self.method,
            "attribution": {k: round(v, 6) for k, v in self.attribution.items()},
            "diagnostics": {k: round(v, 6) for k, v in self.diagnostics.items()},
            "extra": self.extra,
        }


@dataclass
class DetectionResult:
    """Hasil deteksi lengkap untuk satu ticker."""

    ticker: str
    events: List[AnomalyEvent]
    scores: pd.DataFrame  # index=date, cols=diagnostic components + final score
    model_path: Optional[str] = None
    calibrated: bool = False

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "events": [e.to_dict() for e in self.events],
            "scores": self.scores.to_dict() if self.scores is not None else {},
            "model_path": self.model_path,
            "calibrated": self.calibrated,
        }


# ─────────────────────────────────────────────────────────
# SLIDING WINDOW DATASET
# ─────────────────────────────────────────────────────────

class SlidingWindowDataset:
    """Utility untuk membuat sliding windows dari time series."""

    @staticmethod
    def build(
        df: pd.DataFrame,
        window_size: int,
        horizon: int,
        feature_cols: List[str],
        stride: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex]:
        """
        Build input windows X and target windows Y.

        Returns:
            X: (N, window_size, n_features)
            Y: (N, horizon, n_features)
            dates: index dates for each window (end of window)
        """
        arr = df[feature_cols].values.astype(np.float32)
        n = len(arr)
        windows = []
        targets = []
        dates = []

        for i in range(0, n - window_size - horizon + 1, stride):
            x = arr[i : i + window_size]
            y = arr[i + window_size : i + window_size + horizon]
            windows.append(x)
            targets.append(y)
            dates.append(df.index[i + window_size - 1])

        if not windows:
            return torch.empty(0, window_size, len(feature_cols)), torch.empty(0, horizon, len(feature_cols)), pd.DatetimeIndex([])

        return (
            torch.from_numpy(np.stack(windows)),
            torch.from_numpy(np.stack(targets)),
            pd.DatetimeIndex(dates),
        )


# ─────────────────────────────────────────────────────────
# PURIFICATION (STAGE I)
# ─────────────────────────────────────────────────────────

class Purifier:
    """Reconstruction-based outlier purification (Stage I)."""

    def __init__(
        self,
        quantile: float = 0.97,
        max_remove_frac: float = 0.30,
        max_iter: int = 5,
        device: str = "cpu",
    ):
        self.quantile = quantile
        self.max_remove_frac = max_remove_frac
        self.max_iter = max_iter
        self.device = device

    def purify(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        input_dim: int,
        window_size: int,
        horizon: int,
    ) -> torch.Tensor:
        """
        Iteratively remove windows with extreme reconstruction error.

        Returns:
            indices: torch.LongTensor of purified indices
        """
        indices = torch.arange(len(X), device=self.device)
        n_total = len(X)
        max_remove = int(self.max_remove_frac * n_total)
        removed_total = 0

        for iteration in range(self.max_iter):
            if len(indices) < window_size + horizon:
                logger.warning("Purification: too few samples left, stopping early.")
                break

            # Build a simple autoencoder for purification
            purify_model = self._build_purify_ae(input_dim, window_size, horizon).to(self.device)
            purify_model.train()
            optimizer = torch.optim.Adam(purify_model.parameters(), lr=1e-3)
            dataset = TensorDataset(X[indices], Y[indices])
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            # Quick training (lightweight)
            for epoch in range(20):
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    _, recon = purify_model(xb)
                    loss = F.mse_loss(recon, xb)
                    loss.backward()
                    optimizer.step()

            # Compute reconstruction errors
            purify_model.eval()
            errors = []
            with torch.no_grad():
                for xb, _ in DataLoader(TensorDataset(X[indices]), batch_size=128):
                    xb = xb[0].to(self.device)
                    _, recon = purify_model(xb)
                    e = F.mse_loss(recon, xb, reduction="none").mean(dim=(1, 2))
                    errors.append(e.cpu())
            errors = torch.cat(errors).numpy()

            threshold = np.quantile(errors, self.quantile)
            keep_mask = errors < threshold
            n_removed = (~keep_mask).sum()

            if n_removed == 0:
                logger.info(f"Purification converged at iteration {iteration + 1}.")
                break

            if removed_total + n_removed > max_remove:
                allowed = max_remove - removed_total
                # Remove only the top `allowed` worst
                worst_idx = np.argsort(errors)[-allowed:]
                keep_mask = np.ones(len(errors), dtype=bool)
                keep_mask[worst_idx] = False
                n_removed = allowed

            indices = indices[keep_mask]
            removed_total += n_removed
            logger.info(
                f"Purification iter {iteration + 1}: removed {n_removed} windows, "
                f"remaining {len(indices)} / {n_total}"
            )

        return indices.cpu()

    def _build_purify_ae(self, input_dim: int, window_size: int, horizon: int) -> nn.Module:
        """Simple autoencoder for purification."""
        class SimpleAE(nn.Module):
            def __init__(self, d_in, w, h):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(d_in * w, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, d_in * w),
                )
                self.d_in = d_in
                self.w = w

            def forward(self, x):
                b = x.size(0)
                z = self.encoder(x.view(b, -1))
                recon = self.decoder(z).view(b, self.w, self.d_in)
                return z, recon

        return SimpleAE(input_dim, window_size, horizon)


# ─────────────────────────────────────────────────────────
# ENSEMBLE SCORER (STAGE III)
# ─────────────────────────────────────────────────────────

class EnsembleScorer:
    """
    Compute and aggregate multiple diagnostic scores.

    Components:
      1. Refined forecast residual magnitude
      2. Reconstruction residual magnitude
      3. k-NN latent density score
      4. Latent residual dynamics deviation
      5. Mahalanobis distance from baseline latent distribution
      6. Dispersion of forecast residuals
    """

    def __init__(
        self,
        k: int = 20,
        ewma_span: int = 5,
        epsilon: float = 1e-6,
        score_weights: Optional[List[float]] = None,
    ):
        self.k = k
        self.ewma_span = ewma_span
        self.epsilon = epsilon
        self.score_weights = score_weights

        # Calibration stats (computed on calibration set)
        self.medians: Optional[np.ndarray] = None
        self.iqrs: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.latent_mean: Optional[np.ndarray] = None
        self.latent_cov: Optional[np.ndarray] = None
        self.precision: Optional[np.ndarray] = None
        self.baseline_latents: Optional[np.ndarray] = None

    def fit_calibration(
        self,
        diagnostics: np.ndarray,
        latent_vectors: np.ndarray,
        alpha: float = 0.05,
        use_rank: bool = True,
    ) -> None:
        """
        Fit robust normalization and threshold on calibration data.

        Args:
            diagnostics: (N, n_components) array of raw diagnostic scores
            latent_vectors: (N, latent_dim) array of latent embeddings
            alpha: top fraction to flag
            use_rank: if True, use rank-based cutoff; else fixed threshold
        """
        # Robust standardization stats
        self.medians = np.median(diagnostics, axis=0)
        q75 = np.quantile(diagnostics, 0.75, axis=0)
        q25 = np.quantile(diagnostics, 0.25, axis=0)
        self.iqrs = q75 - q25
        self.iqrs = np.where(self.iqrs < self.epsilon, self.epsilon, self.iqrs)

        # Standardize calibration diagnostics
        standardized = self._standardize(diagnostics)

        # Aggregate
        weights = self._get_weights(standardized.shape[1])
        aggregated = (standardized * weights).sum(axis=1)

        # EWMA smoothing
        smoothed = self._ewma(aggregated)

        # Threshold
        if use_rank:
            self.threshold = np.quantile(smoothed, 1 - alpha)
        else:
            self.threshold = 3.0  # z-score-like

        # Latent baseline for Mahalanobis
        self.latent_mean = latent_vectors.mean(axis=0)
        self.baseline_latents = latent_vectors

        # Shrinkage covariance (Ledoit-Wolf)
        try:
            lw = LedoitWolf().fit(latent_vectors)
            self.precision = lw.precision_
        except Exception:
            # Fallback: diagonal
            var = latent_vectors.var(axis=0)
            var = np.where(var < self.epsilon, self.epsilon, var)
            self.precision = np.diag(1.0 / var)

        logger.info(
            f"Calibration fitted: medians={self.medians.round(4)}, "
            f"threshold={self.threshold:.4f}"
        )

    def compute_diagnostics(
        self,
        model: ReGENTADBackbone,
        X: torch.Tensor,
        Y: torch.Tensor,
        device: str,
    ) -> np.ndarray:
        """
        Compute all 6 diagnostic components for each window.

        Returns:
            diagnostics: (N, 6) numpy array
        """
        model.eval()
        N = len(X)
        all_z = []
        all_f1_err = []
        all_f2_err = []
        all_recon_err = []
        all_resid_vol = []

        batch_size = 128
        with torch.no_grad():
            for i in range(0, N, batch_size):
                xb = X[i : i + batch_size].to(device)
                yb = Y[i : i + batch_size].to(device)
                z, f1, f2, x_hat = model(xb)

                # 1. Refined forecast residual
                f2_err = (f2 - yb).pow(2).mean(dim=(1, 2)).cpu().numpy()
                # 2. Reconstruction residual
                recon_err = (x_hat - xb).pow(2).mean(dim=(1, 2)).cpu().numpy()
                # 3. Latent vectors
                z_np = z.cpu().numpy()
                # 6. Residual volatility (std of forecast errors across features)
                resid_vol = (f2 - yb).std(dim=2).mean(dim=1).cpu().numpy()

                all_f2_err.append(f2_err)
                all_recon_err.append(recon_err)
                all_z.append(z_np)
                all_resid_vol.append(resid_vol)

        f2_err = np.concatenate(all_f2_err)
        recon_err = np.concatenate(all_recon_err)
        z_all = np.vstack(all_z)
        resid_vol = np.concatenate(all_resid_vol)

        # 3. k-NN latent density
        knn_scores = self._knn_density(z_all)

        # 4. Latent dynamics deviation
        latent_dyn = self._latent_dynamics(z_all)

        # 5. Mahalanobis distance
        if self.latent_mean is not None and self.precision is not None:
            mahal = self._mahalanobis(z_all)
        else:
            mahal = np.zeros(N)

        diagnostics = np.column_stack([f2_err, recon_err, knn_scores, latent_dyn, mahal, resid_vol])
        return diagnostics, z_all

    def score(
        self,
        diagnostics: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize, aggregate, and smooth diagnostics.

        Returns:
            final_scores: (N,)
            standardized: (N, n_components)
        """
        standardized = self._standardize(diagnostics)
        weights = self._get_weights(standardized.shape[1])
        aggregated = (standardized * weights).sum(axis=1)
        smoothed = self._ewma(aggregated)
        return smoothed, standardized

    def predict(self, final_scores: np.ndarray) -> np.ndarray:
        """Binary anomaly labels."""
        if self.threshold is None:
            raise RuntimeError("Scorer not calibrated. Call fit_calibration() first.")
        return (final_scores >= self.threshold).astype(int)

    def _standardize(self, diagnostics: np.ndarray) -> np.ndarray:
        if self.medians is None:
            # Fallback: use robust stats from current batch
            med = np.median(diagnostics, axis=0)
            q75 = np.quantile(diagnostics, 0.75, axis=0)
            q25 = np.quantile(diagnostics, 0.25, axis=0)
            iqr = q75 - q25
            iqr = np.where(iqr < self.epsilon, self.epsilon, iqr)
            return np.abs(diagnostics - med) / iqr
        return np.abs(diagnostics - self.medians) / self.iqrs

    def _get_weights(self, n_components: int) -> np.ndarray:
        if self.score_weights is not None:
            w = np.array(self.score_weights, dtype=float)
        else:
            w = np.ones(n_components)
        return w / w.sum()

    def _ewma(self, x: np.ndarray) -> np.ndarray:
        """Exponentially weighted moving average smoothing."""
        span = self.ewma_span
        alpha_ewma = 2.0 / (span + 1)
        result = np.zeros_like(x)
        result[0] = x[0]
        for i in range(1, len(x)):
            result[i] = alpha_ewma * x[i] + (1 - alpha_ewma) * result[i - 1]
        return result

    def _knn_density(self, z: np.ndarray) -> np.ndarray:
        """k-NN distance as density score."""
        if self.baseline_latents is None:
            # Use self as baseline (for calibration phase)
            baseline = z
        else:
            baseline = self.baseline_latents

        # Compute distances to baseline
        dists = cdist(z, baseline, metric="euclidean")
        # Exclude self-distance if same array
        if baseline is z:
            np.fill_diagonal(dists, np.inf)
        # k-th nearest neighbor distance
        k = min(self.k, len(baseline) - 1)
        knn_dist = np.partition(dists, k, axis=1)[:, k]
        return knn_dist

    def _latent_dynamics(self, z: np.ndarray, lag: int = 5) -> np.ndarray:
        """Deviation from recent latent trajectory."""
        if len(z) <= lag:
            return np.zeros(len(z))
        # Rolling mean of recent latents
        dyn = np.zeros(len(z))
        for i in range(lag, len(z)):
            recent_mean = z[i - lag : i].mean(axis=0)
            dyn[i] = np.linalg.norm(z[i] - recent_mean)
        # Fill initial values with 0
        dyn[:lag] = dyn[lag:lag*2].mean() if len(z) > lag * 2 else 0
        return dyn

    def _mahalanobis(self, z: np.ndarray) -> np.ndarray:
        """Mahalanobis distance from baseline latent distribution."""
        diff = z - self.latent_mean
        return np.sqrt(np.sum(diff @ self.precision * diff, axis=1))


# ─────────────────────────────────────────────────────────
# FACTOR ATTRIBUTION
# ─────────────────────────────────────────────────────────

class FactorAttributor:
    """
    Factor-level attribution for detected anomalies.

    Computes:
      Delta_j = |x_bar_j - mu_j| / sigma_j   (baseline deviation)
      Gamma_j = mean gradient of latent anomaly score w.r.t factor j
      C_j = Delta_j * Gamma_j
    """

    def __init__(self, feature_cols: List[str], epsilon: float = 1e-6):
        self.feature_cols = feature_cols
        self.epsilon = epsilon
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None

    def fit(self, X_windows: np.ndarray) -> None:
        """Fit baseline statistics from normal windows."""
        # X_windows: (N, window_size, n_features)
        window_means = X_windows.mean(axis=1)  # (N, n_features)
        self.baseline_mean = window_means.mean(axis=0)
        self.baseline_std = window_means.std(axis=0)
        self.baseline_std = np.where(self.baseline_std < self.epsilon, self.epsilon, self.baseline_std)

    def attribute(
        self,
        x_window: np.ndarray,
        model: ReGENTADBackbone,
        device: str,
    ) -> Dict[str, float]:
        """
        Compute factor attribution for a single anomalous window.

        Args:
            x_window: (window_size, n_features) numpy array

        Returns:
            dict mapping feature name -> attribution score
        """
        model.eval()
        x_t = torch.from_numpy(x_window).float().unsqueeze(0).to(device)  # (1, L, p)
        x_t.requires_grad = True

        # Baseline deviation
        if self.baseline_mean is not None:
            window_mean = x_window.mean(axis=0)
            delta = np.abs(window_mean - self.baseline_mean) / self.baseline_std
        else:
            delta = np.ones(len(self.feature_cols))

        # Latent sensitivity via gradient
        z = model.encode(x_t)
        # Anomaly score = squared distance from zero (proxy)
        score = z.pow(2).sum()
        score.backward()

        grad = x_t.grad[0].cpu().numpy()  # (L, p)
        gamma = np.abs(grad).mean(axis=0)  # average over time

        # Contribution
        contrib = delta * gamma
        # Normalize to sum=1
        total = contrib.sum()
        if total > 0:
            contrib = contrib / total

        return {name: float(v) for name, v in zip(self.feature_cols, contrib)}


# ─────────────────────────────────────────────────────────
# MAIN DETECTOR CLASS
# ─────────────────────────────────────────────────────────

class ReGENTADDetector:
    """
    Production-ready ReGEN-TAD anomaly detector.

    Usage:
        detector = ReGENTADDetector("BBCA.JK", config={...})
        result = detector.detect(df)  # df with OHLCV columns
    """

    def __init__(
        self,
        ticker: str,
        config: Optional[dict] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.ticker = ticker.upper().strip()
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.feature_cols = feature_cols or ["Open", "High", "Low", "Close", "Volume"]

        # Device
        device_cfg = self.cfg["device"]
        if device_cfg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_cfg
        logger.info(f"[{self.ticker}] Using device: {self.device}")

        # Set seed
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

        # State
        self.model: Optional[ReGENTADBackbone] = None
        self.scorer: Optional[EnsembleScorer] = None
        self.attributor: Optional[FactorAttributor] = None
        self.is_trained: bool = False
        self._model_path: Optional[str] = None

    # ─────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, save_path: Optional[str] = None) -> "ReGENTADDetector":
        """
        Train the ReGEN-TAD model on historical data.

        Args:
            df: DataFrame with OHLCV columns, datetime index
            save_path: optional path to save trained model

        Returns:
            self (fluent interface)
        """
        if len(df) < self.cfg["window_size"] + self.cfg["forecast_horizon"] + 50:
            raise ValueError(
                f"Data too short: {len(df)} rows. Need at least "
                f"{self.cfg['window_size'] + self.cfg['forecast_horizon'] + 50} rows."
            )

        # Build windows
        X_all, Y_all, dates = SlidingWindowDataset.build(
            df,
            self.cfg["window_size"],
            self.cfg["forecast_horizon"],
            self.feature_cols,
            stride=self.cfg["stride"],
        )

        n = len(X_all)
        n_train = int(n * self.cfg["train_split"])
        n_val = int(n * self.cfg["val_split"])
        n_test = n - n_train - n_val

        X_train, Y_train = X_all[:n_train], Y_all[:n_train]
        X_val, Y_val = X_all[n_train : n_train + n_val], Y_all[n_train : n_train + n_val]
        X_test, Y_test = X_all[n_train + n_val :], Y_all[n_train + n_val :]

        logger.info(
            f"[{self.ticker}] Data split: train={n_train}, val={n_val}, test={n_test}"
        )

        # ── Stage I: Purification ──
        purifier = Purifier(
            quantile=self.cfg["purify_quantile"],
            max_remove_frac=self.cfg["max_remove_frac"],
            max_iter=self.cfg["purify_max_iter"],
            device=self.device,
        )
        purified_idx = purifier.purify(
            X_train, Y_train,
            len(self.feature_cols),
            self.cfg["window_size"],
            self.cfg["forecast_horizon"],
        )
        X_pure = X_train[purified_idx]
        Y_pure = Y_train[purified_idx]
        logger.info(f"[{self.ticker}] Purified training set: {len(X_pure)} windows")

        # ── Stage II: Train backbone ──
        self.model = ReGENTADBackbone(
            input_dim=len(self.feature_cols),
            window_size=self.cfg["window_size"],
            forecast_horizon=self.cfg["forecast_horizon"],
            d_model=self.cfg["d_model"],
            n_heads=self.cfg["n_heads"],
            dropout=self.cfg["dropout"],
            lstm_hidden=self.cfg["lstm_hidden"],
            latent_dim=self.cfg["latent_dim"],
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        train_loader = DataLoader(
            TensorDataset(X_pure, Y_pure),
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
        )
        val_loader = DataLoader(
            TensorDataset(X_val, Y_val),
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
        )

        best_val_loss = float("inf")
        patience_counter = 0
        weights = self.cfg["loss_weights"]
        latent_reg = self.cfg["latent_reg"]

        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = self.model.compute_loss(xb, yb, weights=weights, latent_reg=latent_reg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    loss = self.model.compute_loss(xb, yb, weights=weights, latent_reg=latent_reg)
                    val_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            scheduler.step(avg_val)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"[{self.ticker}] Epoch {epoch + 1}/{self.cfg['epochs']} "
                    f"train_loss={avg_train:.6f} val_loss={avg_val:.6f}"
                )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.cfg["patience"]:
                    logger.info(f"[{self.ticker}] Early stopping at epoch {epoch + 1}")
                    break

        # Restore best
        self.model.load_state_dict(best_state)
        logger.info(f"[{self.ticker}] Training complete. Best val_loss={best_val_loss:.6f}")

        # ── Stage III: Calibration ──
        self.scorer = EnsembleScorer(
            k=self.cfg["k_neighbors"],
            ewma_span=self.cfg["ewma_span"],
            epsilon=self.cfg["epsilon"],
            score_weights=self.cfg.get("score_weights"),
        )

        # Compute diagnostics on calibration (val) set
        diag_cal, z_cal = self.scorer.compute_diagnostics(self.model, X_val, Y_val, self.device)
        self.scorer.fit_calibration(
            diag_cal, z_cal,
            alpha=self.cfg["alpha"],
            use_rank=self.cfg["use_rank_based"],
        )

        # Attributor
        self.attributor = FactorAttributor(self.feature_cols, epsilon=self.cfg["epsilon"])
        self.attributor.fit(X_pure.numpy())

        self.is_trained = True

        if save_path:
            self._save(save_path)

        return self

    # ─────────────────────────────────────────────
    # DETECTION
    # ─────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        """
        Detect anomalies on new data.

        Args:
            df: DataFrame with same feature columns, datetime index

        Returns:
            DetectionResult with events, scores, and attribution
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        X, Y, dates = SlidingWindowDataset.build(
            df,
            self.cfg["window_size"],
            self.cfg["forecast_horizon"],
            self.feature_cols,
            stride=1,
        )

        if len(X) == 0:
            logger.warning(f"[{self.ticker}] No windows generated from data.")
            return DetectionResult(ticker=self.ticker, events=[], scores=pd.DataFrame())

        # Compute diagnostics
        diagnostics, z_all = self.scorer.compute_diagnostics(self.model, X, Y, self.device)
        final_scores, standardized = self.scorer.score(diagnostics)
        labels = self.scorer.predict(final_scores)

        # Build scores DataFrame
        score_df = pd.DataFrame(
            {
                "forecast_residual": diagnostics[:, 0],
                "reconstruction_error": diagnostics[:, 1],
                "knn_density": diagnostics[:, 2],
                "latent_dynamics": diagnostics[:, 3],
                "mahalanobis": diagnostics[:, 4],
                "residual_volatility": diagnostics[:, 5],
                "final_score": final_scores,
                "anomaly": labels.astype(bool),
            },
            index=dates,
        )

        # Extract anomaly events
        events: List[AnomalyEvent] = []
        anomaly_idx = np.where(labels == 1)[0]

        for idx in anomaly_idx:
            dt = dates[idx]
            score = float(final_scores[idx])
            severity = self._severity_from_score(score, self.scorer.threshold)

            # Attribution
            x_win = X[idx].cpu().numpy() if hasattr(X[idx], "cpu") else X[idx].numpy()
            attr = self.attributor.attribute(x_win, self.model, self.device)

            # Diagnostics for this point
            diag = {
                "forecast_residual": float(diagnostics[idx, 0]),
                "reconstruction_error": float(diagnostics[idx, 1]),
                "knn_density": float(diagnostics[idx, 2]),
                "latent_dynamics": float(diagnostics[idx, 3]),
                "mahalanobis": float(diagnostics[idx, 4]),
                "residual_volatility": float(diagnostics[idx, 5]),
            }

            # Build description
            top_factors = sorted(attr.items(), key=lambda kv: -kv[1])[:3]
            desc = (
                f"ReGEN-TAD detected structural anomaly (score={score:.3f}). "
                f"Top drivers: {', '.join(f'{k}({v:.2f})' for k, v in top_factors)}. "
                f"Diagnostics: recon_err={diag['reconstruction_error']:.4f}, "
                f"mahal={diag['mahalanobis']:.4f}."
            )

            events.append(
                AnomalyEvent(
                    ticker=self.ticker,
                    date=dt.date() if hasattr(dt, "date") else str(dt)[:10],
                    anomaly_type="regen_tad_anomaly",
                    anomaly_score=score,
                    severity=severity,
                    description=desc,
                    method="ReGEN-TAD",
                    attribution=attr,
                    diagnostics=diag,
                )
            )

        # Sort by score descending
        events.sort(key=lambda e: e.anomaly_score, reverse=True)

        logger.info(f"[{self.ticker}] Detected {len(events)} anomalies from {len(X)} windows")
        return DetectionResult(
            ticker=self.ticker,
            events=events,
            scores=score_df,
            model_path=self._model_path,
            calibrated=True,
        )

    # ─────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────

    def _save(self, path: str) -> None:
        """Save model, scorer, and config."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "config": self.cfg,
            "feature_cols": self.feature_cols,
            "scorer": {
                "medians": self.scorer.medians,
                "iqrs": self.scorer.iqrs,
                "threshold": self.scorer.threshold,
                "latent_mean": self.scorer.latent_mean,
                "precision": self.scorer.precision,
                "baseline_latents": self.scorer.baseline_latents,
            },
            "attributor": {
                "baseline_mean": self.attributor.baseline_mean,
                "baseline_std": self.attributor.baseline_std,
            },
        }
        torch.save(payload, path)
        self._model_path = path
        logger.info(f"[{self.ticker}] Model saved to {path}")

    def load(self, path: str) -> "ReGENTADDetector":
        """Load trained model from disk."""
        payload = torch.load(path, map_location=self.device)

        self.cfg = payload["config"]
        self.feature_cols = payload["feature_cols"]

        self.model = ReGENTADBackbone(
            input_dim=len(self.feature_cols),
            window_size=self.cfg["window_size"],
            forecast_horizon=self.cfg["forecast_horizon"],
            d_model=self.cfg["d_model"],
            n_heads=self.cfg["n_heads"],
            dropout=self.cfg["dropout"],
            lstm_hidden=self.cfg["lstm_hidden"],
            latent_dim=self.cfg["latent_dim"],
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

        self.scorer = EnsembleScorer(
            k=self.cfg["k_neighbors"],
            ewma_span=self.cfg["ewma_span"],
            epsilon=self.cfg["epsilon"],
        )
        s = payload["scorer"]
        self.scorer.medians = s["medians"]
        self.scorer.iqrs = s["iqrs"]
        self.scorer.threshold = s["threshold"]
        self.scorer.latent_mean = s["latent_mean"]
        self.scorer.precision = s["precision"]
        self.scorer.baseline_latents = s["baseline_latents"]

        self.attributor = FactorAttributor(self.feature_cols, epsilon=self.cfg["epsilon"])
        a = payload["attributor"]
        self.attributor.baseline_mean = a["baseline_mean"]
        self.attributor.baseline_std = a["baseline_std"]

        self.is_trained = True
        self._model_path = path
        logger.info(f"[{self.ticker}] Model loaded from {path}")
        return self

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _severity_from_score(score: float, threshold: float) -> str:
        ratio = score / max(threshold, 1e-9)
        if ratio >= 2.0:
            return "critical"
        elif ratio >= 1.5:
            return "high"
        elif ratio >= 1.2:
            return "medium"
        else:
            return "low"

    @staticmethod
    def batch_detect(
        tickers: List[str],
        data_fetcher: Callable[[str], pd.DataFrame],
        config: Optional[dict] = None,
        model_dir: str = "./models",
    ) -> Dict[str, DetectionResult]:
        """
        Batch detection for multiple tickers.

        Args:
            tickers: list of ticker symbols
            data_fetcher: callable(ticker) -> DataFrame
            config: shared config override
            model_dir: directory to save/load models

        Returns:
            dict mapping ticker -> DetectionResult
        """
        results = {}
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        for ticker in tickers:
            try:
                df = data_fetcher(ticker)
                if df is None or len(df) == 0:
                    logger.warning(f"[{ticker}] No data fetched, skipping.")
                    results[ticker] = DetectionResult(ticker=ticker, events=[], scores=pd.DataFrame())
                    continue

                model_path = f"{model_dir}/{ticker}_regentad.pt"
                detector = ReGENTADDetector(ticker, config=config)

                if Path(model_path).exists():
                    detector.load(model_path)
                else:
                    detector.fit(df, save_path=model_path)

                result = detector.detect(df)
                results[ticker] = result
            except Exception as e:
                logger.error(f"[{ticker}] Batch detection failed: {e}", exc_info=True)
                results[ticker] = DetectionResult(ticker=ticker, events=[], scores=pd.DataFrame())

        total_events = sum(len(r.events) for r in results.values())
        logger.info(f"Batch complete: {len(tickers)} tickers, {total_events} anomalies total")
        return results

    @staticmethod
    def print_report(batch_result: Dict[str, DetectionResult]) -> None:
        """Print formatted batch report to stdout."""
        print("=" * 70)
        print("  LAPORAN DETEKSI ANOMALI — ReGEN-TAD")
        print("=" * 70)
        total = 0
        for ticker, result in batch_result.items():
            if not result.events:
                print(f"\n  {ticker}: Tidak ada anomali")
                continue
            print(f"\n  {ticker}: {len(result.events)} anomali")
            print("  " + "-" * 60)
            for ev in result.events:
                print("  " + ev.summary().replace("\n", "\n  "))
            total += len(result.events)
        print(f"\n{'=' * 70}")
        print(f"  Total anomali: {total}")
        print("=" * 70)
