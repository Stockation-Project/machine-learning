"""
ReGEN-TAD Model Backbone
========================
PyTorch implementation of the generative backbone for ReGEN-TAD.

Architecture:
  - Conv1D front-end (2 layers, 64 filters, kernel=3)
  - Positional Encoding
  - Transformer Encoder (6 heads, d_model=128, dropout=0.1)
  - BiLSTM (32 hidden per direction)
  - Latent projection (dim=128)
  - Forecast head (initial + refined)
  - Reconstruction head

Reference: Martinez (2026) — ReGEN-TAD Framework
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal ordering."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class ConvFrontEnd(nn.Module):
    """Temporal convolutional front-end."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x.transpose(1, 2)  # back to (batch, seq_len, features)


class ReGENTADBackbone(nn.Module):
    """
    Generative backbone for ReGEN-TAD.

    Args:
        input_dim: Number of input features (e.g., OHLCV = 5)
        window_size: Length of input window (L)
        forecast_horizon: Number of steps to forecast (H)
        d_model: Embedding dimension (default 128)
        n_heads: Transformer heads (default 6)
        dropout: Dropout rate (default 0.1)
        lstm_hidden: BiLSTM hidden dim per direction (default 32)
        latent_dim: Final latent dimension (default 128)
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        forecast_horizon: int,
        d_model: int = 128,
        n_heads: int = 6,
        dropout: float = 0.1,
        lstm_hidden: int = 32,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.latent_dim = latent_dim

        # --- Front-end ---
        self.conv_frontend = ConvFrontEnd(
            in_channels=input_dim,
            out_channels=64,
            kernel_size=3,
            dropout=dropout,
        )
        self.input_proj = nn.Linear(64, d_model)

        # --- Positional Encoding ---
        self.pos_enc = PositionalEncoding(d_model, max_len=window_size + 100, dropout=dropout)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- BiLSTM ---
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- Latent projection ---
        lstm_out_dim = lstm_hidden * 2
        self.latent_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Forecast heads ---
        # Initial forecast
        self.forecast_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, forecast_horizon * input_dim),
        )
        # Refinement network
        self.refine_net = nn.Sequential(
            nn.Linear(latent_dim + forecast_horizon * input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, forecast_horizon * input_dim),
        )

        # --- Reconstruction head ---
        self.recon_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, window_size * input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input window to latent representation.

        Args:
            x: (batch, window_size, input_dim)

        Returns:
            z: (batch, latent_dim)
        """
        # Conv front-end
        x = self.conv_frontend(x)  # (batch, window_size, 64)
        x = self.input_proj(x)     # (batch, window_size, d_model)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer
        x = self.transformer(x)    # (batch, window_size, d_model)

        # BiLSTM
        lstm_out, _ = self.lstm(x)   # (batch, window_size, lstm_hidden*2)

        # Pooling: mean over time + last timestep
        mean_pool = lstm_out.mean(dim=1)
        last_pool = lstm_out[:, -1, :]
        pooled = mean_pool + last_pool

        # Latent projection
        z = self.latent_proj(pooled)  # (batch, latent_dim)
        return z

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning all outputs.

        Args:
            x: (batch, window_size, input_dim)

        Returns:
            z: latent representation (batch, latent_dim)
            f1: initial forecast (batch, forecast_horizon, input_dim)
            f2: refined forecast (batch, forecast_horizon, input_dim)
            x_hat: reconstruction (batch, window_size, input_dim)
        """
        z = self.encode(x)

        # Initial forecast
        f1_flat = self.forecast_head(z)
        f1 = f1_flat.view(-1, self.forecast_horizon, self.input_dim)

        # Refined forecast
        refine_input = torch.cat([z, f1_flat], dim=-1)
        f2_flat = self.refine_net(refine_input)
        f2 = f2_flat.view(-1, self.forecast_horizon, self.input_dim)

        # Reconstruction
        x_hat_flat = self.recon_head(z)
        x_hat = x_hat_flat.view(-1, self.window_size, self.input_dim)

        return z, f1, f2, x_hat

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights: Tuple[float, float, float] = (0.2, 0.8, 0.5),
        latent_reg: float = 0.0,
    ) -> torch.Tensor:
        """
        Composite training loss.

        Args:
            x: input window (batch, window_size, input_dim)
            y: target future window (batch, forecast_horizon, input_dim)
            weights: (w1, w2, w_r) for initial forecast, refined forecast, reconstruction
            latent_reg: L2 regularization on latent space

        Returns:
            loss scalar
        """
        z, f1, f2, x_hat = self.forward(x)
        w1, w2, wr = weights

        loss_f1 = F.mse_loss(f1, y)
        loss_f2 = F.mse_loss(f2, y)
        loss_recon = F.mse_loss(x_hat, x)

        loss = w1 * loss_f1 + w2 * loss_f2 + wr * loss_recon
        if latent_reg > 0:
            loss += latent_reg * z.pow(2).mean()
        return loss
