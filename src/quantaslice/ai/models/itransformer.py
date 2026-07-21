"""iTransformer multi-task detector.

iTransformer (inverted Transformer) is a state-of-the-art Transformer-based model
for multivariate time series forecasting. It utilizes:
1. Inverted Tokenization: Treat the entire sequence of each individual variable/channel
   as a token (instead of time steps as tokens).
2. Time-Dimension Projection: Project the lookback sequence length W into a hidden representation D.
3. Multi-Head Attention: Apply self-attention across the variable tokens (correlations between features).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from quantaslice.core.exceptions import SolverError
from quantaslice.ai.models.tcn import _sigmoid_stable

__all__ = ["MultiTaskiTransformer", "iTransformerDetector"]


class iTransformerEncoder(nn.Module):
    """iTransformer Encoder treating independent channels as tokens."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Project sequence length W of each channel to hidden_dim
        self.proj = nn.Linear(seq_len, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, W, M)
        B, W, M = x.shape

        # 1. Transpose to (B, M, W) to treat each channel as a token
        x = x.transpose(1, 2)

        # 2. Project W dimension to hidden_dim
        enc = self.proj(x)  # (B, M, hidden_dim)
        enc = enc + self.pos_embed
        enc = self.dropout(enc)

        # 3. Apply Transformer attention across variable tokens (M)
        out = self.transformer(enc)  # (B, M, hidden_dim)
        return out


class MultiTaskiTransformer(nn.Module):
    """Shared iTransformer encoder + 2 heads (flag, priority)."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = iTransformerEncoder(
            seq_len=seq_len,
            n_features=n_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.pooling = pooling
        self.flat_dim = hidden_dim if pooling in ("mean", "last") else n_features * hidden_dim
        self.flag_head = nn.Linear(self.flat_dim, 1)
        self.prio_head = nn.Linear(self.flat_dim, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, W, F)
        enc = self.encoder(x)  # (B, F, hidden_dim)

        if self.pooling == "mean":
            h = enc.mean(dim=1)  # (B, hidden_dim)
        elif self.pooling == "last":
            h = enc[:, -1, :]  # (B, hidden_dim)
        else:
            h = enc.reshape(enc.size(0), -1)  # (B, F * hidden_dim)

        flag_logit = self.flag_head(h).squeeze(-1)
        priority = nn.functional.softplus(self.prio_head(h))
        return flag_logit, priority


class iTransformerDetector:
    """Wrapper around MultiTaskiTransformer to match QuantaSlice runtime interface."""

    backend_name = "itransformer"

    def __init__(self, model: MultiTaskiTransformer, *, lookback: int) -> None:
        self.model = model.eval()
        self.scaler = None
        self.temperature: float = 1.0
        self.threshold: float = 0.5
        self.window_lookback: int = lookback
        self.ts_feature_names: tuple[str, ...] = ()

    def _prep(self, seq: np.ndarray) -> torch.Tensor:
        n, w, f = seq.shape
        flat = seq.reshape(n * w, f)
        if self.scaler is not None:
            flat = self.scaler.transform(flat)
        x = flat.reshape(n, w, f)  # (N, W, F)
        return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)

    @torch.no_grad()
    def logits(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if seq.shape[0] == 0:
            return np.empty(0), np.empty((0, 3))
        flag_logit, prio = self.model(self._prep(seq))
        return flag_logit.cpu().numpy(), prio.cpu().numpy()

    def predict_windows(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flag_logit, prio = self.logits(seq)
        probs = _sigmoid_stable(flag_logit / max(self.temperature, 1e-3))
        return probs, np.clip(prio, 0.0, None)

    def save(self, path: str) -> None:
        import joblib
        self.model = self.model.eval()
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> iTransformerDetector:
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, iTransformerDetector):
            raise SolverError(f"Artifact at {path} is not an iTransformerDetector")
        return obj
