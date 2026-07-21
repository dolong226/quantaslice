"""TSMixer multi-task detector.

TSMixer (Time-Series Mixer) is a lightweight MLP-based architecture that mixes features
across the temporal dimension and the feature dimension. It is extremely fast to train
and run compared to Transformers, while maintaining strong performance on time-series.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from quantaslice.core.exceptions import SolverError
from quantaslice.ai.models.tcn import _sigmoid_stable

__all__ = ["MultiTaskTSMixer", "TSMixerDetector"]


class TSMixerBlock(nn.Module):
    """TSMixer block consisting of temporal and feature-wise MLP mixing."""

    def __init__(self, seq_len: int, n_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(n_features)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.feature_norm = nn.LayerNorm(n_features)
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, W, F)
        # 1. Temporal mixing
        res = x
        x = self.temporal_norm(x)
        x = x.transpose(1, 2)  # (N, F, W)
        x = self.temporal_mlp(x)
        x = x.transpose(1, 2)  # (N, W, F)
        x = x + res

        # 2. Feature mixing
        res = x
        x = self.feature_norm(x)
        x = self.feature_mlp(x)
        x = x + res
        return x


class MultiTaskTSMixer(nn.Module):
    """Shared TSMixer encoder + 2 heads (flag, priority)."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.blocks = nn.ModuleList([
            TSMixerBlock(seq_len, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.pooling = pooling
        self.flat_dim = hidden_dim if pooling in ("mean", "last") else seq_len * hidden_dim
        self.flag_head = nn.Linear(self.flat_dim, 1)
        self.prio_head = nn.Linear(self.flat_dim, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (N, W, F)
        x = self.input_proj(x)  # (N, W, hidden_dim)
        for block in self.blocks:
            x = block(x)

        if self.pooling == "mean":
            h = x.mean(dim=1)
        elif self.pooling == "last":
            h = x[:, -1, :]
        else:
            h = x.reshape(x.size(0), -1)

        flag_logit = self.flag_head(h).squeeze(-1)
        priority = nn.functional.softplus(self.prio_head(h))
        return flag_logit, priority


class TSMixerDetector:
    """Wrapper around MultiTaskTSMixer to match QuantaSlice's runtime detector interface."""

    backend_name = "tsmixer"

    def __init__(self, model: MultiTaskTSMixer, *, lookback: int) -> None:
        self.model = model.eval()
        self.scaler = None
        self.temperature: float = 1.0
        self.threshold: float = 0.5
        self.window_lookback: int = lookback
        self.ts_feature_names: tuple[str, ...] = ()

    def _prep(self, seq: np.ndarray) -> torch.Tensor:
        """Scale sequence features and prepare as PyTorch tensor."""
        n, w, f = seq.shape
        flat = seq.reshape(n * w, f)
        if self.scaler is not None:
            flat = self.scaler.transform(flat)
        x = flat.reshape(n, w, f)  # (N, W, F)
        return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)

    @torch.no_grad()
    def logits(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute raw logit predictions."""
        if seq.shape[0] == 0:
            return np.empty(0), np.empty((0, 3))
        flag_logit, prio = self.model(self._prep(seq))
        return flag_logit.cpu().numpy(), prio.cpu().numpy()

    def predict_windows(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict probabilities and priorities for input windows."""
        flag_logit, prio = self.logits(seq)
        probs = _sigmoid_stable(flag_logit / max(self.temperature, 1e-3))
        return probs, np.clip(prio, 0.0, None)

    def save(self, path: str) -> None:
        import joblib
        self.model = self.model.eval()
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> TSMixerDetector:
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, TSMixerDetector):
            raise SolverError(f"Artifact at {path} is not a TSMixerDetector")
        return obj
