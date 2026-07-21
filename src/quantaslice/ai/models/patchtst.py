"""PatchTST multi-task detector.

PatchTST (Patch Time Series Transformer) is a state-of-the-art Transformer-based model
for time series. It utilizes:
1. Channel Independence: Treat each time series feature channel as an independent batch item.
2. Patching: Extract local semantic patches along the time dimension to capture local correlations
   and reduce Transformer sequence length.
3. Transformer Encoder: Apply multi-head self-attention across the patches.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from quantaslice.core.exceptions import SolverError
from quantaslice.ai.models.tcn import _sigmoid_stable

__all__ = ["MultiTaskPatchTST", "PatchTSTDetector"]


class PatchTSTEncoder(nn.Module):
    """PatchTST Encoder featuring channel independence, sequence patching, and Transformer encoding."""

    def __init__(
        self,
        seq_len: int,
        patch_len: int = 8,
        stride: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Calculate number of patches and padding needed to cover the sequence
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.padding_len = seq_len - ((self.num_patches - 1) * stride + patch_len)
        if self.padding_len > 0:
            self.num_patches += 1
            self.padding_len = (self.num_patches - 1) * stride + patch_len - seq_len

        self.proj = nn.Linear(patch_len, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        
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
        # x shape: (B, W, M) where M is n_features
        B, W, M = x.shape

        # 1. Channel Independence: transpose and reshape to (B * M, W)
        x = x.transpose(1, 2)  # (B, M, W)
        x = x.reshape(B * M, W)  # (B * M, W)

        # 2. Replicate pad sequence if it does not fit the patch stride perfectly
        if self.padding_len > 0:
            x = nn.functional.pad(x, (self.padding_len, 0), "replicate")

        # 3. Extract overlapping patches: (B * M, num_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # 4. Project and add positional embeddings
        enc = self.proj(patches)  # (B * M, num_patches, hidden_dim)
        enc = enc + self.pos_embed
        enc = self.dropout(enc)

        # 5. Attention across patches
        out = self.transformer(enc)  # (B * M, num_patches, hidden_dim)

        # 6. Reshape back: (B, M, num_patches, hidden_dim)
        out = out.reshape(B, M, self.num_patches, -1)
        return out


class MultiTaskPatchTST(nn.Module):
    """Shared PatchTST encoder + 2 heads (flag, priority)."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 8,
        stride: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = PatchTSTEncoder(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.pooling = pooling
        
        # Determine representation dimension after pooling
        # If mean/last pooling over patches, dimension is n_features * hidden_dim
        # If flattening, dimension is n_features * num_patches * hidden_dim
        num_patches = self.encoder.num_patches
        self.flat_dim = n_features * hidden_dim if pooling in ("mean", "last") else n_features * num_patches * hidden_dim
        
        self.flag_head = nn.Linear(self.flat_dim, 1)
        self.prio_head = nn.Linear(self.flat_dim, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, W, F)
        enc = self.encoder(x)  # (B, F, num_patches, hidden_dim)

        if self.pooling == "mean":
            h = enc.mean(dim=2)  # (B, F, hidden_dim)
            h = h.reshape(h.size(0), -1)  # (B, F * hidden_dim)
        elif self.pooling == "last":
            h = enc[:, :, -1, :]  # (B, F, hidden_dim)
            h = h.reshape(h.size(0), -1)  # (B, F * hidden_dim)
        else:
            h = enc.reshape(enc.size(0), -1)  # (B, F * num_patches * hidden_dim)

        flag_logit = self.flag_head(h).squeeze(-1)
        priority = nn.functional.softplus(self.prio_head(h))
        return flag_logit, priority


class PatchTSTDetector:
    """Wrapper around MultiTaskPatchTST to match QuantaSlice runtime interface."""

    backend_name = "patchtst"

    def __init__(self, model: MultiTaskPatchTST, *, lookback: int) -> None:
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
    def load(path: str) -> PatchTSTDetector:
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, PatchTSTDetector):
            raise SolverError(f"Artifact at {path} is not a PatchTSTDetector")
        return obj
