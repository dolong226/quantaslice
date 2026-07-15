"""TCN multi-task detector — §4 Tier A (primary) của plan ML.

TCN (dilated causal convolution) là ứng viên production ưu tiên: song
song hoá được (nhanh hơn RNN khi inference), receptive field dài qua
dilation, latency ổn định — hợp near-RT xApp. Kiến trúc đa nhiệm (§0,
§4): shared encoder + 2 head chia sẻ biểu diễn::

    KPM window (W, F) ─► TCN encoder ─► rep (C)
                                        ├─► flag head  (logit, sigmoid)
                                        └─► priority head (softplus, 3-dim ≥0)

``TCNDetector`` bọc model + scaler + nhiệt độ hiệu chỉnh + ngưỡng, phơi
bày CÙNG giao diện ``predict_windows(seq)`` như GradientBoostingDetector
để ``MLPredictionProvider`` dùng thống nhất (nhận cửa sổ thô (N, W, F)).

Module này import torch ở top -> chỉ nạp khi thực sự cần TCN (train deep
hoặc load artifact TCN); package ``ai`` KHÔNG import nó ở tầng __init__.
CHỈ import từ ``core`` + ``ai`` nội bộ + torch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from quantaslice.core.exceptions import SolverError

__all__ = ["MultiTaskTCN", "TCNDetector"]


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Sigmoid ổn định số (tránh overflow exp với logit lớn)."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class _Chomp1d(nn.Module):
    """Cắt phần padding thừa bên phải để giữ tính NHÂN QUẢ (causal): output
    tại bước t chỉ phụ thuộc input ≤ t."""

    def __init__(self, chomp: int) -> None:
        super().__init__()
        self._chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self._chomp] if self._chomp > 0 else x


class _TemporalBlock(nn.Module):
    """Khối TCN: 2 lớp conv1d giãn nở (dilated) + residual."""

    def __init__(self, in_c: int, out_c: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_c, out_c, kernel, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MultiTaskTCN(nn.Module):
    """Encoder TCN dùng chung + 2 head (flag, priority)."""

    def __init__(
        self, n_features: int, channels: tuple[int, ...] = (32, 32),
        kernel: int = 3, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_c = n_features
        for i, ch in enumerate(channels):
            layers.append(_TemporalBlock(in_c, ch, kernel, dilation=2**i, dropout=dropout))
            in_c = ch
        self.encoder = nn.Sequential(*layers)
        self.flag_head = nn.Linear(in_c, 1)
        self.prio_head = nn.Linear(in_c, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (N, F, W). Lấy biểu diễn tại bước CUỐI (nowcast).
        h = self.encoder(x)[:, :, -1]
        flag_logit = self.flag_head(h).squeeze(-1)          # (N,)
        priority = nn.functional.softplus(self.prio_head(h))  # (N, 3) ≥ 0
        return flag_logit, priority


class TCNDetector:
    """Bọc ``MultiTaskTCN`` để dùng như một detector runtime.

    Vòng đời: ``ai.train.deep.run_training_tcn`` fit model + scaler +
    temperature + threshold rồi gán vào đây; runtime gọi
    ``predict_windows``.
    """

    backend_name = "tcn"

    def __init__(self, model: MultiTaskTCN, *, lookback: int) -> None:
        self.model = model.eval()
        self.scaler = None            # sklearn RobustScaler (fit trên feature thô)
        self.temperature: float = 1.0  # nhiệt độ hiệu chỉnh (§5.2)
        self.threshold: float = 0.5
        self.window_lookback: int = lookback
        self.ts_feature_names: tuple[str, ...] = ()

    # ── Suy luận ──────────────────────────────────────────────────────
    def _prep(self, seq: np.ndarray) -> torch.Tensor:
        """(N, W, F) -> tensor (N, F, W) đã scale per-feature."""
        n, w, f = seq.shape
        flat = seq.reshape(n * w, f)
        if self.scaler is not None:
            flat = self.scaler.transform(flat)
        x = flat.reshape(n, w, f).transpose(0, 2, 1)  # (N, F, W)
        return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)

    @torch.no_grad()
    def logits(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Trả (flag_logit thô, priority) — logit CHƯA chia nhiệt độ, dùng
        cho hiệu chỉnh (§5.2)."""
        if seq.shape[0] == 0:
            return np.empty(0), np.empty((0, 3))
        flag_logit, prio = self.model(self._prep(seq))
        return flag_logit.cpu().numpy(), prio.cpu().numpy()

    def predict_windows(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Giao diện thống nhất với baseline: (N, W, F) -> (probs (N,),
        priorities (N, 3)). Áp nhiệt độ hiệu chỉnh vào logit."""
        flag_logit, prio = self.logits(seq)
        probs = _sigmoid_stable(flag_logit / max(self.temperature, 1e-3))
        return probs, np.clip(prio, 0.0, None)

    # ── Lưu / nạp ─────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        import joblib
        self.model = self.model.eval()
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TCNDetector":
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, TCNDetector):
            raise SolverError(f"Artifact tại {path} không phải TCNDetector")
        return obj
