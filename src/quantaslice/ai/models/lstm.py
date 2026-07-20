"""LSTM đa nhiệm — baseline chuỗi kinh điển (plan gốc "LSTM detector").

Plan §4 đã chuyển "beyond LSTM" sang TCN/GRU/Transformer vì LSTM không
còn là mặc định tốt nhất, nhưng LSTM vẫn là mốc so sánh bắt buộc. Module
này mirror ``models/tcn.py``: shared LSTM encoder + 2 head (flag +
priority), phơi bày CÙNG giao diện ``predict_windows(seq (N,W,F))`` nên
là drop-in thay TCN trong toàn pipeline.

Khác TCN ở tensor layout: LSTM dùng ``(N, W, F)`` (batch_first), TCN dùng
``(N, F, W)``. Import torch ở top -> chỉ nạp khi thực sự dùng.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from quantaslice.core.exceptions import SolverError
from quantaslice.ai.models.tcn import _sigmoid_stable

__all__ = ["MultiTaskLSTM", "LSTMDetector"]


class MultiTaskLSTM(nn.Module):
    """Encoder LSTM dùng chung + 2 head (flag, priority)."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.flag_head = nn.Linear(hidden, 1)
        self.prio_head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (N, W, F). Lấy biểu diễn tại bước CUỐI (nowcast/forecast).
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        flag_logit = self.flag_head(h_last).squeeze(-1)
        priority = nn.functional.softplus(self.prio_head(h_last))
        return flag_logit, priority


class LSTMDetector:
    """Bọc ``MultiTaskLSTM`` — cùng giao diện với ``TCNDetector``."""

    backend_name = "lstm"

    def __init__(self, model: MultiTaskLSTM, *, lookback: int) -> None:
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
        x = flat.reshape(n, w, f)  # (N, W, F) — batch_first
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
    def load(path: str) -> "LSTMDetector":
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, LSTMDetector):
            raise SolverError(f"Artifact tại {path} không phải LSTMDetector")
        return obj
