"""Hiệu chỉnh xác suất (calibration) — §5.2 của plan ML.

Vì flag là TRIGGER, sai-hiệu-chuẩn nguy hiểm hơn kém-chính-xác-nhưng-
hiệu-chuẩn-đúng (§6.2). **Temperature scaling** (Guo et al. 2017): học
một vô hướng ``T > 0`` chia vào logit để tối thiểu hoá NLL trên tập val,
KHÔNG đổi thứ hạng (nên không đổi ROC/PR-AUC) mà chỉ nắn lại độ tin cậy
-> giảm ECE.

Tách riêng khỏi ``deep.py`` để tái dùng cho mọi model có logit (TCN, và
Transformer về sau).
"""

from __future__ import annotations

import numpy as np

__all__ = ["temperature_scale"]


def temperature_scale(
    logits: np.ndarray, labels: np.ndarray, *, max_iter: int = 200, lr: float = 0.05
) -> float:
    """Tìm nhiệt độ ``T > 0`` tối thiểu hoá BCE(logits / T, labels) trên
    val. Trả 1.0 nếu không đủ dữ liệu hoặc chỉ một lớp (không hiệu chỉnh
    được)."""
    import torch

    y = labels.astype(np.float32)
    if y.size == 0 or not (0 < y.sum() < y.size):
        return 1.0

    z = torch.as_tensor(logits, dtype=torch.float32)
    t_labels = torch.as_tensor(y, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) > 0
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def _closure() -> "torch.Tensor":
        optimizer.zero_grad()
        loss = loss_fn(z / torch.exp(log_t), t_labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    temperature = float(torch.exp(log_t).item())
    # Chặn giá trị bất thường (phân kỳ) -> lùi về 1.0 an toàn.
    if not np.isfinite(temperature) or temperature <= 0:
        return 1.0
    return float(np.clip(temperature, 0.05, 100.0))
