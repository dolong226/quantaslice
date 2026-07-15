"""Metric đánh giá — §6 của plan ML.

Nguyên tắc §6.1: TRÁNH point-adjusted F1 (đã được chứng minh thổi phồng
điểm, đôi khi cả với dự đoán ngẫu nhiên). Ở foundation dùng:

* **PR-AUC (average precision)** — threshold-free, bất biến với imbalance,
  là metric chính để so baseline vs deep model (§6.6 ablation).
* **ECE** (Expected Calibration Error) — vì flag là TRIGGER, sai-hiệu-chuẩn
  nguy hiểm hơn kém-chính-xác-nhưng-hiệu-chuẩn-đúng (§6.2).
* **Precision/Recall/F1 tại ngưỡng** đã chốt — để báo cáo vận hành.

VUS-PR/ROC và affiliation-based (§6.1) là mốc nâng cấp sau; PR-AUC +
block-split (§3.5) đã đủ để foundation không bị ảo giác kết quả.

Cost-sensitive threshold (§5.2): miss ≫ false alarm, nên chọn ngưỡng
theo ma trận chi phí thay vì mặc định 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DetectionMetrics", "evaluate_flag", "cost_sensitive_threshold",
           "expected_calibration_error", "lead_time_recall"]


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    ece: float
    threshold: float
    positive_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "pr_auc": self.pr_auc, "roc_auc": self.roc_auc,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "ece": self.ece, "threshold": self.threshold,
            "positive_rate": self.positive_rate,
        }


def evaluate_flag(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> DetectionMetrics:
    """Tính bộ metric phát hiện tại một ngưỡng cho trước."""
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y = y_true.astype(int)
    pos_rate = float(y.mean()) if y.size else 0.0
    # PR/ROC-AUC cần cả 2 lớp; nếu chỉ 1 lớp -> không xác định, trả NaN.
    if 0 < y.sum() < y.size:
        pr_auc = float(average_precision_score(y, y_prob))
        roc_auc = float(roc_auc_score(y, y_prob))
    else:
        pr_auc = roc_auc = float("nan")

    pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    ece = expected_calibration_error(y, y_prob)
    return DetectionMetrics(
        pr_auc=pr_auc, roc_auc=roc_auc, precision=float(prec), recall=float(rec),
        f1=float(f1), ece=ece, threshold=float(threshold), positive_rate=pos_rate,
    )


def cost_sensitive_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, *, cost_miss: float = 10.0, cost_fa: float = 1.0
) -> float:
    """Chọn ngưỡng tối thiểu hoá chi phí kỳ vọng ``c_miss·FN + c_fa·FP``.

    Mặc định ``cost_miss=10 ≫ cost_fa=1``: bỏ sót emergency đắt hơn nhiều
    một báo động giả (§0, §5.2). Trả 0.5 nếu chỉ có 1 lớp."""
    y = y_true.astype(int)
    if not (0 < y.sum() < y.size):
        return 0.5
    candidates = np.unique(np.concatenate([y_prob, [0.0, 1.0]]))
    best_thr, best_cost = 0.5, np.inf
    for thr in candidates:
        pred = y_prob >= thr
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        cost = cost_miss * fn + cost_fa * fp
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)
    return best_thr


def lead_time_recall(
    future_positive: np.ndarray,
    currently_breached: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> float:
    """Recall trên các bước "SẮP vi phạm (trong H bước tới) nhưng HIỆN
    TẠI chưa vi phạm" — thước đo TRUNG THỰC giá trị của detector: khả năng
    bắt SỚM (lead-time §6.3). Luật "đang vi phạm chưa?" theo định nghĩa
    đạt ~0 ở đây; một model học precursor mới ghi điểm.

    Trả NaN nếu không có bước onset nào (không có precursor để dự báo)."""
    onset = future_positive.astype(bool) & (~currently_breached.astype(bool))
    if onset.sum() == 0:
        return float("nan")
    return float(((scores >= threshold) & onset).sum() / onset.sum())


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: |accuracy - confidence| trung bình có trọng số trên các bin
    xác suất (§6.2)."""
    y = y_true.astype(int)
    if y.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y[mask].mean())
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)
