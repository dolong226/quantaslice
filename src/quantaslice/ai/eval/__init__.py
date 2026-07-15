"""Đánh giá detector (§6): PR-AUC, ECE, cost-sensitive threshold."""

from quantaslice.ai.eval.metrics import (
    DetectionMetrics,
    cost_sensitive_threshold,
    evaluate_flag,
    expected_calibration_error,
)

__all__ = [
    "DetectionMetrics",
    "evaluate_flag",
    "cost_sensitive_threshold",
    "expected_calibration_error",
]
