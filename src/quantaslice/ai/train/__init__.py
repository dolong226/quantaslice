"""Huấn luyện detector (§5): baseline gradient boosting và TCN đa nhiệm.

``deep`` và ``calibrate`` import torch bên trong (không ở tầng module
này) để ``import quantaslice.ai.train`` không ép phụ thuộc torch cho
người chỉ dùng baseline."""

from quantaslice.ai.train.baseline import TrainResult, run_training

__all__ = ["TrainResult", "run_training", "run_training_tcn", "DeepTrainResult"]


def __getattr__(name: str):  # lazy: chỉ nạp torch khi thực sự dùng deep
    if name in ("run_training_tcn", "DeepTrainResult"):
        from quantaslice.ai.train import deep
        return getattr(deep, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
