"""Model cho ML detector: GradientBoostingDetector (baseline §4) và TCN
đa nhiệm (§4 Tier A primary). Cả hai phơi bày cùng giao diện
``predict_windows(seq)`` để runtime dùng thống nhất.

KHÔNG import ``tcn`` ở đây để không ép phụ thuộc torch cho mọi người
dùng baseline; ``load_detector`` unpickle sẽ tự nạp lớp cần thiết.
"""

from quantaslice.ai.models.baselines import GradientBoostingDetector
from quantaslice.ai.models.arima import GlobalARDetector, LocalARIMADetector

__all__ = ["GradientBoostingDetector", "GlobalARDetector", "LocalARIMADetector", "load_detector"]


def load_detector(path: str) -> object:
    """Nạp bất kỳ detector nào (baseline hoặc TCN) từ artifact joblib.

    Duck-typing: chỉ yêu cầu có ``predict_windows``. Nếu artifact là TCN,
    unpickle sẽ tự import ``quantaslice.ai.models.tcn`` (cần torch)."""
    import joblib

    from quantaslice.core.exceptions import ConfigurationError

    obj = joblib.load(path)
    if not hasattr(obj, "predict_windows"):
        raise ConfigurationError(
            f"Artifact tại {path} không phải detector hợp lệ (thiếu predict_windows)"
        )
    return obj
