"""``quantaslice.ai`` — package phát hiện emergency (ML detector).

Bề mặt public: ``provider_registry`` (đúng pattern
``quantum.solver_registry`` / ``orchestrator.orchestrator_registry``).
Detector thật đăng ký dưới tên ``"ml"``; ``DependencyContainer`` tra
theo tên lấy từ ``Configuration.prediction_provider`` — đổi 1 dòng config
là chuyển từ bootstrap provider (mock/threshold) sang ML thật, KHÔNG sửa
``Runner``.

Ràng buộc kiến trúc: package này CHỈ import từ ``quantaslice.core`` —
KHÔNG import ``quantum``/``orchestrator``/``pipeline``.

Provider ``"ml"`` cần đường dẫn artifact (model đã train). Lấy theo thứ
tự: ``Configuration.extra["ml_artifact"]`` -> biến môi trường
``QUANTASLICE_ML_ARTIFACT``. Huấn luyện artifact bằng
``python -m quantaslice.ai.train.baseline``.
"""

from __future__ import annotations

import os

from quantaslice.core.exceptions import ConfigurationError
from quantaslice.core.protocols import PredictionProvider
from quantaslice.core.registry import Registry

__all__ = ["provider_registry"]

provider_registry: Registry[PredictionProvider] = Registry("prediction_provider")


def _make_ml_provider(config: object | None = None) -> PredictionProvider:
    """Factory cho provider ``"ml"``: nạp detector từ artifact."""
    artifact = None
    if config is not None:
        artifact = getattr(config, "extra", {}).get("ml_artifact")
    artifact = artifact or os.environ.get("QUANTASLICE_ML_ARTIFACT")
    if not artifact:
        raise ConfigurationError(
            "Provider 'ml' cần đường dẫn artifact: đặt Configuration.extra['ml_artifact'] "
            "hoặc biến môi trường QUANTASLICE_ML_ARTIFACT (train bằng "
            "python -m quantaslice.ai.train.baseline)."
        )
    from quantaslice.ai.provider import MLPredictionProvider

    return MLPredictionProvider(artifact)


provider_registry.register_instance_factory("ml", _make_ml_provider)
