"""Registry cho các "trục biến thiên" mà package ``ai`` (chưa tồn tại
dưới dạng package riêng) sẽ implement.

Tạm thời đặt ở đây (pipeline) làm nơi trú tạm — đúng tinh thần "hệ thống
phải chạy được trước khi LSTM thật tồn tại". Khi Member A hoàn thiện
package ``quantaslice.ai``, họ chỉ cần định nghĩa registry riêng theo
đúng pattern ``quantum.solver_registry`` / ``orchestrator.orchestrator_registry``
và ``DependencyContainer`` chuyển sang trỏ tới registry đó — ``Runner``
không cần đổi gì.

Lưu ý: registry cho orchestrator KHÔNG còn nằm ở đây nữa — đã chuyển
sang ``quantaslice.orchestrator.orchestrator_registry`` (package riêng,
xem lịch sử: trước đây từng tạm trú ở pipeline như prediction_provider
hiện tại).
"""

from __future__ import annotations

from quantaslice.core.protocols import PredictionProvider
from quantaslice.core.registry import Registry

__all__ = ["prediction_provider_registry"]

prediction_provider_registry: Registry[PredictionProvider] = Registry("prediction_provider")
