"""Orchestrator chạy trên hạ tầng OpenRAN thật qua E2 interface
(E2SM-RC) — theo slide System Architecture: "Pushes PRB reassignment
via O-RAN E2 interface", "Enforces URLLC QoS (PDCP/RLC params)".

Hiện là placeholder — cùng implement
:class:`~quantaslice.core.protocols.SliceOrchestratorPort` như
``MockOranOrchestrator`` để có thể hoán đổi qua registry (đổi 1 dòng
config ``orchestrator: "e2_interface"``) mà KHÔNG cần sửa
``quantaslice.pipeline``, khi hạ tầng O-RAN thật sẵn sàng (Roadmap
Phase II, M7-M12: "Deploy on real O-RAN gNB emulator").
"""

from __future__ import annotations

from quantaslice.core.exceptions import QuantaSliceError
from quantaslice.core.types import OptimizationResult

__all__ = ["E2Interface"]


class E2Interface:
    """Placeholder — raise lỗi rõ ràng khi gọi ``apply()``/``rollback()``
    thay vì âm thầm không làm gì."""

    def __init__(self, *, gnb_endpoint: str = "localhost:36421") -> None:
        self._gnb_endpoint = gnb_endpoint

    def apply(self, result: OptimizationResult) -> None:
        raise QuantaSliceError(
            "E2Interface chưa triển khai (Roadmap Phase II: 'Deploy on "
            "real O-RAN gNB emulator'). Dùng orchestrator='mock_oran' "
            "cho MVP hiện tại."
        )

    def rollback(self) -> None:
        raise QuantaSliceError("E2Interface chưa triển khai (Roadmap Phase II).")
