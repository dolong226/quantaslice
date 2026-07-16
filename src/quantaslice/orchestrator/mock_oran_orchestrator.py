"""Orchestrator mock (in-memory) — implement
:class:`~quantaslice.core.protocols.SliceOrchestratorPort` mà không đụng
hạ tầng thật. Dùng cho demo/CI trước khi có :class:`E2Interface` thật
(Roadmap Phase II: "Deploy on real O-RAN gNB emulator").
"""

from __future__ import annotations

import copy

from quantaslice.core.exceptions import InfeasibleAllocationError
from quantaslice.core.types import OptimizationResult

__all__ = ["MockOranOrchestrator"]


class MockOranOrchestrator:
    """``apply()`` thay thế toàn bộ trạng thái phân bổ hiện tại;
    ``rollback()`` khôi phục về trạng thái ngay trước lần ``apply()``
    gần nhất — đúng mô tả "Rollback logic if allocation is infeasible"
    trong slide System Architecture.

    Bất biến quan trọng: ``apply()`` snapshot ``_previous`` VÔ ĐIỀU
    KIỆN trước khi validate feasibility, để nếu ``apply()`` raise,
    ``rollback()`` gọi ngay sau đó luôn là no-op an toàn (không vô tình
    ghi đè ``current`` bằng một ``_previous`` cũ hơn từ lần apply thành
    công trước đó).
    """

    def __init__(self) -> None:
        self._current: OptimizationResult | None = None
        self._previous: OptimizationResult | None = None

    def apply(self, result: OptimizationResult) -> None:
        self._previous = copy.deepcopy(self._current)
        if result.allocations and result.n_unserved == len(result.allocations):
            raise InfeasibleAllocationError(
                "OptimizationResult không phục vụ được bất kỳ slice nào — "
                "coi như infeasible, orchestrator từ chối áp dụng."
            )
        self._current = result

    def rollback(self) -> None:
        self._current = self._previous

    @property
    def current_allocation(self) -> OptimizationResult | None:
        """Trạng thái phân bổ hiện hành — dashboard/CLI tương lai có thể
        đọc qua đây để hiển thị, không cần biết gì về QAOA/LSTM."""
        return self._current
