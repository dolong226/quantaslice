"""Orchestrator mock (in-memory) — implement
:class:`~quantaslice.core.protocols.SliceOrchestratorPort` mà không đụng
hạ tầng thật. Dùng cho demo/CI trước khi có :class:`E2Interface` thật
(Roadmap Phase II: "Deploy on real O-RAN gNB emulator").
"""

from __future__ import annotations

import copy

from quantaslice.core.exceptions import InfeasibleAllocationError
from quantaslice.core.types import BaseStation, OptimizationResult, SliceRequest
from quantaslice.orchestrator.state import OrchestratorState

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

    ``stations``/``slices`` là THAM SỐ TUỲ CHỌN (mặc định None, giữ
    nguyên hành vi cũ — backward compatible với
    ``MockOranOrchestrator()`` không tham số). Khi truyền vào,
    orchestrator tự dựng thêm
    :class:`~quantaslice.orchestrator.state.OrchestratorState` để theo
    dõi state machine (idle/emergency/resolved) từng gNB, phục vụ
    ``get_dashboard_state()``.
    """

    def __init__(
        self,
        *,
        stations: tuple[BaseStation, ...] | None = None,
        slices: tuple[SliceRequest, ...] | None = None,
    ) -> None:
        self._current: OptimizationResult | None = None
        self._previous: OptimizationResult | None = None
        self._emergency_gnb: str | None = None
        self._state: OrchestratorState | None = (
            OrchestratorState(stations=stations, slices=slices) if stations and slices else None
        )

    @property
    def state(self) -> OrchestratorState | None:
        """``None`` nếu orchestrator được tạo không kèm ``stations``/
        ``slices`` (chế độ tối giản, chỉ theo dõi allocation)."""
        return self._state

    def set_emergency_gnb(self, gnb_id: str | None) -> None:
        """Đánh dấu gNB nào đang khẩn cấp cho lần ``apply()`` KẾ TIẾP —
        "sticky" cho tới khi gọi lại (đổi giá trị khác hoặc None), hoặc
        tới khi ``reset()``."""
        self._emergency_gnb = gnb_id

    def apply(self, result: OptimizationResult) -> None:
        self._previous = copy.deepcopy(self._current)
        if result.allocations and result.n_unserved == len(result.allocations):
            raise InfeasibleAllocationError(
                "OptimizationResult không phục vụ được bất kỳ slice nào — "
                "coi như infeasible, orchestrator từ chối áp dụng."
            )
        self._current = result
        if self._state is not None:
            self._state.apply_result(result, emergency_gnb=self._emergency_gnb)

    def rollback(self) -> None:
        self._current = self._previous

    def reset(self) -> None:
        """Đưa orchestrator về trạng thái ban đầu: xoá allocation hiện
        hành VÀ đưa state machine (nếu có) về IDLE toàn bộ, xoá luôn
        emergency_gnb đang "sticky"."""
        self._current = None
        self._previous = None
        self._emergency_gnb = None
        if self._state is not None:
            self._state.reset()

    def get_dashboard_state(self) -> dict:
        """JSON-serializable dict cho dashboard — dict rỗng có cấu trúc
        tối thiểu nếu orchestrator không có ``state`` (tạo không kèm
        stations/slices), thay vì raise lỗi."""
        if self._state is None:
            return {"stations": {}, "event_log": [], "solver_name": "", "timestamp": None}
        return self._state.to_dashboard_dict()

    @property
    def current_allocation(self) -> OptimizationResult | None:
        """Trạng thái phân bổ hiện hành — dashboard/CLI tương lai có thể
        đọc qua đây để hiển thị, không cần biết gì về QAOA/LSTM."""
        return self._current
