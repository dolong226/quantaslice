"""Orchestrator mock (in-memory) — implement
:class:`~quantaslice.core.protocols.SliceOrchestratorPort` mà không đụng
hạ tầng thật. Dùng cho demo/CI trước khi có :class:`E2Interface` thật
(Roadmap Phase II: "Deploy on real O-RAN gNB emulator").

Phiên bản nâng cấp:
* **PRB capacity validation** — kiểm tra tổng PRB per-gNB không vượt quá
  capacity trước khi áp dụng.
* **Tích hợp OrchestratorState** — cập nhật state machine per-gNB
  (idle/emergency/resolved) và event log cho dashboard.
* **E2 control message logging** — build và log E2SM-RC-style messages
  cho mỗi allocation change.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict

from quantaslice.core.exceptions import InfeasibleAllocationError
from quantaslice.core.types import BaseStation, OptimizationResult, SliceRequest

__all__ = ["MockOranOrchestrator"]

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        stations: tuple[BaseStation, ...] | None = None,
        slices: tuple[SliceRequest, ...] | None = None,
    ) -> None:
        self._current: OptimizationResult | None = None
        self._previous: OptimizationResult | None = None

        # PRB capacity map — nếu stations được cung cấp, cho phép
        # validate tổng PRB per-gNB.
        self._stations: dict[str, BaseStation] = {}
        if stations:
            self._stations = {s.gnb_id: s for s in stations}

        # State machine cho dashboard — lazy init để backward-compatible
        # với code cũ không truyền stations/slices.
        self._state: _OrchestratorStateRef | None = None
        if stations:
            from quantaslice.orchestrator.state import OrchestratorState

            self._state = _OrchestratorStateRef(
                OrchestratorState(stations=stations, slices=slices)
            )

        # Emergency context — set bởi pipeline/runner khi có emergency.
        self._emergency_gnb: str | None = None

    def apply(self, result: OptimizationResult) -> None:
        """Áp dụng kết quả tối ưu.

        1. Snapshot ``_previous`` cho rollback.
        2. Validate: tất cả slices đều unserved → infeasible.
        3. Validate: tổng PRB per-gNB ≤ capacity (nếu có stations).
        4. Cập nhật ``OrchestratorState`` cho dashboard.
        5. Log E2 control messages.
        """
        self._previous = copy.deepcopy(self._current)

        # ── Feasibility check: toàn bộ unserved ──
        if result.allocations and result.n_unserved == len(result.allocations):
            raise InfeasibleAllocationError(
                "OptimizationResult không phục vụ được bất kỳ slice nào — "
                "coi như infeasible, orchestrator từ chối áp dụng."
            )

        # ── PRB capacity check (nếu có station info) ──
        if self._stations:
            per_gnb_prbs: dict[str, float] = defaultdict(float)
            for alloc in result.allocations:
                if alloc.gnb_id is None:
                    continue
                # Lấy prb_required từ metadata hoặc mặc định.
                prb_key = f"prb_{alloc.slice_id}"
                prb_count = result.metadata.get(prb_key, 0)
                per_gnb_prbs[alloc.gnb_id] += prb_count

            for gnb_id, total_prb in per_gnb_prbs.items():
                station = self._stations.get(gnb_id)
                if station and total_prb > station.prb_capacity:
                    raise InfeasibleAllocationError(
                        f"{gnb_id}: tổng PRB ({total_prb}) vượt quá "
                        f"capacity ({station.prb_capacity})"
                    )

        # ── Apply ──
        self._current = result

        # ── Cập nhật state machine cho dashboard ──
        if self._state is not None:
            emergency_gnb = (
                self._emergency_gnb
                or result.metadata.get("emergency_gnb")
            )
            self._state.ref.apply_result(result, emergency_gnb=emergency_gnb)

        # ── Log E2 control messages ──
        from quantaslice.orchestrator.e2_interface import E2Interface

        for alloc in result.allocations:
            if alloc.gnb_id is not None:
                msg = E2Interface.build_control_message(alloc)
                logger.debug("E2 Control → %s: %s", alloc.gnb_id, msg)

        logger.info(
            "MockOranOrchestrator: applied %d allocations (solver=%s, "
            "unserved=%d, objective=%.4f)",
            len(result.allocations),
            result.solver_name,
            result.n_unserved,
            result.objective_value,
        )

    def rollback(self) -> None:
        """Khôi phục allocation trước đó."""
        self._current = self._previous
        if self._state is not None and self._previous is not None:
            self._state.ref.apply_result(self._previous)
        logger.info("MockOranOrchestrator: rolled back to previous allocation.")

    def set_emergency_gnb(self, gnb_id: str | None) -> None:
        """Set emergency context — gọi bởi Runner trước khi apply()."""
        self._emergency_gnb = gnb_id
        if self._state is not None and gnb_id:
            self._state.ref.set_emergency(gnb_id)

    def reset(self) -> None:
        """Reset state machine về idle — dùng cho nút Reset trên dashboard."""
        self._emergency_gnb = None
        if self._state is not None:
            self._state.ref.reset()
        logger.info("MockOranOrchestrator: reset to idle.")

    # ── Properties cho dashboard / API ─────────────────────────────────

    @property
    def current_allocation(self) -> OptimizationResult | None:
        """Trạng thái phân bổ hiện hành — dashboard/CLI tương lai có thể
        đọc qua đây để hiển thị, không cần biết gì về QAOA/LSTM."""
        return self._current

    @property
    def state(self):
        """OrchestratorState cho dashboard. Trả None nếu chưa init."""
        return self._state.ref if self._state is not None else None

    def get_dashboard_state(self) -> dict:
        """Trả về dict cho ``GET /state`` endpoint.

        Nếu chưa có OrchestratorState (backward-compat mode), trả về
        dict tối thiểu từ ``current_allocation``.
        """
        if self._state is not None:
            return self._state.ref.to_dashboard_dict()

        # Fallback cho backward compatibility.
        if self._current is None:
            return {"stations": {}, "event_log": [], "solver_name": "", "timestamp": None}

        return {
            "stations": {},
            "event_log": [],
            "solver_name": self._current.solver_name,
            "timestamp": None,
            "allocations": [
                {"slice_id": a.slice_id, "gnb_id": a.gnb_id}
                for a in self._current.allocations
            ],
        }


class _OrchestratorStateRef:
    """Wrapper để tránh circular import khi dùng OrchestratorState."""

    __slots__ = ("ref",)

    def __init__(self, ref: object) -> None:
        self.ref = ref
