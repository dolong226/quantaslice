"""``OrchestratorState`` — quản lý trạng thái phân bổ per-gNB và cung
cấp dữ liệu tuần tự hoá cho dashboard.

Ba trạng thái thị giác (mục 3 wireframe UX):

* **idle**      — hoạt động bình thường, không cần can thiệp.
* **emergency** — phát hiện sự kiện khẩn cấp, đang tái phân bổ.
* **resolved**  — đã tái phân bổ xong, URLLC đã chuyển đúng trạm.

Module này CHỈ import từ ``quantaslice.core`` — KHÔNG import ngược lại
bất kỳ package con nào (ai, quantum, simulation, pipeline).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from quantaslice.core.types import (
    Allocation,
    BaseStation,
    OptimizationResult,
    SliceRequest,
)

__all__ = ["OrchestratorState", "EventLogEntry"]

logger = logging.getLogger(__name__)

# Giới hạn event log giữ trong bộ nhớ — tránh tràn RAM khi chạy lâu.
_MAX_EVENT_LOG = 200

# Số lịch sử emergency flag gần nhất giữ per-gNB (hiển thị trên dashboard
# dưới dạng ``history ●●●○○●●●●●``).
_HISTORY_LENGTH = 10


# ── Trạng thái gNB ────────────────────────────────────────────────────
IDLE = "idle"
EMERGENCY = "emergency"
RESOLVED = "resolved"


@dataclass(slots=True)
class EventLogEntry:
    """Một dòng trong bảng EVENT LOG trên dashboard."""

    timestamp: datetime
    gnb_id: str
    status: str
    allocation_summary: str
    solver_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.timestamp.strftime("%H:%M:%S.%f")[:-3],
            "gnb": self.gnb_id,
            "status": self.status,
            "allocation": self.allocation_summary,
            "solver": self.solver_name,
        }


@dataclass
class _GNBState:
    """Trạng thái nội bộ cho một gNB."""

    gnb_id: str
    status: str = IDLE
    prb_capacity: float = 0.0
    slice_allocations: dict[str, int | float] = field(default_factory=dict)
    # slice_id -> prb_count (0 nếu slice không gán cho gNB này)
    slice_types: dict[str, str] = field(default_factory=dict)
    # slice_id -> slice_type name (eMBB / URLLC / mMTC)
    history: deque[int] = field(
        default_factory=lambda: deque([0] * _HISTORY_LENGTH, maxlen=_HISTORY_LENGTH)
    )


class OrchestratorState:
    """Quản lý state tổng thể cho tất cả gNB, cung cấp ``to_dashboard_dict()``
    cho endpoint ``GET /state``.

    Usage::

        state = OrchestratorState(stations, slices)
        # ... trong pipeline ...
        state.apply_result(result, emergency_gnb="gnb-1")
        # ... dashboard endpoint ...
        return state.to_dashboard_dict()
    """

    def __init__(
        self,
        stations: tuple[BaseStation, ...] | list[BaseStation],
        slices: tuple[SliceRequest, ...] | list[SliceRequest] | None = None,
    ) -> None:
        self._gnb_states: dict[str, _GNBState] = {}
        for st in stations:
            self._gnb_states[st.gnb_id] = _GNBState(
                gnb_id=st.gnb_id,
                prb_capacity=st.prb_capacity,
            )

        # Lưu thông tin slice type nếu có.
        if slices:
            for s in slices:
                for gs in self._gnb_states.values():
                    gs.slice_types[s.slice_id] = s.slice_type.value

        self._event_log: deque[EventLogEntry] = deque(maxlen=_MAX_EVENT_LOG)
        self._solver_name: str = ""
        self._last_update: datetime | None = None

    # ── Public API ────────────────────────────────────────────────────

    def apply_result(
        self,
        result: OptimizationResult,
        emergency_gnb: str | None = None,
    ) -> None:
        """Cập nhật state dựa trên kết quả tối ưu mới.

        Parameters
        ----------
        result:
            Kết quả từ QAOA/greedy solver.
        emergency_gnb:
            ``gnb_id`` của trạm đang ở trạng thái khẩn cấp (nếu có).
            Nếu ``None``, tất cả các gNB về trạng thái bình thường.
        """
        now = datetime.now()
        self._solver_name = result.solver_name
        self._last_update = now

        # 1. Clear các allocation cũ.
        for gs in self._gnb_states.values():
            gs.slice_allocations.clear()

        # 2. Ghi allocation mới.
        for alloc in result.allocations:
            if alloc.gnb_id is None:
                continue  # unserved slice
            gs = self._gnb_states.get(alloc.gnb_id)
            if gs is None:
                logger.warning(
                    "Allocation cho gNB '%s' không tồn tại trong state, bỏ qua.",
                    alloc.gnb_id,
                )
                continue
            # Lấy prb_required từ metadata nếu có, mặc định = 0.
            prb_count = result.metadata.get(
                f"prb_{alloc.slice_id}", 0
            )
            gs.slice_allocations[alloc.slice_id] = prb_count

        # 3. Cập nhật trạng thái + history per-gNB.
        for gs in self._gnb_states.values():
            if emergency_gnb and gs.gnb_id == emergency_gnb:
                gs.status = EMERGENCY
                gs.history.append(1)
            else:
                # Nếu trước đó EMERGENCY, chuyển sang RESOLVED;
                # nếu IDLE/RESOLVED, giữ IDLE.
                if gs.status == EMERGENCY:
                    gs.status = RESOLVED
                else:
                    gs.status = IDLE
                gs.history.append(0)

        # 4. Ghi event log.
        alloc_summary = self._format_allocation_summary(result.allocations)
        for gs in self._gnb_states.values():
            self._event_log.appendleft(
                EventLogEntry(
                    timestamp=now,
                    gnb_id=gs.gnb_id,
                    status=gs.status.upper() if gs.status == EMERGENCY else "normal",
                    allocation_summary=alloc_summary,
                    solver_name=result.solver_name,
                )
            )

        logger.info(
            "OrchestratorState: applied result (solver=%s, emergency_gnb=%s)",
            result.solver_name,
            emergency_gnb,
        )

    def reset(self) -> None:
        """Reset tất cả gNB về trạng thái idle — khi user bấm nút Reset
        trên dashboard."""
        now = datetime.now()
        for gs in self._gnb_states.values():
            gs.status = IDLE
            gs.history.append(0)
        self._event_log.appendleft(
            EventLogEntry(
                timestamp=now,
                gnb_id="system",
                status="RESET",
                allocation_summary="All stations reset to idle",
            )
        )
        logger.info("OrchestratorState: RESET to idle.")

    def set_emergency(self, gnb_id: str) -> None:
        """Đánh dấu một gNB là emergency (dùng bởi ManualOverrideProvider)."""
        gs = self._gnb_states.get(gnb_id)
        if gs:
            gs.status = EMERGENCY
            gs.history.append(1)

    # ── Dashboard serialization ───────────────────────────────────────

    def to_dashboard_dict(self) -> dict[str, Any]:
        """Tuần tự hoá state cho ``GET /state`` — format khớp với wireframe.

        Returns
        -------
        dict
            JSON-serializable dict với keys: ``stations``, ``event_log``,
            ``solver_name``, ``timestamp``.
        """
        stations: dict[str, Any] = {}
        for gs in self._gnb_states.values():
            prb_used = sum(gs.slice_allocations.values())
            slices_list = []
            for slice_id, prbs in gs.slice_allocations.items():
                slices_list.append(
                    {
                        "id": slice_id,
                        "type": gs.slice_types.get(slice_id, "unknown"),
                        "prbs": prbs,
                    }
                )
            stations[gs.gnb_id] = {
                "status": gs.status,
                "prb_used": prb_used,
                "prb_capacity": gs.prb_capacity,
                "slices": slices_list,
                "history": list(gs.history),
            }

        return {
            "stations": stations,
            "event_log": [e.to_dict() for e in self._event_log],
            "solver_name": self._solver_name,
            "timestamp": (
                self._last_update.isoformat() if self._last_update else None
            ),
        }

    @property
    def event_log(self) -> list[EventLogEntry]:
        """Danh sách event log (mới nhất trước)."""
        return list(self._event_log)

    @property
    def gnb_ids(self) -> list[str]:
        return list(self._gnb_states.keys())

    def get_gnb_status(self, gnb_id: str) -> str:
        """Trả về trạng thái hiện tại của một gNB."""
        gs = self._gnb_states.get(gnb_id)
        return gs.status if gs else IDLE

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _format_allocation_summary(allocations: tuple[Allocation, ...]) -> str:
        """Tạo chuỗi tóm tắt: ``s1→gnb-2, s2→gnb-1, s3→gnb-1``."""
        parts = []
        for a in allocations:
            target = a.gnb_id if a.gnb_id else "unserved"
            parts.append(f"{a.slice_id}→{target}")
        return ", ".join(parts) if parts else "(empty)"
