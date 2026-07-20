"""State machine + dashboard state cho orchestrator — theo dõi trạng
thái từng gNB (idle/emergency/resolved), lịch sử emergency, và event
log, phục vụ ``MockOranOrchestrator.get_dashboard_state()``.

Đây là bản CHÍNH THỨC hoá của logic mà ``examples/run_web_demo.py``
(``SharedState``) từng tự làm ở tầng demo — chuyển vào core package để
mọi orchestrator (mock lẫn E2 thật sau này) đều dùng chung, thay vì mỗi
demo script tự làm lại.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from quantaslice.core.types import BaseStation, OptimizationResult, SliceRequest

__all__ = ["IDLE", "EMERGENCY", "RESOLVED", "OrchestratorState"]

IDLE = "idle"
EMERGENCY = "emergency"
RESOLVED = "resolved"

_HISTORY_LENGTH = 10
_EVENT_LOG_LENGTH = 100


class OrchestratorState:
    """State machine mỗi gNB: ``IDLE -> EMERGENCY`` (khi được đánh dấu
    khẩn cấp) ``-> RESOLVED`` (apply bình thường ngay sau emergency)
    ``-> IDLE`` (apply bình thường lần tiếp theo nữa).

    ``RESOLVED`` là trạng thái trung gian có chủ đích — cho dashboard 1
    nhịp để hiển thị "vừa xử lý xong" trước khi quay lại idle hoàn
    toàn, thay vì nhảy thẳng EMERGENCY -> IDLE (dễ bị người xem bỏ lỡ).
    """

    def __init__(self, *, stations: tuple[BaseStation, ...], slices: tuple[SliceRequest, ...]) -> None:
        self._stations = {st.gnb_id: st for st in stations}
        self._slices = {s.slice_id: s for s in slices}
        self._status: dict[str, str] = {gnb_id: IDLE for gnb_id in self._stations}
        self._history: dict[str, deque[int]] = {
            gnb_id: deque([0] * _HISTORY_LENGTH, maxlen=_HISTORY_LENGTH) for gnb_id in self._stations
        }
        self._current_allocations: dict[str, str | None] = {s.slice_id: None for s in slices}
        self._event_log: deque[dict] = deque(maxlen=_EVENT_LOG_LENGTH)
        self._solver_name: str = ""
        self._last_timestamp: str | None = None

    @property
    def event_log(self) -> list[dict]:
        return list(self._event_log)

    def get_gnb_status(self, gnb_id: str) -> str:
        return self._status.get(gnb_id, IDLE)

    def apply_result(self, result: OptimizationResult, emergency_gnb: str | None = None) -> None:
        """Cập nhật allocation hiện hành + state machine từng gNB.

        ``emergency_gnb``: gNB nào (nếu có) đang được coi là khẩn cấp
        NGAY LẦN GỌI NÀY — quyết định cả history (1/0) lẫn transition
        trạng thái cho mọi gNB.
        """
        for alloc in result.allocations:
            self._current_allocations[alloc.slice_id] = alloc.gnb_id
        self._solver_name = result.solver_name
        self._last_timestamp = datetime.now(timezone.utc).isoformat()

        for gnb_id in self._stations:
            is_emergency_now = gnb_id == emergency_gnb
            prev_status = self._status[gnb_id]
            if is_emergency_now:
                new_status = EMERGENCY
            elif prev_status == EMERGENCY:
                new_status = RESOLVED
            elif prev_status == RESOLVED:
                new_status = IDLE
            else:
                new_status = IDLE
            self._status[gnb_id] = new_status
            self._history[gnb_id].append(1 if is_emergency_now else 0)

        self._event_log.append(
            {
                "timestamp": self._last_timestamp,
                "emergency_gnb": emergency_gnb,
                "solver_name": self._solver_name,
                "allocation_summary": ", ".join(
                    f"{sid}->{gid or '(none)'}" for sid, gid in self._current_allocations.items()
                ),
            }
        )

    def reset(self) -> None:
        """Đưa mọi gNB về IDLE — KHÔNG xoá lịch sử allocation/event log
        (chỉ trạng thái state machine, để dashboard vẫn còn ngữ cảnh)."""
        for gnb_id in self._stations:
            self._status[gnb_id] = IDLE

    def to_dashboard_dict(self) -> dict:
        """Format JSON khớp wireframe dashboard — mỗi gNB có status,
        PRB used/capacity, danh sách slice đang phục vụ, và history."""
        stations_out: dict[str, dict] = {}
        for gnb_id, station in self._stations.items():
            served_ids = [sid for sid, g in self._current_allocations.items() if g == gnb_id]
            used = sum(self._slices[sid].prb_required for sid in served_ids)
            stations_out[gnb_id] = {
                "status": self._status[gnb_id],
                "prb_used": used,
                "prb_capacity": station.prb_capacity,
                "slices": served_ids,
                "history": list(self._history[gnb_id]),
            }
        return {
            "stations": stations_out,
            "event_log": list(self._event_log),
            "solver_name": self._solver_name,
            "timestamp": self._last_timestamp,
        }
