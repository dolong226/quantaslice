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
from quantaslice.core.types import Allocation, OptimizationResult

__all__ = ["E2Interface"]

# Mapping chính sách lập lịch -> tên chuẩn ColO-RAN (round_robin,
# waterfilling, proportionally_fair) — khớp thứ tự policy dùng trong
# dataset gốc (0/1/2).
_SCHEDULING_POLICY_NAMES: dict[int, str] = {
    0: "round_robin",
    1: "waterfilling",
    2: "proportionally_fair",
}


class E2Interface:
    """Placeholder — ``apply()``/``rollback()`` raise lỗi rõ ràng thay
    vì âm thầm không làm gì (Roadmap Phase II).

    3 hàm ``build_*_message()`` là static method THUẦN (không cần kết
    nối gì) — dựng đúng cấu trúc message E2SM-RC/E2SM-KPM chuẩn O-RAN,
    độc lập với việc kết nối E2 thật đã sẵn sàng hay chưa. Có thể dùng
    ngay hôm nay để log/kiểm thử định dạng message, trước khi có hạ
    tầng thật để gửi đi.
    """

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

    @staticmethod
    def build_control_message(allocation: Allocation) -> dict:
        """RIC Control message (E2SM-RC) áp dụng 1 quyết định allocation
        — 5 RAN parameter: trạm đích, slice, loại hành động, mức PRB
        share, mức ưu tiên QoS."""
        return {
            "e2sm_rc": {
                "ric_control_header": {
                    "ric_style_type": 1,
                    "action_type": "PRB_REALLOCATION",
                },
                "ric_control_message": {
                    "ran_parameter_list": [
                        {"ran_parameter_name": "target_cell_global_id", "ran_parameter_value": allocation.gnb_id},
                        {"ran_parameter_name": "slice_id", "ran_parameter_value": allocation.slice_id},
                        {"ran_parameter_name": "action_type", "ran_parameter_value": "PRB_REALLOCATION"},
                        {"ran_parameter_name": "prb_share", "ran_parameter_value": None},
                        {"ran_parameter_name": "qos_priority_level", "ran_parameter_value": None},
                    ]
                },
            }
        }

    @staticmethod
    def build_policy_message(gnb_id: str, slice_id: str, policy: int) -> dict:
        """RIC Policy message (E2SM-RC) — đổi chính sách lập lịch của 1
        slice tại 1 gNB (0=round_robin, 1=waterfilling, 2=proportionally_fair,
        khớp 3 policy dùng trong dataset ColO-RAN gốc)."""
        return {
            "e2sm_rc": {
                "ric_policy_message": {
                    "cell_global_id": gnb_id,
                    "slice_id": slice_id,
                    "scheduling_policy": policy,
                    "scheduling_policy_name": _SCHEDULING_POLICY_NAMES.get(policy, "unknown"),
                }
            }
        }

    @staticmethod
    def build_indication_message(gnb_id: str, kpms: dict) -> dict:
        """RIC Indication message (E2SM-KPM) — báo cáo KPM đo được từ 1
        gNB, dùng ``kpms`` nguyên dạng (không biến đổi) làm measurement
        data."""
        return {
            "e2sm_kpm": {
                "ric_indication_header": {"cell_global_id": gnb_id},
                "ric_indication_message": {"measurement_data": kpms},
            }
        }
