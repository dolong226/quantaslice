"""Orchestrator chạy trên hạ tầng OpenRAN thật qua E2 interface
(E2SM-RC) — theo slide System Architecture: "Pushes PRB reassignment
via O-RAN E2 interface", "Enforces URLLC QoS (PDCP/RLC params)".

Phiên bản nâng cấp:
* ``build_control_message()`` — tạo E2SM-RC RAN Control Request cấu trúc
  (mock) cho mỗi allocation change. Dùng để log và demo.
* ``build_policy_message()`` — tạo E2SM-RC Policy message cho scheduling
  policy change.
* ``apply()`` / ``rollback()`` — vẫn raise lỗi (Phase II), nhưng giờ
  có thêm ``build_control_message()`` static method có thể dùng bởi
  ``MockOranOrchestrator`` để log messages.
"""

from __future__ import annotations

import logging
from typing import Any

from quantaslice.core.exceptions import QuantaSliceError
from quantaslice.core.types import Allocation, OptimizationResult

__all__ = ["E2Interface"]

logger = logging.getLogger(__name__)


class E2Interface:
    """E2SM-RC control message builder + placeholder orchestrator.

    Static methods ``build_control_message()`` và ``build_policy_message()``
    có thể dùng bởi bất kỳ orchestrator nào (kể cả ``MockOranOrchestrator``)
    để build và log messages mô phỏng E2 interface thật.
    """

    def __init__(self, *, gnb_endpoint: str = "localhost:36421") -> None:
        self._gnb_endpoint = gnb_endpoint

    def apply(self, result: OptimizationResult) -> None:
        """Placeholder — raise lỗi rõ ràng khi gọi. Phase II."""
        raise QuantaSliceError(
            "E2Interface chưa triển khai (Roadmap Phase II: 'Deploy on "
            "real O-RAN gNB emulator'). Dùng orchestrator='mock_oran' "
            "cho MVP hiện tại."
        )

    def rollback(self) -> None:
        """Placeholder — Phase II."""
        raise QuantaSliceError("E2Interface chưa triển khai (Roadmap Phase II).")

    # ── Static builders cho E2SM-RC messages ──────────────────────────

    @staticmethod
    def build_control_message(allocation: Allocation) -> dict[str, Any]:
        """Build E2SM-RC RAN Control Request structure cho một allocation.

        Đây là cấu trúc mô phỏng E2SM-RC Control Header + Control Message
        theo O-RAN WG3 E2SM-RC v1.0. Trong MVP, chỉ build dict và log —
        không gửi qua gRPC/SCTP thật.

        Parameters
        ----------
        allocation:
            Một ``Allocation(slice_id, gnb_id)`` cần áp dụng.

        Returns
        -------
        dict
            E2SM-RC-style control message dict.
        """
        return {
            "e2sm_rc": {
                "ric_style_type": 1,  # RAN Control Request
                "ric_control_header": {
                    "ric_control_message_priority": 1,  # High priority
                    "ue_id": None,  # Broadcast to all UEs on this slice
                    "ric_control_action_id": 6,  # Slice PRB reconfig
                },
                "ric_control_message": {
                    "ran_parameter_list": [
                        {
                            "ran_parameter_name": "target_cell_global_id",
                            "ran_parameter_value": allocation.gnb_id,
                        },
                        {
                            "ran_parameter_name": "slice_id",
                            "ran_parameter_value": allocation.slice_id,
                        },
                        {
                            "ran_parameter_name": "min_prb_ratio",
                            "ran_parameter_value": 0,  # To be filled from context
                        },
                        {
                            "ran_parameter_name": "max_prb_ratio",
                            "ran_parameter_value": 100,  # To be filled
                        },
                        {
                            "ran_parameter_name": "dedicated_prb_ratio",
                            "ran_parameter_value": 0,
                        },
                    ],
                },
            }
        }

    @staticmethod
    def build_policy_message(
        gnb_id: str,
        slice_id: str,
        scheduling_policy: int,
    ) -> dict[str, Any]:
        """Build E2SM-RC Policy message cho scheduling policy change.

        Parameters
        ----------
        gnb_id:
            Target gNB.
        slice_id:
            Target slice.
        scheduling_policy:
            Policy ID: 0=Round-Robin, 1=Waterfilling, 2=Proportionally Fair.
        """
        policy_names = {0: "round_robin", 1: "waterfilling", 2: "proportionally_fair"}
        return {
            "e2sm_rc": {
                "ric_style_type": 2,  # RAN Policy
                "ric_policy_header": {
                    "target_cell": gnb_id,
                },
                "ric_policy_message": {
                    "slice_id": slice_id,
                    "scheduling_policy": scheduling_policy,
                    "scheduling_policy_name": policy_names.get(
                        scheduling_policy, "unknown"
                    ),
                },
            }
        }

    @staticmethod
    def build_indication_message(
        gnb_id: str,
        kpms: dict[str, Any],
    ) -> dict[str, Any]:
        """Build E2SM-KPM Indication Message structure (for reference).

        Mô phỏng E2 Indication message mà gNB gửi về near-RT RIC chứa
        KPMs — đây là chiều ngược lại (gNB → RIC) so với control message.
        """
        return {
            "e2sm_kpm": {
                "ric_indication_header": {
                    "cell_global_id": gnb_id,
                    "collection_start_time": None,
                },
                "ric_indication_message": {
                    "measurement_data": kpms,
                },
            }
        }
