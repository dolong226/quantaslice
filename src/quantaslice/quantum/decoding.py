"""Giải mã bitstring nhị phân (kết quả đo QAOA hoặc lời giải cổ điển)
thành :class:`~quantaslice.core.types.OptimizationResult`, và các hàm
tiện ích đánh giá QUBO — dùng chung cho mọi solver trong package
``quantum`` (tránh lặp logic decode ở từng solver).
"""

from __future__ import annotations

import numpy as np

from quantaslice.core.types import (
    Allocation,
    AllocationProblem,
    OptimizationResult,
    QUBOProblem,
)
from quantaslice.quantum.qubo.builder import SLACK_CAPACITY_TAG, SLACK_UNIQUENESS_TAG

__all__ = ["evaluate_qubo_objective", "decode_bitstring", "compute_achieved_objective"]


def evaluate_qubo_objective(q_matrix: np.ndarray, x: np.ndarray) -> float:
    """Tính f(x) = x^T Q x cho một bitstring cụ thể — dùng để so sánh
    chất lượng các mẫu đo được trong vòng lặp QAOA."""
    return float(x @ q_matrix @ x)


def decode_bitstring(
    bitstring: np.ndarray,
    qubo: QUBOProblem,
    problem: AllocationProblem,
    solver_name: str,
    metadata: dict | None = None,
) -> OptimizationResult:
    """Chuyển bitstring x* (theo ``qubo.variable_map``) thành
    ``OptimizationResult``. Bỏ qua các biến slack (tag ``__slack_*``).

    ``objective_value`` được đặt tạm = 0.0; caller nên gọi
    :func:`compute_achieved_objective` sau đó và cập nhật bằng
    ``dataclasses.replace`` (xem cách dùng trong các solver), vì giá trị
    objective "nghiệp vụ" (tổng ưu tiên đạt được) độc lập với chi tiết
    penalty nội bộ của QUBO.
    """
    assignments: dict[str, list[str]] = {s.slice_id: [] for s in problem.slices}
    for idx, value in enumerate(bitstring):
        if value != 1:
            continue
        mapped = qubo.variable_map.get(idx)
        if mapped is None:
            continue
        slice_id, gnb_id = mapped
        if slice_id in (SLACK_UNIQUENESS_TAG, SLACK_CAPACITY_TAG):
            continue
        if slice_id in assignments:
            assignments[slice_id].append(gnb_id)

    allocations: list[Allocation] = []
    violations: dict[str, list[str]] = {}
    for slice_id, gnb_ids in assignments.items():
        if len(gnb_ids) == 0:
            allocations.append(Allocation(slice_id=slice_id, gnb_id=None))
        else:
            if len(gnb_ids) > 1:
                # Nghiệm xấp xỉ vi phạm ràng buộc uniqueness (thường gặp ở
                # QAOA độ sâu p thấp) -> chọn trạm đầu tiên, ghi log vào
                # metadata để giám sát chất lượng nghiệm thay vì raise lỗi.
                violations[slice_id] = gnb_ids
            allocations.append(Allocation(slice_id=slice_id, gnb_id=gnb_ids[0]))

    result_metadata = dict(metadata or {})
    if violations:
        result_metadata["uniqueness_violations"] = violations

    return OptimizationResult(
        allocations=tuple(allocations),
        objective_value=0.0,
        approximation_ratio=None,
        solver_name=solver_name,
        metadata=result_metadata,
    )


def compute_achieved_objective(problem: AllocationProblem, result: OptimizationResult) -> float:
    """Tổng trọng số ưu tiên thực sự đạt được bởi một OptimizationResult
    — dùng làm thước đo chung, độc lập với chi tiết QUBO/penalty, để so
    sánh công bằng giữa QAOA và ``ClassicalGreedySolver``."""
    slices_by_id = {s.slice_id: s for s in problem.slices}
    total = 0.0
    for alloc in result.allocations:
        if alloc.gnb_id is None:
            continue
        slice_ = slices_by_id[alloc.slice_id]
        pred = problem.prediction_for(alloc.gnb_id)
        weight = pred.priority.weight_for(slice_.slice_type) if pred is not None else 1.0
        total += weight
    return total
