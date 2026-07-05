"""Baseline cổ điển (greedy) để so sánh với QAOA — không dùng QUBO/Ising,
giải trực tiếp trên ``AllocationProblem``. Xem Roadmap Phase II:
"Benchmark QAOA vs greedy & LP".

Không phụ thuộc Qiskit, nên luôn khả dụng kể cả khi chưa cài extras
``quantaslice[quantum]`` — hữu ích làm fallback/demo nhanh.
"""

from __future__ import annotations

import dataclasses

from quantaslice.core.runtime import Configuration
from quantaslice.core.types import Allocation, AllocationProblem, OptimizationResult
from quantaslice.quantum.decoding import compute_achieved_objective

__all__ = ["ClassicalGreedySolver"]


class ClassicalGreedySolver:
    """Implement :class:`~quantaslice.core.protocols.OptimizationSolver`.

    Thuật toán: liệt kê mọi cặp (slice, trạm) cùng trọng số ưu tiên, sắp
    xếp giảm dần, gán tham lam trong giới hạn PRB còn lại của từng trạm.
    Không đảm bảo tối ưu toàn cục (khác QAOA/branch-and-bound) nhưng
    chạy tức thời — dùng làm đường tham chiếu (baseline) chất lượng.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        self._config = config or Configuration()

    def solve(self, problem: AllocationProblem) -> OptimizationResult:
        remaining_capacity = {st.gnb_id: st.prb_capacity for st in problem.stations}

        candidates = [
            (self._weight(problem, s, st), s, st)
            for s in problem.slices
            for st in problem.stations
        ]
        candidates.sort(key=lambda c: c[0], reverse=True)

        assigned: dict[str, str | None] = {s.slice_id: None for s in problem.slices}
        for _weight, s, st in candidates:
            if assigned[s.slice_id] is not None:
                continue
            if remaining_capacity[st.gnb_id] >= s.prb_required:
                assigned[s.slice_id] = st.gnb_id
                remaining_capacity[st.gnb_id] -= s.prb_required

        allocations = tuple(
            Allocation(slice_id=slice_id, gnb_id=gnb_id) for slice_id, gnb_id in assigned.items()
        )
        result = OptimizationResult(
            allocations=allocations,
            objective_value=0.0,
            approximation_ratio=None,
            solver_name="classical_greedy",
            metadata={"remaining_capacity": remaining_capacity},
        )
        objective = compute_achieved_objective(problem, result)
        return dataclasses.replace(result, objective_value=objective)

    @staticmethod
    def _weight(problem: AllocationProblem, slice_, station) -> float:
        pred = problem.prediction_for(station.gnb_id)
        return pred.priority.weight_for(slice_.slice_type) if pred is not None else 1.0
