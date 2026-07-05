"""Solver QAOA chạy trên phần cứng IBM Quantum thật.

Hiện là placeholder — cùng implement
:class:`~quantaslice.core.protocols.OptimizationSolver` như
``QAOAAerSolver`` để có thể hoán đổi qua registry (đổi 1 dòng config
``solver: "qaoa_ibmq"``) mà KHÔNG cần sửa code gọi, khi hạ tầng sẵn sàng
(Roadmap Phase III, M13-M18: "First IBM Quantum hardware run").

Lý do chưa triển khai đầy đủ: hàng đợi cloud hiện tại của IBM Quantum
là phút-giờ, không tương thích mục tiêu latency 500ms của QuantaSlice
(xem slide "Hardware Queue Latency" trong Limitations) — MVP dùng
:class:`~quantaslice.quantum.solvers.qaoa_aer_solver.QAOAAerSolver`.
"""

from __future__ import annotations

from quantaslice.core.exceptions import SolverError
from quantaslice.core.runtime import Configuration
from quantaslice.core.types import AllocationProblem, OptimizationResult

__all__ = ["QAOAIBMQSolver"]


class QAOAIBMQSolver:
    """Placeholder — raise :class:`SolverError` khi gọi ``solve()``."""

    def __init__(
        self,
        config: Configuration | None = None,
        *,
        backend_name: str = "ibmq_qasm_simulator",
    ) -> None:
        self._config = config or Configuration()
        self._backend_name = backend_name

    def solve(self, problem: AllocationProblem) -> OptimizationResult:
        raise SolverError(
            "QAOAIBMQSolver chưa triển khai (Roadmap Phase III). "
            "Dùng solver_name='qaoa_aer' (mô phỏng) hoặc "
            "'classical_greedy' (baseline) cho MVP hiện tại."
        )
