"""API public DUY NHẤT của package ``quantum``::

    allocation = quantaslice.quantum.solve(problem)

Mọi chi tiết bên trong (QUBOBuilder, qubo_to_ising, mạch QAOA, vòng lặp
COBYLA...) đều ẩn phía sau hàm này. Không package nào khác (pipeline,
dashboard, cli, ai) nên import trực tiếp từ ``quantum.qubo``,
``quantum.hamiltonian`` hay ``quantum.solvers`` — chỉ import từ đây.
"""

from __future__ import annotations

from quantaslice.core.runtime import Configuration
from quantaslice.core.types import AllocationProblem, OptimizationResult

__all__ = ["solve"]


def solve(
    problem: AllocationProblem,
    *,
    solver_name: str = "qaoa_aer",
    config: Configuration | None = None,
) -> OptimizationResult:
    """Giải bài toán phân bổ slice.

    Args:
        problem: mô tả slice/trạm/prediction (xem
            :class:`~quantaslice.core.types.AllocationProblem`).
        solver_name: tên solver đã đăng ký trong ``solver_registry`` —
            ``"qaoa_aer"`` (mặc định), ``"qaoa_ibmq"``, hoặc
            ``"classical_greedy"``.
        config: cấu hình runtime (độ sâu QAOA, số shots...). Nếu None,
            dùng :class:`~quantaslice.core.runtime.Configuration` mặc định.

    Returns:
        :class:`~quantaslice.core.types.OptimizationResult` — allocation
        tối ưu (hoặc gần tối ưu) tìm được.

    Raises:
        ~quantaslice.core.exceptions.ProviderNotFoundError: ``solver_name``
            chưa được đăng ký.
        ~quantaslice.core.exceptions.SolverError: lỗi backend khi giải.
        ~quantaslice.core.exceptions.InfeasibleAllocationError: không tìm
            được nghiệm khả thi nào.
    """
    from quantaslice.quantum import solver_registry  # import trễ, tránh circular import

    solver = solver_registry.create(solver_name, config=config)
    return solver.solve(problem)
