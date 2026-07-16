"""``quantaslice.quantum`` — package độc lập cho QUBO/Hamiltonian/QAOA.

Bề mặt public DUY NHẤT: :func:`quantaslice.quantum.solve`. Mọi thứ khác
(``qubo``, ``hamiltonian``, ``solvers``, ``decoding``) là chi tiết triển
khai nội bộ.

Ràng buộc kiến trúc quan trọng nhất: package này CHỈ import từ
``quantaslice.core`` — KHÔNG BAO GIỜ import ``quantaslice.ai``.
"""

from quantaslice.core.protocols import OptimizationSolver
from quantaslice.core.registry import Registry
from quantaslice.quantum.solvers import ClassicalGreedySolver, QAOAAerSolver, QAOAIBMQSolver, QAOAQuappSolver

solver_registry: Registry[OptimizationSolver] = Registry("solver")
solver_registry.register_instance_factory("qaoa_aer", QAOAAerSolver)
solver_registry.register_instance_factory("qaoa_ibmq", QAOAIBMQSolver)
solver_registry.register_instance_factory("qaoa_quapp", QAOAQuappSolver)
solver_registry.register_instance_factory("classical_greedy", ClassicalGreedySolver)

# Import sau khi solver_registry đã sẵn sàng, vì api.solve() tra cứu
# solver_registry lúc gọi (không phải lúc import) -> tránh circular import.
from quantaslice.quantum.api import solve  # noqa: E402

__all__ = ["solve", "solver_registry"]
