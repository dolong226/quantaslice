"""Các implementation cụ thể của
:class:`~quantaslice.core.protocols.OptimizationSolver` — chi tiết nội
bộ của package ``quantum``, được đăng ký vào ``solver_registry`` ở
``quantaslice/quantum/__init__.py``, không export trực tiếp ra ngoài."""

from quantaslice.quantum.solvers.classical_greedy_solver import ClassicalGreedySolver
from quantaslice.quantum.solvers.qaoa_aer_solver import QAOAAerSolver
from quantaslice.quantum.solvers.qaoa_ibmq_solver import QAOAIBMQSolver

__all__ = ["QAOAAerSolver", "QAOAIBMQSolver", "ClassicalGreedySolver"]
