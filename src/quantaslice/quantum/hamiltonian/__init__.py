"""Chuyển đổi QUBO -> Ising Hamiltonian — chi tiết nội bộ của package
``quantum``, không export ra ngoài."""

from quantaslice.quantum.hamiltonian.ising import qubo_to_ising

__all__ = ["qubo_to_ising"]
