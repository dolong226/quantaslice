"""Chuyển ``QUBOProblem`` (ma trận Q) sang biểu diễn Ising Hamiltonian
thưa (sparse) — theo mục 3 tài liệu "QUBO Formulation & QAOA".

Đặt x_k = (1 - z_k) / 2, z_k in {-1, +1}. Với f(x) = x^T Q x, Q đối
xứng, khai triển trực tiếp cho:

    h_k        = -R_k / 2         (R_k = tổng hàng thứ k của Q)
    J_kl (k<l) = Q_kl / 2
    offset     = (sum(Q) + trace(Q)) / 4

sao cho f(x) = sum_k h_k*z_k + sum_{k<l} J_kl*z_k*z_l + offset.

Trả về dict thưa (bỏ hệ số ~0) vì QUBO trong QuantaSlice có cấu trúc
thưa (mỗi x_ij chỉ tương tác với biến cùng slice i hoặc cùng trạm j) —
đúng ghi chú "O(n·M) gates" trong tài liệu QAOA mục 4.2.
"""

from __future__ import annotations

import numpy as np

from quantaslice.core.types import QUBOProblem

__all__ = ["qubo_to_ising"]

_ZERO_TOL = 1e-10


def qubo_to_ising(
    qubo: QUBOProblem,
) -> tuple[dict[int, float], dict[tuple[int, int], float], float]:
    """Trả về (h, J, offset) — biểu diễn Ising thưa của QUBOProblem.

    ``h``: dict {qubit_index: hệ số tuyến tính}.
    ``J``: dict {(k, l) với k<l: hệ số tương tác ZZ}.
    ``offset``: hằng số cộng thêm (không ảnh hưởng nghiệm tối ưu).
    """
    q_matrix = qubo.q_matrix
    n = qubo.n_qubits
    row_sum = q_matrix.sum(axis=1)

    h: dict[int, float] = {}
    for k in range(n):
        hk = -row_sum[k] / 2.0
        if abs(hk) > _ZERO_TOL:
            h[k] = float(hk)

    j: dict[tuple[int, int], float] = {}
    for k in range(n):
        for l in range(k + 1, n):
            jkl = q_matrix[k, l] / 2.0
            if abs(jkl) > _ZERO_TOL:
                j[(k, l)] = float(jkl)

    offset = float((q_matrix.sum() + np.trace(q_matrix)) / 4.0)
    return h, j, offset
