"""Hàm tiện ích chung để thêm điều khoản phạt (penalty) dạng
``weight * (sum_m a_m*z_m + constant)^2`` vào ma trận QUBO — dùng cho cả
ràng buộc uniqueness (P1) lẫn capacity (P2), theo mục 2.3 tài liệu
"QUBO Formulation & QAOA".

Khai triển tổng quát (z_m nhị phân, do đó z_m^2 = z_m):

    S = sum_m a_m*z_m + constant
    S^2 = sum_m (a_m^2 + 2*constant*a_m) * z_m
          + 2 * sum_{m<n} a_m*a_n * z_m*z_n
          + constant^2

Với quy ước ma trận Q đối xứng sao cho f(x) = x^T Q x, hệ số của cặp
z_m*z_n (m<n) được chia đôi vào Q[m,n] và Q[n,m] để khi khai triển
x^T Q x (đếm cả hai chiều) tái tạo đúng hệ số gốc.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = ["add_quadratic_penalty"]


def add_quadratic_penalty(
    q_matrix: np.ndarray,
    terms: list[tuple[int, float]],
    constant: float,
    weight: float,
) -> float:
    """Cộng dồn (in-place) vào ``q_matrix`` các hệ số sinh ra từ khai
    triển ``weight * (sum_m a_m*z_m + constant)^2``.

    Args:
        q_matrix: ma trận QUBO (n x n), sẽ bị chỉnh sửa in-place.
        terms: danh sách (chỉ số biến, hệ số a_m).
        constant: hằng số cộng thêm trong biểu thức trước khi bình phương.
        weight: trọng số phạt (thường là lambda1 hoặc lambda2).

    Returns:
        Phần hằng số (offset) sinh ra từ khai triển — không phụ thuộc
        biến nên không ảnh hưởng nghiệm tối ưu, nhưng hữu ích nếu cần
        tính giá trị objective tuyệt đối chính xác.
    """
    offset = weight * constant**2
    for idx, coeff in terms:
        q_matrix[idx, idx] += weight * (coeff**2 + 2 * constant * coeff)
    for (idx_m, coeff_m), (idx_n, coeff_n) in itertools.combinations(terms, 2):
        pair_coeff = weight * 2 * coeff_m * coeff_n
        q_matrix[idx_m, idx_n] += pair_coeff / 2
        q_matrix[idx_n, idx_m] += pair_coeff / 2
    return offset
