"""Xây dựng :class:`~quantaslice.core.types.QUBOProblem` từ
:class:`~quantaslice.core.types.AllocationProblem`, theo đúng mục 2 của
tài liệu "QUBO Formulation & QAOA".

Layout biến sau khi flatten (mục 2.2, 2.3):

    [0, N*M)              : x_ij  — slice i được gán vào trạm j
    [N*M, N*M+N)           : y_i   — slack uniqueness (1 qubit / slice)
    [N*M+N, n)             : s_{j,t} — slack capacity, mã hoá nhị phân
                              (ceil(log2(c_j+1)) qubit / trạm)

Đây là module DUY NHẤT biết cấu trúc nội bộ Q; phần còn lại của hệ
thống chỉ thấy :func:`quantaslice.quantum.solve`.
"""

from __future__ import annotations

import math

import numpy as np

from quantaslice.core.types import AllocationProblem, QUBOProblem
from quantaslice.quantum.qubo.penalties import add_quadratic_penalty

__all__ = ["QUBOBuilder", "SLACK_UNIQUENESS_TAG", "SLACK_CAPACITY_TAG"]

# Tiền tố đánh dấu biến slack trong variable_map, để decoding.py biết
# bỏ qua khi giải mã bitstring thành Allocation thực sự.
SLACK_UNIQUENESS_TAG = "__slack_uniqueness__"
SLACK_CAPACITY_TAG = "__slack_capacity__"


class QUBOBuilder:
    """Chuyển ``AllocationProblem`` thành ``QUBOProblem`` (ma trận Q)."""

    def build(self, problem: AllocationProblem) -> QUBOProblem:
        slices, stations = problem.slices, problem.stations
        n_slices, n_stations = len(slices), len(stations)
        n_main = n_slices * n_stations

        def x_index(i: int, j: int) -> int:
            return i * n_stations + j

        variable_map: dict[int, tuple[str, str]] = {}
        for i, s in enumerate(slices):
            for j, st in enumerate(stations):
                variable_map[x_index(i, j)] = (s.slice_id, st.gnb_id)

        y_index = {i: n_main + i for i in range(n_slices)}
        for i, s in enumerate(slices):
            variable_map[y_index[i]] = (SLACK_UNIQUENESS_TAG, s.slice_id)

        offset = n_main + n_slices
        capacity_bits: dict[int, list[int]] = {}
        for j, st in enumerate(stations):
            n_bits = self._n_capacity_bits(st.prb_capacity)
            bits = list(range(offset, offset + n_bits))
            capacity_bits[j] = bits
            for t, bit_idx in enumerate(bits):
                variable_map[bit_idx] = (SLACK_CAPACITY_TAG, f"{st.gnb_id}:bit{t}")
            offset += n_bits

        n_qubits = offset
        q_matrix = np.zeros((n_qubits, n_qubits))

        weights = self._compute_normalized_weights(problem)
        for i in range(n_slices):
            for j in range(n_stations):
                # Mục tiêu gốc là MAX(sum w_ij x_ij); QUBO là bài toán MIN
                # -> đổi dấu (mục 2.4: "đổi dấu vì QUBO là bài toán min").
                q_matrix[x_index(i, j), x_index(i, j)] += -weights[i, j]

        max_w = float(weights.max()) if weights.size else 0.0
        lambda1 = problem.penalty_lambda1 if problem.penalty_lambda1 is not None else 3.0 * max_w
        lambda2 = problem.penalty_lambda2 if problem.penalty_lambda2 is not None else lambda1

        # P1 - uniqueness: (sum_j x_ij - y_i)^2, mỗi slice tối đa 1 trạm.
        for i in range(n_slices):
            terms = [(x_index(i, j), 1.0) for j in range(n_stations)]
            terms.append((y_index[i], -1.0))
            add_quadratic_penalty(q_matrix, terms, constant=0.0, weight=lambda1)

        # P2 - capacity: (sum_i r_i x_ij + s_j - c_j)^2.
        for j, st in enumerate(stations):
            terms = [(x_index(i, j), slices[i].prb_required) for i in range(n_slices)]
            for t, bit_idx in enumerate(capacity_bits[j]):
                terms.append((bit_idx, float(2**t)))
            add_quadratic_penalty(q_matrix, terms, constant=-st.prb_capacity, weight=lambda2)

        # Rescale TOÀN BỘ Q về cùng thang đo (chia cho max|Q|) trước khi
        # trả về — KHÔNG đổi nghiệm tối ưu (argmin f(x) bất biến khi nhân
        # f với hằng số dương), nhưng quan trọng cho QAOA: nếu không
        # rescale, hệ số liên quan capacity/PRB (r_i, c_j chưa chuẩn hoá,
        # có thể ~10-30) sau khi bình phương trong penalty có thể lớn
        # hơn hệ số objective (đã chuẩn hoá [0,1]) tới hàng trăm lần.
        # Với QAOA dùng 1 giá trị gamma DÙNG CHUNG cho mọi hệ số trong 1
        # layer, chênh lệch biên độ lớn khiến các cổng RZ ứng với hệ số
        # lớn quay pha nhiều vòng 2π gần như ngẫu nhiên trong khi hệ số
        # nhỏ gần như không đổi pha — COBYLA không thể tìm gamma tốt cho
        # cả 2 nhóm cùng lúc. Đã verify bằng thực nghiệm: h_k dao động
        # -931 đến 1.5 trước khi rescale.
        max_abs = np.max(np.abs(q_matrix))
        if max_abs > 0:
            q_matrix = q_matrix / max_abs

        return QUBOProblem(
            q_matrix=q_matrix,
            variable_map=variable_map,
            n_qubits=n_qubits,
            lambda1=lambda1,
            lambda2=lambda2,
        )

    @staticmethod
    def _n_capacity_bits(capacity: float) -> int:
        """Số qubit cần để mã hoá nhị phân slack s_j trong khoảng
        [0, 2^bits - 1] >= capacity (mục 2.3b)."""
        if capacity <= 0:
            return 1
        return max(1, math.ceil(math.log2(capacity + 1)))

    @staticmethod
    def _compute_normalized_weights(problem: AllocationProblem) -> np.ndarray:
        """w_ij chuẩn hoá về [0, 1] (mục 2.3, "Cần chuẩn hóa wi,j").

        Lấy từ priority vector p của Prediction ứng với gNB j — đúng cơ
        chế "p nhân trực tiếp vào hệ số ưu tiên slice" mô tả ở tài liệu
        LSTM mục 8. Baseline p0=(1,1,1) khi gNB đó không có Prediction
        (không khẩn cấp).
        """
        n_slices, n_stations = len(problem.slices), len(problem.stations)
        raw = np.ones((n_slices, n_stations))
        for j, st in enumerate(problem.stations):
            pred = problem.prediction_for(st.gnb_id)
            if pred is None:
                continue
            for i, s in enumerate(problem.slices):
                raw[i, j] = pred.priority.weight_for(s.slice_type)
        max_w = raw.max()
        return raw / max_w if max_w > 0 else raw
