"""Các kiểu dữ liệu (contract) miền bài toán (domain) dùng chung giữa mọi
package của QuantaSlice: telemetry -> prediction -> allocation problem ->
optimization result.

Đây là "shared kernel": CHỈ chứa định nghĩa dữ liệu (dataclass, enum),
KHÔNG chứa logic nghiệp vụ. Mọi package con (ai, quantum, orchestrator,
simulation, pipeline) được phép import module này; module này không
import ngược lại bất kỳ package con nào -> không có circular dependency.

Mọi dataclass đều ``frozen=True, slots=True``:
- frozen: đối tượng bất biến sau khi tạo, an toàn khi truyền qua nhiều
  module/thread mà không lo side-effect ẩn.
- slots: tiết kiệm bộ nhớ, tránh gán nhầm field không tồn tại.

Xem thêm :mod:`quantaslice.core.runtime` cho các kiểu dữ liệu cấp hệ
thống (Configuration, SimulationFrame) — được tách riêng để mỗi file
giữ dưới 300 dòng và tách bạch domain-type khỏi system-type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from quantaslice.core.exceptions import SchemaValidationError

__all__ = [
    "SliceType",
    "PriorityVector",
    "FeatureWindow",
    "Prediction",
    "SliceRequest",
    "BaseStation",
    "AllocationProblem",
    "QUBOProblem",
    "Allocation",
    "OptimizationResult",
]


class SliceType(str, Enum):
    """Ba loại network slice theo chuẩn 5G, dùng xuyên suốt hệ thống."""

    EMBB = "eMBB"
    URLLC = "URLLC"
    MMTC = "mMTC"


@dataclass(frozen=True, slots=True)
class PriorityVector:
    """Vector ưu tiên p = (p_eMBB, p_URLLC, p_mMTC), đầu ra softplus của LSTM.

    Xem tài liệu Emergency Detector, mục 5.2 (công thức 2): p ∈ R^3_{≥0}.
    """

    embb: float
    urllc: float
    mmtc: float

    def __post_init__(self) -> None:
        for name, value in (("embb", self.embb), ("urllc", self.urllc), ("mmtc", self.mmtc)):
            if value < 0:
                raise SchemaValidationError(
                    f"PriorityVector.{name} phải >= 0 (softplus output), nhận được {value}"
                )

    def as_tuple(self) -> tuple[float, float, float]:
        """Trả về (p_eMBB, p_URLLC, p_mMTC) theo đúng thứ tự tài liệu."""
        return (self.embb, self.urllc, self.mmtc)

    def weight_for(self, slice_type: SliceType) -> float:
        """Lấy trọng số ưu tiên tương ứng với một loại slice cụ thể."""
        mapping = {
            SliceType.EMBB: self.embb,
            SliceType.URLLC: self.urllc,
            SliceType.MMTC: self.mmtc,
        }
        return mapping[slice_type]


@dataclass(frozen=True, slots=True)
class FeatureWindow:
    """Cửa sổ đặc trưng trượt X_i(t) ∈ R^(T x d_in) cho một gNB tại một
    mốc thời gian — input chuẩn cho :class:`PredictionProvider`.

    Xem tài liệu LSTM, mục 3.3 (Sliding window): T=50 mặc định ứng với
    5 giây lịch sử ở tần số 100ms/bước.
    """

    gnb_id: str
    timestamp: datetime
    features: np.ndarray  # shape (T, d_in)
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise SchemaValidationError(
                f"FeatureWindow.features phải có shape (T, d_in), nhận ndim={self.features.ndim}"
            )
        t, d_in = self.features.shape
        if d_in != len(self.feature_names):
            raise SchemaValidationError(
                "Số cột trong features "
                f"({d_in}) không khớp số lượng feature_names ({len(self.feature_names)})"
            )

    @property
    def window_length(self) -> int:
        """Độ dài chuỗi T của cửa sổ."""
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        """Số chiều đặc trưng d_in."""
        return self.features.shape[1]


@dataclass(frozen=True, slots=True)
class Prediction:
    """Hợp đồng seam quan trọng nhất của toàn hệ thống: output DUY NHẤT
    của package ``ai``, input DUY NHẤT (theo gNB) của package ``quantum``.

    Xem tài liệu LSTM, mục 1 (Interface contract): ê ∈ {0,1}, p ∈ R^3_{≥0}.
    """

    gnb_id: str
    timestamp: datetime
    emergency_flag: bool
    emergency_prob: float  # xác suất sigmoid thô, trước khi áp ngưỡng theta_e
    priority: PriorityVector

    def __post_init__(self) -> None:
        if not (0.0 <= self.emergency_prob <= 1.0):
            raise SchemaValidationError(
                f"Prediction.emergency_prob phải trong [0, 1], nhận được {self.emergency_prob}"
            )


@dataclass(frozen=True, slots=True)
class SliceRequest:
    """Một yêu cầu network slice cần được gán vào một trạm gốc."""

    slice_id: str
    slice_type: SliceType
    prb_required: float

    def __post_init__(self) -> None:
        if self.prb_required < 0:
            raise SchemaValidationError(
                f"SliceRequest.prb_required phải >= 0, nhận được {self.prb_required}"
            )


@dataclass(frozen=True, slots=True)
class BaseStation:
    """Một trạm gốc (gNB) với dung lượng PRB khả dụng."""

    gnb_id: str
    prb_capacity: float

    def __post_init__(self) -> None:
        if self.prb_capacity < 0:
            raise SchemaValidationError(
                f"BaseStation.prb_capacity phải >= 0, nhận được {self.prb_capacity}"
            )


@dataclass(frozen=True, slots=True)
class AllocationProblem:
    """Input tổng hợp mà pipeline truyền vào ``quantaslice.quantum.solve()``.

    Đây là biên (boundary) input duy nhất của package quantum — package
    quantum không cần biết gì thêm ngoài object này.
    """

    slices: tuple[SliceRequest, ...]
    stations: tuple[BaseStation, ...]
    predictions: tuple[Prediction, ...] = field(default_factory=tuple)
    penalty_lambda1: float | None = None  # None => auto: lambda = 3 * max(w_ij)
    penalty_lambda2: float | None = None

    def __post_init__(self) -> None:
        if not self.slices:
            raise SchemaValidationError("AllocationProblem.slices không được rỗng")
        if not self.stations:
            raise SchemaValidationError("AllocationProblem.stations không được rỗng")

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    def prediction_for(self, gnb_id: str) -> Prediction | None:
        """Tìm Prediction ứng với một gNB cụ thể, None nếu không có."""
        for pred in self.predictions:
            if pred.gnb_id == gnb_id:
                return pred
        return None


@dataclass(frozen=True, slots=True)
class QUBOProblem:
    """Biểu diễn QUBO nội bộ (Q matrix). KHÔNG được export ra ngoài
    package ``quantum``, nhưng vẫn định nghĩa ở core để test-friendly.

    Xem tài liệu QUBO, mục 2.5: f(x) = x^T Q x, Q đối xứng.
    """

    q_matrix: np.ndarray  # shape (n, n), đối xứng
    variable_map: dict[int, tuple[str, str]]  # k -> (slice_id, gnb_id)
    n_qubits: int
    lambda1: float
    lambda2: float

    def __post_init__(self) -> None:
        rows, cols = self.q_matrix.shape
        if rows != cols:
            raise SchemaValidationError(
                f"QUBOProblem.q_matrix phải vuông, nhận shape ({rows}, {cols})"
            )
        if rows != self.n_qubits:
            raise SchemaValidationError(
                f"QUBOProblem.q_matrix có kích thước {rows} nhưng n_qubits={self.n_qubits}"
            )
        if not np.allclose(self.q_matrix, self.q_matrix.T, atol=1e-8):
            raise SchemaValidationError("QUBOProblem.q_matrix phải đối xứng (Q = Q^T)")


@dataclass(frozen=True, slots=True)
class Allocation:
    """Kết quả gán 1 slice vào 1 trạm (hoặc None nếu không được phục vụ,
    tương ứng biến slack y_i = 0 trong tài liệu QUBO mục 2.3a)."""

    slice_id: str
    gnb_id: str | None


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Output DUY NHẤT của ``quantaslice.quantum.solve()`` — biên
    (boundary) output của package quantum."""

    allocations: tuple[Allocation, ...]
    objective_value: float
    approximation_ratio: float | None
    solver_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def allocation_for(self, slice_id: str) -> Allocation | None:
        for alloc in self.allocations:
            if alloc.slice_id == slice_id:
                return alloc
        return None

    @property
    def n_unserved(self) -> int:
        """Số slice không được phục vụ (gnb_id is None) — dùng để log/giám sát
        mức độ hy sinh eMBB để cứu URLLC trong tình huống khẩn cấp."""
        return sum(1 for a in self.allocations if a.gnb_id is None)



