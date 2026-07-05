"""Protocol (interface hành vi) dùng chung — nền tảng của Dependency
Inversion trong QuantaSlice.

Dùng ``typing.Protocol`` (structural typing / duck-typing tĩnh) thay vì
``abc.ABC`` để:

1. Provider cụ thể (vd ``LSTMPredictionProvider``) không bị ép phải kế
   thừa từ package ``core`` — chỉ cần khớp chữ ký phương thức.
2. Dễ viết fake/stub trong unit test mà không cần import class thật.
3. Cho phép package ``ai`` implement ``PredictionProvider`` mà KHÔNG cần
   import bất kỳ thứ gì từ package ``quantum``, và ngược lại — cả hai chỉ
   cùng phụ thuộc vào ``core.protocols`` và ``core.types``.

Mọi package con (ai, quantum, orchestrator, simulation) implement các
Protocol này; package ``pipeline`` chỉ thao tác qua Protocol, không bao
giờ import class cụ thể ngoài lúc đăng ký ở registry.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from quantaslice.core.types import (
    AllocationProblem,
    FeatureWindow,
    OptimizationResult,
    Prediction,
)

__all__ = [
    "PredictionProvider",
    "OptimizationSolver",
    "SliceOrchestratorPort",
    "TelemetrySource",
]


@runtime_checkable
class PredictionProvider(Protocol):
    """Bề mặt duy nhất mà package ``ai`` phải hiện thực hoá.

    Implementation: ``MockPredictionProvider``, ``RandomPredictionProvider``,
    ``CSVPredictionProvider``, và sau cùng ``LSTMPredictionProvider`` — tất
    cả có thể hoán đổi cho nhau mà không cần đổi code gọi (Liskov
    substitution).
    """

    def predict(self, window: FeatureWindow) -> Prediction:
        """Nhận một cửa sổ đặc trưng của 1 gNB, trả về Prediction (ê, p).

        Implementation không được raise exception cho input hợp lệ; nếu
        không đủ dữ liệu để dự đoán, nên trả về Prediction với
        ``emergency_flag=False`` và priority baseline (1, 1, 1) thay vì
        raise, để pipeline không phải xử lý trường hợp đặc biệt.
        """
        ...


@runtime_checkable
class OptimizationSolver(Protocol):
    """Bề mặt duy nhất mà package ``quantum`` phải hiện thực hoá bên
    trong (QAOAAerSolver, QAOAIBMQSolver, ClassicalGreedySolver).

    Đây KHÔNG phải là API public của package quantum — API public duy
    nhất là hàm ``quantaslice.quantum.solve()``, hàm đó nội bộ chọn một
    instance của Protocol này từ registry rồi gọi ``.solve()``.
    """

    def solve(self, problem: AllocationProblem) -> OptimizationResult:
        """Giải bài toán phân bổ, trả về OptimizationResult.

        Implementation phải raise ``SolverError`` nếu backend lỗi, và
        raise ``InfeasibleAllocationError`` nếu không tìm được bất kỳ
        nghiệm khả thi nào sau khi giải mã.
        """
        ...


@runtime_checkable
class SliceOrchestratorPort(Protocol):
    """Bề mặt mà package ``orchestrator`` hiện thực hoá để áp dụng kết
    quả phân bổ lên hạ tầng (thật hoặc mock)."""

    def apply(self, result: OptimizationResult) -> None:
        """Áp dụng OptimizationResult lên hạ tầng slice hiện tại."""
        ...

    def rollback(self) -> None:
        """Hoàn tác về trạng thái phân bổ trước đó nếu ``apply`` thất bại
        hoặc bị phát hiện infeasible sau khi áp dụng."""
        ...


@runtime_checkable
class TelemetrySource(Protocol):
    """Bề mặt mà package ``simulation`` (hoặc nguồn telemetry thật) hiện
    thực hoá để phát sinh luồng FeatureWindow."""

    def stream(self) -> Iterator[FeatureWindow]:
        """Sinh liên tục các FeatureWindow theo thời gian (generator).

        Với dữ liệu demo (Italian Telecom dataset), implementation áp
        dụng interpolation + resample về 100ms như mô tả trong tài liệu
        LSTM mục 3.1.
        """
        ...
