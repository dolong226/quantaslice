"""``Runner`` — điều phối luồng end-to-end theo đúng Phase 7 (Execution
Flow) của tài liệu kiến trúc:

    FeatureWindow -> Prediction -> (nếu emergency) AllocationProblem
    -> OptimizationResult -> orchestrator.apply() -> SimulationFrame

Đây là module duy nhất trong toàn hệ thống thực sự "biết" cả
``PredictionProvider``, ``OptimizationSolver`` và ``SliceOrchestratorPort``
— nhưng chỉ qua Protocol (``quantaslice.core.protocols``), không bao giờ
import class cụ thể của ``ai``/``quantum``/``orchestrator`` ở đây (việc
đó là của ``container.py``).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Iterator

from quantaslice.core.exceptions import QuantaSliceError
from quantaslice.core.protocols import (
    OptimizationSolver,
    PredictionProvider,
    SliceOrchestratorPort,
    TelemetrySource,
)
from quantaslice.core.runtime import Configuration, SimulationFrame
from quantaslice.core.types import (
    AllocationProblem,
    BaseStation,
    FeatureWindow,
    OptimizationResult,
    Prediction,
    SliceRequest,
)

__all__ = ["Runner"]

logger = logging.getLogger(__name__)


class Runner:
    """Vòng lặp reactive: mỗi ``FeatureWindow`` mới (1 gNB) đi qua
    ``run_once()`` hoặc liên tục qua ``run_forever()``.

    ``slices``/``stations`` mô tả "topology" hiện hành (yêu cầu slice
    nào cần phục vụ, trạm nào có bao nhiêu PRB) — độc lập với luồng
    telemetry/predictions, và có thể cập nhật qua ``update_*()`` bất cứ
    lúc nào (dashboard/CLI tương lai gọi khi topology thay đổi).
    """

    def __init__(
        self,
        *,
        provider: PredictionProvider,
        solver: OptimizationSolver,
        orchestrator: SliceOrchestratorPort,
        slices: tuple[SliceRequest, ...],
        stations: tuple[BaseStation, ...],
        config: Configuration | None = None,
    ) -> None:
        self._provider = provider
        self._solver = solver
        self._orchestrator = orchestrator
        self._slices = slices
        self._stations = stations
        self._config = config or Configuration()
        self._predictions_by_gnb: dict[str, Prediction] = {}
        self._last_result: OptimizationResult | None = None

    # ------------------------------------------------------------------
    # Cấu hình động — dashboard/CLI tương lai gọi khi topology thay đổi
    # ------------------------------------------------------------------
    def update_slices(self, slices: tuple[SliceRequest, ...]) -> None:
        self._slices = slices

    def update_stations(self, stations: tuple[BaseStation, ...]) -> None:
        self._stations = stations

    @property
    def last_result(self) -> OptimizationResult | None:
        """OptimizationResult hiện hành (None nếu chưa tối ưu lần nào)."""
        return self._last_result

    @property
    def latest_predictions(self) -> tuple[Prediction, ...]:
        """Prediction gần nhất theo từng gNB đã từng thấy."""
        return tuple(self._predictions_by_gnb.values())

    # ------------------------------------------------------------------
    # Luồng chính
    # ------------------------------------------------------------------
    def run_once(self, window: FeatureWindow, *, force_reoptimize: bool = False) -> SimulationFrame:
        """Xử lý 1 FeatureWindow — bước 2 đến 7 trong Phase 7 (mục
        "Luồng runtime")."""
        prediction = self._provider.predict(window)
        prediction = self._apply_threshold(prediction)
        self._predictions_by_gnb[prediction.gnb_id] = prediction

        should_reoptimize = (
            force_reoptimize or prediction.emergency_flag or self._last_result is None
        )
        if should_reoptimize:
            self._reoptimize()
        # else: giữ nguyên allocation hiện tại — đúng note trong tài
        # liệu LSTM mục 8: "tránh tái tối ưu không cần thiết".

        return SimulationFrame(
            timestamp=window.timestamp,
            windows=(window,),
            predictions=tuple(self._predictions_by_gnb.values()),
            result=self._last_result,
        )

    def run_forever(self, telemetry: TelemetrySource) -> Iterator[SimulationFrame]:
        """Sinh liên tục SimulationFrame theo từng FeatureWindow phát ra
        từ nguồn telemetry (thật hoặc ``simulation.StreamSimulator``)."""
        for window in telemetry.stream():
            yield self.run_once(window)

    # ------------------------------------------------------------------
    # Nội bộ
    # ------------------------------------------------------------------
    def _apply_threshold(self, prediction: Prediction) -> Prediction:
        """Cho phép tinh chỉnh ngưỡng quyết định theta_e ở tầng pipeline
        (tài liệu LSTM mục 8: "hiệu chỉnh theo đường cong precision–recall")
        mà không cần huấn luyện lại provider — provider chỉ cần trả về
        xác suất thô ``emergency_prob``."""
        flag = prediction.emergency_prob >= self._config.emergency_threshold
        if flag == prediction.emergency_flag:
            return prediction
        return dataclasses.replace(prediction, emergency_flag=flag)

    def _reoptimize(self) -> None:
        problem = AllocationProblem(
            slices=self._slices,
            stations=self._stations,
            predictions=tuple(self._predictions_by_gnb.values()),
        )
        try:
            result = self._solver.solve(problem)
        except QuantaSliceError as exc:
            # Solver lỗi/infeasible -> giữ nguyên allocation hiện tại
            # thay vì crash toàn bộ pipeline (an toàn hơn cho vòng lặp
            # reactive chạy real-time) — nhưng PHẢI log rõ lý do, không
            # được im lặng (lỗi thiết kế đã sửa: trước đây chỉ "return"
            # khiến không debug được vì sao allocation không đổi).
            logger.warning("Solver '%s' thất bại, giữ nguyên allocation hiện tại: %s", self._solver, exc)
            return

        try:
            self._orchestrator.apply(result)
        except QuantaSliceError as exc:
            logger.warning("Orchestrator.apply() thất bại, rollback: %s", exc)
            self._orchestrator.rollback()
            return

        self._last_result = result
