"""Bootstrap/wiring: đọc ``Configuration`` + topology, tra registry, dựng
sẵn một ``Runner`` có thể chạy ngay — đúng "Luồng khởi động (bootstrap /
DI)" ở Phase 7 tài liệu kiến trúc.

Đây là module DUY NHẤT của toàn hệ thống thực sự import cả
``quantaslice.quantum`` lẫn ``quantaslice.orchestrator`` (cùng bootstrap
provider tạm của ``ai``) — mọi nơi khác chỉ thao tác qua ``Runner`` hoặc
qua Protocol.
"""

from __future__ import annotations

# Import để đăng ký (side-effect) các provider bootstrap mặc định vào
# registry trước khi DependencyContainer.build_runner() tra.
import quantaslice.pipeline.bootstrap_providers  # noqa: F401
from quantaslice.core.runtime import Configuration
from quantaslice.core.types import BaseStation, SliceRequest
from quantaslice.orchestrator import orchestrator_registry
from quantaslice.pipeline.registries import prediction_provider_registry
from quantaslice.pipeline.runner import Runner
from quantaslice.quantum import solver_registry

__all__ = ["DependencyContainer"]


class DependencyContainer:
    """Factory tĩnh: ``Configuration`` + topology (slices/stations) ->
    ``Runner`` sẵn sàng chạy.

    Ví dụ dùng (xem thêm ``examples/run_mock_demo.py``)::

        config = load_configuration("configs/default.yaml")
        runner = DependencyContainer.build_runner(
            config, slices=my_slices, stations=my_stations,
        )
        frame = runner.run_once(window)
    """

    @staticmethod
    def build_runner(
        config: Configuration,
        *,
        slices: tuple[SliceRequest, ...],
        stations: tuple[BaseStation, ...],
    ) -> Runner:
        provider = prediction_provider_registry.create(config.prediction_provider)
        solver = solver_registry.create(config.solver, config=config)
        orchestrator = orchestrator_registry.create(config.orchestrator)

        return Runner(
            provider=provider,
            solver=solver,
            orchestrator=orchestrator,
            slices=slices,
            stations=stations,
            config=config,
        )
