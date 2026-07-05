"""``quantaslice.pipeline`` — Application/Orchestration layer (Layer 3).

Đây là package DUY NHẤT được phép import cả ``quantaslice.quantum`` lẫn
``quantaslice.orchestrator`` (cùng bootstrap provider tạm của ``ai``) —
mọi package khác (dashboard, cli, tests, ví dụ) chỉ nên import từ đây,
không import trực tiếp ``quantum``/``ai``/``orchestrator``.

Bề mặt public:

    from quantaslice.pipeline import DependencyContainer, Runner

    runner = DependencyContainer.build_runner(config, slices=..., stations=...)
    frame = runner.run_once(window)
"""

from quantaslice.orchestrator import orchestrator_registry
from quantaslice.pipeline.container import DependencyContainer
from quantaslice.pipeline.registries import prediction_provider_registry
from quantaslice.pipeline.runner import Runner

__all__ = [
    "Runner",
    "DependencyContainer",
    "prediction_provider_registry",
    "orchestrator_registry",
]
