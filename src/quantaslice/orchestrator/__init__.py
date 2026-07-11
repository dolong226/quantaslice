"""``quantaslice.orchestrator`` — package độc lập cho slice orchestration.

Bề mặt public: ``orchestrator_registry`` (tra bằng tên qua
:class:`~quantaslice.core.runtime.Configuration.orchestrator`, đúng
pattern ``quantum.solver_registry``).

Ràng buộc kiến trúc: package này CHỈ import từ ``quantaslice.core`` —
KHÔNG import ``quantaslice.ai`` hay ``quantaslice.quantum``.

Modules:
* ``mock_oran_orchestrator`` — in-memory orchestrator cho demo/CI.
* ``e2_interface`` — E2SM-RC message builder + placeholder.
* ``state`` — state machine per-gNB cho dashboard (idle/emergency/resolved).
* ``coloran_loader`` — loader cho ColO-RAN dataset.
"""

from quantaslice.core.protocols import SliceOrchestratorPort
from quantaslice.core.registry import Registry
from quantaslice.orchestrator.e2_interface import E2Interface
from quantaslice.orchestrator.mock_oran_orchestrator import MockOranOrchestrator
from quantaslice.orchestrator.state import OrchestratorState

orchestrator_registry: Registry[SliceOrchestratorPort] = Registry("orchestrator")
orchestrator_registry.register_instance_factory("mock_oran", MockOranOrchestrator)
orchestrator_registry.register_instance_factory("e2_interface", E2Interface)

__all__ = [
    "orchestrator_registry",
    "MockOranOrchestrator",
    "E2Interface",
    "OrchestratorState",
]
