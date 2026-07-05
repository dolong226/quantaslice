"""``quantaslice.core`` — shared kernel: dataclass contracts, protocols,
registry, exceptions, config loader.

Đây là package DUY NHẤT mà mọi package khác (ai, quantum, orchestrator,
simulation, dashboard, cli, pipeline) được phép import trực tiếp class cụ
thể từ đó. Package này không import ngược lại bất kỳ package con nào.

Bề mặt public re-export ở đây; các submodule (types, protocols, registry,
config, exceptions) vẫn có thể import trực tiếp nếu cần rõ ràng hơn.
"""

from quantaslice.core.exceptions import (
    ConfigurationError,
    DuplicateRegistrationError,
    InfeasibleAllocationError,
    ProviderNotFoundError,
    QuantaSliceError,
    SchemaValidationError,
    SolverError,
)
from quantaslice.core.protocols import (
    OptimizationSolver,
    PredictionProvider,
    SliceOrchestratorPort,
    TelemetrySource,
)
from quantaslice.core.registry import Registry
from quantaslice.core.runtime import Configuration, SimulationFrame
from quantaslice.core.types import (
    Allocation,
    AllocationProblem,
    BaseStation,
    FeatureWindow,
    OptimizationResult,
    Prediction,
    PriorityVector,
    QUBOProblem,
    SliceRequest,
    SliceType,
)

__all__ = [
    # types
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
    "Configuration",
    "SimulationFrame",
    # protocols
    "PredictionProvider",
    "OptimizationSolver",
    "SliceOrchestratorPort",
    "TelemetrySource",
    # registry
    "Registry",
    # exceptions
    "QuantaSliceError",
    "ConfigurationError",
    "ProviderNotFoundError",
    "DuplicateRegistrationError",
    "SchemaValidationError",
    "InfeasibleAllocationError",
    "SolverError",
]

__version__ = "0.1.0"
