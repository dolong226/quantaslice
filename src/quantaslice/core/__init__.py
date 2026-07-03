"""Shared core abstractions for QuantaSlice.

Provides the common data models, protocols, configuration utilities,
registry, and exceptions used across the project.
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
