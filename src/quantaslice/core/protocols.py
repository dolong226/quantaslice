"""Shared protocol definitions for QuantaSlice
Define the interfaces userd acress the project with 'typing.Protocol'.
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
    def predict(self, window: FeatureWindow) -> Prediction:
        """Predict from a specific window and return the result.
        """
        ...


@runtime_checkable
class OptimizationSolver(Protocol):
    """Interface for allocation problem solvers.
    """

    def solve(self, problem: AllocationProblem) -> OptimizationResult:
        """
        
        """
        ...


@runtime_checkable
class SliceOrchestratorPort(Protocol):
    """Interface for applying allocation results to the system."""

    def apply(self, result: OptimizationResult) -> None:
        """Applying OptimizationResult to current slice system."""
        ...

    def rollback(self) -> None:
        """Rollback the previous state if 'apply' fails or is found infeasible after application."""
        ...


@runtime_checkable
class TelemetrySource(Protocol):
    """Interface for telemetry data source.
    The data source can come from a simulator or a real system."""

    def stream(self) -> Iterator[FeatureWindow]:
        """Generating FeatureWindow continuosly over time (generator).
        """
        ...
