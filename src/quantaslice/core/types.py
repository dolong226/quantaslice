"""Shared data models used throughout QuantaSlice.
This module defines the common dataclass and enums shared across all packages. 
All dataclasses use ``frozen=True`` and ``slots=True`` for immutability 
and lower memory usage.
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
    EMBB = "eMBB" # Enhanced Mobile Broadband
    URLLC = "URLLC" # Ultra-Reliable Low-latency Communication
    MMTC = "mMTC" # Massive Machine-Type Communication


@dataclass(frozen=True, slots=True)
class PriorityVector:
    """p = (p_eMBB, p_URLLC, p_mMTC), LSTM softplus output.
    """

    embb: float
    urllc: float
    mmtc: float

    def __post_init__(self) -> None:
        for name, value in (("embb", self.embb), ("urllc", self.urllc), ("mmtc", self.mmtc)):
            if value < 0:
                raise SchemaValidationError(
                    f"PriorityVector.{name} must be >= 0 (softplus output), receive {value}"
                )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.embb, self.urllc, self.mmtc)

    def weight_for(self, slice_type: SliceType) -> float:
        mapping = {
            SliceType.EMBB: self.embb,
            SliceType.URLLC: self.urllc,
            SliceType.MMTC: self.mmtc,
        }
        return mapping[slice_type]


@dataclass(frozen=True, slots=True)
class FeatureWindow:
    """Sliding feature window X_i(t) in R^(T x d_in) for a gNB at a time point - standard input for class `PredictionProvider`.
    T=50 defaults to 5 seconds of history at a frequency of 100ms/step.
    """

    gnb_id: str
    timestamp: datetime
    features: np.ndarray  # shape (T, d_in)
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise SchemaValidationError(
                f"FeatureWindow.features must be shape (T, d_in), receive ndim={self.features.ndim}"
            )
        t, d_in = self.features.shape
        if d_in != len(self.feature_names):
            raise SchemaValidationError(
                "The number of row in features "
                f"({d_in}) does not match the number of feature_names ({len(self.feature_names)})"
            )

    @property
    def window_length(self) -> int:
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        """The characteristic dimension d_in."""
        return self.features.shape[1]


@dataclass(frozen=True, slots=True)
class Prediction:
    """The contract of the system: the output of the `ai` package, 
    and input (according to gNB) of the `quantum` package.
    """

    gnb_id: str
    timestamp: datetime
    emergency_flag: bool
    emergency_prob: float  # raw sigmoid pro, before applying the theta_e threshold
    priority: PriorityVector

    def __post_init__(self) -> None:
        if not (0.0 <= self.emergency_prob <= 1.0):
            raise SchemaValidationError(
                f"Prediction.emergency_prob must be within the range [0, 1], receive {self.emergency_prob}"
            )


@dataclass(frozen=True, slots=True)
class SliceRequest:
    """A network slice request needs to be assigned to a base station."""

    slice_id: str
    slice_type: SliceType
    prb_required: float

    def __post_init__(self) -> None:
        if self.prb_required < 0:
            raise SchemaValidationError(
                f"SliceRequest.prb_required must be >= 0, receive {self.prb_required}"
            )


@dataclass(frozen=True, slots=True)
class BaseStation:
    """Một trạm gốc (gNB) với dung lượng PRB khả dụng."""
    """A base station (gNB) with available PRB capacity"""
    gnb_id: str
    prb_capacity: float

    def __post_init__(self) -> None:
        if self.prb_capacity < 0:
            raise SchemaValidationError(
                f"BaseStation.prb_capacity must be >= 0, receive {self.prb_capacity}"
            )


@dataclass(frozen=True, slots=True)
class AllocationProblem:
    """The aggregated input that the pipeline passes in ``quantaslice.quantum.solve()``.

    This is only input boundary of the quantum package - the quantum package doesn't need to know anything else besides this object.
    """

    slices: tuple[SliceRequest, ...]
    stations: tuple[BaseStation, ...]
    predictions: tuple[Prediction, ...] = field(default_factory=tuple)
    penalty_lambda1: float | None = None  # None => auto: lambda = 3 * max(w_ij)
    penalty_lambda2: float | None = None

    def __post_init__(self) -> None:
        if not self.slices:
            raise SchemaValidationError("AllocationProblem.slices must be not empty")
        if not self.stations:
            raise SchemaValidationError("AllocationProblem.stations must be not empty")

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    def prediction_for(self, gnb_id: str) -> Prediction | None:
        """Find the prediction corresponding to a specific gNB, None if none."""
        for pred in self.predictions:
            if pred.gnb_id == gnb_id:
                return pred
        return None


@dataclass(frozen=True, slots=True)
class QUBOProblem:
    """Internal QUBO representation (Q matrix). Do not export it outside the package 'quantum',
    but define it in the 'core' for test-friendliness.
    """

    q_matrix: np.ndarray  # shape (n, n), symmetry
    variable_map: dict[int, tuple[str, str]]  # k -> (slice_id, gnb_id)
    n_qubits: int
    lambda1: float
    lambda2: float

    def __post_init__(self) -> None:
        rows, cols = self.q_matrix.shape
        if rows != cols:
            raise SchemaValidationError(
                f"QUBOProblem.q_matrix must be square, accept shaped ({rows}, {cols})"
            )
        if rows != self.n_qubits:
            raise SchemaValidationError(
                f"QUBOProblem.q_matrix has size {rows} but n_qubits={self.n_qubits}"
            )
        if not np.allclose(self.q_matrix, self.q_matrix.T, atol=1e-8):
            raise SchemaValidationError("QUBOProblem.q_matrix must be symmetry.")


@dataclass(frozen=True, slots=True)
class Allocation:
    """The result assigns 1 slice to 1 station (None if not served),
    correspondingly the slack variable y_i = 0.
    """

    slice_id: str
    gnb_id: str | None


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """The only output of ``quantaslice.quantum.solve()`` - output boundary of quantum package."""

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
        """The number of slices not served (gnb_id is None) - used to log/monitor
        the level of eMBB sacrifice to save URLLC in an emergency situation.
        """
        return sum(1 for a in self.allocations if a.gnb_id is None)



