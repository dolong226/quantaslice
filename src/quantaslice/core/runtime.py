from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from quantaslice.core.exceptions import SchemaValidationError
from quantaslice.core.types import FeatureWindow, OptimizationResult, Prediction

__all__ = ["Configuration", "SimulationFrame"]


@dataclass(frozen=True, slots=True)
class Configuration:
    """Cấu hình runtime của toàn hệ thống, load từ YAML/env qua
    :mod:`quantaslice.core.config`."""

    prediction_provider: str = "mock"  # "mock" | "random" | "csv" | "lstm"
    solver: str = "qaoa_aer"  # "qaoa_aer" | "qaoa_ibmq" | "classical_greedy"
    orchestrator: str = "mock_oran"
    emergency_threshold: float = 0.5
    qaoa_depth: int = 2
    qaoa_shots: int = 1024
    qaoa_max_iterations: int = 100
    window_length: int = 50
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.emergency_threshold <= 1.0):
            raise SchemaValidationError(
                "Configuration.emergency_threshold phải trong [0, 1], "
                f"nhận được {self.emergency_threshold}"
            )
        if self.qaoa_depth < 1:
            raise SchemaValidationError(
                f"Configuration.qaoa_depth phải >= 1, nhận được {self.qaoa_depth}"
            )
        if self.window_length < 1:
            raise SchemaValidationError(
                f"Configuration.window_length phải >= 1, nhận được {self.window_length}"
            )


@dataclass(frozen=True, slots=True)
class SimulationFrame:
    """Một frame trạng thái toàn hệ thống tại một mốc thời gian, dùng
    cho dashboard/visualization. """

    timestamp: datetime
    windows: tuple[FeatureWindow, ...]
    predictions: tuple[Prediction, ...] | None = None
    result: OptimizationResult | None = None
