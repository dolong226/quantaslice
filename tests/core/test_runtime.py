"""Unit test cho quantaslice.core.runtime (Configuration, SimulationFrame)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quantaslice.core.exceptions import SchemaValidationError
from quantaslice.core.runtime import Configuration, SimulationFrame

NOW = datetime(2026, 7, 3, tzinfo=timezone.utc)


class TestConfiguration:
    def test_defaults_are_valid(self):
        cfg = Configuration()
        assert cfg.prediction_provider == "mock"
        assert cfg.solver == "qaoa_aer"

    def test_invalid_threshold_rejected(self):
        with pytest.raises(SchemaValidationError):
            Configuration(emergency_threshold=1.5)

    def test_invalid_qaoa_depth_rejected(self):
        with pytest.raises(SchemaValidationError):
            Configuration(qaoa_depth=0)

    def test_invalid_window_length_rejected(self):
        with pytest.raises(SchemaValidationError):
            Configuration(window_length=0)


class TestSimulationFrame:
    def test_minimal_construction(self):
        frame = SimulationFrame(timestamp=NOW, windows=())
        assert frame.predictions is None
        assert frame.result is None
