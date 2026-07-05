"""Unit test cho quantaslice.core.types — đảm bảo mọi dataclass contract
validate đúng và bất biến (frozen)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import numpy as np
import pytest

from quantaslice.core.exceptions import SchemaValidationError
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

NOW = datetime(2026, 7, 3, tzinfo=timezone.utc)


def make_priority(embb=1.0, urllc=1.0, mmtc=1.0) -> PriorityVector:
    return PriorityVector(embb=embb, urllc=urllc, mmtc=mmtc)


class TestPriorityVector:
    def test_valid_construction(self):
        p = make_priority(0.5, 9.0, 0.1)
        assert p.as_tuple() == (0.5, 9.0, 0.1)

    def test_negative_value_rejected(self):
        with pytest.raises(SchemaValidationError):
            make_priority(-1.0, 1.0, 1.0)

    def test_weight_for_slice_type(self):
        p = make_priority(1.0, 8.0, 0.2)
        assert p.weight_for(SliceType.URLLC) == 8.0
        assert p.weight_for(SliceType.EMBB) == 1.0

    def test_is_frozen(self):
        p = make_priority()
        with pytest.raises(FrozenInstanceError):
            p.embb = 5.0  # type: ignore[misc]


class TestFeatureWindow:
    def test_valid_construction(self):
        features = np.zeros((50, 9))
        names = tuple(f"f{i}" for i in range(9))
        window = FeatureWindow(gnb_id="gnb-1", timestamp=NOW, features=features, feature_names=names)
        assert window.window_length == 50
        assert window.n_features == 9

    def test_mismatched_feature_names_rejected(self):
        features = np.zeros((10, 5))
        with pytest.raises(SchemaValidationError):
            FeatureWindow(gnb_id="gnb-1", timestamp=NOW, features=features, feature_names=("a", "b"))

    def test_wrong_ndim_rejected(self):
        features = np.zeros((10,))
        with pytest.raises(SchemaValidationError):
            FeatureWindow(gnb_id="gnb-1", timestamp=NOW, features=features, feature_names=("a",))


class TestPrediction:
    def test_valid_construction(self):
        pred = Prediction(
            gnb_id="gnb-1",
            timestamp=NOW,
            emergency_flag=True,
            emergency_prob=0.87,
            priority=make_priority(1.0, 9.5, 0.3),
        )
        assert pred.emergency_flag is True

    def test_prob_out_of_range_rejected(self):
        with pytest.raises(SchemaValidationError):
            Prediction(
                gnb_id="gnb-1",
                timestamp=NOW,
                emergency_flag=False,
                emergency_prob=1.5,
                priority=make_priority(),
            )


class TestAllocationProblem:
    def test_requires_non_empty_slices_and_stations(self):
        with pytest.raises(SchemaValidationError):
            AllocationProblem(slices=(), stations=(BaseStation("gnb-1", 100.0),))
        with pytest.raises(SchemaValidationError):
            AllocationProblem(
                slices=(SliceRequest("s1", SliceType.URLLC, 10.0),), stations=()
            )

    def test_prediction_lookup(self):
        pred = Prediction("gnb-1", NOW, True, 0.9, make_priority(1, 9, 0.2))
        problem = AllocationProblem(
            slices=(SliceRequest("s1", SliceType.URLLC, 10.0),),
            stations=(BaseStation("gnb-1", 100.0),),
            predictions=(pred,),
        )
        assert problem.prediction_for("gnb-1") is pred
        assert problem.prediction_for("gnb-does-not-exist") is None
        assert problem.n_slices == 1
        assert problem.n_stations == 1


class TestQUBOProblem:
    def test_valid_symmetric_matrix(self):
        q = np.array([[1.0, 2.0], [2.0, 3.0]])
        problem = QUBOProblem(
            q_matrix=q, variable_map={0: ("s1", "gnb-1"), 1: ("s1", "gnb-2")}, n_qubits=2,
            lambda1=1.0, lambda2=1.0,
        )
        assert problem.n_qubits == 2

    def test_non_square_rejected(self):
        q = np.zeros((2, 3))
        with pytest.raises(SchemaValidationError):
            QUBOProblem(q_matrix=q, variable_map={}, n_qubits=2, lambda1=1.0, lambda2=1.0)

    def test_non_symmetric_rejected(self):
        q = np.array([[1.0, 2.0], [0.0, 3.0]])
        with pytest.raises(SchemaValidationError):
            QUBOProblem(q_matrix=q, variable_map={}, n_qubits=2, lambda1=1.0, lambda2=1.0)

    def test_mismatched_n_qubits_rejected(self):
        q = np.eye(3)
        with pytest.raises(SchemaValidationError):
            QUBOProblem(q_matrix=q, variable_map={}, n_qubits=2, lambda1=1.0, lambda2=1.0)


class TestOptimizationResult:
    def test_allocation_lookup_and_unserved_count(self):
        result = OptimizationResult(
            allocations=(
                Allocation("s1", "gnb-1"),
                Allocation("s2", None),
            ),
            objective_value=42.0,
            approximation_ratio=0.85,
            solver_name="qaoa_aer",
        )
        assert result.allocation_for("s1").gnb_id == "gnb-1"
        assert result.allocation_for("nope") is None
        assert result.n_unserved == 1

