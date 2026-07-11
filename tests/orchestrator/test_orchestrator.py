"""Tests cho ``quantaslice.orchestrator`` package.

Kiểm tra:
1. MockOranOrchestrator.apply() — happy path.
2. apply() — infeasible khi toàn bộ unserved.
3. rollback() — khôi phục allocation trước đó.
4. OrchestratorState — state transitions (idle → emergency → resolved).
5. to_dashboard_dict() — format JSON khớp wireframe spec.
6. E2Interface — build_control_message() cấu trúc đúng.
7. ColORANLoader — create_stations/create_slices đúng.
"""

from __future__ import annotations

import pytest

from quantaslice.core.exceptions import InfeasibleAllocationError
from quantaslice.core.types import (
    Allocation,
    BaseStation,
    OptimizationResult,
    SliceRequest,
    SliceType,
)
from quantaslice.orchestrator.e2_interface import E2Interface
from quantaslice.orchestrator.mock_oran_orchestrator import MockOranOrchestrator
from quantaslice.orchestrator.state import (
    EMERGENCY,
    IDLE,
    RESOLVED,
    OrchestratorState,
)
from quantaslice.orchestrator.coloran_loader import ColORANLoader


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def stations() -> tuple[BaseStation, ...]:
    return (
        BaseStation(gnb_id="gnb-1", prb_capacity=50),
        BaseStation(gnb_id="gnb-2", prb_capacity=50),
    )


@pytest.fixture
def slices() -> tuple[SliceRequest, ...]:
    return (
        SliceRequest(slice_id="s0-eMBB", slice_type=SliceType.EMBB, prb_required=17),
        SliceRequest(slice_id="s1-mMTC", slice_type=SliceType.MMTC, prb_required=17),
        SliceRequest(slice_id="s2-URLLC", slice_type=SliceType.URLLC, prb_required=17),
    )


@pytest.fixture
def valid_result() -> OptimizationResult:
    return OptimizationResult(
        allocations=(
            Allocation(slice_id="s0-eMBB", gnb_id="gnb-1"),
            Allocation(slice_id="s1-mMTC", gnb_id="gnb-2"),
            Allocation(slice_id="s2-URLLC", gnb_id="gnb-1"),
        ),
        objective_value=-3.14,
        approximation_ratio=0.95,
        solver_name="classical_greedy",
        metadata={
            "prb_s0-eMBB": 20,
            "prb_s1-mMTC": 15,
            "prb_s2-URLLC": 15,
        },
    )


@pytest.fixture
def all_unserved_result() -> OptimizationResult:
    return OptimizationResult(
        allocations=(
            Allocation(slice_id="s0-eMBB", gnb_id=None),
            Allocation(slice_id="s1-mMTC", gnb_id=None),
            Allocation(slice_id="s2-URLLC", gnb_id=None),
        ),
        objective_value=0.0,
        approximation_ratio=None,
        solver_name="qaoa_aer",
    )


# ── MockOranOrchestrator tests ────────────────────────────────────────


class TestMockOranOrchestrator:
    """Tests cho MockOranOrchestrator."""

    def test_apply_valid_allocation(self, stations, slices, valid_result):
        """Happy path — apply() thành công, current_allocation được cập nhật."""
        orch = MockOranOrchestrator(stations=stations, slices=slices)
        orch.apply(valid_result)
        assert orch.current_allocation is valid_result

    def test_apply_no_stations_backward_compat(self, valid_result):
        """Backward compat — gọi MockOranOrchestrator() không tham số vẫn hoạt động."""
        orch = MockOranOrchestrator()
        orch.apply(valid_result)
        assert orch.current_allocation is valid_result

    def test_apply_all_unserved_raises(self, stations, slices, all_unserved_result):
        """Toàn bộ unserved → InfeasibleAllocationError."""
        orch = MockOranOrchestrator(stations=stations, slices=slices)
        with pytest.raises(InfeasibleAllocationError):
            orch.apply(all_unserved_result)

    def test_rollback_restores_previous(self, stations, slices, valid_result):
        """rollback() khôi phục allocation trước đó."""
        orch = MockOranOrchestrator(stations=stations, slices=slices)

        # First apply — previous = None.
        orch.apply(valid_result)
        assert orch.current_allocation is valid_result

        # Second apply.
        result2 = OptimizationResult(
            allocations=(Allocation(slice_id="s0-eMBB", gnb_id="gnb-2"),),
            objective_value=-2.0,
            approximation_ratio=0.90,
            solver_name="qaoa_aer",
        )
        orch.apply(result2)
        assert orch.current_allocation is result2

        # Rollback.
        orch.rollback()
        # After rollback, current should be deep copy of first result.
        assert orch.current_allocation is not None
        assert orch.current_allocation.solver_name == valid_result.solver_name

    def test_rollback_after_infeasible(self, stations, slices, all_unserved_result, valid_result):
        """Sau apply() infeasible, rollback() an toàn."""
        orch = MockOranOrchestrator(stations=stations, slices=slices)
        orch.apply(valid_result)

        with pytest.raises(InfeasibleAllocationError):
            orch.apply(all_unserved_result)

        # Rollback should restore to valid_result.
        orch.rollback()
        assert orch.current_allocation is not None

    def test_set_emergency_gnb(self, stations, slices, valid_result):
        """set_emergency_gnb() cập nhật state machine."""
        orch = MockOranOrchestrator(stations=stations, slices=slices)
        orch.set_emergency_gnb("gnb-1")
        orch.apply(valid_result)

        state = orch.state
        assert state is not None
        assert state.get_gnb_status("gnb-1") == EMERGENCY

    def test_reset(self, stations, slices, valid_result):
        """reset() đưa tất cả gNB về idle."""
        orch = MockOranOrchestrator(stations=stations, slices=slices)
        orch.set_emergency_gnb("gnb-1")
        orch.apply(valid_result)
        orch.reset()

        state = orch.state
        assert state is not None
        assert state.get_gnb_status("gnb-1") == IDLE

    def test_get_dashboard_state(self, stations, slices, valid_result):
        """get_dashboard_state() trả về dict có đủ keys."""
        orch = MockOranOrchestrator(stations=stations, slices=slices)
        orch.apply(valid_result)

        dashboard = orch.get_dashboard_state()
        assert "stations" in dashboard
        assert "event_log" in dashboard
        assert "solver_name" in dashboard
        assert "timestamp" in dashboard
        assert dashboard["solver_name"] == "classical_greedy"


# ── OrchestratorState tests ───────────────────────────────────────────


class TestOrchestratorState:
    """Tests cho OrchestratorState."""

    def test_initial_state_all_idle(self, stations, slices):
        """Ban đầu, tất cả gNB ở trạng thái idle."""
        state = OrchestratorState(stations=stations, slices=slices)
        for gid in ["gnb-1", "gnb-2"]:
            assert state.get_gnb_status(gid) == IDLE

    def test_apply_result_with_emergency(self, stations, slices, valid_result):
        """apply_result() với emergency_gnb → trạm đó chuyển sang EMERGENCY."""
        state = OrchestratorState(stations=stations, slices=slices)
        state.apply_result(valid_result, emergency_gnb="gnb-1")

        assert state.get_gnb_status("gnb-1") == EMERGENCY
        assert state.get_gnb_status("gnb-2") == IDLE

    def test_emergency_then_normal_becomes_resolved(self, stations, slices, valid_result):
        """EMERGENCY → apply bình thường → RESOLVED."""
        state = OrchestratorState(stations=stations, slices=slices)

        # First: emergency.
        state.apply_result(valid_result, emergency_gnb="gnb-1")
        assert state.get_gnb_status("gnb-1") == EMERGENCY

        # Second: no emergency.
        state.apply_result(valid_result, emergency_gnb=None)
        assert state.get_gnb_status("gnb-1") == RESOLVED

    def test_resolved_then_normal_becomes_idle(self, stations, slices, valid_result):
        """RESOLVED → apply bình thường → IDLE."""
        state = OrchestratorState(stations=stations, slices=slices)

        state.apply_result(valid_result, emergency_gnb="gnb-1")  # → EMERGENCY
        state.apply_result(valid_result, emergency_gnb=None)  # → RESOLVED
        state.apply_result(valid_result, emergency_gnb=None)  # → IDLE

        assert state.get_gnb_status("gnb-1") == IDLE

    def test_reset_all_to_idle(self, stations, slices, valid_result):
        """reset() đưa tất cả gNB về IDLE."""
        state = OrchestratorState(stations=stations, slices=slices)
        state.apply_result(valid_result, emergency_gnb="gnb-1")
        state.reset()
        assert state.get_gnb_status("gnb-1") == IDLE

    def test_event_log_grows(self, stations, slices, valid_result):
        """Event log ghi nhận mỗi apply_result()."""
        state = OrchestratorState(stations=stations, slices=slices)
        assert len(state.event_log) == 0

        state.apply_result(valid_result)
        assert len(state.event_log) > 0

    def test_to_dashboard_dict_format(self, stations, slices, valid_result):
        """to_dashboard_dict() trả về format khớp wireframe."""
        state = OrchestratorState(stations=stations, slices=slices)
        state.apply_result(valid_result)

        d = state.to_dashboard_dict()
        assert "stations" in d
        assert "gnb-1" in d["stations"]
        assert "gnb-2" in d["stations"]

        gnb1 = d["stations"]["gnb-1"]
        assert "status" in gnb1
        assert "prb_used" in gnb1
        assert "prb_capacity" in gnb1
        assert gnb1["prb_capacity"] == 50
        assert "slices" in gnb1
        assert "history" in gnb1
        assert len(gnb1["history"]) == 10  # _HISTORY_LENGTH

        assert "event_log" in d
        assert "solver_name" in d
        assert "timestamp" in d

    def test_history_tracks_emergency(self, stations, slices, valid_result):
        """History list ghi đúng 1 (emergency) / 0 (normal)."""
        state = OrchestratorState(stations=stations, slices=slices)

        state.apply_result(valid_result, emergency_gnb="gnb-1")
        d = state.to_dashboard_dict()
        history = d["stations"]["gnb-1"]["history"]
        assert history[-1] == 1  # Last entry is emergency

        state.apply_result(valid_result, emergency_gnb=None)
        d = state.to_dashboard_dict()
        history = d["stations"]["gnb-1"]["history"]
        assert history[-1] == 0  # Resolved, not emergency


# ── E2Interface tests ─────────────────────────────────────────────────


class TestE2Interface:
    """Tests cho E2Interface message builders."""

    def test_build_control_message_structure(self):
        """build_control_message() trả về đúng E2SM-RC structure."""
        alloc = Allocation(slice_id="s2-URLLC", gnb_id="gnb-1")
        msg = E2Interface.build_control_message(alloc)

        assert "e2sm_rc" in msg
        assert "ric_control_header" in msg["e2sm_rc"]
        assert "ric_control_message" in msg["e2sm_rc"]

        params = msg["e2sm_rc"]["ric_control_message"]["ran_parameter_list"]
        assert len(params) == 5

        target_cell = next(p for p in params if p["ran_parameter_name"] == "target_cell_global_id")
        assert target_cell["ran_parameter_value"] == "gnb-1"

        slice_param = next(p for p in params if p["ran_parameter_name"] == "slice_id")
        assert slice_param["ran_parameter_value"] == "s2-URLLC"

    def test_build_policy_message(self):
        """build_policy_message() trả về scheduling policy message."""
        msg = E2Interface.build_policy_message("gnb-1", "s0-eMBB", 2)
        assert msg["e2sm_rc"]["ric_policy_message"]["scheduling_policy"] == 2
        assert msg["e2sm_rc"]["ric_policy_message"]["scheduling_policy_name"] == "proportionally_fair"

    def test_build_indication_message(self):
        """build_indication_message() trả về KPM indication."""
        kpms = {"dl_buffer": 50000, "tx_brate": 12.5}
        msg = E2Interface.build_indication_message("gnb-1", kpms)
        assert msg["e2sm_kpm"]["ric_indication_header"]["cell_global_id"] == "gnb-1"
        assert msg["e2sm_kpm"]["ric_indication_message"]["measurement_data"] == kpms


# ── ColORANLoader tests ──────────────────────────────────────────────


class TestColORANLoader:
    """Tests cho ColORANLoader factory methods."""

    def test_create_stations_default(self):
        """create_stations() mặc định 7 BSs, 50 PRBs."""
        stations = ColORANLoader.create_stations()
        assert len(stations) == 7
        assert stations[0].gnb_id == "bs-1"
        assert stations[0].prb_capacity == 50
        assert stations[6].gnb_id == "bs-7"

    def test_create_stations_custom(self):
        """create_stations() cho phép tùy chỉnh."""
        stations = ColORANLoader.create_stations(n_stations=2, prb_capacity=25)
        assert len(stations) == 2
        assert stations[0].prb_capacity == 25

    def test_create_slices(self):
        """create_slices() tạo 3 slices eMBB/mMTC/URLLC."""
        slices = ColORANLoader.create_slices()
        assert len(slices) == 3
        types = {s.slice_type for s in slices}
        assert types == {SliceType.EMBB, SliceType.MMTC, SliceType.URLLC}

    def test_create_slices_custom_prb(self):
        """create_slices() cho phép tùy chỉnh PRB per slice."""
        slices = ColORANLoader.create_slices(prb_per_slice={
            "s0-eMBB": 25,
            "s1-mMTC": 10,
            "s2-URLLC": 15,
        })
        assert slices[0].prb_required == 25  # eMBB
        assert slices[1].prb_required == 10  # mMTC
        assert slices[2].prb_required == 15  # URLLC
