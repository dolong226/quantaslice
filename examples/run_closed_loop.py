"""Closed-loop evaluation (§6.4): vector ưu tiên ``p`` của detector có
cải thiện tỉ lệ thoả SLA so với trọng số TĨNH không?

Đây là số đo GIÁ TRỊ THẬT của tầng AI — độc lập với việc phát hiện khó
hay dễ. Lập luận: trong mạng thật, slice nào "nguy cấp" THAY ĐỔI theo
thời gian; một chính sách ưu tiên cố định không thể luôn đúng, còn ``p``
thích ứng thì có thể. Khi tài nguyên khan hiếm (phải rớt bớt slice),
chọn rớt đúng slice ít nguy cấp sẽ giảm vi phạm SLA.

Chống "rigging" (đánh giá vòng tròn, plan §2):
  * criticality THẬT = lịch tiêm emergency (ground truth, luân phiên slice);
  * ``p`` (adaptive) = detector đọc severity độ-trễ từ KPM tiêm (physical);
  * metric = chi phí vi phạm dùng criticality THẬT, KHÔNG dùng ``p``.
So 3 chính sách: STATIC (cố định) / ADAPTIVE (detector) / ORACLE (biết thật).

LƯU Ý trung thực: đây là kịch bản MÔ HÌNH HOÁ (ColO-RAN static không có
emergency thật). Độ lớn cải thiện phụ thuộc thiết kế kịch bản (mức độ
luân phiên criticality, độ khan hiếm). Nguyên lý — "thích ứng thắng cố
định KHI criticality biến thiên" — là thật; nếu criticality cố định thì
một chính sách tĩnh chỉnh đúng sẽ hoà.

Chạy::

    python -m examples.run_closed_loop
    python -m examples.run_closed_loop --solver qaoa_aer   # dùng QAOA thật
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

import quantaslice.quantum  # noqa: F401  (đăng ký solver)
from quantaslice.ai.data.labeling import LabelConfig, _slice_severity
from quantaslice.ai.data.loaders import RAW_METRICS, BSFrame
from quantaslice.ai.data.scenario import EmergencyEvent, inject_emergencies
from quantaslice.core.types import (
    AllocationProblem, BaseStation, Prediction, PriorityVector, SliceRequest, SliceType,
)
from quantaslice.quantum import solve

_BUF = RAW_METRICS.index("dl_buffer")
_BRATE = RAW_METRICS.index("tx_brate_dl")
_SLICE_TYPES = (SliceType.EMBB, SliceType.MMTC, SliceType.URLLC)  # index 0,1,2
_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class PolicyResult:
    sla_cost: float
    critical_served_rate: float


def build_scenario(n_steps: int = 420, seed: int = 0):
    """Nền sạch 3 slice (buffer rỗng, runway lớn) + tiêm emergency LUÂN
    PHIÊN eMBB→mMTC→URLLC. Trả (frame, crit_true (T,3), demand (T,3))."""
    series = np.zeros((n_steps, 3, len(RAW_METRICS)))
    for s in range(3):
        series[:, s, _BUF] = 2000.0
        series[:, s, _BRATE] = 4.0
        series[:, s, 2] = 10.0; series[:, s, 3] = 10.0; series[:, s, 4] = 12.0
    frame = BSFrame(0, 0, 1, "bs1", np.arange(n_steps) * 0.25, series, np.array([2., 2., 2.]))

    crit = np.zeros((n_steps, 3))
    events: dict[int, list[EmergencyEvent]] = {0: [], 1: [], 2: []}
    seg = n_steps // 6
    for k, onset in enumerate(range(20, n_steps - 70, seg)):
        s = k % 3
        events[s].append(EmergencyEvent(s, onset, ramp_steps=40, hold_steps=25, intensity=2.4))
        crit[onset:onset + 65, s] = 1.0
    for s, evs in events.items():
        if evs:
            frame, _ = inject_emergencies(frame, tuple(evs))

    demand = 18.0 + 4.0 * crit
    return frame, crit, demand


def _priority(policy: str, sev_t: np.ndarray, crit_t: np.ndarray) -> np.ndarray:
    if policy == "static":
        return np.array([2.0, 1.0, 3.0])           # cố định: URLLC>eMBB>mMTC
    if policy == "adaptive":
        return 1.0 + 9.0 * sev_t                    # detector
    return 1.0 + 9.0 * crit_t                       # oracle


def _pv(vec: np.ndarray) -> PriorityVector:
    # vec theo [eMBB, mMTC, URLLC] -> PriorityVector(embb, urllc, mmtc)
    return PriorityVector(embb=float(vec[0]), mmtc=float(vec[1]), urllc=float(vec[2]))


def evaluate_policy(
    frame: BSFrame, crit: np.ndarray, demand: np.ndarray, policy: str, *,
    stations=(BaseStation("g1", 25.0), BaseStation("g2", 25.0)),
    step: int = 6, solver_name: str = "classical_greedy",
) -> PolicyResult:
    """Chạy QUBO theo từng bước dưới một chính sách ưu tiên, trả chi phí
    vi phạm SLA (thấp = tốt) và tỉ lệ slice nguy cấp được phục vụ."""
    sev = np.column_stack([
        _slice_severity(frame, s, LabelConfig(latency_budgets=(0.3, 0.3, 0.3)))
        for s in range(3)
    ])
    cost = 0.0
    served_crit = 0
    crit_steps = 0
    for t in range(0, frame.n_steps, step):
        p = _priority(policy, sev[t], crit[t])
        slices = tuple(
            SliceRequest(f"s{i}", _SLICE_TYPES[i], float(demand[t, i])) for i in range(3)
        )
        preds = tuple(Prediction(st.gnb_id, _TS, True, 1.0, _pv(p)) for st in stations)
        result = solve(AllocationProblem(slices, tuple(stations), preds), solver_name=solver_name)
        dropped = {a.slice_id for a in result.allocations if a.gnb_id is None}
        for i in range(3):
            c = crit[t, i]
            if c > 0:
                crit_steps += 1
                if f"s{i}" in dropped:
                    cost += c * 5.0        # rớt slice nguy cấp = rất đắt
                else:
                    served_crit += 1
            elif f"s{i}" in dropped:
                cost += 1.0                # rớt slice thường = rẻ
    return PolicyResult(cost, served_crit / max(crit_steps, 1))


def run_comparison(n_steps: int = 420, seed: int = 0, solver_name: str = "classical_greedy"):
    frame, crit, demand = build_scenario(n_steps, seed)
    return {
        pol: evaluate_policy(frame, crit, demand, pol, solver_name=solver_name)
        for pol in ("static", "adaptive", "oracle")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantaSlice closed-loop QUBO evaluation")
    parser.add_argument("--solver", default="classical_greedy",
                        choices=["classical_greedy", "qaoa_aer"])
    parser.add_argument("--steps", type=int, default=420)
    args = parser.parse_args()

    res = run_comparison(args.steps, solver_name=args.solver)
    print(f"{'Policy':<12}{'SLA cost':>12}{'critical-served':>18}")
    print("-" * 42)
    for pol in ("static", "adaptive", "oracle"):
        r = res[pol]
        print(f"{pol:<12}{r.sla_cost:>12.1f}{r.critical_served_rate * 100:>17.1f}%")
    s, a, o = (res[p].sla_cost for p in ("static", "adaptive", "oracle"))
    print(f"\nAdaptive vs static: giảm {(s - a) / s * 100:.0f}% SLA cost; "
          f"lấp {(s - a) / (s - o + 1e-9) * 100:.0f}% khoảng cách tới oracle.")


if __name__ == "__main__":
    main()
