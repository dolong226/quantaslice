"""Baseline CỔ ĐIỂN CHO BÀI TOÁN TỐI ƯU (không phải AI detector) — dùng
để trả lời đúng câu hỏi mentor: so QAOA với nghiệm TỐI ƯU THẬT, tính
được **approximation ratio** thay vì chỉ "feasible%"/"target-correct%".

Hai baseline:

1. **MIP (Mixed Integer Programming)** — dùng ``PuLP`` + CBC (mã nguồn
   mở, cài qua pip, KHÔNG cần license). Mentor gợi ý Gurobi — Gurobi
   cho tốc độ tốt hơn ở quy mô lớn nhưng cần license thương mại
   (academic license miễn phí nếu anh có email trường). Script này TỰ
   ĐỘNG DÙNG GUROBI nếu ``gurobipy`` cài được VÀ có license hợp lệ,
   ngược lại rơi về CBC — cả 2 đều cho nghiệm TỐI ƯU CHÍNH XÁC (exact),
   chỉ khác tốc độ, không khác chất lượng nghiệm.

2. **MaxSAT (weighted partial MaxSAT)** — dùng ``python-sat`` (PySAT) +
   RC2 solver + ``pypblib`` (mã hoá ràng buộc capacity dạng
   pseudo-boolean thành CNF). Đúng "SAT encoding" mentor nhắc tới.

Cả 2 baseline giải ĐÚNG bài toán BIP gốc (không qua QUBO/penalty) —
dùng để kiểm chứng QAOA/classical_greedy đạt bao nhiêu % so với tối ưu
thật (approximation ratio = objective_QAOA / objective_optimal).

Chạy bằng::

    python -m examples.run_classical_baselines --synthetic
    python -m examples.run_classical_baselines \\
        --data-root colosseum-oran-coloran-dataset --sched 0 --tr 0 --exp 1 --n-stations 2
"""

from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime, timezone

import numpy as np

from quantaslice.core.runtime import Configuration
from quantaslice.core.types import Allocation, AllocationProblem, BaseStation, OptimizationResult, Prediction, PriorityVector, SliceRequest, SliceType
from quantaslice.quantum import solve
from quantaslice.quantum.qubo.builder import QUBOBuilder

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Dựng bài toán — synthetic (nhanh, không cần dataset) hoặc thật
# ----------------------------------------------------------------------


def build_synthetic_problem() -> AllocationProblem:
    slices = (
        SliceRequest("s0", SliceType.URLLC, prb_required=10.0),
        SliceRequest("s1", SliceType.EMBB, prb_required=10.0),
        SliceRequest("s2", SliceType.MMTC, prb_required=10.0),
    )
    stations = (BaseStation("gnb-0", prb_capacity=25.0), BaseStation("gnb-1", prb_capacity=25.0))
    predictions = (
        Prediction("gnb-0", datetime.now(timezone.utc), True, 0.9, PriorityVector(embb=1.0, urllc=9.0, mmtc=1.0)),
    )
    return AllocationProblem(slices=slices, stations=stations, predictions=predictions)


def build_real_problem(data_root: str, *, sched: int, tr: int, exp: int, n_stations: int) -> AllocationProblem:
    from quantaslice.ai.data.labeling import label_frame
    from quantaslice.ai.data.loaders import iter_frames
    from quantaslice.orchestrator.coloran_loader import ColORANLoader

    frames = list(iter_frames(data_root, scheds=(sched,), trs=(tr,), exps=(exp,), limit=n_stations))
    if not frames:
        raise SystemExit(f"Không tìm thấy frame tại sched{sched}/tr{tr}/exp{exp} trong '{data_root}'.")

    stations = ColORANLoader.create_stations(n_stations=n_stations)
    slices = ColORANLoader.create_slices()

    labels = label_frame(frames[0])
    severity_total = labels.severity.sum(axis=1)
    idx = int(np.argmax(severity_total * labels.flag)) if labels.flag.any() else int(np.argmax(severity_total))
    p = labels.priority[idx]
    ts = datetime.fromtimestamp(float(frames[0].time[idx]), tz=timezone.utc)
    prediction = Prediction(
        gnb_id=stations[0].gnb_id, timestamp=ts, emergency_flag=bool(labels.flag[idx]),
        emergency_prob=float(labels.severity[idx].max()),
        priority=PriorityVector(embb=float(p[0]), mmtc=float(p[1]), urllc=float(p[2])),
    )
    return AllocationProblem(slices=slices, stations=stations, predictions=(prediction,))


def _true_objective(problem: AllocationProblem, allocations: tuple[Allocation, ...]) -> float:
    """Tính objective THẬT (dùng weight đã chuẩn hoá gốc, không phải
    weight đã làm tròn/scale cho MIP/SAT solver) — để so sánh công bằng
    giữa MIP/SAT/QAOA/greedy trên CÙNG 1 thang đo."""
    weights = QUBOBuilder._compute_normalized_weights(problem)
    slice_index = {s.slice_id: i for i, s in enumerate(problem.slices)}
    station_index = {st.gnb_id: j for j, st in enumerate(problem.stations)}
    total = 0.0
    for alloc in allocations:
        if alloc.gnb_id is None:
            continue
        total += weights[slice_index[alloc.slice_id], station_index[alloc.gnb_id]]
    return total


# ----------------------------------------------------------------------
# Baseline 1: MIP (PuLP + CBC, hoặc Gurobi nếu có license)
# ----------------------------------------------------------------------


def solve_mip(problem: AllocationProblem) -> tuple[float, tuple[Allocation, ...], float, str]:
    import pulp

    weights = QUBOBuilder._compute_normalized_weights(problem)
    slices, stations = problem.slices, problem.stations
    n, m = len(slices), len(stations)

    prob = pulp.LpProblem("QuantaSlice_MIP", pulp.LpMaximize)
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for i in range(n) for j in range(m)}

    prob += pulp.lpSum(weights[i, j] * x[i, j] for i in range(n) for j in range(m))
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(m)) <= 1
    for j in range(m):
        prob += pulp.lpSum(slices[i].prb_required * x[i, j] for i in range(n)) <= stations[j].prb_capacity

    solver_name = "CBC (open-source)"
    solver_cmd = pulp.PULP_CBC_CMD(msg=False)
    try:
        import gurobipy  # noqa: F401

        solver_cmd = pulp.GUROBI_CMD(msg=False)
        solver_name = "Gurobi"
    except Exception:
        pass  # gurobipy chưa cài hoặc chưa có license -> giữ CBC

    t0 = time.perf_counter()
    prob.solve(solver_cmd)
    elapsed = time.perf_counter() - t0

    allocations = []
    for i, s in enumerate(slices):
        assigned = None
        for j, st in enumerate(stations):
            if pulp.value(x[i, j]) and pulp.value(x[i, j]) > 0.5:
                assigned = st.gnb_id
        allocations.append(Allocation(slice_id=s.slice_id, gnb_id=assigned))

    objective = _true_objective(problem, tuple(allocations))
    return objective, tuple(allocations), elapsed, solver_name


# ----------------------------------------------------------------------
# Baseline 2: MaxSAT (PySAT + RC2 + pypblib cho ràng buộc capacity)
# ----------------------------------------------------------------------


def solve_maxsat(problem: AllocationProblem, *, weight_scale: float = 1000.0) -> tuple[float, tuple[Allocation, ...], float]:
    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF
    from pysat.pb import PBEnc

    weights = QUBOBuilder._compute_normalized_weights(problem)
    slices, stations = problem.slices, problem.stations
    n, m = len(slices), len(stations)

    def vid(i: int, j: int) -> int:
        return i * m + j + 1

    top_id = n * m
    wcnf = WCNF()

    # Uniqueness: tối đa 1 trong {x_i0..x_i(m-1)} true -> pairwise NOT-both (hard)
    for i in range(n):
        lits = [vid(i, j) for j in range(m)]
        for a in range(len(lits)):
            for b in range(a + 1, len(lits)):
                wcnf.append([-lits[a], -lits[b]])

    # Capacity: sum_i r_i*x_ij <= c_j -> pseudo-boolean "at most", mã hoá qua PBEnc (hard)
    for j in range(m):
        lits = [vid(i, j) for i in range(n)]
        w_int = [max(1, int(round(slices[i].prb_required))) for i in range(n)]
        bound = int(stations[j].prb_capacity)
        enc = PBEnc.leq(lits=lits, weights=w_int, bound=bound, top_id=top_id)
        top_id = max(top_id, enc.nv)
        for cl in enc.clauses:
            wcnf.append(cl)

    # Objective (soft): maximize sum w_ij * x_ij
    for i in range(n):
        for j in range(m):
            w = max(1, int(round(weights[i, j] * weight_scale)))
            wcnf.append([vid(i, j)], weight=w)

    t0 = time.perf_counter()
    with RC2(wcnf) as rc2:
        model = rc2.compute()
    elapsed = time.perf_counter() - t0

    true_vars = set(v for v in (model or []) if v > 0)
    allocations = []
    for i, s in enumerate(slices):
        assigned = None
        for j, st in enumerate(stations):
            if vid(i, j) in true_vars:
                assigned = st.gnb_id
        allocations.append(Allocation(slice_id=s.slice_id, gnb_id=assigned))

    objective = _true_objective(problem, tuple(allocations))
    return objective, tuple(allocations), elapsed


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def _fmt(name: str, objective: float, elapsed: float, ratio: float | None = None) -> str:
    ratio_str = f"  ratio={ratio:.1%}" if ratio is not None else ""
    return f"{name:<22} objective={objective:.3f}  time={elapsed:.3f}s{ratio_str}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline cổ điển (MIP/MaxSAT) cho bài toán tối ưu QuantaSlice")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data-root", default="colosseum-oran-coloran-dataset")
    parser.add_argument("--sched", type=int, default=0)
    parser.add_argument("--tr", type=int, default=0)
    parser.add_argument("--exp", type=int, default=1)
    parser.add_argument("--n-stations", type=int, default=2)
    parser.add_argument("--qaoa-depth", type=int, default=2)
    parser.add_argument("--qaoa-shots", type=int, default=256)
    parser.add_argument("--qaoa-max-iterations", type=int, default=30)
    args = parser.parse_args()

    problem = (
        build_synthetic_problem()
        if args.synthetic
        else build_real_problem(args.data_root, sched=args.sched, tr=args.tr, exp=args.exp, n_stations=args.n_stations)
    )
    print(f"Bài toán: {len(problem.slices)} slice x {len(problem.stations)} trạm")
    if problem.predictions:
        print(f"Prediction: {problem.predictions[0]}")
    print()

    # --- MIP (nghiệm tối ưu chính xác - dùng làm mốc 100% cho approximation ratio) ---
    mip_obj, mip_alloc, mip_time, mip_solver = solve_mip(problem)
    print(_fmt(f"MIP ({mip_solver})", mip_obj, mip_time))

    # --- MaxSAT ---
    sat_obj, sat_alloc, sat_time = solve_maxsat(problem)
    print(_fmt("MaxSAT (RC2)", sat_obj, sat_time, ratio=sat_obj / mip_obj if mip_obj else None))

    # --- classical_greedy ---
    config = Configuration(qaoa_depth=args.qaoa_depth, qaoa_shots=args.qaoa_shots, qaoa_max_iterations=args.qaoa_max_iterations)
    t0 = time.perf_counter()
    greedy_result: OptimizationResult = solve(problem, solver_name="classical_greedy", config=config)
    greedy_time = time.perf_counter() - t0
    greedy_obj = _true_objective(problem, greedy_result.allocations)
    print(_fmt("classical_greedy", greedy_obj, greedy_time, ratio=greedy_obj / mip_obj if mip_obj else None))

    # --- QAOA (Aer) ---
    t0 = time.perf_counter()
    qaoa_result: OptimizationResult = solve(problem, solver_name="qaoa_aer", config=config)
    qaoa_time = time.perf_counter() - t0
    qaoa_obj = _true_objective(problem, qaoa_result.allocations)
    print(_fmt("qaoa_aer", qaoa_obj, qaoa_time, ratio=qaoa_obj / mip_obj if mip_obj else None))

    print()
    print(f"Approximation ratio = objective / objective_MIP_optimal (mốc 100% = nghiệm tối ưu thật)")


if __name__ == "__main__":
    main()
