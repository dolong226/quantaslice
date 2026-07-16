"""Thực nghiệm tìm ``lambda1``/``lambda2`` (hệ số phạt QUBO) tốt cho
QAOA — quét qua nhiều giá trị λ cố định (dùng ``λ1 = λ2 = λ``), đo
feasibility rate / URLLC-correct rate / latency ở mỗi giá trị, lặp
nhiều lần (QAOA ngẫu nhiên), xuất CSV.

BỐI CẢNH: ``QUBOBuilder`` mặc định dùng ``λ = 3.0 × max(w)`` — nhưng
``w`` đã được chuẩn hoá về [0,1] TRƯỚC khi tính λ, nên ``max(w)`` LUÔN
xấp xỉ 1.0 bất kể kích thước bài toán → λ mặc định LUÔN ~3.0. Đây chính
là "heuristic chưa từng tune" — script này tune nó bằng thực nghiệm
thay vì đoán, dùng field có sẵn ``AllocationProblem.penalty_lambda1/2``
(không cần sửa ``QUBOBuilder``).

Chạy bằng::

    python -m examples.run_lambda_sweep
    python -m examples.run_lambda_sweep --lambdas 1,2,3,5,8,12,20 --repeats 15
    python -m examples.run_lambda_sweep --sizes 3x2,4x2 --output lambda_results.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
import warnings
from datetime import datetime, timezone

from quantaslice.core.runtime import Configuration
from quantaslice.core.types import (
    AllocationProblem,
    BaseStation,
    OptimizationResult,
    Prediction,
    PriorityVector,
    SliceRequest,
    SliceType,
)
from quantaslice.quantum import solve

warnings.filterwarnings("ignore")  # Qiskit hay in DeprecationWarning, gây nhiễu output


def build_problem(n_slices: int, n_stations: int, lambda1: float, lambda2: float) -> AllocationProblem:
    """Cùng bài toán chuẩn với ``run_benchmark.py`` (slice đầu là URLLC,
    trạm đầu đang khẩn cấp) — CHỈ khác: truyền thẳng ``lambda1``/
    ``lambda2`` thay vì để ``QUBOBuilder`` tự tính heuristic mặc định."""
    slices = tuple(
        SliceRequest(f"s{i}", SliceType.URLLC if i == 0 else SliceType.EMBB, prb_required=10.0)
        for i in range(n_slices)
    )
    stations = tuple(BaseStation(f"gnb-{j}", prb_capacity=30.0) for j in range(n_stations))
    predictions = (
        Prediction("gnb-0", datetime.now(timezone.utc), True, 0.9, PriorityVector(1.0, 9.0, 1.0)),
    )
    return AllocationProblem(
        slices=slices,
        stations=stations,
        predictions=predictions,
        penalty_lambda1=lambda1,
        penalty_lambda2=lambda2,
    )


def evaluate_run(problem: AllocationProblem, result: OptimizationResult) -> dict:
    n_served = sum(1 for a in result.allocations if a.gnb_id is not None)
    n_total = len(result.allocations)
    urllc_slice = next((s for s in problem.slices if s.slice_type == SliceType.URLLC), None)
    urllc_correct = None
    if urllc_slice is not None and problem.predictions:
        target_gnb = problem.predictions[0].gnb_id
        alloc = result.allocation_for(urllc_slice.slice_id)
        urllc_correct = alloc is not None and alloc.gnb_id == target_gnb
    return {
        "feasible": n_served == n_total,
        "n_served": n_served,
        "n_total": n_total,
        "urllc_correct": urllc_correct,
        "objective": result.objective_value,
    }


def run_for_lambda(
    n_slices: int, n_stations: int, lam: float, repeats: int, config: Configuration
) -> list[dict]:
    problem = build_problem(n_slices, n_stations, lam, lam)
    rows = []
    for i in range(repeats):
        t0 = time.perf_counter()
        result = solve(problem, solver_name="qaoa_aer", config=config)
        elapsed = time.perf_counter() - t0
        metrics = evaluate_run(problem, result)
        metrics.update(
            {"lambda": lam, "n_slices": n_slices, "n_stations": n_stations, "repeat": i, "elapsed_s": round(elapsed, 4)}
        )
        rows.append(metrics)
    return rows


def _summarize(rows: list[dict]) -> str:
    feas_rate = sum(r["feasible"] for r in rows) / len(rows)
    urllc_rows = [r for r in rows if r["urllc_correct"] is not None]
    urllc_rate = sum(r["urllc_correct"] for r in urllc_rows) / len(urllc_rows) if urllc_rows else float("nan")
    avg_time = statistics.mean(r["elapsed_s"] for r in rows)
    return f"feasible={feas_rate:.0%}  urllc_correct={urllc_rate:.0%}  avg_time={avg_time:.2f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quét lambda1/lambda2 để tìm giá trị tốt cho QAOA")
    parser.add_argument("--lambdas", type=str, default="1,2,3,5,8,12,20")
    parser.add_argument("--sizes", type=str, default="3x2")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--qaoa-depth", type=int, default=2)
    parser.add_argument("--qaoa-shots", type=int, default=256)
    parser.add_argument("--qaoa-max-iterations", type=int, default=30)
    parser.add_argument("--output", default="lambda_sweep_results.csv")
    args = parser.parse_args()

    config = Configuration(
        qaoa_depth=args.qaoa_depth, qaoa_shots=args.qaoa_shots, qaoa_max_iterations=args.qaoa_max_iterations
    )
    lambdas = [float(x) for x in args.lambdas.split(",")]

    all_rows: list[dict] = []
    for size_str in args.sizes.split(","):
        n_slices, n_stations = map(int, size_str.lower().split("x"))
        print(f"--- {n_slices} slices x {n_stations} stations ---")
        for lam in lambdas:
            rows = run_for_lambda(n_slices, n_stations, lam, args.repeats, config)
            all_rows.extend(rows)
            print(f"  lambda={lam:<6} {_summarize(rows)}")

    if all_rows:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nĐã lưu {len(all_rows)} dòng kết quả vào {args.output}")


if __name__ == "__main__":
    main()
