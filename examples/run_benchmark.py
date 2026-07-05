"""Benchmark có hệ thống: so sánh ``classical_greedy`` vs ``qaoa_aer``
qua nhiều kích thước bài toán, lặp lại nhiều lần (vì QAOA có tính ngẫu
nhiên — 1 lần chạy không nói lên điều gì), xuất kết quả ra CSV để phân
tích thêm bằng pandas/Excel/Google Sheets.

Chạy bằng::

    python -m examples.run_benchmark
    python -m examples.run_benchmark --repeats 10 --sizes 2x2,3x2,4x2
    python -m examples.run_benchmark --qaoa-depth 3 --qaoa-shots 512 --output results_depth3.csv

LƯU Ý: mỗi lần gọi ``quantaslice.quantum.solve()`` tạo MỚI 1 solver qua
registry (không dùng lại warm-start cache giữa các lần) — benchmark này
đo QAOA "cold start", phản ánh đúng khả năng thật của thuật toán chứ
không bị lợi thế nhờ may mắn warm-start từ lần chạy trước.
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


def build_problem(n_slices: int, n_stations: int) -> AllocationProblem:
    """Bài toán benchmark chuẩn: slice đầu tiên là URLLC (slice quan
    trọng nhất cần theo dõi), còn lại eMBB; trạm đầu tiên (``gnb-0``)
    đang trong tình trạng khẩn cấp."""
    slices = tuple(
        SliceRequest(f"s{i}", SliceType.URLLC if i == 0 else SliceType.EMBB, prb_required=10.0)
        for i in range(n_slices)
    )
    stations = tuple(BaseStation(f"gnb-{j}", prb_capacity=30.0) for j in range(n_stations))
    predictions = (
        Prediction(
            "gnb-0", datetime.now(timezone.utc), True, 0.9, PriorityVector(1.0, 9.0, 1.0)
        ),
    )
    return AllocationProblem(slices=slices, stations=stations, predictions=predictions)


def evaluate_run(problem: AllocationProblem, result: OptimizationResult) -> dict:
    """Tính các chỉ số đánh giá 1 lần chạy — dùng chung cho mọi solver để
    so sánh công bằng."""
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


def run_experiment(
    solver_name: str, n_slices: int, n_stations: int, repeats: int, config: Configuration
) -> list[dict]:
    problem = build_problem(n_slices, n_stations)
    rows = []
    for i in range(repeats):
        t0 = time.perf_counter()
        result = solve(problem, solver_name=solver_name, config=config)
        elapsed = time.perf_counter() - t0
        metrics = evaluate_run(problem, result)
        metrics.update(
            {
                "solver": solver_name,
                "n_slices": n_slices,
                "n_stations": n_stations,
                "repeat": i,
                "elapsed_s": round(elapsed, 4),
            }
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
    parser = argparse.ArgumentParser(description="Benchmark classical_greedy vs qaoa_aer")
    parser.add_argument("--repeats", type=int, default=10, help="Số lần lặp lại mỗi cấu hình (QAOA ngẫu nhiên)")
    parser.add_argument("--output", default="benchmark_results.csv")
    parser.add_argument("--sizes", type=str, default="2x2,3x2,4x2", help="n_slices x n_stations, cách nhau bởi dấu phẩy")
    parser.add_argument("--solvers", type=str, default="classical_greedy,qaoa_aer")
    parser.add_argument("--qaoa-depth", type=int, default=2)
    parser.add_argument("--qaoa-shots", type=int, default=256)
    parser.add_argument("--qaoa-max-iterations", type=int, default=30)
    args = parser.parse_args()

    config = Configuration(
        qaoa_depth=args.qaoa_depth, qaoa_shots=args.qaoa_shots, qaoa_max_iterations=args.qaoa_max_iterations
    )
    solvers = args.solvers.split(",")

    all_rows: list[dict] = []
    for size_str in args.sizes.split(","):
        n_slices, n_stations = map(int, size_str.lower().split("x"))
        print(f"--- {n_slices} slices x {n_stations} stations ---")
        for solver_name in solvers:
            rows = run_experiment(solver_name, n_slices, n_stations, args.repeats, config)
            all_rows.extend(rows)
            print(f"  {solver_name:20} {_summarize(rows)}")

    if all_rows:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nĐã lưu {len(all_rows)} dòng kết quả vào {args.output}")


if __name__ == "__main__":
    main()
