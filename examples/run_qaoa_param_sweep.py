"""Quét tham số QAOA (``depth``, ``shots``, ``max_iterations``,
``lambda``) trên bài toán XÂY TỪ DỮ LIỆU THẬT (dataset ColO-RAN) — khác
``run_benchmark.py``/``run_lambda_sweep.py`` vốn dùng bài toán synthetic
tự tạo.

Cách xây bài toán "thật":

1. Đọc 1 ``BSFrame`` thật từ dataset (``iter_frames``).
2. ``label_frame()`` cho ``severity``/``priority`` THẬT suy ra từ KPM
   thật (``dl_buffer``, ``tx_brate_dl``...) — không phải số gõ tay —
   tại đúng mốc thời gian căng thẳng nhất trong chuỗi thật đó.
3. ``ColORANLoader`` dựng topology chuẩn, CHỈ LẤY SUBSET ``n_stations``
   trạm (mặc định 2) để giữ số qubit khả thi cho QAOA trong thời gian
   hợp lý — dùng đủ 7 trạm thật của dataset sẽ vượt xa "simulation
   ceiling" đã ghi nhận trước đó (~30 qubit).
4. Trạm đầu tiên trong subset mang Prediction THẬT; các trạm còn lại
   dùng baseline p0=(1,1,1) — đúng hành vi mặc định của hệ thống khi 1
   gNB không có Prediction (xem ``QUBOBuilder._compute_normalized_weights``).

Chạy bằng::

    # Quét depth, giữ shots/max_iterations/lambda cố định
    python -m examples.run_qaoa_param_sweep --sweep depth --values 1,2,3,4 --repeats 15

    # Quét shots
    python -m examples.run_qaoa_param_sweep --sweep shots --values 128,256,512,1024 --repeats 15

    # Quét max_iterations
    python -m examples.run_qaoa_param_sweep --sweep max_iter --values 10,30,60,100 --repeats 15

    # Quét lambda1=lambda2
    python -m examples.run_qaoa_param_sweep --sweep lambda --values 1,2,3,5,8,12,20 --repeats 15
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import statistics
import time
import warnings
from datetime import datetime, timezone

import numpy as np

from quantaslice.ai.data.labeling import label_frame
from quantaslice.ai.data.loaders import iter_frames
from quantaslice.core.runtime import Configuration
from quantaslice.core.types import (
    AllocationProblem,
    OptimizationResult,
    Prediction,
    PriorityVector,
    SliceType,
)
from quantaslice.orchestrator.coloran_loader import ColORANLoader
from quantaslice.quantum import solve

warnings.filterwarnings("ignore")  # Qiskit hay in DeprecationWarning, gây nhiễu output

N_STATIONS = 2
def build_real_problem(
    data_root: str, *, sched: int, tr: int, exp: int, n_stations: int
) -> AllocationProblem:
    """Dựng 1 ``AllocationProblem`` từ dữ liệu ColO-RAN thật — xem
    docstring module để biết chi tiết cách chọn mốc thời gian/trạm."""
    frames = list(iter_frames(data_root, scheds=(sched,), trs=(tr,), exps=(exp,), limit=n_stations))
    if not frames:
        raise SystemExit(
            f"Không tìm thấy frame nào tại sched{sched}/tr{tr}/exp{exp} trong "
            f"'{data_root}' — kiểm tra lại --data-root/--sched/--tr/--exp."
        )

    stations = ColORANLoader.create_stations(n_stations=n_stations)
    slices = ColORANLoader.create_slices()

    emergency_frame = frames[0]
    labels = label_frame(emergency_frame)
    # Chọn mốc TỔNG SEVERITY CAO NHẤT (ưu tiên bước đã "sustain" đủ lâu
    # để flag=True). ĐÃ THỬ chọn riêng theo URLLC (tuyệt đối lẫn tương
    # đối so với 2 slice còn lại) — nhưng dữ liệu thật
    # (rome_static_medium/sched0) cho thấy eMBB mới là slice căng
    # thẳng nhất nhất quán qua mọi cấu hình tr (URLLC gần như luôn ~0
    # vì demand per-UE quá nhỏ theo thiết kế traffic ColO-RAN — 0.01
    # Mbps/UE). Đây là đặc điểm THẬT của dataset, không phải lỗi chọn
    # mốc — nên script này theo dõi đúng "slice nào thực sự ưu tiên
    # nhất theo dữ liệu thật" (xem ``evaluate_run``) thay vì ép cứng
    # phải là URLLC.
    severity = labels.severity  # (T, 3) = (eMBB, mMTC, URLLC)
    severity_total = severity.sum(axis=1)
    idx = int(np.argmax(severity_total * labels.flag)) if labels.flag.any() else int(np.argmax(severity_total))

    p = labels.priority[idx]  # (3,) theo thứ tự (eMBB, mMTC, URLLC) — xem FrameLabels
    ts = datetime.fromtimestamp(float(emergency_frame.time[idx]), tz=timezone.utc)

    prediction = Prediction(
        gnb_id=stations[0].gnb_id,
        timestamp=ts,
        emergency_flag=bool(labels.flag[idx]),
        emergency_prob=float(labels.severity[idx].max()),
        priority=PriorityVector(embb=float(p[0]), mmtc=float(p[1]), urllc=float(p[2])),
    )
    return AllocationProblem(slices=slices, stations=stations, predictions=(prediction,))


def evaluate_run(problem: AllocationProblem, result: OptimizationResult) -> dict:
    n_served = sum(1 for a in result.allocations if a.gnb_id is not None)
    n_total = len(result.allocations)

    # Slice mục tiêu = loại có priority CAO NHẤT trong Prediction thật —
    # không ép cứng URLLC, vì dữ liệu thật có thể cho thấy slice khác
    # (VD eMBB) mới thực sự là slice cần ưu tiên tại mốc đã chọn (xem
    # ghi chú trong build_real_problem).
    target_correct = None
    target_slice_type = None
    if problem.predictions:
        pred = problem.predictions[0]
        weights = {
            SliceType.EMBB: pred.priority.embb,
            SliceType.MMTC: pred.priority.mmtc,
            SliceType.URLLC: pred.priority.urllc,
        }
        target_slice_type = max(weights, key=weights.get)
        target_slice = next((s for s in problem.slices if s.slice_type == target_slice_type), None)
        if target_slice is not None:
            alloc = result.allocation_for(target_slice.slice_id)
            target_correct = alloc is not None and alloc.gnb_id == pred.gnb_id

    return {
        "feasible": n_served == n_total,
        "target_slice": target_slice_type.value if target_slice_type else None,
        "target_correct": target_correct,
        "objective": result.objective_value,
    }


def run_for_params(
    problem: AllocationProblem, *, depth: int, shots: int, max_iter: int, lam: float | None, repeats: int
) -> list[dict]:
    config = Configuration(qaoa_depth=depth, qaoa_shots=shots, qaoa_max_iterations=max_iter)
    problem_run = (
        dataclasses.replace(problem, penalty_lambda1=lam, penalty_lambda2=lam) if lam is not None else problem
    )
    rows = []
    for i in range(repeats):
        t0 = time.perf_counter()
        result = solve(problem_run, solver_name="qaoa_aer", config=config)
        elapsed = time.perf_counter() - t0
        metrics = evaluate_run(problem_run, result)
        metrics.update(
            {
                "depth": depth,
                "shots": shots,
                "max_iter": max_iter,
                "lambda": lam if lam is not None else "auto",
                "repeat": i,
                "elapsed_s": round(elapsed, 4),
            }
        )
        rows.append(metrics)
    return rows


def _summarize(rows: list[dict]) -> str:
    feas_rate = sum(r["feasible"] for r in rows) / len(rows)
    target_rows = [r for r in rows if r["target_correct"] is not None]
    target_rate = sum(r["target_correct"] for r in target_rows) / len(target_rows) if target_rows else float("nan")
    avg_time = statistics.mean(r["elapsed_s"] for r in rows)
    target_name = rows[0]["target_slice"] if rows else "?"
    return f"feasible={feas_rate:.0%}  {target_name}-correct={target_rate:.0%}  avg_time={avg_time:.2f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quét tham số QAOA trên bài toán từ dữ liệu ColO-RAN thật")
    parser.add_argument("--data-root", default="colosseum-oran-coloran-dataset")
    parser.add_argument("--sched", type=int, default=0)
    parser.add_argument("--tr", type=int, default=0)
    parser.add_argument("--exp", type=int, default=1)
    parser.add_argument("--n-stations", type=int, default=N_STATIONS, help="Số trạm thật lấy subset (giữ số qubit khả thi)")

    parser.add_argument("--sweep", choices=["depth", "shots", "max_iter", "lambda"], required=True)
    parser.add_argument("--values", type=str, required=True, help="Danh sách giá trị cần quét, cách nhau dấu phẩy")
    parser.add_argument("--repeats", type=int, default=15)

    # Giá trị GIỮ CỐ ĐỊNH cho 3 trục còn lại (không phải trục đang --sweep)
    parser.add_argument("--fixed-depth", type=int, default=2)
    parser.add_argument("--fixed-shots", type=int, default=256)
    parser.add_argument("--fixed-max-iter", type=int, default=30)
    parser.add_argument(
        "--fixed-lambda", type=float, default=None, help="None = dùng heuristic mặc định của QUBOBuilder (~3.0)"
    )

    parser.add_argument("--output", default="qaoa_param_sweep_results.csv")
    args = parser.parse_args()

    problem = build_real_problem(
        args.data_root, sched=args.sched, tr=args.tr, exp=args.exp, n_stations=args.n_stations
    )
    print(f"Bài toán thật: {len(problem.slices)} slice x {len(problem.stations)} trạm")
    print(f"Prediction thật: {problem.predictions[0]}")
    print()

    values = [float(v) for v in args.values.split(",")]
    all_rows: list[dict] = []

    for v in values:
        depth = int(v) if args.sweep == "depth" else args.fixed_depth
        shots = int(v) if args.sweep == "shots" else args.fixed_shots
        max_iter = int(v) if args.sweep == "max_iter" else args.fixed_max_iter
        lam = v if args.sweep == "lambda" else args.fixed_lambda

        rows = run_for_params(problem, depth=depth, shots=shots, max_iter=max_iter, lam=lam, repeats=args.repeats)
        all_rows.extend(rows)
        print(f"  {args.sweep}={v:<8} {_summarize(rows)}")

    if all_rows:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nĐã lưu {len(all_rows)} dòng kết quả vào {args.output}")


if __name__ == "__main__":
    main()
