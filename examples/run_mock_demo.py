"""Demo end-to-end: dữ liệu giả lập (StreamSimulator) -> Runner
(ThresholdPredictionProvider tạm thời thay LSTM, phát hiện bất thường
thống kê THẬT trên dữ liệu — không random) -> QUBO -> QAOA/greedy ->
MockOranOrchestrator.

Chạy bằng::

    pip install -e ".[quantum]"
    python -m examples.run_mock_demo                    # classical_greedy (mặc định, tức thời)
    python -m examples.run_mock_demo --solver qaoa_aer   # QAOA thật trên Aer (chậm hơn, cần cài extras)
    python -m examples.run_mock_demo --solver qaoa_aer --qaoa-depth 3 --qaoa-shots 512

LƯU Ý: dataset thật ghi nhận mỗi 10 phút (``interval`` mặc định của
``generate_synthetic``), nhưng demo này cố tình rút ngắn xuống 2 giây/
mốc để burst khẩn cấp injected lọt vào phạm vi ``max_steps`` mà không
cần in ra hàng chục nghìn dòng — chỉ ảnh hưởng tốc độ demo, không ảnh
hưởng logic nội suy/resample của ``StreamSimulator``.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

from quantaslice.core.runtime import Configuration
from quantaslice.core.types import BaseStation, OptimizationResult, SliceRequest, SliceType
from quantaslice.pipeline import DependencyContainer
from quantaslice.simulation import ItalianTelecomDatasetLoader, StreamSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantaSlice demo end-to-end")
    parser.add_argument(
        "--solver",
        choices=["classical_greedy", "qaoa_aer"],
        default="classical_greedy",
        help="Solver dùng để giải AllocationProblem (mặc định: classical_greedy, tức thời)",
    )
    parser.add_argument("--qaoa-depth", type=int, default=2, help="Độ sâu p của mạch QAOA (chỉ áp dụng khi --solver qaoa_aer)")
    parser.add_argument("--qaoa-shots", type=int, default=256, help="Số shots đo mỗi vòng lặp QAOA")
    parser.add_argument("--qaoa-max-iterations", type=int, default=30, help="Số vòng lặp tối đa của COBYLA")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Configuration:
    return Configuration(
        # "threshold" đọc dữ liệu thật (z-score bất thường, mục 4.1 tài
        # liệu LSTM) — khác "random" (bỏ qua dữ liệu, chỉ để stress-test
        # cơ chế reoptimize). Thay bằng "lstm" khi Member A hoàn thành ai.
        prediction_provider="threshold",
        solver=args.solver,
        orchestrator="mock_oran",  # đổi thành "e2_interface" khi có OpenRAN thật
        emergency_threshold=0.5,
        qaoa_depth=args.qaoa_depth,
        qaoa_shots=args.qaoa_shots,
        qaoa_max_iterations=args.qaoa_max_iterations,
    )


def build_topology() -> tuple[tuple[SliceRequest, ...], tuple[BaseStation, ...]]:
    slices = (
        SliceRequest("s1", SliceType.EMBB, prb_required=20.0),
        SliceRequest("s2", SliceType.URLLC, prb_required=15.0),
        SliceRequest("s3", SliceType.MMTC, prb_required=5.0),
    )
    stations = (
        BaseStation("gnb-1", prb_capacity=30.0),
        BaseStation("gnb-2", prb_capacity=20.0),
    )
    return slices, stations


def build_stream() -> StreamSimulator:
    records = ItalianTelecomDatasetLoader.generate_synthetic(
        cell_ids=("gnb-1", "gnb-2"),
        start=datetime(2026, 7, 4, tzinfo=timezone.utc),
        interval=timedelta(seconds=2),  # rút ngắn từ 10 phút thật, xem docstring module
        n_intervals=12,  # tổng 24s dữ liệu gốc
        emergency_cell_id="gnb-1",
        emergency_start_interval=5,  # burst bắt đầu ở giây thứ 10
        emergency_duration_intervals=2,  # kéo dài tới giây thứ 14
        emergency_multiplier=4.0,
        seed=42,
    )
    # max_steps=240 (24s / 100ms) để phủ hết toàn bộ 24s dữ liệu gốc,
    # bao gồm cả burst khẩn cấp ở giây 10-14 — khác bản demo trước chỉ
    # phủ 6s đầu nên KHÔNG BAO GIỜ chạm tới burst.
    return StreamSimulator(records, window_length=10, max_steps=240, seed=1)


def _format_allocation(result: OptimizationResult | None) -> str:
    if result is None:
        return "(chưa có)"
    return ", ".join(f"{a.slice_id}->{a.gnb_id or '(none)'}" for a in result.allocations)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    slices, stations = build_topology()
    stream = build_stream()

    print(f"Solver: {args.solver}" + (f" (depth={args.qaoa_depth}, shots={args.qaoa_shots})" if args.solver == "qaoa_aer" else ""))
    try:
        runner = DependencyContainer.build_runner(config, slices=slices, stations=stations)
    except Exception as exc:  # noqa: BLE001 - demo script, in lỗi thân thiện rồi thoát
        print(f"Không khởi tạo được runner: {exc}")
        print("Gợi ý: pip install -e '.[quantum]' nếu dùng --solver qaoa_aer")
        return

    print(f"{'timestamp':<30} {'gNB':<8} {'emergency':<10} {'allocation'}")
    print("-" * 90)

    last_printed_allocation: str | None = None
    n_reoptimize = 0
    t_start = time.perf_counter()
    for frame in runner.run_forever(stream):
        window_gnb_id = frame.windows[0].gnb_id
        current_pred = next((p for p in frame.predictions if p.gnb_id == window_gnb_id), None)
        allocation_str = _format_allocation(frame.result)
        is_emergency = bool(current_pred and current_pred.emergency_flag)

        if allocation_str != last_printed_allocation:
            n_reoptimize += 1

        # Chỉ in khi có gì đó THAY ĐỔI (emergency vừa xảy ra, hoặc
        # allocation vừa đổi khác lần in trước) — giữ output demo ngắn
        # gọn, dễ đọc thay vì spam mọi window 100ms.
        if is_emergency or allocation_str != last_printed_allocation:
            print(
                f"{frame.timestamp.isoformat():<30} "
                f"{window_gnb_id:<8} "
                f"{str(is_emergency):<10} "
                f"{allocation_str}"
            )
            last_printed_allocation = allocation_str

    elapsed = time.perf_counter() - t_start
    print("-" * 90)
    print(f"Xong: {n_reoptimize} lần allocation thay đổi, tổng thời gian chạy {elapsed:.2f}s")


if __name__ == "__main__":
    main()
