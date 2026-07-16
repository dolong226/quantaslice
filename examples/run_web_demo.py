"""Demo trực quan qua trình duyệt: chạy pipeline ở background thread,
serve 1 trang HTML/JS đơn (poll ``/state`` mỗi 300ms) hiển thị trạm nào
đang giữ slice nào, đổi ra sao theo thời gian, và báo hiệu khi có
emergency — dùng CHỈ thư viện chuẩn (``http.server``), không cần cài gì
thêm ngoài ``quantaslice[quantum]`` (nếu muốn thử ``--solver qaoa_aer``).

HAI CHẾ ĐỘ EMERGENCY, hoạt động SONG SONG (không loại trừ nhau):

1. **Auto-detect (mặc định BẬT)** — ``ThresholdPredictionProvider`` chạy
   nền, tự đọc dữ liệu thật và phát hiện bất thường (đúng câu chuyện
   end-to-end: AI/ML component thật đang chạy, không phải nút bấm giả).
2. **Nút Trigger trên giao diện** — công cụ CANH THỜI ĐIỂM demo (vì
   burst tự nhiên xảy ra ngẫu nhiên ~8-10s sau khi mở, khó canh đúng
   lúc trước mặt giám khảo). Khi bấm, chỉ ép emergency cho 1 gNB trong
   ~2s rồi tự tắt — các gNB khác vẫn được auto-detect xử lý bình
   thường suốt lúc đó.

Chạy bằng::

    python -m examples.run_web_demo                       # auto-detect BẬT + nút Trigger (mặc định)
    python -m examples.run_web_demo --solver qaoa_aer
    python -m examples.run_web_demo --no-auto-detect        # CHỈ nút Trigger — dùng khi usability test cần kiểm soát hoàn toàn
    python -m examples.run_web_demo --port 9000 --tick-delay 0.3

Sau đó mở trình duyệt tới địa chỉ được in ra (mặc định
http://127.0.0.1:8765).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from quantaslice.core.runtime import Configuration, SimulationFrame
from quantaslice.core.types import BaseStation, FeatureWindow, Prediction, PriorityVector, SliceRequest
from quantaslice.orchestrator import orchestrator_registry
from quantaslice.pipeline.registries import prediction_provider_registry
from quantaslice.pipeline.runner import Runner
from quantaslice.quantum import solver_registry
from examples.run_mock_demo import build_stream, build_topology

_ASSETS_DIR = Path(__file__).resolve().parent / "web_demo_assets"
_HISTORY_LENGTH = 40
_EVENT_LOG_LENGTH = 60
_TRIGGER_DURATION_WINDOWS = 20  # ~2s ở output_interval mặc định 100ms
_TRIGGER_URLLC_BOOST = 9.0


class ManualOverridePredictionProvider:
    """Bọc quanh 1 ``PredictionProvider`` khác — cho phép NGƯỜI DÙNG chủ
    động kích hoạt "emergency" qua nút bấm trên giao diện (``trigger()``),
    thay vì chỉ chờ dữ liệu tự nhiên trong stream sinh ra.

    Đây là mảnh ghép còn thiếu để web demo trở thành "clickable
    prototype" đúng nghĩa — trước đây hệ thống chỉ TỰ chạy 1 kịch bản có
    sẵn, người dùng không tương tác được gì cả.

    Trong lúc override đang active cho 1 gNB, MỌI cửa sổ của gNB đó
    được coi là khẩn cấp trong ``duration_windows`` cửa sổ liên tiếp,
    bất kể dữ liệu thật trong cửa sổ là gì — sau đó tự tắt và quay lại
    dùng ``wrapped`` provider bình thường.
    """

    def __init__(
        self,
        wrapped,
        *,
        duration_windows: int = _TRIGGER_DURATION_WINDOWS,
        urllc_boost: float = _TRIGGER_URLLC_BOOST,
    ) -> None:
        self._wrapped = wrapped
        self._duration = duration_windows
        self._urllc_boost = urllc_boost
        self._lock = threading.Lock()
        self._active_gnb: str | None = None
        self._remaining = 0

    def trigger(self, gnb_id: str) -> None:
        with self._lock:
            self._active_gnb = gnb_id
            self._remaining = self._duration

    def reset(self) -> None:
        with self._lock:
            self._active_gnb = None
            self._remaining = 0

    def predict(self, window: FeatureWindow) -> Prediction:
        with self._lock:
            if self._active_gnb == window.gnb_id and self._remaining > 0:
                self._remaining -= 1
                return Prediction(
                    gnb_id=window.gnb_id,
                    timestamp=window.timestamp,
                    emergency_flag=True,
                    emergency_prob=0.95,
                    priority=PriorityVector(embb=1.0, urllc=1.0 + self._urllc_boost, mmtc=1.0),
                )
        return self._wrapped.predict(window)


class SharedState:
    """Trạng thái dùng chung giữa thread chạy pipeline và thread HTTP
    server — bảo vệ bằng 1 lock đơn giản (đủ dùng vì tần suất update
    thấp, không cần cấu trúc phức tạp hơn)."""

    def __init__(self, stations: tuple[BaseStation, ...], slices: tuple[SliceRequest, ...], solver_name: str) -> None:
        self._lock = threading.Lock()
        self._solver_name = solver_name
        self._stations = {st.gnb_id: st for st in stations}
        self._slices = {s.slice_id: s for s in slices}
        self._history: dict[str, deque[bool]] = {
            gnb_id: deque([False] * _HISTORY_LENGTH, maxlen=_HISTORY_LENGTH) for gnb_id in self._stations
        }
        self._events: deque[dict] = deque(maxlen=_EVENT_LOG_LENGTH)
        self._current_allocations: dict[str, str | None] = {s.slice_id: None for s in slices}
        self._last_timestamp: str | None = None
        # --- Metric "core value" (Week 4, task "add 1 metric to track core
        # value"): thời gian thật (wall-clock, đo bằng time.perf_counter,
        # KHÔNG phải ước lượng) từ lúc 1 cửa sổ khẩn cấp tới lúc hệ thống
        # tính xong + áp dụng allocation mới. Đây là số liệu THẬT, không
        # phải con số "tiết kiệm được X phút so với thao tác tay" bịa ra
        # (không có baseline thao tác tay thật để so sánh công bằng).
        self._reallocation_latencies_ms: deque[float] = deque(maxlen=200)

    def update(
        self, frame: SimulationFrame, window_gnb_id: str, is_emergency: bool, latency_ms: float | None
    ) -> None:
        with self._lock:
            self._history[window_gnb_id].append(is_emergency)
            self._last_timestamp = frame.timestamp.isoformat()
            if frame.result is not None:
                for alloc in frame.result.allocations:
                    self._current_allocations[alloc.slice_id] = alloc.gnb_id

            if latency_ms is not None:
                self._reallocation_latencies_ms.append(latency_ms)

            allocation_summary = ", ".join(
                f"{slice_id}->{gnb_id or '(none)'}" for slice_id, gnb_id in self._current_allocations.items()
            )
            self._events.appendleft(
                {
                    "timestamp": self._last_timestamp,
                    "gnb_id": window_gnb_id,
                    "emergency": is_emergency,
                    "allocation_summary": allocation_summary,
                    "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
                }
            )

    def snapshot(self) -> dict:
        with self._lock:
            stations_out = []
            for gnb_id, station in self._stations.items():
                served_ids = [sid for sid, g in self._current_allocations.items() if g == gnb_id]
                used = sum(self._slices[sid].prb_required for sid in served_ids)
                history = self._history[gnb_id]
                stations_out.append(
                    {
                        "gnb_id": gnb_id,
                        "capacity": station.prb_capacity,
                        "used": used,
                        "emergency": history[-1] if history else False,
                        "history": list(history),
                        "slices": [
                            {"slice_id": sid, "slice_type": str(self._slices[sid].slice_type.value)}
                            for sid in served_ids
                        ],
                    }
                )

            unserved = [
                {"slice_id": sid, "slice_type": str(self._slices[sid].slice_type.value)}
                for sid, gnb_id in self._current_allocations.items()
                if gnb_id is None
            ]

            latencies = list(self._reallocation_latencies_ms)
            metric = {
                "last_latency_ms": round(latencies[-1], 1) if latencies else None,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
                "n_reallocations": len(latencies),
            }

            return {
                "connected": True,
                "solver": self._solver_name,
                "last_update": self._last_timestamp,
                "stations": stations_out,
                "unserved_slices": unserved,
                "events": list(self._events),
                "station_ids": list(self._stations.keys()),
                "metric": metric,
            }


def _run_pipeline(runner: Runner, state: SharedState, tick_delay: float) -> None:
    """Lặp vô hạn: khi 1 lượt ``StreamSimulator`` phát hết dữ liệu, sinh
    lượt mới và chạy tiếp — để phiên demo không tự dừng giữa chừng lúc
    người dùng đang bấm nút thử nghiệm.

    Tự lặp qua ``stream.stream()`` thay vì dùng ``runner.run_forever()``
    trực tiếp (2 cách tương đương về hành vi) — CHỈ để có chỗ đo
    ``time.perf_counter()`` quanh mỗi lần gọi ``run_once()``, phục vụ
    metric "reallocation latency" (Week 4: "add 1 metric to track core
    value"). Không cần sửa gì trong ``Runner``/``pipeline`` — đo hoàn
    toàn từ bên ngoài, qua API public sẵn có.
    """
    while True:
        stream = build_stream()
        for window in stream.stream():
            prev_result = runner.last_result
            t0 = time.perf_counter()
            frame = runner.run_once(window)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            window_gnb_id = frame.windows[0].gnb_id
            current_pred = next((p for p in frame.predictions if p.gnb_id == window_gnb_id), None)
            is_emergency = bool(current_pred and current_pred.emergency_flag)
            # Chỉ tính là 1 "reallocation event" nếu Runner thực sự chạy
            # lại solve()+apply() ở lần gọi này (so sánh identity, không
            # phải giá trị — mỗi lần solve() luôn tạo OptimizationResult
            # mới dù allocation có trùng giá trị hay không).
            reoptimized = frame.result is not prev_result

            state.update(frame, window_gnb_id, is_emergency, elapsed_ms if reoptimized else None)
            if tick_delay > 0:
                time.sleep(tick_delay)


def _make_handler(state: SharedState, manual_provider: ManualOverridePredictionProvider):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # im lặng, tránh spam console
            pass

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._serve_file(_ASSETS_DIR / "index.html", "text/html; charset=utf-8")
            elif self.path == "/state":
                self._send_json(state.snapshot())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/trigger":
                gnb_id = parse_qs(parsed.query).get("gnb", [None])[0]
                if gnb_id:
                    manual_provider.trigger(gnb_id)
                    self._send_json({"ok": True, "triggered": gnb_id})
                else:
                    self.send_response(400)
                    self.end_headers()
            elif parsed.path == "/reset":
                manual_provider.reset()
                self._send_json({"ok": True})
            else:
                self.send_response(404)
                self.end_headers()

        def _serve_file(self, path: Path, content_type: str) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantaSlice — demo trực quan qua trình duyệt")
    parser.add_argument("--solver", choices=["classical_greedy", "qaoa_aer"], default="classical_greedy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--tick-delay",
        type=float,
        default=0.15,
        help="Giãn cách (giây) giữa các frame để mắt người theo dõi kịp — không ảnh hưởng logic pipeline",
    )
    parser.add_argument(
        "--no-auto-detect",
        dest="auto_detect",
        action="store_false",
        help="Tắt ThresholdPredictionProvider chạy nền — CHỈ dùng nút Trigger để demo, "
        "hữu ích cho usability test có kiểm soát (tránh emergency ngẫu nhiên gây nhiễu). "
        "Mặc định: BẬT (AI thật tự đọc dữ liệu và phát hiện bất thường ở nền).",
    )
    parser.set_defaults(auto_detect=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Configuration(solver=args.solver, emergency_threshold=0.5)
    slices, stations = build_topology()

    try:
        base_provider = prediction_provider_registry.create("threshold" if args.auto_detect else "mock")
        solver = solver_registry.create(args.solver, config=config)
        orchestrator = orchestrator_registry.create("mock_oran")
    except Exception as exc:  # noqa: BLE001 - demo script, in lỗi thân thiện rồi thoát
        print(f"Không khởi tạo được runner: {exc}")
        print("Gợi ý: pip install -e '.[quantum]' nếu dùng --solver qaoa_aer")
        return

    manual_provider = ManualOverridePredictionProvider(base_provider)
    runner = Runner(
        provider=manual_provider,
        solver=solver,
        orchestrator=orchestrator,
        slices=slices,
        stations=stations,
        config=config,
    )

    state = SharedState(stations, slices, solver_name=args.solver)
    worker = threading.Thread(target=_run_pipeline, args=(runner, state, args.tick_delay), daemon=True)
    worker.start()

    server = HTTPServer((args.host, args.port), _make_handler(state, manual_provider))
    url = f"http://{args.host}:{args.port}"
    print(f"Solver: {args.solver}  |  auto-detect nền: {'BẬT' if args.auto_detect else 'TẮT (chỉ trigger qua nút)'}")
    print(f"Mở {url} trên trình duyệt để xem demo trực quan (Ctrl+C để dừng)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")


if __name__ == "__main__":
    main()
