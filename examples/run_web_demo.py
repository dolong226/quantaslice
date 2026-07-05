"""Demo trực quan qua trình duyệt: chạy pipeline ở background thread,
serve 1 trang HTML/JS đơn (poll ``/state`` mỗi 300ms) hiển thị trạm nào
đang giữ slice nào, đổi ra sao theo thời gian, và báo hiệu khi có
emergency — dùng CHỈ thư viện chuẩn (``http.server``), không cần cài gì
thêm ngoài ``quantaslice[quantum]`` (nếu muốn thử ``--solver qaoa_aer``).

Chạy bằng::

    python -m examples.run_web_demo
    python -m examples.run_web_demo --solver qaoa_aer
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
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from quantaslice.core.runtime import Configuration, SimulationFrame
from quantaslice.core.types import BaseStation, SliceRequest
from quantaslice.pipeline import DependencyContainer
from examples.run_mock_demo import build_stream, build_topology

_ASSETS_DIR = Path(__file__).resolve().parent / "web_demo_assets"
_HISTORY_LENGTH = 40
_EVENT_LOG_LENGTH = 60


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

    def update(self, frame: SimulationFrame, window_gnb_id: str, is_emergency: bool) -> None:
        with self._lock:
            self._history[window_gnb_id].append(is_emergency)
            self._last_timestamp = frame.timestamp.isoformat()
            if frame.result is not None:
                for alloc in frame.result.allocations:
                    self._current_allocations[alloc.slice_id] = alloc.gnb_id

            allocation_summary = ", ".join(
                f"{slice_id}->{gnb_id or '(none)'}" for slice_id, gnb_id in self._current_allocations.items()
            )
            self._events.appendleft(
                {
                    "timestamp": self._last_timestamp,
                    "gnb_id": window_gnb_id,
                    "emergency": is_emergency,
                    "allocation_summary": allocation_summary,
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

            return {
                "connected": True,
                "solver": self._solver_name,
                "last_update": self._last_timestamp,
                "stations": stations_out,
                "unserved_slices": unserved,
                "events": list(self._events),
            }


def _run_pipeline(runner, stream, state: SharedState, tick_delay: float) -> None:
    for frame in runner.run_forever(stream):
        window_gnb_id = frame.windows[0].gnb_id
        current_pred = next((p for p in frame.predictions if p.gnb_id == window_gnb_id), None)
        is_emergency = bool(current_pred and current_pred.emergency_flag)
        state.update(frame, window_gnb_id, is_emergency)
        if tick_delay > 0:
            time.sleep(tick_delay)


def _make_handler(state: SharedState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # im lặng, tránh spam console
            pass

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._serve_file(_ASSETS_DIR / "index.html", "text/html; charset=utf-8")
            elif self.path == "/state":
                body = json.dumps(state.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Configuration(
        prediction_provider="threshold",
        solver=args.solver,
        orchestrator="mock_oran",
        emergency_threshold=0.5,
    )
    slices, stations = build_topology()
    stream = build_stream()

    try:
        runner = DependencyContainer.build_runner(config, slices=slices, stations=stations)
    except Exception as exc:  # noqa: BLE001 - demo script, in lỗi thân thiện rồi thoát
        print(f"Không khởi tạo được runner: {exc}")
        print("Gợi ý: pip install -e '.[quantum]' nếu dùng --solver qaoa_aer")
        return

    state = SharedState(stations, slices, solver_name=args.solver)
    worker = threading.Thread(target=_run_pipeline, args=(runner, stream, state, args.tick_delay), daemon=True)
    worker.start()

    server = HTTPServer((args.host, args.port), _make_handler(state))
    url = f"http://{args.host}:{args.port}"
    print(f"Solver: {args.solver}")
    print(f"Mở {url} trên trình duyệt để xem demo trực quan (Ctrl+C để dừng)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")


if __name__ == "__main__":
    main()
