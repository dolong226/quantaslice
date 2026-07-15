"""Đọc dataset ColO-RAN (`rome_static_medium`) thành các ``BSFrame`` —
chuỗi KPM per-slice đã align về một trục thời gian chung cho mỗi base
station, đúng §1.1 và §3.1 của plan ML.

Cấu trúc thư mục thật::

    rome_static_medium/sched{0,1,2}/tr{0..27}/exp{1..5}/bs{1..7}/
        slices_bs{n}/<IMSI>_metrics.csv   # per-UE, 2 UE/slice, cadence 250ms

Mỗi file ``*_metrics.csv`` là chuỗi KPM của MỘT UE (một slice cố định).
Loader gộp các UE cùng slice của một BS lại (sum throughput/buffer/PRB),
nội suy về lưới thời gian đều (cadence chung), rồi xếp 3 slice thành một
mảng ``(T, 3, R)``.

Module này CHỈ import từ ``quantaslice.core`` — KHÔNG import ngược lại
package con nào khác (đúng ràng buộc kiến trúc như quantum/orchestrator).
"""

from __future__ import annotations

import csv
import glob
import logging
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from quantaslice.core.exceptions import ConfigurationError

__all__ = [
    "RAW_METRICS",
    "SLICE_IDS",
    "SLICE_NAMES",
    "BSFrame",
    "load_bs_frame",
    "iter_frames",
    "generate_synthetic",
]

logger = logging.getLogger(__name__)

# Thứ tự metric thô trong trục cuối của BSFrame.series — CỐ ĐỊNH (labeling
# và features phụ thuộc vào thứ tự này).
RAW_METRICS: tuple[str, ...] = (
    "dl_buffer",       # dl_buffer [bytes] — hàng đợi downlink (proxy độ trễ)
    "tx_brate_dl",     # tx_brate downlink [Mbps] — throughput đạt được
    "requested_prbs",  # sum_requested_prbs — nhu cầu PRB
    "granted_prbs",    # sum_granted_prbs — PRB được cấp
    "slice_prb",       # slice_prb — số PRB phân bổ cho slice (hằng theo config tr)
)
_R = len(RAW_METRICS)

# slice_id trong dataset: 0=eMBB, 1=MTC(mMTC), 2=URLLC (theo README ColO-RAN).
SLICE_IDS: tuple[int, int, int] = (0, 1, 2)
SLICE_NAMES: tuple[str, str, str] = ("eMBB", "mMTC", "URLLC")

# Ánh xạ tên cột CSV thật -> khoá metric nội bộ.
_CSV_COLUMNS = {
    "dl_buffer": "dl_buffer [bytes]",
    "tx_brate_dl": "tx_brate downlink [Mbps]",
    "requested_prbs": "sum_requested_prbs",
    "granted_prbs": "sum_granted_prbs",
    "slice_prb": "slice_prb",
}
# Metric gộp theo UE bằng phép cộng (aggregate) — slice_prb thì lấy trung
# bình vì nó là hằng số per-slice (mọi UE cùng slice có cùng giá trị).
_SUM_METRICS = frozenset({"dl_buffer", "tx_brate_dl", "requested_prbs", "granted_prbs"})


@dataclass(frozen=True, slots=True)
class BSFrame:
    """Chuỗi KPM đã align của một (sched, tr, exp, BS).

    ``series`` có shape ``(T, 3, R)``: T bước thời gian, 3 slice
    (eMBB/mMTC/URLLC theo :data:`SLICE_IDS`), R metric theo
    :data:`RAW_METRICS`. ``n_ue`` là số UE mỗi slice (dùng để tính
    demand throughput trong labeling)."""

    sched: int
    tr: int
    exp: int
    bs: str
    time: np.ndarray       # (T,) giây, tương đối từ mốc đầu
    series: np.ndarray     # (T, 3, R)
    n_ue: np.ndarray       # (3,) số UE mỗi slice

    def __post_init__(self) -> None:
        if self.series.ndim != 3 or self.series.shape[1:] != (3, _R):
            raise ConfigurationError(
                f"BSFrame.series phải có shape (T, 3, {_R}), nhận {self.series.shape}"
            )

    @property
    def n_steps(self) -> int:
        return self.series.shape[0]

    def slice_series(self, slice_idx: int) -> np.ndarray:
        """Trả về ``(T, R)`` cho một slice (0/1/2)."""
        return self.series[:, slice_idx, :]

    @property
    def config_id(self) -> str:
        """Định danh cấu hình dùng cho block-split (§3.5)."""
        return f"sched{self.sched}-tr{self.tr}"


# ── Đọc dataset thật ──────────────────────────────────────────────────
def _read_ue_metrics(path: str) -> tuple[int, np.ndarray, dict[str, np.ndarray]]:
    """Đọc một file UE metrics -> (slice_id, timestamps_ms, {metric: array})."""
    ts: list[float] = []
    cols: dict[str, list[float]] = {k: [] for k in _CSV_COLUMNS}
    slice_id = -1
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["Timestamp"])
            except (KeyError, ValueError, TypeError):
                continue
            ts.append(t)
            if slice_id < 0:
                try:
                    slice_id = int(float(row.get("slice_id", -1)))
                except (ValueError, TypeError):
                    slice_id = -1
            for key, csv_name in _CSV_COLUMNS.items():
                try:
                    cols[key].append(float(row.get(csv_name, 0.0) or 0.0))
                except (ValueError, TypeError):
                    cols[key].append(0.0)
    return slice_id, np.asarray(ts), {k: np.asarray(v) for k, v in cols.items()}


def load_bs_frame(
    bs_dir: str, *, sched: int, tr: int, exp: int, cadence_ms: float = 250.0
) -> BSFrame | None:
    """Đọc một thư mục ``bs{n}/`` -> ``BSFrame``.

    Gộp các UE cùng slice (sum các metric throughput/buffer/PRB, mean với
    slice_prb), nội suy tuyến tính về lưới thời gian đều ``cadence_ms``.
    Trả về None nếu không có dữ liệu hợp lệ."""
    bs_name = os.path.basename(bs_dir.rstrip("/"))
    slices_dir = os.path.join(bs_dir, f"slices_{bs_name}")
    metric_files = sorted(glob.glob(os.path.join(slices_dir, "*_metrics.csv")))
    if not metric_files:
        return None

    per_ue: list[tuple[int, np.ndarray, dict[str, np.ndarray]]] = []
    t_min, t_max = np.inf, -np.inf
    for path in metric_files:
        sid, ts, cols = _read_ue_metrics(path)
        if sid not in SLICE_IDS or ts.size < 2:
            continue
        per_ue.append((sid, ts, cols))
        t_min, t_max = min(t_min, ts[0]), max(t_max, ts[-1])
    if not per_ue or not np.isfinite(t_min) or t_max <= t_min:
        return None

    grid = np.arange(t_min, t_max, cadence_ms)
    if grid.size < 2:
        return None
    series = np.zeros((grid.size, 3, _R))
    n_ue = np.zeros(3)

    for sid, ts, cols in per_ue:
        s_idx = SLICE_IDS.index(sid)
        n_ue[s_idx] += 1
        for m_idx, metric in enumerate(RAW_METRICS):
            if metric not in cols:
                continue
            interp = np.interp(grid, ts, cols[metric])
            if metric in _SUM_METRICS:
                series[:, s_idx, m_idx] += interp
            else:  # slice_prb: lấy mean giữa các UE cùng slice
                series[:, s_idx, m_idx] = interp

    time = (grid - grid[0]) / 1000.0
    return BSFrame(sched=sched, tr=tr, exp=exp, bs=bs_name,
                   time=time, series=series, n_ue=n_ue)


def iter_frames(
    root: str,
    *,
    scheds: tuple[int, ...] = (0, 1, 2),
    trs: tuple[int, ...] | None = None,
    exps: tuple[int, ...] = (1,),
    bss: tuple[str, ...] | None = None,
    cadence_ms: float = 250.0,
    limit: int | None = None,
) -> Iterator[BSFrame]:
    """Duyệt cây thư mục ColO-RAN, sinh ``BSFrame`` theo từng (sched, tr,
    exp, BS). Cho phép giới hạn subset để load nhẹ (dataset ~17k file)."""
    base = os.path.join(root, "rome_static_medium")
    if not os.path.isdir(base):
        base = root  # cho phép trỏ thẳng vào rome_static_medium
    count = 0
    for sched in scheds:
        tr_list = trs if trs is not None else _discover_trs(base, sched)
        for tr in tr_list:
            for exp in exps:
                exp_dir = os.path.join(base, f"sched{sched}", f"tr{tr}", f"exp{exp}")
                if not os.path.isdir(exp_dir):
                    continue
                bs_dirs = sorted(glob.glob(os.path.join(exp_dir, "bs*")))
                for bs_dir in bs_dirs:
                    if bss is not None and os.path.basename(bs_dir) not in bss:
                        continue
                    frame = load_bs_frame(bs_dir, sched=sched, tr=tr, exp=exp,
                                          cadence_ms=cadence_ms)
                    if frame is None:
                        continue
                    yield frame
                    count += 1
                    if limit is not None and count >= limit:
                        return


def _discover_trs(base: str, sched: int) -> tuple[int, ...]:
    sched_dir = os.path.join(base, f"sched{sched}")
    trs = []
    for d in glob.glob(os.path.join(sched_dir, "tr*")):
        name = os.path.basename(d)
        if name.startswith("tr") and name[2:].isdigit():
            trs.append(int(name[2:]))
    return tuple(sorted(trs))


# ── Sinh dữ liệu synthetic (cho test, cùng contract với dữ liệu thật) ──
def generate_synthetic(
    *,
    n_frames: int = 12,
    n_steps: int = 400,
    seed: int = 0,
    emergency_fraction: float = 0.4,
) -> list[BSFrame]:
    """Sinh danh sách ``BSFrame`` giả lập cùng shape/thang đo với ColO-RAN.

    Một phần frame được tiêm "starvation": PRB cấp cho URLLC bị bóp trong
    khi buffer dồn và throughput sụt — mô phỏng đúng động lực mà nhãn
    QoS-violation (§2) cần bắt. Dùng cho test/CI khi CHƯA có dataset thật.
    """
    rng = np.random.default_rng(seed)
    frames: list[BSFrame] = []
    for k in range(n_frames):
        series = np.zeros((n_steps, 3, _R))
        time = np.arange(n_steps) * 0.25
        # Thang giống ColO-RAN: eMBB buffer khổng lồ nhưng độ trễ dưới ngân
        # sách (không vi phạm lúc bình thường); mMTC/URLLC gần như rỗng.
        # dl_buffer (bytes), tx_brate (Mbps), requested/granted PRB, slice_prb.
        cfg = [
            dict(buf=(366000, 3000), br=(3.5, 0.15), prb=12),   # eMBB
            dict(buf=(0, 80),        br=(0.10, 0.02), prb=6),    # mMTC (idle)
            dict(buf=(0, 120),       br=(0.20, 0.03), prb=8),    # URLLC (idle)
        ]
        for s_idx, c in enumerate(cfg):
            series[:, s_idx, 0] = np.abs(rng.normal(*c["buf"], n_steps))
            series[:, s_idx, 1] = np.abs(rng.normal(*c["br"], n_steps))
            series[:, s_idx, 2] = np.abs(rng.normal(c["prb"] + 4, 2, n_steps))
            series[:, s_idx, 3] = np.abs(rng.normal(c["prb"], 2, n_steps))
            series[:, s_idx, 4] = c["prb"]

        is_emerg = k < int(round(n_frames * emergency_fraction))
        if is_emerg:
            # Tiêm tắc nghẽn THẬT vào URLLC: hàng đợi dồn lên vài KB trong
            # khi throughput sụt -> độ trễ vượt ngân sách 80ms (severity->1).
            t0 = int(rng.integers(n_steps // 4, n_steps // 2))
            t1 = min(n_steps, t0 + int(rng.integers(40, 100)))
            series[t0:t1, 2, 0] = np.linspace(1500, 6000, t1 - t0)  # buffer KB
            series[t0:t1, 2, 1] = 0.12                               # throughput sụt
            series[t0:t1, 2, 3] *= 0.3                               # granted bị bóp
        frames.append(BSFrame(
            sched=k % 3, tr=k, exp=1, bs=f"bs{(k % 7) + 1}",
            time=time, series=series, n_ue=np.array([2.0, 2.0, 2.0]),
        ))
    return frames
