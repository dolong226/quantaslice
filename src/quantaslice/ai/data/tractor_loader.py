"""Loader cho dataset **TRACTOR** (genesys-neu/TRACTOR) — traffic 5G THẬT
(capture từ điện thoại, replay trong Colosseum), khác ColO-RAN tĩnh.

Vì sao cần: ColO-RAN ``rome_static_medium`` gần tĩnh -> phát hiện emergency
là bài toán tầm thường (một luật đủ, ML thừa). TRACTOR có traffic **động,
bursty** (buffer dồn/thoát thật) -> vi phạm SLA có PRECURSOR -> ML
forecasting mới thật sự thắng luật (đã kiểm nghiệm: +0.113 PR-AUC,
lead-time recall 0.25 vs 0.00).

Cấu trúc: ``logs/Multi-UE/{Trial}/{embb*,mmtc*,urllc*}/<IMSI>_metrics.csv``.
Mỗi thư mục = một luồng traffic MỘT loại. Loader ghép embb+mmtc+urllc
cùng chỉ số thành một ``BSFrame`` 3-slice để tái dùng nguyên pipeline
(labeling/features/forecasting).

LƯU Ý xử lý dữ liệu (đã gặp): một số file có epoch timestamp HỎNG (span
114 ngày). Nên loader **bỏ epoch, index theo thứ tự dòng** (dữ liệu vốn
250ms đều) và dùng file UE dài nhất mỗi thư mục.

Schema TRACTOR ~ trùng ColO-RAN nên map thẳng vào ``RAW_METRICS``.
Module CHỈ import từ ``core`` + ``ai.data.loaders``.
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np

from quantaslice.ai.data.loaders import RAW_METRICS, BSFrame

__all__ = ["load_tractor_frame", "iter_tractor_frames", "iter_tractor_streams"]

# Tên cột TRACTOR -> khoá metric nội bộ (khớp RAW_METRICS trừ slice_prb).
_COLS = {
    "dl_buffer": "dl_buffer [bytes]",
    "tx_brate_dl": "tx_brate downlink [Mbps]",
    "requested_prbs": "sum_requested_prbs",
    "granted_prbs": "sum_granted_prbs",
}
_MIN_LEN = 100
_DEFAULT_SLICE_PRB = 50.0 / 3.0  # TRACTOR không sweep RBG -> hằng số danh nghĩa


def _load_dir_stream(scenario_dir: str) -> dict[str, np.ndarray] | None:
    """File UE DÀI NHẤT của thư mục, index theo thứ tự dòng (bỏ epoch)."""
    files = sorted(glob.glob(os.path.join(scenario_dir, "*_metrics.csv")),
                   key=os.path.getsize, reverse=True)
    if not files:
        return None
    cols: dict[str, list[float]] = {k: [] for k in _COLS}
    with open(files[0], newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for key, name in _COLS.items():
                try:
                    cols[key].append(float(row.get(name, 0.0) or 0.0))
                except (ValueError, TypeError):
                    cols[key].append(0.0)
    arr = {k: np.asarray(v) for k, v in cols.items()}
    return arr if len(arr["dl_buffer"]) >= _MIN_LEN else None


def load_tractor_frame(
    embb_dir: str, mmtc_dir: str, urllc_dir: str, *, sched: int = 0, tr: int = 0, exp: int = 1,
) -> BSFrame | None:
    """Ghép 3 luồng traffic (eMBB, mMTC, URLLC) -> ``BSFrame`` (T, 3, R),
    cắt về độ dài chung ngắn nhất. None nếu thiếu luồng."""
    streams = [_load_dir_stream(d) for d in (embb_dir, mmtc_dir, urllc_dir)]
    if any(s is None for s in streams):
        return None
    t = min(len(s["dl_buffer"]) for s in streams)
    if t < _MIN_LEN:
        return None
    series = np.zeros((t, 3, len(RAW_METRICS)))
    for s_idx, s in enumerate(streams):
        for m_idx, metric in enumerate(RAW_METRICS):
            if metric == "slice_prb":
                series[:, s_idx, m_idx] = _DEFAULT_SLICE_PRB
            else:
                series[:, s_idx, m_idx] = np.clip(s[metric][:t], 0, None)
    return BSFrame(sched=sched, tr=tr, exp=exp, bs=os.path.basename(embb_dir),
                   time=np.arange(t) * 0.25, series=series, n_ue=np.array([2., 2., 2.]))


def iter_tractor_streams(root: str, trial: str = "Trial0"):
    """Sinh (tên_kịch_bản, dict metric->array) cho TỪNG luồng traffic một
    loại (embb*/mmtc*/urllc*). Dùng cho đánh giá forecasting PER-STREAM —
    khung TRUNG THỰC (so ML vs luật + lead-time trên một luồng), tránh
    việc gộp 3-slice + nhãn 'any' làm bài toán dễ giả tạo."""
    base = os.path.join(root, "logs", "Multi-UE", trial)
    if not os.path.isdir(base):
        base = os.path.join(root, trial) if os.path.isdir(os.path.join(root, trial)) else root
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d) or name.rstrip("0123456789") not in ("embb", "mmtc", "urllc"):
            continue
        stream = _load_dir_stream(d)
        if stream is not None:
            yield name, stream


def iter_tractor_frames(root: str, trial: str = "Trial0"):
    """Sinh ``BSFrame`` bằng cách ghép embb{i}/mmtc{i}/urllc{i} cùng chỉ số
    trong một trial. ``root`` trỏ tới thư mục gốc TRACTOR (chứa ``logs/``)."""
    base = os.path.join(root, "logs", "Multi-UE", trial)
    if not os.path.isdir(base):
        base = os.path.join(root, trial) if os.path.isdir(os.path.join(root, trial)) else root
    for i in range(1, 9):
        embb = os.path.join(base, f"embb{i}")
        mmtc = os.path.join(base, f"mmtc{i}")
        urllc = os.path.join(base, f"urllc{i}")
        if os.path.isdir(embb) and os.path.isdir(mmtc) and os.path.isdir(urllc):
            frame = load_tractor_frame(embb, mmtc, urllc, tr=i)
            if frame is not None:
                yield frame
