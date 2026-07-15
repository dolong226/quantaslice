"""Feature engineering + windowing — §3.2, §3.3 của plan ML.

Từ ``BSFrame`` (chuỗi KPM thô per-slice) sinh:

1. **Feature per-timestep** ``(T, F_ts)`` giàu tín hiệu hơn KPM thô:
   buffer growth rate, PRB utilization, throughput deficit, tỉ trọng
   liên-slice (emergency thường là hiện tượng TƯƠNG QUAN giữa các slice).
2. **Cửa sổ trượt** lookback ``W`` -> tensor ``(N, W, F_ts)`` cho model
   dạng chuỗi (TCN/Transformer sau này).
3. **Tóm tắt cửa sổ** ``(N, F_ts * n_stats)`` (mean/std/max/last/slope)
   cho model dạng tabular (LightGBM baseline §4).

Nhãn của một cửa sổ lấy tại bước CUỐI cửa sổ (nowcast) — sẵn sàng mở
rộng horizon ``H`` cho dự báo lead-time (§6.3) sau này.

Module CHỈ import từ ``core`` + ``ai.data`` nội bộ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantaslice.ai.data.labeling import PER_UE_DEMAND_MBPS, FrameLabels
from quantaslice.ai.data.loaders import RAW_METRICS, SLICE_IDS, SLICE_NAMES, BSFrame

__all__ = [
    "FEATURE_TS_NAMES",
    "SUMMARY_STATS",
    "WindowConfig",
    "frame_features",
    "summary_feature_names",
    "build_tabular",
    "build_sequences",
    "summarize_window",
]

_BUF = RAW_METRICS.index("dl_buffer")
_BRATE = RAW_METRICS.index("tx_brate_dl")
_REQ = RAW_METRICS.index("requested_prbs")
_GRANT = RAW_METRICS.index("granted_prbs")
_PRB_CAPACITY = 50.0  # 10 MHz = 50 PRB (ColO-RAN)
_GROWTH_W = 8         # ~2s @250ms cho buffer growth feature

# Tên feature per-timestep, sinh theo thứ tự trong frame_features().
_PER_SLICE_FEATS = ("log_buffer", "brate", "buffer_growth", "prb_util_req",
                    "prb_util_cap", "tput_deficit")
_INTERSLICE_FEATS = ("brate_share", "buffer_share")
FEATURE_TS_NAMES: tuple[str, ...] = tuple(
    f"{name}_{feat}" for name in SLICE_NAMES for feat in _PER_SLICE_FEATS
) + tuple(f"{name}_{feat}" for name in SLICE_NAMES for feat in _INTERSLICE_FEATS)

# Thống kê tóm tắt cửa sổ cho nhánh tabular.
SUMMARY_STATS = ("mean", "std", "max", "last", "slope")


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Tham số cửa sổ trượt."""

    lookback: int = 20   # W bước (~5s @250ms)
    stride: int = 5      # bước nhảy khi train (stride=1 khi online)
    horizon: int = 0     # H: nhãn tại bước (cuối + H); 0 = nowcast


def _windowed_delta(x: np.ndarray, w: int) -> np.ndarray:
    prev = np.concatenate([np.full(min(w, x.size), x[0]), x[:-w]]) if x.size > w \
        else np.full_like(x, x[0])
    return x - prev


def frame_features(frame: BSFrame) -> np.ndarray:
    """Sinh ma trận feature per-timestep ``(T, F_ts)`` từ một ``BSFrame``."""
    t = frame.n_steps
    cols: list[np.ndarray] = []

    brate_all = np.clip(frame.series[:, :, _BRATE], 0, None)   # (T, 3)
    buf_all = np.clip(frame.series[:, :, _BUF], 0, None)
    total_brate = brate_all.sum(axis=1) + 1e-6
    total_buf = buf_all.sum(axis=1) + 1.0

    # Feature per-slice.
    for s_idx in range(3):
        s = frame.series[:, s_idx, :]
        buf = np.clip(s[:, _BUF], 0, None)
        brate = np.clip(s[:, _BRATE], 0, None)
        req = np.clip(s[:, _REQ], 0, None)
        grant = np.clip(s[:, _GRANT], 0, None)
        demand = PER_UE_DEMAND_MBPS[SLICE_IDS[s_idx]] * max(frame.n_ue[s_idx], 1.0)

        cols.append(np.log1p(buf))
        cols.append(brate)
        cols.append(_windowed_delta(buf, _GROWTH_W) / (np.median(buf) + 1.0))
        # PRB utilization = cấp / yêu cầu; = 1.0 khi không có yêu cầu
        # (np.divide có where= để tránh cảnh báo chia 0).
        util = np.divide(grant, req, out=np.ones_like(grant), where=req > 1e-6)
        cols.append(np.clip(util, 0.0, 2.0))
        cols.append(grant / _PRB_CAPACITY)
        cols.append(np.clip((demand - brate) / demand, 0, 1) if demand > 0.1 else np.zeros(t))

    # Feature liên-slice: tỉ trọng throughput / buffer của mỗi slice.
    for s_idx in range(3):
        cols.append(brate_all[:, s_idx] / total_brate)
    for s_idx in range(3):
        cols.append(buf_all[:, s_idx] / total_buf)

    return np.column_stack(cols)


def summary_feature_names(ts_names: tuple[str, ...] = FEATURE_TS_NAMES) -> list[str]:
    """Tên feature của vector tóm tắt cửa sổ (tabular)."""
    return [f"{n}_{stat}" for n in ts_names for stat in SUMMARY_STATS]


def summarize_window(win: np.ndarray) -> np.ndarray:
    """Tóm tắt một cửa sổ ``(W, F_ts)`` -> vector ``(F_ts * 5,)``:
    mean, std, max, last, slope (hệ số góc hồi quy tuyến tính theo thời
    gian) cho mỗi feature."""
    w = win.shape[0]
    mean = win.mean(axis=0)
    std = win.std(axis=0)
    mx = win.max(axis=0)
    last = win[-1]
    # slope: cov(t, x)/var(t), t = 0..W-1.
    tt = np.arange(w)
    tt_c = tt - tt.mean()
    denom = (tt_c**2).sum() or 1.0
    slope = (tt_c[:, None] * (win - mean)).sum(axis=0) / denom
    # Xếp xen kẽ theo feature để khớp summary_feature_names.
    return np.stack([mean, std, mx, last, slope], axis=1).reshape(-1)


def _window_starts(t: int, wcfg: WindowConfig) -> np.ndarray:
    last_start = t - wcfg.lookback - wcfg.horizon
    if last_start < 0:
        return np.empty(0, dtype=int)
    return np.arange(0, last_start + 1, wcfg.stride)


def build_sequences(
    frames: list[BSFrame], labels: list[FrameLabels], wcfg: WindowConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cửa sổ dạng chuỗi cho model deep: X ``(N, W, F_ts)``, y_flag
    ``(N,)``, y_priority ``(N, 3)``, groups ``(N,)`` (config_id để split)."""
    wcfg = wcfg or WindowConfig()
    xs, yf, yp, grp = [], [], [], []
    for frame, lab in zip(frames, labels):
        feats = frame_features(frame)
        for start in _window_starts(frame.n_steps, wcfg):
            end = start + wcfg.lookback
            target = end - 1 + wcfg.horizon
            xs.append(feats[start:end])
            yf.append(float(lab.flag[target]))
            yp.append(lab.priority[target])
            grp.append(frame.config_id)
    if not xs:
        return (np.empty((0, wcfg.lookback, len(FEATURE_TS_NAMES))),
                np.empty(0), np.empty((0, 3)), np.empty(0, dtype=object))
    return np.asarray(xs), np.asarray(yf), np.asarray(yp), np.asarray(grp, dtype=object)


def build_tabular(
    frames: list[BSFrame], labels: list[FrameLabels], wcfg: WindowConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Đại diện tabular: X ``(N, F_ts*5)`` (tóm tắt cửa sổ), y_flag, y_priority,
    groups — dùng cho LightGBM baseline (§4)."""
    x_seq, y_flag, y_prio, groups = build_sequences(frames, labels, wcfg)
    if x_seq.shape[0] == 0:
        return (np.empty((0, len(FEATURE_TS_NAMES) * len(SUMMARY_STATS))),
                y_flag, y_prio, groups)
    x_tab = np.stack([summarize_window(win) for win in x_seq])
    return x_tab, y_flag, y_prio, groups
