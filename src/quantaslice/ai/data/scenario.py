"""Tiêm emergency VẬT LÝ vào frame nền để tạo bài toán DỰ BÁO có ý nghĩa
— giải quyết hạn chế cốt lõi: ColO-RAN ``rome_static_medium`` gần TĨNH,
không chứa emergency thật (UE bất động, traffic đều), nên không thể
validate một detector "khẩn cấp" trên nó (xem audit §2).

Cơ chế (không phải nhiễu ngẫu nhiên — tránh circular evaluation §2):
một **traffic surge** làm tải đến của slice tăng dần vượt dung lượng
phục vụ; buffer tích tụ như một *leaky integrator*::

    buffer[t] = max(0, buffer[t-1] + arrival_bytes[t] - capacity_bytes)

Khi tải > dung lượng, buffer dốc lên; độ trễ hàng đợi vượt ngưỡng SAU
một khoảng (runway) tỉ lệ với ``ramp_steps``. Nhờ vậy có **precursor**
(buffer đang dốc) xuất hiện TRƯỚC khi vi phạm — điều kiện để tầng ML
mua được lead-time (dự báo H bước trước), thứ mà luật "đang vi phạm
chưa?" không làm được.

TRUNG THỰC / PROVENANCE: đây là dữ liệu **bán-tổng-hợp** (KPM nền thật +
sự kiện tiêm). Nhãn sự kiện = lịch tiêm (độc lập với KPM), KHÔNG phải
ngưỡng trên chính KPM -> không circular. Giá trị ML (lead-time) CHỈ có
khi slice có "runway" đủ dài (throughput cao, ngân sách độ trễ rộng như
eMBB); với URLLC (ngân sách ~ms) vi phạm gần như tức thì, dự báo ≈
nowcast — đã kiểm nghiệm.

Module CHỈ import từ ``core`` + ``ai.data.loaders``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantaslice.ai.data.loaders import RAW_METRICS, BSFrame

__all__ = ["EmergencyEvent", "inject_emergencies", "generate_emergency_scenario"]

_BUF = RAW_METRICS.index("dl_buffer")
_BRATE = RAW_METRICS.index("tx_brate_dl")
_REQ = RAW_METRICS.index("requested_prbs")
_DT = 0.25  # giây/bước (cadence ColO-RAN)


@dataclass(frozen=True, slots=True)
class EmergencyEvent:
    """Một sự kiện tắc nghẽn tiêm vào một slice."""

    slice_idx: int      # 0=eMBB, 1=mMTC, 2=URLLC
    onset: int          # bước bắt đầu
    ramp_steps: int     # số bước tải tăng dần (precursor — càng dài lead-time càng lớn)
    hold_steps: int     # số bước giữ đỉnh trước khi thoát
    intensity: float    # bội số tải đỉnh so với dung lượng (>1 mới gây dồn)


def inject_emergencies(
    frame: BSFrame,
    events: tuple[EmergencyEvent, ...],
    *,
    capacity_headroom: float = 1.1,
) -> tuple[BSFrame, np.ndarray]:
    """Tiêm các ``events`` vào ``frame``; trả (frame mới, mask sự kiện).

    ``mask[t]`` = True nếu có sự kiện đang diễn ra tại bước t (ground-truth
    độc lập với KPM). ``capacity_headroom`` đặt dung lượng phục vụ =
    ``headroom × throughput nền đỉnh`` (>1 để lúc bình thường không dồn)."""
    series = frame.series.copy()
    mask = np.zeros(frame.n_steps, dtype=bool)

    for ev in events:
        s = ev.slice_idx
        brate = np.clip(series[:, s, _BRATE], 0, None)
        cap_mbps = max(np.percentile(brate, 90) * capacity_headroom, 0.1)
        cap_bytes = cap_mbps * 1e6 / 8 * _DT
        buf = series[:, s, _BUF].copy()

        t1 = min(frame.n_steps, ev.onset + ev.ramp_steps + ev.hold_steps)
        for t in range(ev.onset, t1):
            phase = t - ev.onset
            if phase < ev.ramp_steps:
                factor = 1.0 + (ev.intensity - 1.0) * (phase / max(ev.ramp_steps, 1))
            else:
                factor = ev.intensity
            load_mbps = brate[t] * factor
            arrival_bytes = load_mbps * 1e6 / 8 * _DT
            buf[t] = max(0.0, buf[t - 1] + arrival_bytes - cap_bytes)
            series[t, s, _BRATE] = min(load_mbps, cap_mbps)   # throughput bão hoà ở cap
            series[t, s, _REQ] = series[t, s, _REQ] * factor  # nhu cầu PRB tăng theo
            mask[t] = True
        # Thoát: buffer drain tự nhiên (arrival nền < cap) — để lại "đuôi".
        for t in range(t1, min(frame.n_steps, t1 + ev.ramp_steps + ev.hold_steps)):
            arrival_bytes = np.clip(series[t, s, _BRATE], 0, None) * 1e6 / 8 * _DT
            buf[t] = max(0.0, buf[t - 1] + arrival_bytes - cap_bytes)
        series[:, s, _BUF] = buf

    return (
        BSFrame(sched=frame.sched, tr=frame.tr, exp=frame.exp, bs=frame.bs,
                time=frame.time, series=series, n_ue=frame.n_ue),
        mask,
    )


def generate_emergency_scenario(
    base_frames: list[BSFrame],
    *,
    seed: int = 0,
    fraction: float = 0.5,
    slice_idx: int = 0,
    intensity: tuple[float, float] = (1.8, 2.6),
    ramp_steps: tuple[int, int] = (40, 70),
) -> list[tuple[BSFrame, np.ndarray]]:
    """Tiêm surge vào ``fraction`` số frame nền (mặc định slice eMBB — có
    'runway' đủ dài để dự báo có ý nghĩa). Trả list (frame, mask).

    Frame không tiêm -> mask toàn False (mẫu âm). Dùng ``slice_idx``,
    ``intensity``, ``ramp_steps`` để điều chỉnh độ dốc precursor."""
    rng = np.random.default_rng(seed)
    out: list[tuple[BSFrame, np.ndarray]] = []
    n_emerg = int(round(len(base_frames) * fraction))
    for i, fr in enumerate(base_frames):
        if i < n_emerg and fr.n_steps > 160:
            onset = int(rng.integers(30, fr.n_steps - 140))
            ev = EmergencyEvent(
                slice_idx=slice_idx, onset=onset,
                ramp_steps=int(rng.integers(*ramp_steps)),
                hold_steps=int(rng.integers(20, 50)),
                intensity=float(rng.uniform(*intensity)),
            )
            out.append(inject_emergencies(fr, (ev,)))
        else:
            out.append((fr, np.zeros(fr.n_steps, dtype=bool)))
    return out
