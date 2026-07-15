"""Tests cho tiêm emergency (scenario.py) + metric lead-time.

Kiểm tra:
1. inject_emergencies — buffer DỐC LÊN trong sự kiện và độ trễ vượt ngưỡng
   SAU onset (có runway/precursor); mask ground-truth khớp cửa sổ tiêm.
2. generate_emergency_scenario — frame tiêm có buffer cao hơn nền; frame
   không tiêm giữ nguyên (mask toàn False).
3. lead_time_recall — thưởng model bắt sớm, phạt luật 'đang vi phạm chưa'.
"""

from __future__ import annotations

import numpy as np

from quantaslice.ai.data.loaders import BSFrame, RAW_METRICS, generate_synthetic
from quantaslice.ai.data.scenario import (
    EmergencyEvent,
    generate_emergency_scenario,
    inject_emergencies,
)
from quantaslice.ai.eval.metrics import lead_time_recall

_BUF = RAW_METRICS.index("dl_buffer")
_BRATE = RAW_METRICS.index("tx_brate_dl")


def _clean_frame(T=300):
    """Frame nền 'sạch': slice 0 buffer gần rỗng + throughput cao (runway
    lớn) — đúng điều kiện để tiêm tạo precursor chậm, không như eMBB thật
    (đã bão hoà) hay URLLC (ngân sách ms, vi phạm tức thì)."""
    series = np.zeros((T, 3, len(RAW_METRICS)))
    series[:, 0, _BUF] = 2000.0     # ~2 KB (gần rỗng)
    series[:, 0, _BRATE] = 4.0      # 4 Mbps
    series[:, 0, 2] = 10.0; series[:, 0, 3] = 10.0; series[:, 0, 4] = 12.0
    return BSFrame(sched=0, tr=0, exp=1, bs="bs1",
                   time=np.arange(T) * 0.25, series=series, n_ue=np.array([2., 2., 2.]))


def test_injection_builds_up_and_breaches_after_onset():
    ev = EmergencyEvent(slice_idx=0, onset=60, ramp_steps=50, hold_steps=30, intensity=2.5)
    frame, mask = inject_emergencies(_clean_frame(), (ev,))

    assert mask[60:140].all() and not mask[:60].any()
    buf = frame.slice_series(0)[:, _BUF]
    assert buf[100] > buf[55] * 5      # buffer dồn mạnh trong ramp
    # Độ trễ vượt ngưỡng XẢY RA SAU onset (có precursor), không tức thì.
    brate = np.clip(frame.slice_series(0)[:, _BRATE], 1e-3, None)
    delay = buf * 8 / (brate * 1e6)
    breach = np.where(delay >= 0.3)[0]
    assert breach.size > 0 and breach.min() > 60


def test_scenario_marks_only_injected_frames():
    base = generate_synthetic(n_frames=6, n_steps=260, seed=1, emergency_fraction=0.0)
    scen = generate_emergency_scenario(base, seed=2, fraction=0.5, slice_idx=0)
    n_emerg = sum(1 for _, m in scen if m.any())
    assert n_emerg == 3  # đúng 50%
    for fr, m in scen:
        if not m.any():
            continue
        buf = fr.slice_series(0)[:, _BUF]
        assert buf[m].max() > buf[~m].mean() if (~m).any() else True


def test_lead_time_recall_rewards_early_detection():
    # 4 bước onset (sắp vi phạm nhưng chưa). Model bắt được 3/4; luật 0/4.
    future = np.array([1, 1, 1, 1, 1, 0])
    now = np.array([0, 0, 0, 0, 1, 0])          # bước 5 mới thực sự vi phạm
    ml = np.array([0.9, 0.8, 0.7, 0.2, 0.9, 0.1])
    rule = now.astype(float)                     # luật = đang vi phạm
    assert lead_time_recall(future, now, ml, 0.5) == 0.75   # 3/4 onset
    assert lead_time_recall(future, now, rule, 0.5) == 0.0  # luật bỏ lỡ hết
