"""Tests cho tầng data của ML detector: loader, labeling, features, split.

Kiểm tra:
1. generate_synthetic — shape/contract đúng.
2. labeling — emergency là sự kiện CẤP TÍNH (frame tiêm burst có rate cao
   hơn hẳn frame bình thường); priority boost đúng slice bị tiêm.
3. features — shape ma trận feature + số tên feature khớp.
4. summarize_window — số chiều = F_ts * số thống kê.
5. split — leave-scheduler-out / leave-config-out phân hoạch đúng, không
   chồng lấn (chống rò rỉ §3.5).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from quantaslice.ai.data.features import (
    FEATURE_TS_NAMES,
    SUMMARY_STATS,
    WindowConfig,
    build_tabular,
    frame_features,
    summarize_window,
    summary_feature_names,
)
from quantaslice.ai.data.labeling import label_frame
from quantaslice.ai.data.loaders import RAW_METRICS, generate_synthetic, iter_frames
from quantaslice.ai.data.split import leave_config_out, leave_scheduler_out

_DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "colosseum-oran-coloran-dataset"
)
_HAS_DATASET = os.path.isdir(os.path.join(_DATASET_ROOT, "rome_static_medium"))


def test_synthetic_frame_contract():
    frames = generate_synthetic(n_frames=6, n_steps=200, seed=0)
    assert len(frames) == 6
    fr = frames[0]
    assert fr.series.shape == (200, 3, len(RAW_METRICS))
    assert fr.time.shape == (200,)
    assert fr.n_ue.shape == (3,)
    assert fr.slice_series(2).shape == (200, len(RAW_METRICS))


def test_labeling_separates_violation():
    """Nhãn SLA (độ trễ hàng đợi) phải tách rõ frame có tắc nghẽn tiêm vào
    (4 frame đầu, URLLC dồn buffer) khỏi frame nhàn rỗi (slice rỗng dưới
    sàn buffer -> không vi phạm)."""
    frames = generate_synthetic(n_frames=10, n_steps=400, seed=1, emergency_fraction=0.4)
    rates = [label_frame(fr).emergency_rate for fr in frames]
    assert np.mean(rates[:4]) > 0.02   # frame có tắc nghẽn -> có vi phạm
    assert np.mean(rates[4:]) < 0.01   # frame nhàn rỗi -> gần như không


def test_labeling_priority_boosts_injected_slice():
    """Frame 0 tiêm burst vào URLLC (slice-index 2) -> priority URLLC bị
    boost mạnh, eMBB/mMTC giữ baseline."""
    frames = generate_synthetic(n_frames=4, n_steps=400, seed=1)
    lab = label_frame(frames[0])
    peak = lab.priority.max(axis=0)
    assert peak[2] > peak[0] and peak[2] > peak[1]
    assert peak[2] > 2.0


def test_features_shapes_and_names():
    frames = generate_synthetic(n_frames=2, n_steps=120, seed=2)
    feats = frame_features(frames[0])
    assert feats.shape == (120, len(FEATURE_TS_NAMES))
    assert not np.isnan(feats).any()
    summ = summarize_window(feats[:20])
    assert summ.shape == (len(FEATURE_TS_NAMES) * len(SUMMARY_STATS),)
    assert len(summary_feature_names()) == summ.shape[0]


def test_build_tabular_dims():
    frames = generate_synthetic(n_frames=4, n_steps=200, seed=3)
    labels = [label_frame(fr) for fr in frames]
    x, yf, yp, groups = build_tabular(frames, labels, WindowConfig(lookback=20, stride=10))
    assert x.shape[0] == yf.shape[0] == yp.shape[0] == groups.shape[0]
    assert x.shape[1] == len(FEATURE_TS_NAMES) * len(SUMMARY_STATS)
    assert yp.shape[1] == 3
    assert set(np.unique(yf)).issubset({0.0, 1.0})


def test_splits_partition_without_overlap():
    frames = generate_synthetic(n_frames=9, n_steps=100, seed=4)  # sched = k%3
    train, test = leave_scheduler_out(frames, test_scheds=(2,))
    assert all(f.sched != 2 for f in train)
    assert all(f.sched == 2 for f in test)
    assert len(train) + len(test) == len(frames)

    train2, test2 = leave_config_out(frames, test_trs=(0, 1))
    assert {f.tr for f in test2} == {0, 1}
    assert all(f.tr not in (0, 1) for f in train2)
    assert len(train2) + len(test2) == len(frames)


@pytest.mark.skipif(not _HAS_DATASET, reason="Chưa có dataset ColO-RAN")
def test_real_loader_and_labeling():
    """Loader đọc dữ liệu THẬT: shape đúng, slice_prb phản ánh config tr
    (tr0 mMTC-heavy, tr27 URLLC-heavy), nhãn SLA hợp lệ ∈ [0,1] và biến
    thiên theo config (không phải hằng số)."""
    frames = list(iter_frames(_DATASET_ROOT, scheds=(0,), trs=(0, 13, 27),
                              exps=(1,), bss=("bs1",)))
    assert len(frames) == 3
    rates = []
    for fr in frames:
        assert fr.series.shape[1:] == (3, len(RAW_METRICS))
        assert fr.n_steps > 100
        rate = label_frame(fr).emergency_rate
        assert 0.0 <= rate <= 1.0
        rates.append(rate)
    assert min(rates) < max(rates)  # nhãn phụ thuộc config, không phải hằng
    # slice_prb: tr0 phân nhiều RBG cho mMTC (slice 1); tr27 cho URLLC (2).
    prb0 = frames[0].slice_series(1)[:, RAW_METRICS.index("slice_prb")].mean()
    prb27 = frames[2].slice_series(2)[:, RAW_METRICS.index("slice_prb")].mean()
    assert prb0 > 20 and prb27 > 20
