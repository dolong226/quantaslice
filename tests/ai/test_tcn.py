"""Tests cho TCN đa nhiệm + calibration (§4 Tier A, §5.1–§5.2).

Kiểm tra:
1. MultiTaskTCN — shape output đúng, tính nhân quả (causal): dự đoán tại
   bước cuối KHÔNG đổi khi thay đổi dữ liệu ở TƯƠNG LAI.
2. temperature_scale — không đổi thứ hạng (ROC bất biến), giảm/không tăng
   ECE trên tập lệch hiệu chuẩn.
3. run_training_tcn (synthetic) — trả TCNDetector + metrics; hiệu chỉnh
   nhiệt độ giảm ECE trên val; roundtrip save/load giữ nguyên dự đoán.
4. TCNDetector.predict_windows — cùng giao diện baseline; cửa sổ emergency
   có xác suất > cửa sổ bình thường.
5. MLPredictionProvider nạp được artifact TCN (không chỉ baseline).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantaslice.ai.data.features import WindowConfig
from quantaslice.ai.data.labeling import label_frame
from quantaslice.ai.data.loaders import generate_synthetic
from quantaslice.ai.eval.metrics import expected_calibration_error
from quantaslice.ai.models.tcn import MultiTaskTCN, TCNDetector
from quantaslice.ai.provider import MLPredictionProvider, coloran_feature_window
from quantaslice.ai.train.calibrate import temperature_scale
from quantaslice.ai.train.deep import run_training_tcn
from quantaslice.core.types import Prediction


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_tcn_shapes_and_causality():
    torch.manual_seed(0)
    model = MultiTaskTCN(n_features=6, channels=(8, 8), kernel=3).eval()
    x = torch.randn(4, 6, 30)  # (N, F, W)
    with torch.no_grad():
        flag, prio = model(x)
    assert flag.shape == (4,)
    assert prio.shape == (4, 3)
    assert (prio >= 0).all()  # softplus

    # Nhân quả (causal): sửa input tại bước CUỐI chỉ được đổi output tại
    # bước cuối của encoder — mọi vị trí trước đó bất biến (nhờ _Chomp1d).
    with torch.no_grad():
        x2 = x.clone()
        x2[:, :, -1] += 100.0
        h1 = model.encoder(x)
        h2 = model.encoder(x2)
    assert torch.allclose(h1[:, :, :-1], h2[:, :, :-1], atol=1e-5)
    assert not torch.allclose(h1[:, :, -1], h2[:, :, -1])


def test_temperature_scaling_reduces_ece_preserves_ranking():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(float)
    # Logit quá tự tin (overconfident) -> ECE cao, cần T > 1 để làm mềm.
    base = rng.normal(0, 1, 400)
    logits = np.where(y == 1, 4.0 + base, -4.0 + base) * 2.0
    ece_before = expected_calibration_error(y, _sigmoid(logits))
    t = temperature_scale(logits, y)
    ece_after = expected_calibration_error(y, _sigmoid(logits / t))
    assert ece_after <= ece_before + 1e-6
    # Thứ hạng bất biến: chia hằng số dương không đổi thứ tự.
    assert np.array_equal(np.argsort(logits), np.argsort(logits / t))


def test_run_training_tcn_calibrates_and_roundtrips(tmp_path):
    frames = generate_synthetic(n_frames=24, n_steps=400, seed=5)
    result = run_training_tcn(
        frames, test_scheds=(2,), wcfg=WindowConfig(lookback=20, stride=10),
        channels=(16, 16), epochs=12,
    )
    assert result.n_train > 0 and result.n_test > 0
    # Hiệu chỉnh không được làm ECE val tệ hơn (thường giảm rõ rệt).
    assert result.ece_after <= result.ece_before + 1e-3
    assert result.temperature > 0

    path = str(tmp_path / "tcn.joblib")
    result.detector.save(path)
    loaded = TCNDetector.load(path)
    labels = [label_frame(fr) for fr in frames]
    from quantaslice.ai.data.features import build_sequences
    x, _, _, _ = build_sequences(frames, labels, WindowConfig(lookback=20, stride=10))
    p1, _ = result.detector.predict_windows(x[:16])
    p2, _ = loaded.predict_windows(x[:16])
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-6)


def test_tcn_provider_predicts_and_separates(tmp_path):
    frames = generate_synthetic(n_frames=24, n_steps=400, seed=6)
    result = run_training_tcn(
        frames, test_scheds=(2,), wcfg=WindowConfig(lookback=20, stride=10),
        channels=(16, 16), epochs=15,
    )
    path = str(tmp_path / "tcn.joblib")
    result.detector.save(path)

    provider = MLPredictionProvider(path)  # phải nạp được artifact TCN
    emerg_frame = frames[0]
    calm_frame = frames[-1]
    pos = np.where(label_frame(emerg_frame).flag)[0]
    emerg_idx = int(pos[len(pos) // 2])   # bước được gắn nhãn chắc chắn
    p_emerg = provider.predict(coloran_feature_window(emerg_frame, emerg_idx, 20))
    p_calm = provider.predict(coloran_feature_window(calm_frame, 30, 20))
    assert isinstance(p_emerg, Prediction)
    assert 0.0 <= p_emerg.emergency_prob <= 1.0
    assert p_emerg.emergency_prob > p_calm.emergency_prob
