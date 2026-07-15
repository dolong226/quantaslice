"""Tests cho training + eval + runtime provider của ML detector.

Kiểm tra:
1. run_training (synthetic) — trả detector + metrics hợp lệ, save/load
   roundtrip giữ nguyên dự đoán.
2. metrics — cost_sensitive_threshold ưu tiên recall (miss ≫ false alarm);
   ECE trong [0,1].
3. MLPredictionProvider — implement Protocol PredictionProvider, phát
   Prediction hợp lệ; cửa sổ emergency có xác suất > cửa sổ bình thường.
4. Provider trả baseline (không raise) khi FeatureWindow không khớp không
   gian feature ML (tuân thủ Protocol).
5. DependencyContainer wiring — build Runner với provider 'ml' + chạy
   run_once; provider bootstrap 'threshold' vẫn hoạt động.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from quantaslice.ai.data.features import WindowConfig, build_tabular, summary_feature_names
from quantaslice.ai.data.labeling import label_frame
from quantaslice.ai.data.loaders import generate_synthetic
from quantaslice.ai.eval.metrics import cost_sensitive_threshold, expected_calibration_error
from quantaslice.ai.models.baselines import GradientBoostingDetector
from quantaslice.ai.provider import MLPredictionProvider, coloran_feature_window
from quantaslice.ai.train.baseline import run_training
from quantaslice.core.protocols import PredictionProvider
from quantaslice.core.types import FeatureWindow, Prediction


def _fit_detector_on_all(frames, wcfg):
    """Fit detector trên TOÀN BỘ synthetic (không split) — kiểm soát được
    để test provider học được tín hiệu tiêm."""
    labels = [label_frame(fr) for fr in frames]
    x, yf, yp, _ = build_tabular(frames, labels, wcfg)
    det = GradientBoostingDetector(random_state=0)
    det.fit(x, yf, yp, feature_names=tuple(summary_feature_names()))
    det.set_threshold(cost_sensitive_threshold(yf, det.predict_proba(x)))
    det.window_lookback = wcfg.lookback
    from quantaslice.ai.data.features import FEATURE_TS_NAMES
    det.ts_feature_names = FEATURE_TS_NAMES
    return det


def test_run_training_and_roundtrip(tmp_path):
    frames = generate_synthetic(n_frames=18, n_steps=400, seed=5)
    result = run_training(frames, test_scheds=(2,), wcfg=WindowConfig(lookback=20, stride=8))
    assert result.n_train > 0 and result.n_test > 0
    assert 0.0 <= result.metrics.roc_auc <= 1.0 or np.isnan(result.metrics.roc_auc)

    path = str(tmp_path / "det.joblib")
    result.detector.save(path)
    loaded = GradientBoostingDetector.load(path)
    # Roundtrip: cùng input -> cùng xác suất.
    labels = [label_frame(fr) for fr in frames]
    x, _, _, _ = build_tabular(frames, labels, WindowConfig(lookback=20, stride=8))
    np.testing.assert_allclose(
        result.detector.predict_proba(x[:20]), loaded.predict_proba(x[:20])
    )


def test_cost_sensitive_threshold_favors_recall():
    y = np.array([0, 0, 0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.55, 0.5, 0.6])
    # cost_miss lớn -> ngưỡng đủ thấp để bắt cả 2 dương (recall = 1).
    thr = cost_sensitive_threshold(y, prob, cost_miss=100.0, cost_fa=1.0)
    assert (prob >= thr)[y == 1].all()


def test_ece_range():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    prob = np.linspace(0.1, 0.9, 8)
    ece = expected_calibration_error(y, prob)
    assert 0.0 <= ece <= 1.0


def test_provider_is_protocol_and_predicts(tmp_path):
    frames = generate_synthetic(n_frames=12, n_steps=400, seed=6)
    wcfg = WindowConfig(lookback=20, stride=8)
    det = _fit_detector_on_all(frames, wcfg)
    path = str(tmp_path / "det.joblib")
    det.save(path)

    provider = MLPredictionProvider(path)
    assert isinstance(provider, PredictionProvider)  # structural (Protocol)

    # Xác định mốc emergency THẬT từ nhãn (đỉnh severity của frame tiêm
    # burst) và mốc bình thường (severity thấp ở frame không tiêm), thay
    # vì đoán index — burst nằm ngẫu nhiên trong [T/4, T/2].
    emerg_frame = frames[0]
    calm_frame = frames[-1]
    pos = np.where(label_frame(emerg_frame).flag)[0]
    emerg_idx = int(pos[len(pos) // 2])   # bước được gắn nhãn chắc chắn
    calm_idx = 30

    p_emerg = provider.predict(coloran_feature_window(emerg_frame, emerg_idx, 20))
    p_calm = provider.predict(coloran_feature_window(calm_frame, calm_idx, 20))
    assert isinstance(p_emerg, Prediction)
    assert 0.0 <= p_emerg.emergency_prob <= 1.0
    assert p_emerg.priority.urllc >= 0.0
    assert p_emerg.emergency_prob > p_calm.emergency_prob


def test_provider_baseline_on_mismatched_window(tmp_path):
    frames = generate_synthetic(n_frames=8, n_steps=300, seed=7)
    det = _fit_detector_on_all(frames, WindowConfig(lookback=20, stride=8))
    path = str(tmp_path / "det.joblib")
    det.save(path)
    provider = MLPredictionProvider(path)

    # FeatureWindow với feature_names KHÔNG khớp không gian ML.
    win = FeatureWindow(
        gnb_id="bs-x", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        features=np.zeros((10, 2)), feature_names=("foo", "bar"),
    )
    pred = provider.predict(win)  # KHÔNG được raise
    assert pred.emergency_flag is False
    assert pred.emergency_prob == 0.0
    assert pred.priority.as_tuple() == (1.0, 1.0, 1.0)


def test_container_wires_ml_provider(tmp_path):
    frames = generate_synthetic(n_frames=12, n_steps=400, seed=8)
    wcfg = WindowConfig(lookback=20, stride=8)
    det = _fit_detector_on_all(frames, wcfg)
    path = str(tmp_path / "det.joblib")
    det.save(path)

    from quantaslice.core.runtime import Configuration
    from quantaslice.pipeline import DependencyContainer
    from quantaslice.quantum import solver_registry  # noqa: F401 (đăng ký solver)
    from quantaslice.orchestrator.coloran_loader import ColORANLoader

    config = Configuration(
        prediction_provider="ml", solver="classical_greedy", orchestrator="mock_oran",
        extra={"ml_artifact": path},
    )
    slices = ColORANLoader.create_slices()
    stations = ColORANLoader.create_stations(n_stations=2)
    runner = DependencyContainer.build_runner(config, slices=slices, stations=stations)

    win = coloran_feature_window(frames[0], end_idx=250, lookback=20, gnb_id="bs-1")
    frame = runner.run_once(win)
    assert frame.result is not None  # đã tối ưu lần đầu
    assert frame.predictions and frame.predictions[0].gnb_id == "bs-1"


def test_container_threshold_provider_still_works():
    """Provider bootstrap 'threshold' không thuộc ai registry vẫn build được."""
    from quantaslice.core.runtime import Configuration
    from quantaslice.pipeline import DependencyContainer
    from quantaslice.orchestrator.coloran_loader import ColORANLoader

    config = Configuration(prediction_provider="threshold", solver="classical_greedy")
    runner = DependencyContainer.build_runner(
        config, slices=ColORANLoader.create_slices(),
        stations=ColORANLoader.create_stations(n_stations=2),
    )
    assert runner is not None
