"""Tests cho TSMixer đa nhiệm (MultiTaskTSMixer + TSMixerDetector)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantaslice.ai.data.features import WindowConfig
from quantaslice.ai.data.labeling import label_frame
from quantaslice.ai.data.loaders import generate_synthetic
from quantaslice.ai.models.tsmixer import MultiTaskTSMixer, TSMixerDetector
from quantaslice.ai.train.deep import run_training_tcn


def test_tsmixer_shapes():
    torch.manual_seed(0)
    model = MultiTaskTSMixer(n_features=6, seq_len=20, hidden_dim=8, num_blocks=1).eval()
    x = torch.randn(4, 20, 6)  # (N, W, F)
    with torch.no_grad():
        flag, prio = model(x)
    assert flag.shape == (4,)
    assert prio.shape == (4, 3)
    assert (prio >= 0).all()  # softplus


def test_tsmixer_detector_predicts_and_roundtrips(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)

    model = MultiTaskTSMixer(n_features=24, seq_len=20, hidden_dim=16, num_blocks=1)
    detector = TSMixerDetector(model, lookback=20)

    # Reconstruct some features matching RobustScaler requirements
    x = np.random.randn(8, 20, 24)
    y_flag = np.array([1, 0, 1, 0, 1, 0, 1, 0])

    from sklearn.preprocessing import RobustScaler
    detector.scaler = RobustScaler().fit(x.reshape(-1, 24))

    # Predict before save
    p1, prio1 = detector.predict_windows(x)
    assert p1.shape == (8,)
    assert prio1.shape == (8, 3)

    # Save and load roundtrip
    path = str(tmp_path / "tsmixer.joblib")
    detector.save(path)
    loaded = TSMixerDetector.load(path)

    p2, prio2 = loaded.predict_windows(x)
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(prio1, prio2, rtol=1e-5, atol=1e-6)
