"""Tests cho PatchTST và iTransformer đa nhiệm."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.preprocessing import RobustScaler
from quantaslice.ai.models.patchtst import MultiTaskPatchTST, PatchTSTDetector
from quantaslice.ai.models.itransformer import MultiTaskiTransformer, iTransformerDetector


def test_patchtst_shapes():
    torch.manual_seed(0)
    # n_features=6, seq_len=20, patch_len=8, stride=4
    model = MultiTaskPatchTST(
        n_features=6, seq_len=20, patch_len=8, stride=4, hidden_dim=16, num_heads=2, num_layers=1
    ).eval()
    x = torch.randn(4, 20, 6)  # (B, W, F)
    with torch.no_grad():
        flag, prio = model(x)
    assert flag.shape == (4,)
    assert prio.shape == (4, 3)
    assert (prio >= 0).all()  # softplus


def test_itransformer_shapes():
    torch.manual_seed(0)
    # n_features=6, seq_len=20
    model = MultiTaskiTransformer(
        n_features=6, seq_len=20, hidden_dim=16, num_heads=2, num_layers=1
    ).eval()
    x = torch.randn(4, 20, 6)  # (B, W, F)
    with torch.no_grad():
        flag, prio = model(x)
    assert flag.shape == (4,)
    assert prio.shape == (4, 3)
    assert (prio >= 0).all()  # softplus


def test_advanced_transformers_predict_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)

    # PatchTST
    model_patch = MultiTaskPatchTST(n_features=4, seq_len=20, patch_len=8, stride=4, hidden_dim=16, num_heads=2, num_layers=1)
    det_patch = PatchTSTDetector(model_patch, lookback=20)
    x = np.random.randn(8, 20, 4)
    det_patch.scaler = RobustScaler().fit(x.reshape(-1, 4))
    
    p1_patch, prio1_patch = det_patch.predict_windows(x)
    path_patch = str(tmp_path / "patchtst.joblib")
    det_patch.save(path_patch)
    loaded_patch = PatchTSTDetector.load(path_patch)
    p2_patch, prio2_patch = loaded_patch.predict_windows(x)
    np.testing.assert_allclose(p1_patch, p2_patch, rtol=1e-5, atol=1e-6)

    # iTransformer
    model_itrans = MultiTaskiTransformer(n_features=4, seq_len=20, hidden_dim=16, num_heads=2, num_layers=1)
    det_itrans = iTransformerDetector(model_itrans, lookback=20)
    det_itrans.scaler = RobustScaler().fit(x.reshape(-1, 4))
    
    p1_itrans, prio1_itrans = det_itrans.predict_windows(x)
    path_itrans = str(tmp_path / "itransformer.joblib")
    det_itrans.save(path_itrans)
    loaded_itrans = iTransformerDetector.load(path_itrans)
    p2_itrans, prio2_itrans = loaded_itrans.predict_windows(x)
    np.testing.assert_allclose(p1_itrans, p2_itrans, rtol=1e-5, atol=1e-6)
