"""Tests cho GlobalARDetector và LocalARIMADetector."""

from __future__ import annotations

import numpy as np
import pytest

from quantaslice.ai.models.arima import GlobalARDetector, LocalARIMADetector


def test_arima_and_ar_shapes():
    # Sequence layout is (N, W, F)
    # The detectors reconstruct delay using expm1(seq[:, :, 0]) and seq[:, :, 1]
    # Reconstruct formula: (buf * 8) / np.maximum(br * 1e6, 1e4)
    # Let's generate features: log1p(buf) = seq[:, :, 0], br = seq[:, :, 1]
    np.random.seed(0)
    x = np.zeros((4, 20, 4))
    x[:, :, 0] = np.log1p(100.0)  # buf = 100 bytes
    x[:, :, 1] = 1.0  # br = 1.0 Mbps
    y = np.array([1.0, 0.0, 1.0, 0.0])

    # Global AR
    ar = GlobalARDetector(lookback=20, horizon=8, budget=0.1)
    ar.fit(x, y)
    probs, prio = ar.predict_windows(x)
    assert probs.shape == (4,)
    assert prio.shape == (4, 3)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Local ARIMA
    arima = LocalARIMADetector(lookback=20, horizon=8, budget=0.1)
    probs_arima, prio_arima = arima.predict_windows(x)
    assert probs_arima.shape == (4,)
    assert prio_arima.shape == (4, 3)
    assert ((probs_arima == 0.0) | (probs_arima == 1.0)).all()
