"""ARIMA and AutoRegressive (AR) forecasting models.

These statistical baselines provide comparisons for the deep learning models.
`GlobalARDetector` fits a global linear model on delay histories.
`LocalARIMADetector` fits an ARIMA model locally on-the-fly for each window.
"""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression

__all__ = ["GlobalARDetector", "LocalARIMADetector"]


class GlobalARDetector:
    """Global Autoregressive (AR) classifier trained on historical delay sequences."""

    backend_name = "global_ar"

    def __init__(self, lookback: int, horizon: int = 8, budget: float = 0.1) -> None:
        self.window_lookback = lookback
        self.horizon = horizon
        self.budget = budget
        self.clf = LogisticRegression(class_weight="balanced", random_state=0)

    def _extract_delay(self, seq: np.ndarray) -> np.ndarray:
        # Reconstruct delay from the first two features: log1p(buf) and br
        buf = np.expm1(seq[:, :, 0])
        br = seq[:, :, 1]
        return (buf * 8) / np.maximum(br * 1e6, 1e4)

    def fit(self, seq: np.ndarray, y: np.ndarray) -> GlobalARDetector:
        if seq.shape[0] == 0:
            return self
        delays = self._extract_delay(seq)  # (N, W)
        self.clf.fit(delays, y)
        return self

    def predict_windows(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if seq.shape[0] == 0:
            return np.empty(0), np.empty((0, 3))
        delays = self._extract_delay(seq)
        probs = self.clf.predict_proba(delays)[:, 1]
        prio = np.zeros((seq.shape[0], 3))
        return probs, prio


class LocalARIMADetector:
    """Local ARIMA model fitted on-the-fly for each time-series window."""

    backend_name = "local_arima"

    def __init__(self, lookback: int, horizon: int = 8, budget: float = 0.1) -> None:
        self.window_lookback = lookback
        self.horizon = horizon
        self.budget = budget

    def _extract_delay(self, seq: np.ndarray) -> np.ndarray:
        buf = np.expm1(seq[:, :, 0])
        br = seq[:, :, 1]
        return (buf * 8) / np.maximum(br * 1e6, 1e4)

    def predict_windows(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from statsmodels.tsa.arima.model import ARIMA

        if seq.shape[0] == 0:
            return np.empty(0), np.empty((0, 3))

        delays = self._extract_delay(seq)  # (N, W)
        probs = []
        for hist in delays:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(hist, order=(1, 1, 0))
                    res = model.fit()
                    forecast = res.forecast(steps=self.horizon)
                    probs.append(1.0 if (forecast >= self.budget).any() else 0.0)
            except Exception:
                probs.append(1.0 if hist[-1] >= self.budget else 0.0)

        probs = np.array(probs)
        prio = np.zeros((seq.shape[0], 3))
        return probs, prio
