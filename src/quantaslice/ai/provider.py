"""``MLPredictionProvider`` — cầu nối runtime giữa detector đã huấn luyện
và pipeline. Đây là hiện thực THẬT của Protocol
:class:`~quantaslice.core.protocols.PredictionProvider`, thay cho các
bootstrap provider tạm (mock/threshold) khi model ML đã sẵn sàng.

Hợp đồng (§0): nhận ``FeatureWindow`` -> phát ``Prediction`` gồm
emergency flag ``ê`` và priority vector ``p``. ``FeatureWindow.features``
phải nằm trong KHÔNG GIAN FEATURE per-timestep mà detector được train
(khớp ``FEATURE_TS_NAMES``); một ``TelemetrySource`` cho ColO-RAN sẽ dựng
cửa sổ như vậy (helper :func:`coloran_feature_window`).

Tuân thủ Protocol: KHÔNG raise với input hợp lệ — nếu cửa sổ không khớp
không gian feature (vd nguồn telemetry generic), trả ``Prediction``
baseline (không khẩn cấp, p=(1,1,1)) thay vì làm sập pipeline.

Package ``ai`` CHỈ import từ ``core`` + nội bộ ``ai`` — KHÔNG import
quantum/orchestrator/pipeline.
"""

from __future__ import annotations

import logging

import numpy as np

from quantaslice.core.exceptions import ConfigurationError
from quantaslice.core.types import FeatureWindow, Prediction, PriorityVector

__all__ = ["MLPredictionProvider", "coloran_feature_window"]

logger = logging.getLogger(__name__)

_BASELINE_PRIORITY = PriorityVector(embb=1.0, urllc=1.0, mmtc=1.0)


class MLPredictionProvider:
    """Provider suy luận từ ``GradientBoostingDetector`` (hoặc detector
    tương thích) đã lưu ở artifact."""

    def __init__(self, artifact_path: str) -> None:
        from quantaslice.ai.models import load_detector

        # Nạp được cả baseline (GradientBoosting) lẫn TCN — cả hai phơi bày
        # cùng giao diện predict_windows(seq).
        self._detector = load_detector(artifact_path)
        self._ts_names: tuple[str, ...] = tuple(
            getattr(self._detector, "ts_feature_names", ())
        )

    def predict(self, window: FeatureWindow) -> Prediction:
        seq = self._extract_window(window)
        if seq is None:
            return self._baseline(window)

        probs, prios = self._detector.predict_windows(seq[None])
        prob = float(np.clip(probs[0], 0.0, 1.0))
        prio = prios[0]  # (3,) eMBB, mMTC, URLLC
        flag = prob >= self._detector.threshold

        # slice-index (eMBB, mMTC, URLLC) -> PriorityVector (embb, urllc, mmtc).
        priority = PriorityVector(
            embb=float(prio[0]), mmtc=float(prio[1]), urllc=float(prio[2])
        )
        return Prediction(
            gnb_id=window.gnb_id, timestamp=window.timestamp,
            emergency_flag=bool(flag), emergency_prob=prob, priority=priority,
        )

    # ── Nội bộ ────────────────────────────────────────────────────────
    def _extract_window(self, window: FeatureWindow) -> np.ndarray | None:
        """Lấy cửa sổ THÔ ``(W, F_ts)`` khớp không gian feature đã train
        (sắp lại cột nếu khác thứ tự). None nếu không khớp -> baseline."""
        if not self._ts_names:
            return None
        names = tuple(window.feature_names)
        if names == self._ts_names:
            return window.features
        if set(self._ts_names).issubset(names):
            idx = [names.index(n) for n in self._ts_names]
            return window.features[:, idx]
        logger.debug("FeatureWindow không khớp không gian feature ML -> baseline")
        return None

    def _baseline(self, window: FeatureWindow) -> Prediction:
        return Prediction(
            gnb_id=window.gnb_id, timestamp=window.timestamp,
            emergency_flag=False, emergency_prob=0.0, priority=_BASELINE_PRIORITY,
        )


def coloran_feature_window(
    frame, end_idx: int, lookback: int, *, gnb_id: str | None = None
) -> FeatureWindow:
    """Dựng ``FeatureWindow`` từ một ``BSFrame`` tại cửa sổ kết thúc ở
    ``end_idx`` — tiện cho test và cho ``TelemetrySource`` ColO-RAN tương
    lai (đưa dữ liệu về đúng không gian feature detector mong đợi)."""
    from quantaslice.ai.data.features import FEATURE_TS_NAMES, frame_features

    feats = frame_features(frame)
    start = max(0, end_idx - lookback + 1)
    win = feats[start:end_idx + 1]
    if win.shape[0] == 0:
        raise ConfigurationError("coloran_feature_window: cửa sổ rỗng")
    from datetime import datetime, timezone
    ts = datetime.fromtimestamp(float(frame.time[end_idx]), tz=timezone.utc)
    return FeatureWindow(
        gnb_id=gnb_id or frame.bs, timestamp=ts,
        features=win, feature_names=FEATURE_TS_NAMES,
    )
