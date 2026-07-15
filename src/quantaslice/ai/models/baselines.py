"""Baseline gradient boosting — §4 Tier A của plan ML.

Baseline BẮT BUỘC: nhiều paper RAN không vượt được gradient boosting trên
feature cửa sổ đã tabular-hoá. Model deep (TCN...) phải chứng minh thắng
baseline này thì mới đáng dùng (§4 "Khuyến nghị", §10 rủi ro).

Đa nhiệm (§0): một head phân loại (emergency flag) + một head hồi quy
vector ưu tiên 3 chiều, dùng chung feature đầu vào. Ở baseline, "dùng
chung" nghĩa là cùng ma trận feature; deep model sẽ chia sẻ encoder.

Backend: dùng LightGBM nếu cài được, nếu không tự lùi về sklearn
``HistGradientBoosting*`` (cùng họ thuật toán, sẵn có, không cần biên
dịch) — nhờ vậy foundation chạy được ngay cả khi chưa cài lightgbm.

Artifact tự chứa (scaler + models + threshold) để ``MLPredictionProvider``
nạp lại và chạy runtime mà không cần biết chi tiết huấn luyện.
"""

from __future__ import annotations

import logging

import numpy as np

from quantaslice.core.exceptions import SolverError

__all__ = ["GradientBoostingDetector"]

logger = logging.getLogger(__name__)


def _make_backend() -> tuple[str, type, type]:
    """Chọn (tên, lớp classifier, lớp regressor) theo lib khả dụng."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
        return "lightgbm", LGBMClassifier, LGBMRegressor
    except ImportError:
        from sklearn.ensemble import (  # type: ignore
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )
        return "sklearn_histgb", HistGradientBoostingClassifier, HistGradientBoostingRegressor


class GradientBoostingDetector:
    """Detector baseline: emergency flag + priority weights.

    Vòng đời: ``fit`` -> ``set_threshold`` (từ calibrate) -> ``save``;
    runtime: ``load`` -> ``predict_proba`` / ``predict_priority``.
    """

    def __init__(self, *, random_state: int = 0) -> None:
        self._random_state = random_state
        self.backend_name, self._clf_cls, self._reg_cls = _make_backend()
        self.scaler = None            # sklearn RobustScaler, fit lúc train
        self.clf = None               # classifier (flag)
        self.reg = None               # regressor đa đầu ra (priority 3 chiều)
        self.threshold: float = 0.5
        self.feature_names: tuple[str, ...] = ()

    # ── Huấn luyện ────────────────────────────────────────────────────
    def fit(
        self,
        x: np.ndarray,
        y_flag: np.ndarray,
        y_priority: np.ndarray,
        *,
        feature_names: tuple[str, ...] = (),
    ) -> "GradientBoostingDetector":
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.preprocessing import RobustScaler

        if x.shape[0] == 0:
            raise SolverError("GradientBoostingDetector.fit: không có mẫu train nào")

        self.feature_names = tuple(feature_names)
        # §3.4: fit scaler CHỈ trên train; robust scaling vì KPM lệch/outlier.
        self.scaler = RobustScaler()
        xs = self.scaler.fit_transform(x)

        # §3.6: mất cân bằng lớp -> sample_weight nghịch đảo tần suất
        # (tránh oversampling sao chép gây rò rỉ/overfit).
        self.clf = self._clf_cls(random_state=self._random_state)
        self.clf.fit(xs, y_flag.astype(int), sample_weight=_balanced_weights(y_flag))

        self.reg = MultiOutputRegressor(self._reg_cls(random_state=self._random_state))
        self.reg.fit(xs, y_priority)
        return self

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)

    # ── Suy luận ──────────────────────────────────────────────────────
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Xác suất emergency ∈ [0,1] cho mỗi mẫu."""
        xs = self.scaler.transform(x)
        proba = self.clf.predict_proba(xs)
        # Cột xác suất lớp dương (1). Nếu chỉ thấy 1 lớp lúc train -> 0.
        classes = list(self.clf.classes_)
        if 1 not in classes:
            return np.zeros(x.shape[0])
        return proba[:, classes.index(1)]

    def predict_priority(self, x: np.ndarray) -> np.ndarray:
        """Vector ưu tiên ``(N, 3)`` (eMBB, mMTC, URLLC), clip ≥ 0."""
        xs = self.scaler.transform(x)
        return np.clip(self.reg.predict(xs), 0.0, None)

    def predict_windows(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Giao diện thống nhất với TCNDetector: nhận cửa sổ THÔ
        ``(N, W, F_ts)``, tự tóm tắt thành tabular rồi dự đoán -> (probs
        ``(N,)``, priorities ``(N, 3)``). Nhờ vậy ``MLPredictionProvider``
        gọi cùng một hàm cho mọi loại detector."""
        from quantaslice.ai.data.features import summarize_window

        if seq.shape[0] == 0:
            return np.empty(0), np.empty((0, 3))
        x_tab = np.stack([summarize_window(win) for win in seq])
        return self.predict_proba(x_tab), self.predict_priority(x_tab)

    def predict_flag(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x) >= self.threshold

    # ── Lưu / nạp ─────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "GradientBoostingDetector":
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, GradientBoostingDetector):
            raise SolverError(f"Artifact tại {path} không phải GradientBoostingDetector")
        return obj


def _balanced_weights(y_flag: np.ndarray) -> np.ndarray:
    """Trọng số mẫu nghịch đảo tần suất lớp (cân bằng dương/âm)."""
    y = y_flag.astype(int)
    n = y.size
    w = np.ones(n, dtype=float)
    for cls in (0, 1):
        m = y == cls
        cnt = int(m.sum())
        if cnt > 0:
            w[m] = n / (2.0 * cnt)
    return w
