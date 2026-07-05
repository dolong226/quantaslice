"""``StreamSimulator`` — implement
:class:`~quantaslice.core.protocols.TelemetrySource`: nội suy chuỗi CDR
gốc (10 phút/mốc) thành luồng tần số cao (mặc định 100ms/bước, đúng mục
3.1 tài liệu Emergency Detector: "Nội suy tuyến tính... Resample về cửa
sổ 100ms khi phát sinh dữ liệu stream phục vụ demo trực tiếp"), cắt cửa
sổ trượt độ dài T (mục 3.3), và phát ra
:class:`~quantaslice.core.types.FeatureWindow` — kiểu dữ liệu chuẩn ở
``core`` mà ``PredictionProvider`` tiêu thụ.

CHÚ Ý PHẠM VI: StreamSimulator chỉ phát 5 đặc trưng THÔ (smsin, smsout,
callin, callout, internet) đúng schema dataset gốc — KHÔNG tự làm
feature engineering (delta, z-score cục bộ, trung bình lân cận không
gian...). Việc đó thuộc trách nhiệm package ``ai``
(``ai/preprocessing/feature_engineering.py`` theo cây thư mục kiến
trúc) — giữ ranh giới rõ giữa "sinh telemetry thô" và "biến đổi đặc
trưng cho mô hình".
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np

from quantaslice.core.types import FeatureWindow
from quantaslice.simulation.dataset_loader import CDRRecord

__all__ = ["StreamSimulator"]

_RAW_FEATURE_NAMES = ("smsin", "smsout", "callin", "callout", "internet")
_DEFAULT_MAX_STEPS = 2000  # chặn mặc định để demo không chạy vô hạn (7 ngày @ 100ms là quá lớn)

# xs (giây kể từ mốc gốc chung), ys (giá trị) cho 1 đặc trưng của 1 gNB
_Interpolant = tuple[np.ndarray, np.ndarray]


class StreamSimulator:
    """Nội suy + resample + cắt cửa sổ trượt từ danh sách ``CDRRecord``.

    Dùng trực tiếp với ``Runner.run_forever(stream_simulator)`` (xem
    ``quantaslice.pipeline.Runner``).
    """

    def __init__(
        self,
        records: list[CDRRecord],
        *,
        window_length: int = 50,
        output_interval: timedelta = timedelta(milliseconds=100),
        noise_std_ratio: float = 0.05,
        max_steps: int | None = _DEFAULT_MAX_STEPS,
        seed: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("StreamSimulator cần ít nhất 1 CDRRecord")

        self._window_length = window_length
        self._output_interval = output_interval
        self._noise_std_ratio = noise_std_ratio
        self._max_steps = max_steps
        self._rng = random.Random(seed)

        # Mốc thời gian gốc CHUNG cho mọi gNB — bắt buộc để `_sample_at`
        # và các interpolant per-cell dùng cùng 1 hệ quy chiếu thời gian,
        # nếu không các gNB có timestamp bắt đầu khác nhau sẽ bị lệch.
        all_timestamps = [r.timestamp for r in records]
        self._start = min(all_timestamps)
        self._end = max(all_timestamps)

        self._series_by_cell = self._build_interpolants(records, reference_start=self._start)

    # ------------------------------------------------------------------
    def stream(self) -> Iterator[FeatureWindow]:
        """Sinh FeatureWindow liên tục, xen kẽ theo từng gNB, tại từng
        mốc thời gian resample (mục 3.1 bước 3)."""
        buffers: dict[str, deque[np.ndarray]] = {
            cell_id: deque(maxlen=self._window_length) for cell_id in self._series_by_cell
        }

        total_span_seconds = (self._end - self._start).total_seconds()
        n_native_steps = max(1, int(total_span_seconds / self._output_interval.total_seconds()))
        n_steps = min(n_native_steps, self._max_steps) if self._max_steps else n_native_steps

        for step in range(n_steps):
            ts = self._start + step * self._output_interval
            for cell_id, interpolants in sorted(self._series_by_cell.items()):
                raw = self._sample_at(interpolants, ts)
                buffers[cell_id].append(raw)
                if len(buffers[cell_id]) < self._window_length:
                    continue  # chưa đủ lịch sử để tạo cửa sổ hoàn chỉnh (mục 3.3)
                yield FeatureWindow(
                    gnb_id=cell_id,
                    timestamp=ts,
                    features=np.array(buffers[cell_id]),
                    feature_names=_RAW_FEATURE_NAMES,
                )

    # ------------------------------------------------------------------
    def _sample_at(self, interpolants: dict[str, _Interpolant], ts: datetime) -> np.ndarray:
        """Nội suy tuyến tính 5 đặc trưng thô tại mốc ``ts``, cộng nhiễu
        Gauss tỉ lệ biên độ cục bộ (mục 3.1 bước 2: "Cộng nhiễu Gauss tỉ
        lệ với biên độ cục bộ")."""
        t = (ts - self._start).total_seconds()
        values = np.empty(len(_RAW_FEATURE_NAMES))
        for idx, name in enumerate(_RAW_FEATURE_NAMES):
            xs, ys = interpolants[name]
            value = float(np.interp(t, xs, ys))
            noise = self._rng.gauss(0, self._noise_std_ratio * max(value, 1e-6))
            values[idx] = max(0.0, value + noise)
        return values

    @staticmethod
    def _build_interpolants(
        records: list[CDRRecord], *, reference_start: datetime
    ) -> dict[str, dict[str, _Interpolant]]:
        """Nhóm bản ghi theo gNB, dựng mảng (thời gian tính bằng giây kể
        từ ``reference_start``, giá trị) cho từng đặc trưng — sẵn sàng
        cho ``np.interp`` (nội suy tuyến tính, mục 3.1 bước 1)."""
        by_cell: dict[str, list[CDRRecord]] = defaultdict(list)
        for r in records:
            by_cell[r.cell_id].append(r)

        result: dict[str, dict[str, _Interpolant]] = {}
        for cell_id, cell_records in by_cell.items():
            cell_records.sort(key=lambda r: r.timestamp)
            xs = np.array([(r.timestamp - reference_start).total_seconds() for r in cell_records])
            result[cell_id] = {
                "smsin": (xs, np.array([r.smsin for r in cell_records])),
                "smsout": (xs, np.array([r.smsout for r in cell_records])),
                "callin": (xs, np.array([r.callin for r in cell_records])),
                "callout": (xs, np.array([r.callout for r in cell_records])),
                "internet": (xs, np.array([r.internet for r in cell_records])),
            }
        return result
