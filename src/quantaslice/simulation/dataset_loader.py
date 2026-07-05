"""Đọc / sinh dữ liệu theo đúng schema "Italian Telecom Data 2013 (1
week)" — tài liệu Emergency Detector mục 2.1: mỗi ô lưới (CellID), mỗi
mốc 10 phút, ghi nhận ``smsin, smsout, callin, callout, internet``.

Chưa có dataset thật trong tay lúc code module này, nên
:meth:`ItalianTelecomDatasetLoader.generate_synthetic` sinh dữ liệu giả
lập CÙNG SCHEMA (chu kỳ ngày/đêm + nhiễu + burst khẩn cấp tuỳ chọn) để
:mod:`quantaslice.simulation.stream_simulator` chạy được ngay hôm nay.
:meth:`load_csv` đọc file thật cùng schema khi có sẵn — downstream
(``StreamSimulator``) chỉ thấy ``list[CDRRecord]`` giống hệt nhau, không
cần đổi gì khi thay dữ liệu giả lập bằng dữ liệu thật.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

__all__ = ["CDRRecord", "ItalianTelecomDatasetLoader"]

_NATIVE_INTERVAL = timedelta(minutes=10)  # độ phân giải gốc của dataset thật


@dataclass(frozen=True, slots=True)
class CDRRecord:
    """Một bản ghi CDR tổng hợp — đúng 7 cột mô tả ở mục 2.1 tài liệu
    Emergency Detector.

    Đây là kiểu dữ liệu NỘI BỘ của package ``simulation`` (không thuộc
    ``core``): nó không bao giờ đi qua ranh giới sang ``ai``/``quantum``
    — chỉ ``StreamSimulator`` tiêu thụ để dựng
    :class:`~quantaslice.core.types.FeatureWindow` (kiểu dữ liệu chuẩn
    dùng chung, xem ``core.types``).
    """

    cell_id: str
    timestamp: datetime
    smsin: float
    smsout: float
    callin: float
    callout: float
    internet: float


class ItalianTelecomDatasetLoader:
    """Đọc CSV thật hoặc sinh dữ liệu giả lập cùng schema."""

    @staticmethod
    def load_csv(path: str) -> list[CDRRecord]:
        """Đọc file CSV header ``CellID,datetime,smsin,smsout,callin,
        callout,internet`` (mục 2.1). ``datetime`` phải ở định dạng ISO
        8601 (``YYYY-MM-DD HH:MM:SS``)."""
        records: list[CDRRecord] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    CDRRecord(
                        cell_id=row["CellID"],
                        timestamp=datetime.fromisoformat(row["datetime"]),
                        smsin=float(row["smsin"]),
                        smsout=float(row["smsout"]),
                        callin=float(row["callin"]),
                        callout=float(row["callout"]),
                        internet=float(row["internet"]),
                    )
                )
        return records

    @staticmethod
    def generate_synthetic(
        cell_ids: tuple[str, ...],
        *,
        start: datetime,
        n_intervals: int = 6 * 24,  # mặc định 1 ngày ở độ phân giải 10 phút
        interval: timedelta = _NATIVE_INTERVAL,
        emergency_cell_id: str | None = None,
        emergency_start_interval: int | None = None,
        emergency_duration_intervals: int = 3,
        emergency_multiplier: float = 2.5,
        seed: int | None = None,
    ) -> list[CDRRecord]:
        """Sinh dữ liệu giả lập cùng schema thật, mô phỏng:

        - Chu kỳ ngày/đêm (sinusoid) cho cả 5 chỉ số (mục 2.2: "tính chu
          kỳ ngày/đêm"), biên độ baseline khác nhau theo từng ô lưới.
        - Nhiễu Gauss tỉ lệ với baseline của từng ô lưới.
        - (Tuỳ chọn) một burst khẩn cấp kiểu "tụ tập đông người" (mục
          4.2 kịch bản B): nhân ``emergency_multiplier`` lên
          callin/callout/internet tại ``emergency_cell_id`` trong
          ``emergency_duration_intervals`` mốc liên tiếp, bắt đầu từ
          ``emergency_start_interval`` — hữu ích để demo luồng
          emergency-triggered re-optimization end-to-end.
        """
        rng = random.Random(seed)
        records: list[CDRRecord] = []

        baseline = {
            cell_id: {
                "smsin": rng.uniform(5, 20),
                "smsout": rng.uniform(5, 20),
                "callin": rng.uniform(5, 15),
                "callout": rng.uniform(5, 15),
                "internet": rng.uniform(50, 200),
            }
            for cell_id in cell_ids
        }

        for step in range(n_intervals):
            ts = start + step * interval
            hour_of_day = (ts.hour + ts.minute / 60.0) / 24.0
            daily_factor = 0.6 + 0.4 * math.sin(2 * math.pi * (hour_of_day - 0.25))

            for cell_id in cell_ids:
                base = baseline[cell_id]
                is_emergency = (
                    cell_id == emergency_cell_id
                    and emergency_start_interval is not None
                    and emergency_start_interval
                    <= step
                    < emergency_start_interval + emergency_duration_intervals
                )
                burst = emergency_multiplier if is_emergency else 1.0

                def _sample(mean: float, noise_ratio: float = 0.15) -> float:
                    value = mean * daily_factor * burst
                    noise = rng.gauss(0, noise_ratio * max(value, 1e-6))
                    return max(0.0, value + noise)

                records.append(
                    CDRRecord(
                        cell_id=cell_id,
                        timestamp=ts,
                        smsin=_sample(base["smsin"]),
                        smsout=_sample(base["smsout"]),
                        callin=_sample(base["callin"], noise_ratio=0.2 if is_emergency else 0.15),
                        callout=_sample(base["callout"], noise_ratio=0.2 if is_emergency else 0.15),
                        internet=_sample(base["internet"]),
                    )
                )
        return records
