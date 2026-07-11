"""``ColORANLoader`` — đọc dataset ColO-RAN (Colosseum O-RAN) và chuyển
đổi sang các kiểu dữ liệu domain của QuantaSlice.

Dataset gốc: `github.com/wineslab/colosseum-oran-coloran-dataset`

- 7 Base Stations (nodes 1, 8, 15, 22, 29, 36, 43).
- 10 MHz bandwidth = 50 PRBs per BS.
- 3 slices: eMBB (slice 0), MTC (slice 1), URLLC (slice 2).
- 42 UEs (6 per BS).
- CSV columns: ``slice_id, slice_prb, dl_buffer [bytes],
  tx_brate downlink [Mbps], sum_requested_prbs, sum_granted_prbs,
  scheduling_policy, ...``

Module này CHỈ import từ ``quantaslice.core`` — KHÔNG import ngược lại
bất kỳ package con nào (ai, quantum, pipeline).
"""

from __future__ import annotations

import csv
import glob
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from quantaslice.core.types import BaseStation, SliceRequest, SliceType

__all__ = ["ColORANLoader", "ORANRecord"]

logger = logging.getLogger(__name__)

# ColO-RAN experiment parameters.
_COLORAN_PRB_CAPACITY = 50  # 10 MHz = 50 PRBs
_COLORAN_SLICE_MAP = {
    0: ("eMBB", SliceType.EMBB),
    1: ("mMTC", SliceType.MMTC),
    2: ("URLLC", SliceType.URLLC),
}
_COLORAN_BS_NODES = (1, 8, 15, 22, 29, 36, 43)


@dataclass(frozen=True, slots=True)
class ORANRecord:
    """Một bản ghi KPM từ ColO-RAN dataset — per-UE, per-timestep."""

    bs_id: str
    timestamp: datetime
    slice_id: int
    slice_prb: int
    dl_buffer: float
    tx_brate_dl: float
    sum_requested_prbs: float
    sum_granted_prbs: float
    ratio_granted_req: float
    scheduling_policy: int


class ColORANLoader:
    """Đọc ColO-RAN CSV files và chuyển đổi sang QuantaSlice domain types."""

    @staticmethod
    def create_stations(
        n_stations: int = 7,
        prb_capacity: float = _COLORAN_PRB_CAPACITY,
    ) -> tuple[BaseStation, ...]:
        """Tạo danh sách BaseStation theo cấu hình ColO-RAN.

        Parameters
        ----------
        n_stations:
            Số lượng base stations (mặc định 7 theo ColO-RAN).
        prb_capacity:
            PRB capacity per station (mặc định 50 theo 10 MHz).

        Returns
        -------
        tuple[BaseStation, ...]
        """
        return tuple(
            BaseStation(gnb_id=f"bs-{i + 1}", prb_capacity=prb_capacity)
            for i in range(n_stations)
        )

    @staticmethod
    def create_slices(
        prb_per_slice: dict[str, float] | None = None,
    ) -> tuple[SliceRequest, ...]:
        """Tạo 3 SliceRequest theo chuẩn ColO-RAN (eMBB, mMTC, URLLC).

        Parameters
        ----------
        prb_per_slice:
            Mapping ``{slice_id: prb_required}``. Nếu None, dùng phân bổ
            mặc định đều nhau (50/3 ≈ 17 PRBs mỗi slice).
        """
        default_prb = _COLORAN_PRB_CAPACITY / 3.0
        if prb_per_slice is None:
            prb_per_slice = {
                "s0-eMBB": default_prb,
                "s1-mMTC": default_prb,
                "s2-URLLC": default_prb,
            }
        return (
            SliceRequest(
                slice_id="s0-eMBB",
                slice_type=SliceType.EMBB,
                prb_required=prb_per_slice.get("s0-eMBB", default_prb),
            ),
            SliceRequest(
                slice_id="s1-mMTC",
                slice_type=SliceType.MMTC,
                prb_required=prb_per_slice.get("s1-mMTC", default_prb),
            ),
            SliceRequest(
                slice_id="s2-URLLC",
                slice_type=SliceType.URLLC,
                prb_required=prb_per_slice.get("s2-URLLC", default_prb),
            ),
        )

    @staticmethod
    def load_csv(
        csv_path: str,
        bs_id: str | None = None,
    ) -> list[ORANRecord]:
        """Đọc một file CSV ColO-RAN metrics và trả về list[ORANRecord].

        Parameters
        ----------
        csv_path:
            Đường dẫn tới file CSV.
        bs_id:
            Tên BS (nếu None, suy từ filename, ví dụ ``bs1`` → ``bs-1``).
        """
        if bs_id is None:
            basename = os.path.basename(os.path.dirname(csv_path))
            # Expect "bs1", "bs2", ..., "bs7" in path
            if basename.startswith("bs"):
                bs_id = f"bs-{basename[2:]}"
            else:
                bs_id = "bs-unknown"

        records: list[ORANRecord] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_raw = row.get("Timestamp") or row.get("time") or "0"
                    ts_ms = float(ts_raw)
                    ts = datetime(2021, 1, 1) + timedelta(milliseconds=ts_ms)

                    slice_id = int(row.get("slice_id", 0))
                    slice_prb = int(float(row.get("slice_prb", 0)))
                    dl_buffer = float(row.get("dl_buffer [bytes]", 0))
                    tx_brate = float(row.get("tx_brate downlink [Mbps]", 0))
                    req_prbs = float(row.get("sum_requested_prbs", 0))
                    gnt_prbs = float(row.get("sum_granted_prbs", 0))
                    sched = int(float(row.get("scheduling_policy", 0)))

                    ratio = 0.0
                    if req_prbs > 0:
                        ratio = min(gnt_prbs / req_prbs, 1.0)

                    records.append(
                        ORANRecord(
                            bs_id=bs_id,
                            timestamp=ts,
                            slice_id=slice_id,
                            slice_prb=slice_prb,
                            dl_buffer=dl_buffer,
                            tx_brate_dl=tx_brate,
                            sum_requested_prbs=req_prbs,
                            sum_granted_prbs=gnt_prbs,
                            ratio_granted_req=ratio,
                            scheduling_policy=sched,
                        )
                    )
                except (ValueError, KeyError) as exc:
                    logger.debug("Bỏ qua row không hợp lệ: %s (%s)", row, exc)
                    continue

        logger.info("Loaded %d ORANRecords from %s", len(records), csv_path)
        return records

    @staticmethod
    def load_dataset(
        dataset_dir: str,
        subset: str = "slice_traffic",
        max_files_per_bs: int | None = None,
    ) -> list[ORANRecord]:
        """Đọc toàn bộ dataset ColO-RAN từ thư mục gốc.

        Parameters
        ----------
        dataset_dir:
            Đường dẫn tới thư mục gốc ``colosseum-oran-coloran-dataset/``.
        subset:
            Tên thư mục con (``slice_traffic`` hoặc ``slice_mixed``).
        max_files_per_bs:
            Giới hạn số file CSV đọc per BS (None = tất cả).
        """
        all_records: list[ORANRecord] = []
        base_path = os.path.join(dataset_dir, subset)

        if not os.path.isdir(base_path):
            logger.warning("Dataset path not found: %s", base_path)
            return all_records

        for bs_dir in sorted(glob.glob(os.path.join(base_path, "bs*"))):
            bs_name = os.path.basename(bs_dir)
            bs_id = f"bs-{bs_name[2:]}" if bs_name.startswith("bs") else bs_name

            csv_files = sorted(glob.glob(os.path.join(bs_dir, "**", "*.csv"), recursive=True))
            if max_files_per_bs:
                csv_files = csv_files[:max_files_per_bs]

            for csv_file in csv_files:
                records = ColORANLoader.load_csv(csv_file, bs_id=bs_id)
                all_records.extend(records)

        logger.info(
            "Loaded total %d ORANRecords from %s/%s",
            len(all_records),
            dataset_dir,
            subset,
        )
        return all_records

    @staticmethod
    def aggregate_per_bs_stats(
        records: list[ORANRecord],
    ) -> dict[str, dict[str, Any]]:
        """Tính thống kê tổng hợp per-BS từ ORANRecords.

        Hữu ích để calibrate thresholds cho emergency detection.

        Returns
        -------
        dict
            ``{bs_id: {"avg_dl_buffer": ..., "avg_ratio": ..., ...}}``
        """
        from collections import defaultdict

        acc: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in records:
            acc[r.bs_id]["dl_buffer"].append(r.dl_buffer)
            acc[r.bs_id]["tx_brate_dl"].append(r.tx_brate_dl)
            acc[r.bs_id]["ratio_granted_req"].append(r.ratio_granted_req)
            acc[r.bs_id]["slice_prb"].append(r.slice_prb)

        stats: dict[str, dict[str, Any]] = {}
        for bs_id, metrics in acc.items():
            stats[bs_id] = {}
            for metric_name, values in metrics.items():
                if values:
                    avg = sum(values) / len(values)
                    stats[bs_id][f"avg_{metric_name}"] = round(avg, 4)
                    stats[bs_id][f"max_{metric_name}"] = round(max(values), 4)
                    stats[bs_id][f"count"] = len(values)
        return stats
