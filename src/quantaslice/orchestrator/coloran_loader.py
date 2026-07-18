"""Factory dựng ``BaseStation``/``SliceRequest`` chuẩn theo schema
dataset ColO-RAN (``rome_static_medium``: 7 BS, 3 slice eMBB/mMTC/URLLC)
— tách khỏi ``quantaslice.ai.data.loaders`` (đọc CSV thô) vì đây là bước
"dựng topology cho AllocationProblem", không phải "đọc dữ liệu huấn
luyện".
"""

from __future__ import annotations

from quantaslice.core.types import BaseStation, SliceRequest, SliceType

__all__ = ["ColORANLoader"]

_DEFAULT_PRB_PER_SLICE: dict[str, float] = {
    "s0-eMBB": 17,
    "s1-mMTC": 17,
    "s2-URLLC": 17,
}
_SLICE_TYPE_BY_ID: dict[str, SliceType] = {
    "s0-eMBB": SliceType.EMBB,
    "s1-mMTC": SliceType.MMTC,
    "s2-URLLC": SliceType.URLLC,
}


class ColORANLoader:
    """Factory tĩnh — không giữ state, chỉ dựng topology mặc định khớp
    kịch bản thực nghiệm gốc của dataset ColO-RAN (7 BS trong kịch bản
    ``rome_static_medium``, 3 slice chuẩn)."""

    @staticmethod
    def create_stations(n_stations: int = 7, prb_capacity: float = 50) -> tuple[BaseStation, ...]:
        """7 base station mặc định (``bs-1``..``bs-7``), mỗi trạm 50 PRB
        — khớp kịch bản ``rome_static_medium`` của dataset ColO-RAN."""
        return tuple(BaseStation(gnb_id=f"bs-{i + 1}", prb_capacity=prb_capacity) for i in range(n_stations))

    @staticmethod
    def create_slices(prb_per_slice: dict[str, float] | None = None) -> tuple[SliceRequest, ...]:
        """3 slice chuẩn: eMBB, mMTC, URLLC — PRB mỗi slice tuỳ chỉnh
        qua ``prb_per_slice`` (mặc định 17 PRB/slice mỗi loại)."""
        prb = {**_DEFAULT_PRB_PER_SLICE, **(prb_per_slice or {})}
        return tuple(
            SliceRequest(slice_id=slice_id, slice_type=_SLICE_TYPE_BY_ID[slice_id], prb_required=prb[slice_id])
            for slice_id in _DEFAULT_PRB_PER_SLICE
        )
