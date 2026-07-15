"""Chia train/test theo KHỐI — §3.5 (điểm dễ rò rỉ nhất).

TUYỆT ĐỐI không shuffle ngẫu nhiên theo timestep: cửa sổ liền kề chồng
lấn nhau, shuffle sẽ để cửa sổ gần nhau rơi vào cả train lẫn test -> rò
rỉ thời gian, thổi phồng điểm số. Thay vào đó chia ở cấp ĐỘ CẤU HÌNH:

* ``leave_scheduler_out`` — train trên vài scheduler, test scheduler còn
  lại (vd train RR+WF, test PF). Thước đo generalization mạnh nhất trong
  ColO-RAN.
* ``leave_config_out`` — giữ vài config ``tr`` chỉ để test.

Vì mỗi ``BSFrame`` thuộc đúng một (sched, tr), chia ở cấp frame là đủ
đảm bảo không cửa sổ nào bắc cầu giữa train và test.

Module CHỈ import từ ``ai.data.loaders``.
"""

from __future__ import annotations

from quantaslice.ai.data.loaders import BSFrame

__all__ = ["leave_scheduler_out", "leave_config_out"]


def leave_scheduler_out(
    frames: list[BSFrame], test_scheds: tuple[int, ...] = (2,)
) -> tuple[list[BSFrame], list[BSFrame]]:
    """Train = frame KHÔNG thuộc ``test_scheds``; test = phần còn lại."""
    test_set = set(test_scheds)
    train = [f for f in frames if f.sched not in test_set]
    test = [f for f in frames if f.sched in test_set]
    return train, test


def leave_config_out(
    frames: list[BSFrame], test_trs: tuple[int, ...]
) -> tuple[list[BSFrame], list[BSFrame]]:
    """Train = frame có ``tr`` KHÔNG thuộc ``test_trs``; test = phần còn lại."""
    test_set = set(test_trs)
    train = [f for f in frames if f.tr not in test_set]
    test = [f for f in frames if f.tr in test_set]
    return train, test
