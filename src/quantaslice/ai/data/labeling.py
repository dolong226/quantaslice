"""Nhãn "emergency" dẫn xuất từ vi phạm QoS/SLA — §2 của plan ML.

Đây là phần quyết định tính trung thực của cả dự án. ColO-RAN KHÔNG có
nhãn emergency sẵn; tiêm bất thường rồi học phát hiện chính nó =
circular evaluation. Thay vào đó, nhãn ở đây sinh từ **động lực thật của
mạng** đo qua KPM.

Định nghĩa emergency = **vi phạm SLA đo bằng độ trễ hàng đợi (queueing
delay)** — một đại lượng VẬT LÝ, không phải biến đổi chuẩn hoá tuỳ ý:

    queue_delay(giây) = dl_buffer(bytes) * 8 / throughput(bps)

So với ngân sách độ trễ per-slice (URLLC ms, eMBB lớn hơn). ``severity``
= ``queue_delay / budget`` cắt về [0,1]; ``emergency_flag = 1`` khi có
slice bất kỳ vượt ngưỡng KÉO DÀI ≥ k bước.

VÌ SAO KHÔNG dùng biến-thiên-chuẩn-hoá-theo-median như bản trước: chuẩn
hoá ``growth / (median+1)`` khiến slice RỖNG (median≈0, như URLLC/mMTC
nhàn rỗi 80% thời gian) biến một đốm buffer ~200 byte (nhiễu đo) thành
severity=1.0, trong khi slice NGHẼN THẬT (eMBB, buffer ~367 KB) bị chia
nhỏ xuống ~0. Kết quả 75% "emergency" là nhiễu ở slice rỗng còn slice
nghẽn thật bị bỏ qua -> nhãn ≈ nhiễu, model không học được (PR-AUC
in-distribution chỉ ~0.53). Độ trễ hàng đợi sửa triệt để: độ lớn có ý
nghĩa vật lý (214 byte -> vài ms; 367 KB -> ~0.8 s), và **sàn buffer**
loại hẳn các blip dưới ngưỡng nhiễu.

eMBB bị bỏ đói kinh niên (throughput < 4 Mbps/UE, buffer đầy) được coi
là ĐANG vi phạm SLA — đúng bản chất: slice thiếu tài nguyên, QUBO cần
ưu tiên. ``priority`` boost slice có severity cao để QUBO tái ưu tiên
đúng chỗ.

Provenance rõ ràng: nhãn = hàm tường minh, vật lý, của KPM tức thời;
tái lập được, không phụ thuộc mô hình nào. Xem plan §2.

LƯU Ý dataset: ``rome_static_medium`` gần TĨNH (UE bất động, traffic
CBR/Poisson đều) nên trạng thái SLA gần như do config RBG (tr) quyết
định — nhãn này học được rất cao (PR-AUC ~0.97) nhưng đó phản ánh
dataset tĩnh, không phải "model giỏi"; giá trị thật là vector ``p`` và
khả năng generalize giữa config/scheduler.

Module CHỈ import từ ``core`` + ``ai.data.loaders``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantaslice.ai.data.loaders import RAW_METRICS, SLICE_IDS, BSFrame

__all__ = ["LabelConfig", "FrameLabels", "label_frame", "PER_UE_DEMAND_MBPS"]

# Nhu cầu throughput danh định mỗi UE (Mbps), suy từ traffic class ColO-RAN
# (README): eMBB CBR 4 Mbps; mMTC 30 pkt/s × 125 B; URLLC 10 pkt/s × 125 B.
PER_UE_DEMAND_MBPS: dict[int, float] = {
    0: 4.0,                       # eMBB
    1: 30 * 125 * 8 / 1e6,        # mMTC ≈ 0.03 Mbps
    2: 10 * 125 * 8 / 1e6,        # URLLC ≈ 0.01 Mbps
}

# Chỉ số metric trong trục cuối của BSFrame.series (theo RAW_METRICS).
_BUF = RAW_METRICS.index("dl_buffer")
_BRATE = RAW_METRICS.index("tx_brate_dl")
_REQ = RAW_METRICS.index("requested_prbs")
_GRANT = RAW_METRICS.index("granted_prbs")


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """Ngưỡng dẫn xuất nhãn. Tách ra để calibrate được (§5.2) mà không
    sửa công thức."""

    # Ngân sách độ trễ hàng đợi (giây) theo slice-index (eMBB, mMTC, URLLC).
    # URLLC khắt khe nhất; eMBB lớn (buffer khổng lồ nhưng chỉ tính vi phạm
    # khi throughput tụt đủ để độ trễ vượt ngưỡng).
    latency_budgets: tuple[float, float, float] = (2.0, 0.4, 0.08)
    buffer_floor_bytes: float = 1000.0  # sàn: bỏ blip buffer nhỏ hơn (nhiễu đo)
    min_drain_bps: float = 1e4          # sàn throughput khi tính độ trễ (tránh /0)
    emergency_thr: float = 0.5          # severity ≥ ngưỡng này -> đang vi phạm
    sustain_steps: int = 3              # phải kéo dài ≥ k bước mới tính emergency
    priority_boost: float = 8.0         # hệ số boost trọng số theo severity


@dataclass(frozen=True, slots=True)
class FrameLabels:
    """Nhãn cho một ``BSFrame``.

    ``flag`` (T,) bool; ``severity`` (T, 3) ∈ [0,1] theo slice-index
    (eMBB, mMTC, URLLC); ``priority`` (T, 3) trọng số ≥ 0 cùng thứ tự."""

    flag: np.ndarray
    severity: np.ndarray
    priority: np.ndarray

    @property
    def emergency_rate(self) -> float:
        return float(self.flag.mean()) if self.flag.size else 0.0


def _slice_severity(frame: BSFrame, s_idx: int, cfg: LabelConfig) -> np.ndarray:
    """Mức vi phạm SLA ∈ [0,1] cho một slice tại mỗi bước, đo bằng ĐỘ TRỄ
    HÀNG ĐỢI vật lý (buffer / throughput) so với ngân sách per-slice.

    Có SÀN buffer: buffer dưới ``buffer_floor_bytes`` coi như không có
    hàng đợi (loại nhiễu đo sub-KB ở slice nhàn rỗi)."""
    s = frame.series[:, s_idx, :]
    buffer_bytes = np.clip(s[:, _BUF], 0, None)
    brate = np.clip(s[:, _BRATE], 0, None)               # Mbps

    drain_bps = np.maximum(brate * 1e6, cfg.min_drain_bps)
    queue_delay = (buffer_bytes * 8.0) / drain_bps       # giây
    queue_delay = np.where(buffer_bytes >= cfg.buffer_floor_bytes, queue_delay, 0.0)

    budget = cfg.latency_budgets[s_idx]
    return np.clip(queue_delay / budget, 0.0, 1.0)


def _sustained(mask: np.ndarray, k: int) -> np.ndarray:
    """True tại bước t nếu ``mask`` đúng liên tục trong ≥ k bước tính đến t."""
    if k <= 1:
        return mask
    out = np.zeros_like(mask)
    run = 0
    for t in range(mask.size):
        run = run + 1 if mask[t] else 0
        out[t] = run >= k
    return out


def label_frame(frame: BSFrame, cfg: LabelConfig | None = None) -> FrameLabels:
    """Dẫn xuất nhãn (flag, severity, priority) cho một ``BSFrame``."""
    cfg = cfg or LabelConfig()
    t = frame.n_steps
    severity = np.zeros((t, 3))
    for s_idx in range(3):
        severity[:, s_idx] = _slice_severity(frame, s_idx, cfg)

    # Ngưỡng "đang vi phạm" của từng slice.
    violating = severity >= cfg.emergency_thr
    sustained = np.zeros_like(violating)
    for s_idx in range(3):
        sustained[:, s_idx] = _sustained(violating[:, s_idx], cfg.sustain_steps)
    flag = sustained.any(axis=1)

    # Priority: baseline 1, boost theo severity đã sustain (chỉ boost khi
    # emergency để p ổn định lúc bình thường).
    priority = 1.0 + cfg.priority_boost * (severity * sustained)
    return FrameLabels(flag=flag, severity=severity, priority=priority)
