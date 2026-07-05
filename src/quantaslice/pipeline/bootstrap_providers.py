"""Prediction provider tạm thời (bootstrap) để pipeline chạy được ngay
cả khi package ``quantaslice.ai`` (Member A) chưa hoàn thành — đúng yêu
cầu overview: "The system MUST be runnable before the LSTM exists."

Khi ``quantaslice.ai`` hoàn thiện với ``LSTMPredictionProvider`` và
``CSVPredictionProvider`` thật, các provider đó nên đăng ký vào registry
riêng của ``ai`` (theo đúng pattern ``quantum.solver_registry``);
``DependencyContainer`` chỉ cần đổi ``provider_registry=`` mà không cần
sửa ``Runner``. Ba provider ở đây (không cần dữ liệu training, không
cần AI thật) vẫn hữu ích lâu dài làm baseline demo/CI nên có thể giữ
nguyên.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np

from quantaslice.core.types import FeatureWindow, Prediction, PriorityVector
from quantaslice.pipeline.registries import prediction_provider_registry

__all__ = ["MockPredictionProvider", "RandomPredictionProvider", "ThresholdPredictionProvider"]

_BASELINE_PRIORITY = PriorityVector(embb=1.0, urllc=1.0, mmtc=1.0)
# Thứ tự cột đúng theo StreamSimulator._RAW_FEATURE_NAMES: smsin, smsout, callin, callout, internet
_CALLIN_IDX, _CALLOUT_IDX = 2, 3


@prediction_provider_registry.register("mock")
class MockPredictionProvider:
    """Luôn trả Prediction baseline (không khẩn cấp, p0=(1,1,1), xem
    tài liệu LSTM mục 4.2) — dùng để demo/kiểm thử luồng end-to-end mà
    chưa cần bất kỳ logic AI thật nào."""

    def predict(self, window: FeatureWindow) -> Prediction:
        return Prediction(
            gnb_id=window.gnb_id,
            timestamp=window.timestamp,
            emergency_flag=False,
            emergency_prob=0.0,
            priority=_BASELINE_PRIORITY,
        )


@prediction_provider_registry.register("random")
class RandomPredictionProvider:
    """Random emergency theo xác suất cấu hình, KHÔNG đọc nội dung
    ``window`` — dùng để "stress test" luồng re-optimize/orchestrator
    (Runner._reoptimize) mà chưa cần LSTM thật.

    LƯU Ý: vì bỏ qua hoàn toàn dữ liệu, provider này KHÔNG phù hợp để
    demo câu chuyện "burst dữ liệu -> khẩn cấp" — dùng
    :class:`ThresholdPredictionProvider` cho mục đích đó.
    """

    def __init__(self, emergency_probability: float = 0.1, seed: int | None = None) -> None:
        self._emergency_probability = emergency_probability
        self._rng = random.Random(seed)

    def predict(self, window: FeatureWindow) -> Prediction:
        is_emergency = self._rng.random() < self._emergency_probability
        if is_emergency:
            prob = self._rng.uniform(0.5, 1.0)
            priority = PriorityVector(
                embb=self._rng.uniform(0.1, 1.0),
                urllc=self._rng.uniform(5.0, 10.0),
                mmtc=self._rng.uniform(0.1, 1.0),
            )
        else:
            prob = self._rng.uniform(0.0, 0.3)
            priority = _BASELINE_PRIORITY
        return Prediction(
            gnb_id=window.gnb_id,
            timestamp=window.timestamp,
            emergency_flag=is_emergency,
            emergency_prob=prob,
            priority=priority,
        )


@prediction_provider_registry.register("threshold")
class ThresholdPredictionProvider:
    """Phát hiện bất thường thống kê ĐỌC THẬT nội dung ``window`` — proxy
    đơn giản cho "Strategy A" (mục 4.1 tài liệu LSTM: "Phát hiện bất
    thường thống kê... vượt ngưỡng bất thường theo phân phối lịch sử").

    Baseline là trung bình trượt DÀI theo từng gNB, tích luỹ qua nhiều
    lần gọi ``predict()`` (đúng tinh thần mục 3.2: "trung bình trượt cửa
    sổ W") — KHÔNG lấy từ vài điểm cuối trong chính cửa sổ dự đoán hiện
    tại, vì cửa sổ đó quá ngắn (T bước gần nhất) nên dễ bị nhiễu bởi các
    "điểm gãy" (kink) tự nhiên của nội suy tuyến tính giữa 2 mốc dữ liệu
    gốc, gây báo động giả.

    HẠN CHẾ ĐÃ BIẾT (thực nghiệm trên dữ liệu giả lập): baseline rolling
    "đuổi theo" khá nhanh với burst tăng DẦN (ramp, do nội suy tuyến
    tính giữa 2 mốc dữ liệu gốc) — z-score có thể tụt dưới ngưỡng ngay
    giữa lúc burst đang ở đỉnh, vì baseline đã kịp "hấp thụ" phần đầu
    ramp vào chính nó. Với bước nhảy ĐỘT NGỘT (thay vì tăng dần) detector
    sẽ nhạy hơn nhiều. Đây là nhược điểm cố hữu của thống kê rolling-mean
    đơn giản — LSTM thật (học được động lực học thời gian, không chỉ
    theo dõi 1 con số trung bình) sẽ xử lý tốt hơn nhiều. Vẫn có thể có
    vài báo động giả ngắn trong giai đoạn "bình thường" do nhiễu tự
    nhiên — khớp với ghi nhận trong tài liệu LSTM: "bất thường thống kê
    thuần túy có thể là nhiễu đo lường chứ không phải 'emergency'".

    Đây KHÔNG phải LSTM thật (không học được pattern phức tạp), nhưng
    phản ứng thật với dữ liệu — phù hợp để demo "burst -> khẩn cấp ->
    ưu tiên URLLC" mà không cần huấn luyện mô hình. Có state (rolling
    history mỗi gNB) nên MỖI PredictionProvider instance chỉ nên dùng
    cho 1 Runner tại 1 thời điểm.

    Priority vector: tỉ trọng callin+callout trong tổng hoạt động bước
    cuối quyết định mức boost cho URLLC — đúng cơ chế ánh xạ ở mục 2.2b
    tài liệu LSTM ("callin/callout -> proxy cho nhu cầu URLLC").
    """

    def __init__(self, k_sigma: float = 3.0, baseline_length: int = 100, urllc_boost: float = 9.0) -> None:
        self._k_sigma = k_sigma
        self._urllc_boost = urllc_boost
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=baseline_length))

    def predict(self, window: FeatureWindow) -> Prediction:
        current_row = window.features[-1]
        current_total = float(current_row.sum())
        history = self._history[window.gnb_id]

        if len(history) < 5:
            is_emergency, prob = False, 0.0
        else:
            baseline = np.array(history)
            mean_b, std_b = float(baseline.mean()), max(float(baseline.std()), 1e-6)
            z_score = (current_total - mean_b) / std_b
            is_emergency = z_score > self._k_sigma
            prob = float(np.clip(z_score / (2 * self._k_sigma), 0.0, 1.0))

        # Cập nhật baseline SAU khi tính z-score (không để điểm hiện tại
        # tự làm loãng baseline của chính nó).
        history.append(current_total)

        if is_emergency:
            call_share = (current_row[_CALLIN_IDX] + current_row[_CALLOUT_IDX]) / max(
                current_total, 1e-6
            )
            priority = PriorityVector(
                embb=1.0, urllc=1.0 + self._urllc_boost * call_share, mmtc=1.0
            )
        else:
            priority = _BASELINE_PRIORITY

        return Prediction(
            gnb_id=window.gnb_id, timestamp=window.timestamp,
            emergency_flag=is_emergency, emergency_prob=prob, priority=priority,
        )
