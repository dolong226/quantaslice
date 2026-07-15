"""Huấn luyện end-to-end baseline detector — §5 + §9 (milestone tuần 3).

Luồng: load ``BSFrame`` (ColO-RAN thật hoặc synthetic) -> nhãn
QoS-violation (§2) -> feature + windowing (§3) -> block-split
leave-scheduler-out (§3.5) -> fit GradientBoostingDetector -> chốt
ngưỡng cost-sensitive trên val (§5.2) -> đánh giá trên test (§6) -> lưu
artifact cho ``MLPredictionProvider``.

Chạy::

    python -m quantaslice.ai.train.baseline --data-root colosseum-oran-coloran-dataset \\
        --scheds 0,1,2 --trs 0,6,13,21,27 --test-scheds 2 --out artifacts/detector.joblib
    python -m quantaslice.ai.train.baseline --synthetic --out /tmp/detector.joblib
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from quantaslice.ai.data.features import (
    FEATURE_TS_NAMES,
    WindowConfig,
    build_tabular,
    summary_feature_names,
)
from quantaslice.ai.data.labeling import LabelConfig, label_frame
from quantaslice.ai.data.loaders import BSFrame, generate_synthetic, iter_frames
from quantaslice.ai.data.split import leave_config_out, leave_scheduler_out
from quantaslice.ai.eval.metrics import DetectionMetrics, cost_sensitive_threshold, evaluate_flag
from quantaslice.ai.models.baselines import GradientBoostingDetector

__all__ = ["TrainResult", "run_training"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainResult:
    detector: GradientBoostingDetector
    metrics: DetectionMetrics
    n_train: int
    n_test: int


def run_training(
    frames: list[BSFrame],
    *,
    test_scheds: tuple[int, ...] = (2,),
    val_trs: tuple[int, ...] | None = None,
    wcfg: WindowConfig | None = None,
    label_cfg: LabelConfig | None = None,
    cost_miss: float = 10.0,
    random_state: int = 0,
) -> TrainResult:
    """Huấn luyện + đánh giá trên một tập ``BSFrame`` đã load sẵn."""
    wcfg = wcfg or WindowConfig()
    if len(frames) < 2:
        raise ValueError("Cần ≥ 2 frame để train/test")

    # Block-split: giữ scheduler test riêng (OOD generalization §6.5). Nếu
    # dữ liệu chỉ có 1 scheduler, lùi về leave-config-out theo tr.
    train_frames, test_frames = leave_scheduler_out(frames, test_scheds)
    if not train_frames or not test_frames:
        trs = sorted({f.tr for f in frames})
        holdout = tuple(trs[-max(1, len(trs) // 3):])
        train_frames, test_frames = leave_config_out(frames, holdout)
        logger.info("Không đủ scheduler để leave-scheduler-out; dùng leave-config-out %s", holdout)

    # Val (chốt ngưỡng) tách khỏi train theo config để không rò rỉ.
    if val_trs is None:
        # Val trải đều trên dải config (mỗi 4 lấy 1) thay vì "1/4 cuối"
        # để ngưỡng cost-sensitive chốt trên phân phối đại diện.
        train_trs = sorted({f.tr for f in train_frames})
        val_trs = tuple(train_trs[::4]) or (train_trs[-1],)
    fit_frames, val_frames = leave_config_out(train_frames, val_trs)
    if not fit_frames or not val_frames:
        fit_frames, val_frames = train_frames, train_frames  # dữ liệu nhỏ: dùng chung

    labels = {id(f): label_frame(f, label_cfg) for f in frames}
    x_fit, yf_fit, yp_fit, _ = build_tabular(fit_frames, [labels[id(f)] for f in fit_frames], wcfg)
    x_val, yf_val, _, _ = build_tabular(val_frames, [labels[id(f)] for f in val_frames], wcfg)
    x_test, yf_test, _, _ = build_tabular(test_frames, [labels[id(f)] for f in test_frames], wcfg)

    detector = GradientBoostingDetector(random_state=random_state)
    detector.fit(x_fit, yf_fit, yp_fit, feature_names=tuple(summary_feature_names()))

    # Chốt ngưỡng cost-sensitive trên val; nếu khối val không có mẫu
    # dương (block-split có thể rơi vào vùng "bình thường"), lùi về chốt
    # trên chính tập fit để tránh ngưỡng suy biến (recall = 0).
    if yf_val.sum() > 0:
        thr = cost_sensitive_threshold(yf_val, detector.predict_proba(x_val), cost_miss=cost_miss)
    else:
        logger.info("Khối val không có mẫu dương -> chốt ngưỡng trên tập fit")
        thr = cost_sensitive_threshold(yf_fit, detector.predict_proba(x_fit), cost_miss=cost_miss)
    detector.set_threshold(thr)

    metrics = evaluate_flag(yf_test, detector.predict_proba(x_test), thr)

    # Metadata cần cho MLPredictionProvider (đính vào artifact).
    detector.window_lookback = wcfg.lookback           # type: ignore[attr-defined]
    detector.ts_feature_names = FEATURE_TS_NAMES        # type: ignore[attr-defined]
    return TrainResult(detector=detector, metrics=metrics,
                       n_train=x_fit.shape[0], n_test=x_test.shape[0])


# ── CLI ───────────────────────────────────────────────────────────────
def _parse_int_list(s: str | None) -> tuple[int, ...] | None:
    if not s:
        return None
    return tuple(int(x) for x in s.split(","))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train QuantaSlice baseline emergency detector")
    p.add_argument("--data-root", default=None, help="Thư mục colosseum-oran-coloran-dataset")
    p.add_argument("--synthetic", action="store_true", help="Dùng dữ liệu synthetic (không cần dataset)")
    p.add_argument("--scheds", default="0,1,2")
    p.add_argument("--trs", default=None, help="Danh sách tr, vd 0,6,13,21,27 (mặc định: tất cả)")
    p.add_argument("--exps", default="1")
    p.add_argument("--bss", default=None, help="Danh sách bs, vd bs1,bs4 (mặc định: tất cả)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--test-scheds", default="2")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--cost-miss", type=float, default=10.0)
    p.add_argument("--out", default="artifacts/detector.joblib")
    args = p.parse_args()

    if args.synthetic or not args.data_root:
        logger.info("Sinh dữ liệu synthetic")
        frames = generate_synthetic(n_frames=18, n_steps=500)
    else:
        bss = tuple(args.bss.split(",")) if args.bss else None
        frames = list(iter_frames(
            args.data_root,
            scheds=_parse_int_list(args.scheds) or (0, 1, 2),
            trs=_parse_int_list(args.trs),
            exps=_parse_int_list(args.exps) or (1,),
            bss=bss, limit=args.limit,
        ))
    logger.info("Loaded %d frames", len(frames))

    result = run_training(
        frames,
        test_scheds=_parse_int_list(args.test_scheds) or (2,),
        wcfg=WindowConfig(lookback=args.lookback, stride=args.stride),
        cost_miss=args.cost_miss,
    )

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.detector.save(args.out)
    m = result.metrics
    logger.info("Backend: %s | train=%d test=%d", result.detector.backend_name,
                result.n_train, result.n_test)
    logger.info("Test metrics: PR-AUC=%.3f ROC-AUC=%.3f P=%.3f R=%.3f F1=%.3f ECE=%.3f thr=%.3f (pos=%.3f)",
                m.pr_auc, m.roc_auc, m.precision, m.recall, m.f1, m.ece, m.threshold, m.positive_rate)
    logger.info("Đã lưu artifact -> %s", args.out)


if __name__ == "__main__":
    main()
