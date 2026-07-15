"""Huấn luyện TCN đa nhiệm — §5 + §4 Tier A (milestone tuần 4).

Luồng: build_sequences (§3) -> block-split leave-scheduler-out (§3.5) ->
fit RobustScaler trên feature thô (train only, §3.4) -> train TCN với
**uncertainty weighting** (Kendall & Gal, §5.1) tự cân bằng loss flag vs
priority -> early stopping theo **PR-AUC trên val** (§5.3, KHÔNG theo
accuracy) -> **temperature scaling** (§5.2) -> ngưỡng cost-sensitive ->
đánh giá test (§6). Lưu artifact tương thích ``MLPredictionProvider``.

Chạy::

    python -m quantaslice.ai.train.deep --data-root colosseum-oran-coloran-dataset \\
        --scheds 0,1,2 --trs 0,6,13,21,27 --test-scheds 2 --out artifacts/tcn.joblib
    python -m quantaslice.ai.train.deep --synthetic --epochs 15 --out /tmp/tcn.joblib
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np

from quantaslice.ai.data.features import FEATURE_TS_NAMES, WindowConfig, build_sequences
from quantaslice.ai.data.labeling import LabelConfig, label_frame
from quantaslice.ai.data.loaders import BSFrame, generate_synthetic, iter_frames
from quantaslice.ai.data.split import leave_config_out, leave_scheduler_out
from quantaslice.ai.eval.metrics import (
    DetectionMetrics,
    cost_sensitive_threshold,
    evaluate_flag,
    expected_calibration_error,
)
from quantaslice.ai.train.calibrate import temperature_scale

__all__ = ["DeepTrainResult", "run_training_tcn"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeepTrainResult:
    detector: object                 # TCNDetector
    metrics: DetectionMetrics        # trên test (đã hiệu chỉnh)
    temperature: float
    ece_before: float                # ECE trên val TRƯỚC hiệu chỉnh
    ece_after: float                 # ECE trên val SAU hiệu chỉnh
    n_train: int
    n_test: int


def _split_xy(frames, labels_map, wcfg):
    x, yf, yp, _ = build_sequences(frames, [labels_map[id(f)] for f in frames], wcfg)
    return x, yf, yp


def run_training_tcn(
    frames: list[BSFrame],
    *,
    test_scheds: tuple[int, ...] = (2,),
    val_trs: tuple[int, ...] | None = None,
    wcfg: WindowConfig | None = None,
    label_cfg: LabelConfig | None = None,
    channels: tuple[int, ...] = (32, 32),
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    cost_miss: float = 10.0,
    patience: int = 6,
    seed: int = 0,
) -> DeepTrainResult:
    """Huấn luyện TCN + hiệu chỉnh trên tập ``BSFrame`` đã load."""
    import torch
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import RobustScaler

    from quantaslice.ai.models.tcn import MultiTaskTCN, TCNDetector

    torch.manual_seed(seed)
    wcfg = wcfg or WindowConfig()
    if len(frames) < 2:
        raise ValueError("Cần ≥ 2 frame để train/test")

    # Block-split (giống baseline để so sánh công bằng).
    train_frames, test_frames = leave_scheduler_out(frames, test_scheds)
    if not train_frames or not test_frames:
        trs = sorted({f.tr for f in frames})
        holdout = tuple(trs[-max(1, len(trs) // 3):])
        train_frames, test_frames = leave_config_out(frames, holdout)
    if val_trs is None:
        # Chọn val TRẢI ĐỀU trên dải config (mỗi 4 config lấy 1) thay vì
        # "1/4 cuối" — tránh val thiên lệch về nhóm URLLC-heavy (tr21-27)
        # làm early-stopping/calibration mất đại diện, dẫn tới underfit.
        train_trs = sorted({f.tr for f in train_frames})
        val_trs = tuple(train_trs[::4]) or (train_trs[-1],)
    fit_frames, val_frames = leave_config_out(train_frames, val_trs)
    if not fit_frames or not val_frames:
        fit_frames, val_frames = train_frames, train_frames

    labels_map = {id(f): label_frame(f, label_cfg) for f in frames}
    x_fit, yf_fit, yp_fit = _split_xy(fit_frames, labels_map, wcfg)
    x_val, yf_val, _ = _split_xy(val_frames, labels_map, wcfg)
    x_test, yf_test, _ = _split_xy(test_frames, labels_map, wcfg)
    if x_fit.shape[0] == 0 or x_test.shape[0] == 0:
        raise ValueError("Không đủ cửa sổ để train/test (tăng dữ liệu hoặc giảm lookback)")

    n_feat = x_fit.shape[2]
    # §3.4: fit scaler CHỈ trên train, trên feature thô (flatten theo thời gian).
    scaler = RobustScaler().fit(x_fit.reshape(-1, n_feat))

    model = MultiTaskTCN(n_feat, channels=channels, kernel=3, dropout=0.1)
    detector = TCNDetector(model, lookback=wcfg.lookback)
    detector.scaler = scaler
    detector.ts_feature_names = FEATURE_TS_NAMES

    # Uncertainty weighting (§5.1): 2 log-variance học được tự cân bằng
    # loss flag (BCE) và priority (SmoothL1).
    log_var = torch.zeros(2, requires_grad=True)
    params = list(model.parameters()) + [log_var]
    optimizer = torch.optim.Adam(params, lr=lr)

    pos = float(yf_fit.sum())
    neg = float(yf_fit.size - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)])  # §3.6 cân bằng lớp
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    huber = torch.nn.SmoothL1Loss()

    xf = torch.as_tensor(
        scaler.transform(x_fit.reshape(-1, n_feat)).reshape(x_fit.shape).transpose(0, 2, 1),
        dtype=torch.float32,
    )
    yf_t = torch.as_tensor(yf_fit, dtype=torch.float32)
    yp_t = torch.as_tensor(yp_fit, dtype=torch.float32)
    n = xf.shape[0]

    best_ap, best_state, no_improve = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            flag_logit, prio = model(xf[idx])
            l_flag = bce(flag_logit, yf_t[idx])
            l_prio = huber(prio, yp_t[idx])
            # total = Σ exp(-s)·L + s  (s = log σ²).
            loss = (torch.exp(-log_var[0]) * l_flag + log_var[0]
                    + torch.exp(-log_var[1]) * l_prio + log_var[1])
            loss.backward()
            optimizer.step()

        # Early stopping theo PR-AUC trên val (§5.3).
        val_probs, _ = detector.predict_windows(x_val)
        ap = average_precision_score(yf_val, val_probs) if 0 < yf_val.sum() < yf_val.size else 0.0
        if ap > best_ap:
            best_ap, no_improve = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stop tại epoch %d (val PR-AUC=%.3f)", epoch, best_ap)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    detector.model = model.eval()

    # Hiệu chỉnh nhiệt độ (§5.2) trên val, đo ECE trước/sau.
    val_logits, _ = detector.logits(x_val)
    ece_before = expected_calibration_error(yf_val, _sigmoid(val_logits))
    temperature = temperature_scale(val_logits, yf_val)
    ece_after = expected_calibration_error(yf_val, _sigmoid(val_logits / temperature))
    # Guard: nếu hiệu chỉnh làm ECE TỆ HƠN (temperature phân kỳ, vd chạm
    # trần clip), lùi về T=1 — calibration không bao giờ được làm xấu đi.
    if not (ece_after <= ece_before + 1e-6):
        logger.info("Temperature=%.2f làm ECE tệ hơn (%.3f->%.3f) -> giữ T=1",
                    temperature, ece_before, ece_after)
        temperature, ece_after = 1.0, ece_before
    detector.temperature = temperature

    # Ngưỡng cost-sensitive trên val đã hiệu chỉnh (lùi về fit nếu val
    # không có mẫu dương).
    if yf_val.sum() > 0:
        val_probs, _ = detector.predict_windows(x_val)
        thr = cost_sensitive_threshold(yf_val, val_probs, cost_miss=cost_miss)
    else:
        fit_probs, _ = detector.predict_windows(x_fit)
        thr = cost_sensitive_threshold(yf_fit, fit_probs, cost_miss=cost_miss)
    detector.threshold = thr

    test_probs, _ = detector.predict_windows(x_test)
    metrics = evaluate_flag(yf_test, test_probs, thr)
    return DeepTrainResult(
        detector=detector, metrics=metrics, temperature=temperature,
        ece_before=ece_before, ece_after=ece_after,
        n_train=x_fit.shape[0], n_test=x_test.shape[0],
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


# ── CLI ───────────────────────────────────────────────────────────────
def _parse_int_list(s: str | None) -> tuple[int, ...] | None:
    return tuple(int(x) for x in s.split(",")) if s else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train QuantaSlice TCN emergency detector")
    p.add_argument("--data-root", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--scheds", default="0,1,2")
    p.add_argument("--trs", default=None)
    p.add_argument("--exps", default="1")
    p.add_argument("--bss", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--test-scheds", default="2")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--cost-miss", type=float, default=10.0)
    p.add_argument("--out", default="artifacts/tcn.joblib")
    args = p.parse_args()

    if args.synthetic or not args.data_root:
        frames = generate_synthetic(n_frames=24, n_steps=500)
    else:
        bss = tuple(args.bss.split(",")) if args.bss else None
        frames = list(iter_frames(
            args.data_root, scheds=_parse_int_list(args.scheds) or (0, 1, 2),
            trs=_parse_int_list(args.trs), exps=_parse_int_list(args.exps) or (1,),
            bss=bss, limit=args.limit,
        ))
    logger.info("Loaded %d frames", len(frames))

    result = run_training_tcn(
        frames, test_scheds=_parse_int_list(args.test_scheds) or (2,),
        wcfg=WindowConfig(lookback=args.lookback, stride=args.stride),
        epochs=args.epochs, cost_miss=args.cost_miss,
    )

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.detector.save(args.out)
    m = result.metrics
    logger.info("TCN | train=%d test=%d | T=%.3f ECE val %.3f->%.3f",
                result.n_train, result.n_test, result.temperature,
                result.ece_before, result.ece_after)
    logger.info("Test metrics: PR-AUC=%.3f ROC-AUC=%.3f P=%.3f R=%.3f F1=%.3f ECE=%.3f thr=%.3f (pos=%.3f)",
                m.pr_auc, m.roc_auc, m.precision, m.recall, m.f1, m.ece, m.threshold, m.positive_rate)
    logger.info("Đã lưu artifact -> %s", args.out)


if __name__ == "__main__":
    main()
