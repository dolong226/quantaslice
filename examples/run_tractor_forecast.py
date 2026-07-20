"""Forecasting eval trên TRACTOR (traffic 5G THẬT, động) — bằng chứng giá
trị ML mà ColO-RAN tĩnh KHÔNG cho được.

Khung TRUNG THỰC (per-stream): với mỗi luồng traffic một loại, dự báo vi
phạm SLA (độ trễ hàng đợi > ngân sách) H bước TRƯỚC từ cửa sổ KPM hiện
tại. Luôn so ML với LUẬT tầm thường "đang vi phạm chưa" và báo **lead-time
recall** (bắt sớm ở các bước "sắp vi phạm nhưng hiện chưa") — thước đo
giá trị thật, không phải PR-AUC tuyệt đối (dễ gây hiểu nhầm).

Cần tải dữ liệu TRACTOR trước (genesys-neu/TRACTOR, thư mục logs/), trỏ
``--root`` vào đó.

Chạy::

    python -m examples.run_tractor_forecast --root <tractor_root> --horizon 8
"""

from __future__ import annotations

import argparse

import numpy as np

from quantaslice.ai.data.features import summarize_window
from quantaslice.ai.data.tractor_loader import iter_tractor_streams
from quantaslice.ai.eval.metrics import lead_time_recall
from quantaslice.ai.models.baselines import GradientBoostingDetector

_DT = 0.25


def _featurize(stream: dict[str, np.ndarray]):
    """(features per-bước KHÔNG chứa delay, delay tức thời)."""
    buf = np.clip(stream["dl_buffer"], 0, None)
    br = np.clip(stream["tx_brate_dl"], 0, None)
    delay = (buf * 8) / np.maximum(br * 1e6, 1e4)
    w = 8
    prev = np.concatenate([np.full(min(w, len(buf)), buf[0]), buf[:-w]])
    growth = (buf - prev) / (np.median(buf) + 1.0)
    util = np.divide(stream["granted_prbs"], stream["requested_prbs"],
                     out=np.ones_like(br), where=stream["requested_prbs"] > 1e-6)
    feats = np.column_stack([np.log1p(buf), br, growth, np.clip(util, 0, 2)])
    return feats, delay


def _build(streams, *, lookback: int, horizon: int, budget: float, stride: int = 3):
    xs, yf, triv = [], [], []
    for _, s in streams:
        feats, delay = _featurize(s)
        breach = (delay >= budget).astype(int)
        for st in range(0, len(delay) - lookback - horizon, stride):
            t = st + lookback - 1
            xs.append(summarize_window(feats[st:st + lookback]))
            yf.append(1.0 if breach[t + 1:t + 1 + horizon].any() else 0.0)
            triv.append(delay[t])
    return np.array(xs), np.array(yf), np.array(triv)


def run(root: str, *, trial: str = "Trial0", lookback: int = 20, horizon: int = 8,
        budget: float = 0.1):
    from sklearn.metrics import average_precision_score as AP

    streams = list(iter_tractor_streams(root, trial))
    if len(streams) < 2:
        raise SystemExit(f"Cần ≥2 luồng TRACTOR ở {root} (trial {trial}); thấy {len(streams)}. "
                         "Tải dữ liệu genesys-neu/TRACTOR (logs/) trước.")

    # Split theo CHỈ SỐ kịch bản để giữ cân bằng loại traffic (train nhóm
    # lẻ embb1/mmtc1/urllc1..., test nhóm chẵn) — tránh train/test lệch
    # loại. Nếu một phía rỗng thì lùi về chia đôi danh sách.
    def _scenario_idx(name: str) -> int:
        digits = name[len(name.rstrip("0123456789")):]
        return int(digits) if digits else 0

    train = [s for s in streams if _scenario_idx(s[0]) % 2 == 1]
    test = [s for s in streams if _scenario_idx(s[0]) % 2 == 0]
    if not train or not test:
        mid = max(1, len(streams) // 2)
        train, test = streams[:mid], streams[mid:]
    xtr, ytr, _ = _build(train, lookback=lookback, horizon=horizon, budget=budget)
    xte, yte, triv = _build(test, lookback=lookback, horizon=horizon, budget=budget)

    m = GradientBoostingDetector()
    m.fit(xtr, ytr, np.ones((len(ytr), 3)))
    p = m.predict_proba(xte)
    now = triv >= budget
    return {
        "n_test": int(len(yte)), "pos_rate": float(yte.mean()),
        "ml_pr_auc": float(AP(yte, p)) if 0 < yte.sum() < len(yte) else float("nan"),
        "rule_pr_auc": float(AP(yte, triv / budget)) if 0 < yte.sum() < len(yte) else float("nan"),
        "onset_windows": int(((yte == 1) & (~now)).sum()),
        "ml_lead_recall": lead_time_recall(yte, now, p, 0.5),
        "rule_lead_recall": lead_time_recall(yte, now, now.astype(float), 0.5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="TRACTOR forecasting eval (ML vs rule + lead-time)")
    ap.add_argument("--root", required=True, help="Thư mục gốc TRACTOR (chứa logs/)")
    ap.add_argument("--trial", default="Trial0")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.1)
    args = ap.parse_args()
    r = run(args.root, trial=args.trial, lookback=args.lookback, horizon=args.horizon,
            budget=args.budget)
    print(f"TRACTOR forecasting (dự báo {args.horizon} bước ~ {args.horizon * _DT:.1f}s trước)")
    print(f"  test windows={r['n_test']}  pos={r['pos_rate']:.3f}")
    print(f"  PR-AUC:  ML={r['ml_pr_auc']:.3f}  luật={r['rule_pr_auc']:.3f}  "
          f"(ML thắng {r['ml_pr_auc'] - r['rule_pr_auc']:+.3f})")
    print(f"  Lead-time recall (onset n={r['onset_windows']}):  "
          f"ML={r['ml_lead_recall']:.3f}  luật={r['rule_lead_recall']:.3f}")


if __name__ == "__main__":
    main()
