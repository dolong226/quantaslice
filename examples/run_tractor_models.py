"""So sánh MÔ HÌNH trên TRACTOR forecasting: luật / LightGBM / LSTM / TCN.

Cùng dữ liệu (cửa sổ KPM per-stream), cùng nhãn (vi phạm SLA H bước tới),
cùng split. Báo PR-AUC + lead-time recall cho từng mô hình. Trả lời câu
"LSTM chạy ra sao so với TCN/LightGBM trên dữ liệu động thật".

Chạy::

    python -m examples.run_tractor_models --root tractor-repo --trial Trial0 --horizon 8 --epochs 12
"""

from __future__ import annotations

import argparse

import numpy as np

from quantaslice.ai.data.features import summarize_window
from quantaslice.ai.data.tractor_loader import iter_tractor_streams
from quantaslice.ai.eval.metrics import lead_time_recall
from quantaslice.ai.models.baselines import GradientBoostingDetector
from examples.run_tractor_forecast import _featurize


def _build_seq(streams, *, lookback, horizon, budget, stride=3):
    """Cửa sổ chuỗi (N,W,F) + nhãn future-breach + delay tức thời (cho luật)."""
    xs, yf, triv = [], [], []
    for _, s in streams:
        feats, delay = _featurize(s)
        breach = (delay >= budget).astype(int)
        for st in range(0, len(delay) - lookback - horizon, stride):
            t = st + lookback - 1
            xs.append(feats[st:st + lookback])
            yf.append(1.0 if breach[t + 1:t + 1 + horizon].any() else 0.0)
            triv.append(delay[t])
    return np.asarray(xs), np.asarray(yf), np.asarray(triv)


def _train_detector(detector, x_tr, y_tr, *, epochs, lr=1e-3, batch=512):
    import torch
    from sklearn.preprocessing import RobustScaler

    n, w, f = x_tr.shape
    detector.scaler = RobustScaler().fit(x_tr.reshape(-1, f))
    model = detector.model
    xf = detector._prep(x_tr)
    yt = torch.as_tensor(y_tr, dtype=torch.float32)
    pos = float(y_tr.sum()); neg = float(len(y_tr) - pos)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)]))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logit, _ = model(xf[idx])
            loss = bce(logit, yt[idx])
            loss.backward()
            opt.step()
    detector.model = model.eval()
    return detector


def run(root, *, trial="Trial0", lookback=20, horizon=8, budget=0.1, epochs=12, seed=0):
    import torch
    from sklearn.metrics import average_precision_score as AP
    from quantaslice.ai.models.tcn import MultiTaskTCN, TCNDetector
    from quantaslice.ai.models.lstm import MultiTaskLSTM, LSTMDetector

    torch.manual_seed(seed)
    streams = list(iter_tractor_streams(root, trial))
    if len(streams) < 2:
        raise SystemExit(f"Cần ≥2 luồng TRACTOR ở {root}/{trial}; thấy {len(streams)}.")

    def sidx(name):
        d = name[len(name.rstrip("0123456789")):]
        return int(d) if d else 0
    train = [s for s in streams if sidx(s[0]) % 2 == 1] or streams[:len(streams) // 2]
    test = [s for s in streams if sidx(s[0]) % 2 == 0] or streams[len(streams) // 2:]

    xtr, ytr, _ = _build_seq(train, lookback=lookback, horizon=horizon, budget=budget)
    xte, yte, triv = _build_seq(test, lookback=lookback, horizon=horizon, budget=budget)
    now = triv >= budget
    f = xtr.shape[2]
    valid = 0 < yte.sum() < len(yte)

    results = {}

    def record(name, probs):
        results[name] = {
            "pr_auc": float(AP(yte, probs)) if valid else float("nan"),
            "lead_recall": lead_time_recall(yte, now, probs, 0.5),
        }

    # Luật
    record("rule", triv / budget)
    # LightGBM (tabular summary)
    xtr_tab = np.stack([summarize_window(w) for w in xtr])
    xte_tab = np.stack([summarize_window(w) for w in xte])
    gb = GradientBoostingDetector().fit(xtr_tab, ytr, np.ones((len(ytr), 3)))
    record("lightgbm", gb.predict_proba(xte_tab))
    # LSTM
    lstm = _train_detector(LSTMDetector(MultiTaskLSTM(f, hidden=48), lookback=lookback),
                           xtr, ytr, epochs=epochs)
    record("lstm", lstm.predict_windows(xte)[0])
    # TCN
    tcn = _train_detector(TCNDetector(MultiTaskTCN(f, channels=(32, 32)), lookback=lookback),
                          xtr, ytr, epochs=epochs)
    record("tcn", tcn.predict_windows(xte)[0])

    results["_meta"] = {"n_test": int(len(yte)), "pos_rate": float(yte.mean()),
                        "onset": int(((yte == 1) & (~now)).sum())}
    return results


def main():
    ap = argparse.ArgumentParser(description="TRACTOR model comparison (rule/LightGBM/LSTM/TCN)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--trial", default="Trial0")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    r = run(a.root, trial=a.trial, lookback=a.lookback, horizon=a.horizon,
            budget=a.budget, epochs=a.epochs)
    m = r.pop("_meta")
    print(f"TRACTOR forecasting (~{a.horizon * 0.25:.1f}s ahead) | test={m['n_test']} "
          f"pos={m['pos_rate']:.3f} onset={m['onset']}")
    print(f"{'model':<12}{'PR-AUC':>10}{'lead-recall':>14}")
    print("-" * 36)
    for name in ("rule", "lightgbm", "lstm", "tcn"):
        d = r[name]
        print(f"{name:<12}{d['pr_auc']:>10.3f}{d['lead_recall']:>14.3f}")


if __name__ == "__main__":
    main()
