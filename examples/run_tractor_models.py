"""So sánh MÔ HÌNH trên TRACTOR forecasting: luật / LightGBM / LSTM / TCN / TSMixer / ARIMA / PatchTST / iTransformer.

Cùng dữ liệu (cửa sổ KPM per-stream), cùng nhãn (vi phạm SLA H bước tới),
cùng split. Báo PR-AUC + lead-time recall cho từng mô hình.
"""

from __future__ import annotations

import argparse
import warnings
import numpy as np

from quantaslice.ai.data.features import summarize_window
from quantaslice.ai.data.tractor_loader import iter_tractor_streams
from quantaslice.ai.eval.metrics import lead_time_recall
from quantaslice.ai.models.baselines import GradientBoostingDetector
from examples.run_tractor_forecast import _featurize


def _build_seq_with_delay(streams, *, lookback, horizon, budget, stride=3):
    """Cửa sổ chuỗi (N,W,F) + nhãn future-breach + delay tức thời + raw delay history."""
    xs, yf, triv, xs_delay = [], [], [], []
    for _, s in streams:
        feats, delay = _featurize(s)
        breach = (delay >= budget).astype(int)
        for st in range(0, len(delay) - lookback - horizon, stride):
            t = st + lookback - 1
            xs.append(feats[st:st + lookback])
            yf.append(1.0 if breach[t + 1:t + 1 + horizon].any() else 0.0)
            triv.append(delay[t])
            xs_delay.append(delay[st:st + lookback])
    return np.asarray(xs), np.asarray(yf), np.asarray(triv), np.asarray(xs_delay)


def _train_tuned_detector(detector, x_tr, y_tr, *, epochs, lr=1e-3, batch=512, weight_decay=1e-4):
    """Tuned training loop with a validation split (80/20) for early stopping."""
    import torch
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import average_precision_score as AP

    # Split train to train/val internally
    n = len(x_tr)
    n_tr = int(n * 0.8)
    indices = torch.randperm(n)
    tr_idx, val_idx = indices[:n_tr], indices[n_tr:]

    x_train, y_train = x_tr[tr_idx], y_tr[tr_idx]
    x_val, y_val = x_tr[val_idx], y_tr[val_idx]

    n_train = len(x_train)
    detector.scaler = RobustScaler().fit(x_train.reshape(-1, x_train.shape[2]))
    model = detector.model
    
    xf_tr = detector._prep(x_train)
    yt_tr = torch.as_tensor(y_train, dtype=torch.float32)
    xf_val = detector._prep(x_val)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)])
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_ap = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logit, _ = model(xf_tr[idx])
            loss = bce(logit, yt_tr[idx])
            loss.backward()
            opt.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(xf_val)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_ap = AP(y_val, val_probs)

        if val_ap > best_ap:
            best_ap = val_ap
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    detector.model = model.eval()
    return detector


def run(root, *, trial="Trial0", lookback=20, horizon=8, budget=0.1, epochs=20, seed=0):
    import torch
    from sklearn.metrics import average_precision_score as AP
    from quantaslice.ai.models.tcn import MultiTaskTCN, TCNDetector
    from quantaslice.ai.models.lstm import MultiTaskLSTM, LSTMDetector
    from quantaslice.ai.models.tsmixer import MultiTaskTSMixer, TSMixerDetector
    from quantaslice.ai.models.arima import GlobalARDetector, LocalARIMADetector
    from quantaslice.ai.models.patchtst import MultiTaskPatchTST, PatchTSTDetector
    from quantaslice.ai.models.itransformer import MultiTaskiTransformer, iTransformerDetector

    torch.manual_seed(seed)
    np.random.seed(seed)

    streams = list(iter_tractor_streams(root, trial))
    if len(streams) < 2:
        raise SystemExit(f"Cần ≥2 luồng TRACTOR ở {root}/{trial}; thấy {len(streams)}.")

    def sidx(name):
        d = name[len(name.rstrip("0123456789")):]
        return int(d) if d else 0

    train = [s for s in streams if sidx(s[0]) % 2 == 1] or streams[:len(streams) // 2]
    test = [s for s in streams if sidx(s[0]) % 2 == 0] or streams[len(streams) // 2:]

    xtr, ytr, _, xdelay_tr = _build_seq_with_delay(train, lookback=lookback, horizon=horizon, budget=budget)
    xte, yte, triv, xdelay_te = _build_seq_with_delay(test, lookback=lookback, horizon=horizon, budget=budget)
    now = triv >= budget
    f = xtr.shape[2]
    valid = 0 < yte.sum() < len(yte)

    results = {}

    def record(name, probs):
        results[name] = {
            "pr_auc": float(AP(yte, probs)) if valid else float("nan"),
            "lead_recall": lead_time_recall(yte, now, probs, 0.5),
        }

    # 1. Luật
    record("rule", triv / budget)

    # 2. Global AR
    ar = GlobalARDetector(lookback=lookback, horizon=horizon, budget=budget)
    ar.fit(xtr, ytr)
    probs_ar, _ = ar.predict_windows(xte)
    record("global_ar", probs_ar)

    # 3. Local ARIMA (on 500 windows for performance)
    print("Evaluating Local ARIMA on 500 sample windows...")
    n_samples = 500
    step = max(1, len(xdelay_te) // n_samples)
    subset_idx = np.arange(0, len(xdelay_te), step)[:n_samples]
    
    arima_preds = []
    for idx in subset_idx:
        hist = xdelay_te[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(hist, order=(1, 1, 0))
                res = model.fit()
                forecast = res.forecast(steps=horizon)
                arima_preds.append(1.0 if (forecast >= budget).any() else 0.0)
        except Exception:
            arima_preds.append(1.0 if hist[-1] >= budget else 0.0)

    arima_preds = np.array(arima_preds)
    yte_sub = yte[subset_idx]
    now_sub = now[subset_idx]
    results["arima"] = {
        "pr_auc": float(AP(yte_sub, arima_preds)) if (0 < yte_sub.sum() < len(yte_sub)) else float("nan"),
        "lead_recall": lead_time_recall(yte_sub, now_sub, arima_preds, 0.5),
    }

    # 4. LightGBM (tabular summary)
    xtr_tab = np.stack([summarize_window(w) for w in xtr])
    xte_tab = np.stack([summarize_window(w) for w in xte])
    gb = GradientBoostingDetector().fit(xtr_tab, ytr, np.ones((len(ytr), 3)))
    record("lightgbm", gb.predict_proba(xte_tab))

    # 5. Tuned LSTM
    lstm = _train_tuned_detector(
        LSTMDetector(MultiTaskLSTM(f, hidden=64, layers=2, dropout=0.2), lookback=lookback),
        xtr, ytr, epochs=epochs
    )
    record("lstm", lstm.predict_windows(xte)[0])

    # 6. Tuned TCN
    tcn = _train_tuned_detector(
        TCNDetector(MultiTaskTCN(f, channels=(32, 64, 64), kernel=3, dropout=0.2), lookback=lookback),
        xtr, ytr, epochs=epochs
    )
    record("tcn", tcn.predict_windows(xte)[0])

    # 7. TSMixer
    tsm = _train_tuned_detector(
        TSMixerDetector(MultiTaskTSMixer(f, seq_len=lookback, hidden_dim=64, num_blocks=3, dropout=0.1, pooling="mean"), lookback=lookback),
        xtr, ytr, epochs=epochs
    )
    record("tsmixer", tsm.predict_windows(xte)[0])

    # 8. PatchTST
    patchtst = _train_tuned_detector(
        PatchTSTDetector(MultiTaskPatchTST(f, seq_len=lookback, patch_len=8, stride=4, hidden_dim=64, num_heads=4, num_layers=2, pooling="mean"), lookback=lookback),
        xtr, ytr, epochs=epochs
    )
    record("patchtst", patchtst.predict_windows(xte)[0])

    # 9. iTransformer
    itransformer = _train_tuned_detector(
        iTransformerDetector(MultiTaskiTransformer(f, seq_len=lookback, hidden_dim=64, num_heads=4, num_layers=2, pooling="mean"), lookback=lookback),
        xtr, ytr, epochs=epochs
    )
    record("itransformer", itransformer.predict_windows(xte)[0])

    results["_meta"] = {"n_test": int(len(yte)), "pos_rate": float(yte.mean()),
                        "onset": int(((yte == 1) & (~now)).sum()),
                        "arima_n_test": len(yte_sub), "arima_onset": int(((yte_sub == 1) & (~now_sub)).sum())}
    return results


def main():
    ap = argparse.ArgumentParser(description="TRACTOR model comparison (rule/LightGBM/LSTM/TCN/TSMixer/ARIMA/PatchTST/iTransformer)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--trial", default="Trial0")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    
    r = run(a.root, trial=a.trial, lookback=a.lookback, horizon=a.horizon,
            budget=a.budget, epochs=a.epochs)
    m = r.pop("_meta")
    
    print(f"\nTRACTOR forecasting (~{a.horizon * 0.25:.1f}s ahead) | test={m['n_test']} "
          f"pos={m['pos_rate']:.3f} onset={m['onset']}")
    print(f"{'model':<15}{'PR-AUC':>10}{'lead-recall':>14}")
    print("-" * 42)
    for name in ("rule", "global_ar", "lightgbm", "lstm", "tcn", "tsmixer", "patchtst", "itransformer"):
        d = r[name]
        print(f"{name:<15}{d['pr_auc']:>10.3f}{d['lead_recall']:>14.3f}")
        
    d_arima = r["arima"]
    print(f"\nLocal ARIMA (evaluated on {m['arima_n_test']} windows, onset n={m['arima_onset']}):")
    print(f"{'arima':<15}{d_arima['pr_auc']:>10.3f}{d_arima['lead_recall']:>14.3f}")


if __name__ == "__main__":
    main()
