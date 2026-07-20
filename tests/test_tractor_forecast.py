"""Test forecasting trên TRACTOR (traffic 5G thật, động) — bằng chứng
giá trị ML: dự báo vi phạm SLA sớm hơn luật tầm thường (lead-time).

Skip nếu chưa tải dữ liệu TRACTOR vào ``tractor-dataset/`` (không commit).
Tải: các thư mục embb*/mmtc*/urllc* từ genesys-neu/TRACTOR (logs/)."""

from __future__ import annotations

import os

import pytest

_HERE = os.path.dirname(__file__)
# Ưu tiên full data (tractor-repo, sparse clone của genesys-neu/TRACTOR),
# lùi về subset tractor-dataset. Cả hai đều gitignored.
_CANDIDATES = [
    os.path.join(_HERE, "..", "tractor-repo"),
    os.path.join(_HERE, "..", "tractor-dataset"),
]


def _root() -> str | None:
    from quantaslice.ai.data.tractor_loader import iter_tractor_streams
    for root in _CANDIDATES:
        try:
            if len(list(iter_tractor_streams(root))) >= 2:
                return root
        except Exception:
            continue
    return None


_ROOT = _root()


@pytest.mark.skipif(_ROOT is None, reason="Chưa tải dữ liệu TRACTOR (tractor-repo/ hoặc tractor-dataset/)")
def test_ml_beats_rule_and_buys_lead_time():
    from examples.run_tractor_forecast import run

    r = run(_ROOT, horizon=8, budget=0.1)
    # Trên dữ liệu ĐỘNG thật, ML forecasting phải hơn luật "đang vi phạm chưa".
    assert r["ml_pr_auc"] > r["rule_pr_auc"]
    # Và bắt SỚM được các vi phạm sắp xảy ra mà luật (theo định nghĩa) bỏ lỡ.
    assert r["onset_windows"] > 0
    assert r["ml_lead_recall"] > 0.0
    assert r["rule_lead_recall"] == 0.0
