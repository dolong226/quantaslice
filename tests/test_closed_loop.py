"""Test closed-loop QUBO eval: chính sách adaptive (p từ detector) phải
tốt hơn tĩnh và không tệ hơn oracle — bằng chứng giá trị downstream (§6.4).

Đây là test tích hợp cross-cutting (ai + quantum), nên đặt ở tests/ gốc
thay vì tests/ai/."""

from __future__ import annotations

from examples.run_closed_loop import run_comparison


def test_adaptive_beats_static_and_bounded_by_oracle():
    res = run_comparison(n_steps=300, solver_name="classical_greedy")
    static, adaptive, oracle = res["static"], res["adaptive"], res["oracle"]

    # Adaptive giảm chi phí vi phạm SLA so với chính sách tĩnh.
    assert adaptive.sla_cost < static.sla_cost
    # Oracle là trần: adaptive không thể tốt hơn oracle.
    assert oracle.sla_cost <= adaptive.sla_cost
    # Adaptive phục vụ slice nguy cấp tốt hơn tĩnh.
    assert adaptive.critical_served_rate > static.critical_served_rate
    # Oracle phục vụ (gần) trọn vẹn slice nguy cấp.
    assert oracle.critical_served_rate >= adaptive.critical_served_rate
