"""QAOA (Quantum Approximate Optimization Algorithm) chạy trên Aer
simulator — solver mặc định của QuantaSlice hiện tại (MVP), theo mục 4
tài liệu "QUBO Formulation & QAOA".

Import Qiskit/Aer/SciPy được trì hoãn (lazy) trong ``try/except`` ở đầu
module: package ``quantum`` vẫn import được (và
``ClassicalGreedySolver`` vẫn chạy được) trên máy chưa cài extras
``quantaslice[quantum]`` — chỉ khi thực sự gọi ``QAOAAerSolver.solve()``
mới cần các thư viện lượng tử.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from quantaslice.core.exceptions import InfeasibleAllocationError, SolverError
from quantaslice.core.runtime import Configuration
from quantaslice.core.types import AllocationProblem, OptimizationResult
from quantaslice.quantum.decoding import (
    compute_achieved_objective,
    decode_bitstring,
    evaluate_qubo_objective,
)
from quantaslice.quantum.hamiltonian.ising import qubo_to_ising
from quantaslice.quantum.qubo.builder import QUBOBuilder

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from scipy.optimize import minimize

    _QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - phụ thuộc môi trường cài đặt
    _QISKIT_AVAILABLE = False

__all__ = ["QAOAAerSolver"]

_INSTALL_HINT = (
    "qiskit / qiskit-aer / scipy chưa được cài. Cài bằng: "
    "pip install 'quantaslice[quantum]'"
)


class QAOAAerSolver:
    """Implement :class:`~quantaslice.core.protocols.OptimizationSolver`.

    Pipeline nội bộ (ẩn hoàn toàn với caller):
    ``AllocationProblem -> QUBOProblem -> Ising (h, J) -> mạch QAOA
    độ sâu p -> vòng lặp COBYLA -> bitstring tốt nhất -> OptimizationResult``.

    WARM-START (mục "Warm-Start Transferability" trong slide): lưu lại
    ``(gamma*, beta*)`` hội tụ của lần giải TRƯỚC (theo cùng kích thước
    bài toán ``n_qubits`` và cùng ``qaoa_depth``), dùng làm điểm khởi tạo
    cho lần giải SAU thay vì random uniform mỗi lần. Vì các lần
    re-optimize liên tiếp trong 1 phiên chạy thường có topology/priority
    biến động nhẹ quanh cùng một bài toán, tham số hội tụ trước đó là
    điểm khởi tạo tốt hơn nhiều so với random — giảm số vòng COBYLA cần
    thiết và tăng ổn định chất lượng nghiệm giữa các lần gọi liên tiếp
    (khắc phục vấn đề observed: mỗi lần random init cho ra kết quả không
    liên quan gì tới lần trước, dù cùng 1 bài toán gần giống hệt).

    LƯU Ý: vì có state (``_warm_start_cache``), MỖI instance chỉ nên
    dùng cho 1 Runner tại 1 thời điểm — giống ``ThresholdPredictionProvider``.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        if not _QISKIT_AVAILABLE:
            # Fail-fast ngay lúc khởi tạo (không đợi tới solve() đầu
            # tiên) — để DependencyContainer.build_runner() báo lỗi rõ
            # ràng NGAY tại thời điểm wiring, thay vì bị Runner._reoptimize
            # nuốt lỗi trong vòng lặp reactive rồi im lặng giữ allocation
            # None mãi (rất khó debug, xem lịch sử: log "(chưa có)" suốt
            # kèm thời gian chạy 0.02s là dấu hiệu chính xác của lỗi này).
            raise SolverError(_INSTALL_HINT)
        self._config = config or Configuration()
        self._warm_start_cache: dict[tuple[int, int], np.ndarray] = {}

    def solve(self, problem: AllocationProblem) -> OptimizationResult:
        qubo = QUBOBuilder().build(problem)
        h, j_terms, _offset = qubo_to_ising(qubo)
        n = qubo.n_qubits
        depth = self._config.qaoa_depth
        shots = self._config.qaoa_shots
        cache_key = (n, depth)

        build_circuit = _make_circuit_builder(n, depth, h, j_terms)
        best = {"bitstring": None, "value": float("inf")}

        def cost_fn(params: np.ndarray) -> float:
            gammas, betas = params[:depth], params[depth:]
            circuit = build_circuit(gammas, betas)
            counts = _run_circuit(circuit, shots)
            return _expectation_and_track_best(counts, qubo.q_matrix, best)

        cached_params = self._warm_start_cache.get(cache_key)
        initial_params = (
            cached_params if cached_params is not None else np.random.uniform(0, np.pi, size=2 * depth)
        )
        opt_result = minimize(
            cost_fn,
            initial_params,
            method="COBYLA",
            options={"maxiter": self._config.qaoa_max_iterations},
        )
        self._warm_start_cache[cache_key] = opt_result.x

        if best["bitstring"] is None:
            raise InfeasibleAllocationError(
                "QAOA không sinh được bất kỳ mẫu đo nào (0 shots hợp lệ)."
            )

        result = decode_bitstring(
            bitstring=best["bitstring"],
            qubo=qubo,
            problem=problem,
            solver_name="qaoa_aer",
            metadata={
                "qaoa_depth": depth,
                "qaoa_shots": shots,
                "classical_iterations": int(getattr(opt_result, "nfev", 0)),
                "best_qubo_value": best["value"],
                "warm_started": cached_params is not None,
            },
        )
        objective = compute_achieved_objective(problem, result)
        return dataclasses.replace(result, objective_value=objective)


def _make_circuit_builder(n: int, depth: int, h: dict[int, float], j_terms: dict[tuple[int, int], float]):
    """Trả về hàm ``(gammas, betas) -> QuantumCircuit`` dựng mạch QAOA
    độ sâu ``depth`` theo đúng cấu trúc mục 4.3 tài liệu QAOA: Hadamard
    khởi tạo superposition, rồi luân phiên phase separator U_C(gamma)
    (cổng RZ cho h_k, cổng RZZ cho J_kl) và mixer U_B(beta) (RX toàn bộ
    qubit)."""

    def _build(gammas: np.ndarray, betas: np.ndarray) -> "QuantumCircuit":
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        for layer in range(depth):
            gamma = gammas[layer]
            for k, hk in h.items():
                qc.rz(2 * gamma * hk, k)
            for (k, l), jkl in j_terms.items():
                qc.cx(k, l)
                qc.rz(2 * gamma * jkl, l)
                qc.cx(k, l)
            beta = betas[layer]
            qc.rx(2 * beta, range(n))
        qc.measure(range(n), range(n))
        return qc

    return _build


def _run_circuit(circuit: "QuantumCircuit", shots: int) -> dict[str, int]:
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    return job.result().get_counts()


def _expectation_and_track_best(
    counts: dict[str, int], q_matrix: np.ndarray, best: dict
) -> float:
    """Ước lượng <H_C> từ phân phối shots đo được, đồng thời cập nhật
    bitstring tốt nhất từng thấy qua mọi vòng lặp (mục 4.4 bước 5: "trả
    về x* có f(x*) nhỏ nhất - không nhất thiết là mẫu của lần đo cuối
    cùng")."""
    total_shots = sum(counts.values())
    expectation = 0.0
    for bitstring, freq in counts.items():
        # Qiskit trả bitstring little-endian (qubit 0 ở cuối chuỗi).
        x = np.array([int(bit) for bit in bitstring[::-1]])
        value = evaluate_qubo_objective(q_matrix, x)
        probability = freq / total_shots
        expectation += probability * value
        if value < best["value"]:
            best["value"] = value
            best["bitstring"] = x
    return expectation
