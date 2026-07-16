"""QAOA solver chạy circuit trên Quapp Functions (functions.quapp.cloud)
— FaaS chuyên thực thi 1 mạch lượng tử/lần, có thể target QPU thật
(IBM, Rigetti...) hoặc simulator qua cùng 1 API.

KHÁC với ``QAOAAerSolver`` ở ĐÚNG 1 CHỖ: nơi circuit được chạy. Toàn bộ
logic xây QUBO -> Ising -> mạch QAOA -> vòng lặp COBYLA giữ NGUYÊN, tái
sử dụng trực tiếp từ ``qaoa_aer_solver`` — minh chứng giá trị thật của
``OptimizationSolver`` Protocol: đổi "ai chạy circuit" (Aer cục bộ hay
Quapp remote) mà không đổi 1 dòng thuật toán.

TRẠNG THÁI: CHƯA TEST được (chưa có tài khoản/API key Quapp lúc code
phần này). ``_invoke_quapp_function()`` là nơi DUY NHẤT cần điền lệnh
gọi API/SDK thật — mọi phần còn lại (xây mạch, decode, warm-start) đã
verify đúng qua ``QAOAAerSolver``.

CẢNH BÁO QUAN TRỌNG: mỗi lần invoke Quapp = 1 lần chạy circuit (theo
tài liệu chính thức). Vòng lặp COBYLA gọi lại NHIỀU LẦN mỗi lần
``solve()`` — nếu target QPU thật (hàng đợi phút-giờ), một lần
``solve()`` có thể tốn cực nhiều thời gian (số vòng lặp × hàng đợi mỗi
vòng). KHÔNG dùng solver này trong ``Runner.run_forever()`` (vòng lặp
reactive real-time) — chỉ dùng cho benchmark một lần hoặc demo "đã chạy
trên QPU thật" ngoài luồng chính, khớp đúng ghi nhận "Hardware Queue
Latency" trong Limitations.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np

from quantaslice.core.exceptions import InfeasibleAllocationError, SolverError
from quantaslice.core.runtime import Configuration
from quantaslice.core.types import AllocationProblem, OptimizationResult
from quantaslice.quantum.decoding import compute_achieved_objective, decode_bitstring
from quantaslice.quantum.hamiltonian.ising import qubo_to_ising
from quantaslice.quantum.qubo.builder import QUBOBuilder
from quantaslice.quantum.solvers.qaoa_aer_solver import (
    _expectation_and_track_best,
    _make_circuit_builder,
)

try:
    from qiskit.qasm2 import dumps as _qasm2_dumps
    from scipy.optimize import minimize

    _DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DEPS_AVAILABLE = False

__all__ = ["QAOAQuappSolver"]

_INSTALL_HINT = "qiskit / scipy chưa được cài. Cài bằng: pip install 'quantaslice[quantum]'"


class QAOAQuappSolver:
    """Xem docstring module. Đọc ``function_id``/``api_key`` từ tham số
    constructor hoặc biến môi trường ``QUAPP_FUNCTION_ID``/``QUAPP_API_KEY``."""

    def __init__(
        self,
        config: Configuration | None = None,
        *,
        function_id: str | None = None,
        api_key: str | None = None,
        provider: str = "quapp",
        device: str | None = None,
    ) -> None:
        if not _DEPS_AVAILABLE:
            raise SolverError(_INSTALL_HINT)

        self._config = config or Configuration()
        self._function_id = function_id or os.environ.get("QUAPP_FUNCTION_ID")
        self._api_key = api_key or os.environ.get("QUAPP_API_KEY")
        self._provider = provider
        self._device = device or os.environ.get("QUAPP_DEVICE", "simulator")
        self._warm_start_cache: dict[tuple[int, int], np.ndarray] = {}

        if not self._function_id or not self._api_key:
            # Fail-fast ngay lúc khởi tạo, đúng pattern đã áp dụng cho
            # QAOAAerSolver — báo lỗi rõ ràng ngay tại DependencyContainer
            # .build_runner(), không đợi tới solve() đầu tiên.
            raise SolverError(
                "Thiếu function_id/api_key cho Quapp. Đặt biến môi trường "
                "QUAPP_FUNCTION_ID, QUAPP_API_KEY, hoặc truyền trực tiếp vào "
                "constructor. Xem examples/quapp_function/handler.py để "
                "deploy function trước khi dùng solver này."
            )

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
            qasm_str = _qasm2_dumps(circuit)
            counts = self._invoke_quapp_function(qasm_str, shots)
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
            raise InfeasibleAllocationError("QAOA (Quapp) không sinh được bất kỳ mẫu đo nào.")

        result = decode_bitstring(
            bitstring=best["bitstring"],
            qubo=qubo,
            problem=problem,
            solver_name="qaoa_quapp",
            metadata={
                "qaoa_depth": depth,
                "qaoa_shots": shots,
                "provider": self._provider,
                "device": self._device,
            },
        )
        objective = compute_achieved_objective(problem, result)
        return dataclasses.replace(result, objective_value=objective)

    def _invoke_quapp_function(self, qasm_str: str, shots: int) -> dict[str, int]:
        """CHỖ DUY NHẤT cần điền lệnh gọi API/SDK thật của Quapp.

        Theo tài liệu chính thức (docs.quapp.cloud/developer-
        documentation/quantum-function/invoke-function/), 1 lần invoke
        cần: ``function_id``, ``api_key``, ``provider``, ``device``,
        ``shots``, và input payload dạng Raw JSON — ở đây là
        ``{"qasm": qasm_str}``, khớp với ``handler.py`` mẫu ở
        ``examples/quapp_function/handler.py``.

        Response (sau khi qua ``post_processing()`` của handler.py) nên
        là dict counts, ví dụ ``{"0101": 12, "1100": 44, ...}`` — đúng
        định dạng mà ``_expectation_and_track_best()`` cần.

        Raises:
            NotImplementedError: luôn luôn, cho tới khi anh điền lệnh
                gọi thật (chưa test được vì chưa có tài khoản Quapp).
        """
        raise NotImplementedError(
            "Chưa điền lệnh gọi API Quapp thật ở _invoke_quapp_function(). "
            "Tham khảo SDK/API docs của Quapp lúc anh có tài khoản, dùng "
            f"function_id={self._function_id!r}, provider={self._provider!r}, "
            f"device={self._device!r}, shots={shots}."
        )
