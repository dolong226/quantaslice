"""Template ``handler.py`` để upload lên Quapp Functions (runtime template
"qiskit") — KHÔNG chạy trong repo này, chỉ dùng để copy-paste vào function
editor của Quapp (hoặc upload qua SDK/API của họ).

Theo đúng hợp đồng bắt buộc của Quapp (xem docs.quapp.cloud/developer-
documentation/quantum-function/create-function/):

    processing(invocation_input) -> QuantumCircuit   # chỉ DỰNG mạch
    post_processing(job_result)  -> serializable      # biến đổi kết quả

``invocation_input`` ở đây là 1 chuỗi QASM (OpenQASM 2.0) do
``QAOAQuappSolver`` (phía quantaslice) sinh ra từ mạch QAOA đã build cục
bộ — cách này tránh phải viết lại toàn bộ logic xây mạch QAOA (h, J,
depth, gamma/beta) ở phía Quapp, chỉ cần Quapp "chạy hộ" mạch đã dựng
sẵn và trả counts về.
"""

from qiskit import QuantumCircuit


def processing(invocation_input):
    """``invocation_input`` là dict {"qasm": "<chuỗi QASM>"} (Raw Input,
    format JSON) — xem mục 4.4 tài liệu Invoke function."""
    qasm_str = invocation_input["qasm"]
    circuit = QuantumCircuit.from_qasm_str(qasm_str)
    # MANDATORY: trả về QuantumCircuit, KHÔNG tự chạy/submit job ở đây.
    return circuit


def post_processing(job_result):
    """Biến kết quả thô (backend-specific) thành dict counts thuần —
    khớp định dạng mà ``QAOAQuappSolver._invoke_quapp_function()`` phía
    quantaslice mong đợi."""
    counts = job_result.get_counts()
    return dict(counts)
