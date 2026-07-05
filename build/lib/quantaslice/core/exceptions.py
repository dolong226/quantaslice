"""Exception hierarchy dùng chung cho toàn bộ hệ thống QuantaSlice.

Mọi package con (ai, quantum, orchestrator, simulation, pipeline...) nên
raise các exception định nghĩa ở đây (hoặc subclass của chúng) thay vì
raise Exception trần, để pipeline có thể bắt lỗi theo domain một cách
nhất quán.
"""

from __future__ import annotations


class QuantaSliceError(Exception):
    """Lớp gốc cho mọi lỗi phát sinh trong hệ thống QuantaSlice.

    Bất kỳ package nào bắt lỗi tổng quát của hệ thống chỉ cần
    ``except QuantaSliceError`` là đủ, không cần biết lỗi đến từ
    module con nào.
    """


class ConfigurationError(QuantaSliceError):
    """Cấu hình không hợp lệ (thiếu field bắt buộc, giá trị sai kiểu,
    file config không tồn tại, v.v.)."""


class ProviderNotFoundError(QuantaSliceError):
    """Không tìm thấy provider/solver/orchestrator đã đăng ký ứng với
    tên được yêu cầu trong :class:`~quantaslice.core.registry.Registry`.
    """

    def __init__(self, name: str, category: str, available: tuple[str, ...]) -> None:
        self.name = name
        self.category = category
        self.available = available
        super().__init__(
            f"Không tìm thấy '{name}' trong danh mục '{category}'. "
            f"Các tên đã đăng ký: {', '.join(available) or '(rỗng)'}"
        )


class DuplicateRegistrationError(QuantaSliceError):
    """Cố gắng đăng ký hai implementation cùng tên trong cùng một
    :class:`~quantaslice.core.registry.Registry`."""

    def __init__(self, name: str, category: str) -> None:
        self.name = name
        self.category = category
        super().__init__(
            f"'{name}' đã được đăng ký trước đó trong danh mục '{category}'."
        )


class SchemaValidationError(QuantaSliceError):
    """Dữ liệu không khớp với contract (dataclass) mong đợi — ví dụ
    priority vector có giá trị âm, ma trận QUBO không đối xứng, v.v."""


class InfeasibleAllocationError(QuantaSliceError):
    """Bộ giải (QAOA hoặc classical fallback) không tìm được một phân bổ
    khả thi nào (mọi ràng buộc capacity/uniqueness đều bị vi phạm sau khi
    giải mã bitstring tốt nhất)."""


class SolverError(QuantaSliceError):
    """Lỗi phát sinh trong quá trình chạy solver lượng tử/cổ điển (ví dụ
    lỗi backend Aer, lỗi kết nối IBM Quantum, timeout)."""
