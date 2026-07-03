from __future__ import annotations


class QuantaSliceError(Exception):
    """
    """


class ConfigurationError(QuantaSliceError):
    """
    """


class ProviderNotFoundError(QuantaSliceError):
    """
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
    """
    """

    def __init__(self, name: str, category: str) -> None:
        self.name = name
        self.category = category
        super().__init__(
            f"'{name}' đã được đăng ký trước đó trong danh mục '{category}'."
        )


class SchemaValidationError(QuantaSliceError):
    """
    """


class InfeasibleAllocationError(QuantaSliceError):
    """
    """


class SolverError(QuantaSliceError):
    """
    """
