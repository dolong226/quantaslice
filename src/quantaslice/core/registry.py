from __future__ import annotations

from typing import Callable, Generic, TypeVar

from quantaslice.core.exceptions import DuplicateRegistrationError, ProviderNotFoundError

__all__ = ["Registry"]

T = TypeVar("T")

Factory = Callable[[], T]


class Registry(Generic[T]):
    def __init__(self, category: str) -> None:
        """``category`` chỉ dùng để log/thông báo lỗi rõ ràng hơn, ví dụ
        'prediction_provider', 'solver', 'orchestrator'."""
        self._category = category
        self._factories: dict[str, Factory[T]] = {}

    def register(self, name: str) -> Callable[[Factory[T]], Factory[T]]:
        """Decorator đăng ký một class/factory dưới một tên.
        """

        def _decorator(factory: Factory[T]) -> Factory[T]:
            if name in self._factories:
                raise DuplicateRegistrationError(name, self._category)
            self._factories[name] = factory
            return factory

        return _decorator

    def register_instance_factory(self, name: str, factory: Factory[T]) -> None:
        """Đăng ký trực tiếp (không dùng decorator) — tiện cho các factory
        được tạo động (vd closure bao tham số cấu hình)."""
        if name in self._factories:
            raise DuplicateRegistrationError(name, self._category)
        self._factories[name] = factory

    def create(self, name: str) -> T:
        """Tạo một instance mới ứng với tên đã đăng ký.
        """
        if name not in self._factories:
            raise ProviderNotFoundError(name, self._category, tuple(self._factories))
        return self._factories[name]()

    def is_registered(self, name: str) -> bool:
        return name in self._factories

    def available(self) -> tuple[str, ...]:
        """Danh sách tên đã đăng ký, hữu ích cho CLI/dashboard liệt kê
        các lựa chọn khả dụng."""
        return tuple(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def __repr__(self) -> str:  # pragma: no cover - tiện debug
        return f"Registry(category={self._category!r}, registered={self.available()!r})"
