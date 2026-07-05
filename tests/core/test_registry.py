"""Unit test cho quantaslice.core.registry — plugin/registry pattern."""

from __future__ import annotations

import pytest

from quantaslice.core.exceptions import DuplicateRegistrationError, ProviderNotFoundError
from quantaslice.core.registry import Registry


class DummySolver:
    def solve(self, problem):  # pragma: no cover - chỉ để test registry, không test logic
        return "solved"


def test_register_and_create_via_decorator():
    registry: Registry[DummySolver] = Registry("solver")

    @registry.register("dummy")
    class MySolver(DummySolver):
        pass

    instance = registry.create("dummy")
    assert isinstance(instance, MySolver)


def test_duplicate_registration_raises():
    registry: Registry[DummySolver] = Registry("solver")
    registry.register("dummy")(DummySolver)
    with pytest.raises(DuplicateRegistrationError):
        registry.register("dummy")(DummySolver)


def test_create_unknown_name_raises_with_available_list():
    registry: Registry[DummySolver] = Registry("solver")
    registry.register("qaoa_aer")(DummySolver)
    with pytest.raises(ProviderNotFoundError) as exc_info:
        registry.create("does_not_exist")
    assert "qaoa_aer" in exc_info.value.available


def test_register_instance_factory_with_closure():
    registry: Registry[DummySolver] = Registry("solver")

    def factory() -> DummySolver:
        solver = DummySolver()
        return solver

    registry.register_instance_factory("closured", factory)
    assert registry.create("closured") is not None


def test_available_and_contains():
    registry: Registry[DummySolver] = Registry("solver")
    registry.register("a")(DummySolver)
    registry.register("b")(DummySolver)
    assert set(registry.available()) == {"a", "b"}
    assert "a" in registry
    assert "z" not in registry
