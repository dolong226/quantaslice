"""Load :class:`~quantaslice.core.types.Configuration` từ file YAML, dict,
hoặc biến môi trường.

Package này KHÔNG biết gì về provider/solver cụ thể — nó chỉ tạo ra một
``Configuration`` (dữ liệu thuần), việc tra registry theo tên nằm ở
``pipeline``. Nhờ vậy ``config.py`` không phụ thuộc ``ai``/``quantum``.
"""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any

from quantaslice.core.exceptions import ConfigurationError
from quantaslice.core.runtime import Configuration

__all__ = ["load_configuration", "configuration_from_dict"]

_ENV_PREFIX = "QUANTASLICE_"

# Field nào là kiểu gì, dùng để ép kiểu khi đọc từ env var (luôn là str).
_FIELD_CASTERS: dict[str, Any] = {
    "emergency_threshold": float,
    "qaoa_depth": int,
    "qaoa_shots": int,
    "qaoa_max_iterations": int,
    "window_length": int,
}

_KNOWN_FIELDS = {f.name for f in fields(Configuration)}


def configuration_from_dict(data: dict[str, Any]) -> Configuration:
    """Tạo Configuration từ một dict thuần (đã parse sẵn), validate field
    lạ để tránh lỗi gõ nhầm tên (typo) trong YAML bị âm thầm bỏ qua."""
    unknown = set(data) - _KNOWN_FIELDS - {"extra"}
    known_kwargs = {k: v for k, v in data.items() if k in _KNOWN_FIELDS}
    if unknown:
        # Field lạ không raise cứng, mà gom vào `extra` để package con
        # tương lai (dashboard, cli...) có thể đọc cấu hình riêng của họ
        # mà không cần sửa Configuration ở core.
        known_kwargs.setdefault("extra", {})
        known_kwargs["extra"] = {**known_kwargs.get("extra", {}), **{k: data[k] for k in unknown}}
    try:
        return Configuration(**known_kwargs)
    except TypeError as exc:
        raise ConfigurationError(f"Cấu hình không hợp lệ: {exc}") from exc


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml  # import trễ: yaml là optional dependency
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError(
            "Cần cài 'pyyaml' để đọc file config YAML: pip install pyyaml"
        ) from exc

    if not path.exists():
        raise ConfigurationError(f"Không tìm thấy file cấu hình: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Nội dung file cấu hình '{path}' phải là một mapping (dict), nhận {type(data)}"
        )
    return data


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Cho phép override từng field bằng biến môi trường, ví dụ
    ``QUANTASLICE_SOLVER=classical_greedy`` override ``solver`` trong YAML.
    Hữu ích khi chạy demo/CI mà không muốn sửa file YAML."""
    result = dict(data)
    for name in _KNOWN_FIELDS:
        env_key = f"{_ENV_PREFIX}{name.upper()}"
        if env_key in os.environ:
            raw_value = os.environ[env_key]
            caster = _FIELD_CASTERS.get(name, str)
            try:
                result[name] = caster(raw_value)
            except ValueError as exc:
                raise ConfigurationError(
                    f"Không thể ép kiểu biến môi trường {env_key}={raw_value!r}"
                ) from exc
    return result


def load_configuration(path: str | Path | None = None) -> Configuration:
    """Load Configuration từ file YAML (nếu có) rồi áp override từ biến
    môi trường ``QUANTASLICE_*``.

    Nếu ``path`` là None, trả về Configuration mặc định (chỉ áp dụng env
    override) — hữu ích cho demo nhanh không cần file config.
    """
    data: dict[str, Any] = _load_yaml_file(Path(path)) if path is not None else {}
    data = _apply_env_overrides(data)
    return configuration_from_dict(data)
