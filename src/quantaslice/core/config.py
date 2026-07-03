from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any

from quantaslice.core.exceptions import ConfigurationError
from quantaslice.core.runtime import Configuration

__all__ = ["load_configuration", "configuration_from_dict"]

_ENV_PREFIX = "QUANTASLICE_"

_FIELD_CASTERS: dict[str, Any] = {
    "emergency_threshold": float,
    "qaoa_depth": int,
    "qaoa_shots": int,
    "qaoa_max_iterations": int,
    "window_length": int,
}

_KNOWN_FIELDS = {f.name for f in fields(Configuration)}


def configuration_from_dict(data: dict[str, Any]) -> Configuration:
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
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml  # import trễ: yaml là optional dependency
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError(
            "Need to install 'pyyaml' to read YAML configuration files: pip install pyyaml"
        ) from exc

    if not path.exists():
        raise ConfigurationError(f"Not found configuration file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"The contents of configuration file '{path}' must be a mapping (dict), receiving {type(data)}"
        )
    return data


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
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
                    f"Cannot cast env var {env_key}={raw_value!r}"
                ) from exc
    return result


def load_configuration(path: str | Path | None = None) -> Configuration:
    data: dict[str, Any] = _load_yaml_file(Path(path)) if path is not None else {}
    data = _apply_env_overrides(data)
    return configuration_from_dict(data)
