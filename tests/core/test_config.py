"""Unit test cho quantaslice.core.config — load YAML + env override."""

from __future__ import annotations

import os

import pytest

from quantaslice.core.config import configuration_from_dict, load_configuration
from quantaslice.core.exceptions import ConfigurationError


def test_configuration_from_empty_dict_uses_defaults():
    cfg = configuration_from_dict({})
    assert cfg.prediction_provider == "mock"


def test_configuration_from_dict_overrides_fields():
    cfg = configuration_from_dict({"solver": "classical_greedy", "qaoa_depth": 3})
    assert cfg.solver == "classical_greedy"
    assert cfg.qaoa_depth == 3


def test_unknown_field_goes_to_extra_not_raise():
    cfg = configuration_from_dict({"dashboard_theme": "dark"})
    assert cfg.extra["dashboard_theme"] == "dark"


def test_load_configuration_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigurationError):
        load_configuration(missing)


def test_load_configuration_from_yaml_file(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("solver: classical_greedy\nqaoa_shots: 2048\n", encoding="utf-8")
    cfg = load_configuration(yaml_path)
    assert cfg.solver == "classical_greedy"
    assert cfg.qaoa_shots == 2048


def test_env_override_takes_precedence(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("solver: qaoa_aer\n", encoding="utf-8")
    monkeypatch.setenv("QUANTASLICE_SOLVER", "qaoa_ibmq")
    cfg = load_configuration(yaml_path)
    assert cfg.solver == "qaoa_ibmq"


def test_env_override_casts_numeric_types(monkeypatch):
    monkeypatch.setenv("QUANTASLICE_QAOA_DEPTH", "4")
    monkeypatch.setenv("QUANTASLICE_EMERGENCY_THRESHOLD", "0.7")
    cfg = load_configuration(None)
    assert cfg.qaoa_depth == 4
    assert cfg.emergency_threshold == pytest.approx(0.7)
