"""Dữ liệu cho ML detector: loader ColO-RAN, nhãn QoS-violation, feature
engineering + windowing, block-split (§1–§3 plan ML)."""

from quantaslice.ai.data.features import WindowConfig, build_sequences, build_tabular
from quantaslice.ai.data.labeling import LabelConfig, label_frame
from quantaslice.ai.data.loaders import BSFrame, generate_synthetic, iter_frames
from quantaslice.ai.data.scenario import (
    EmergencyEvent,
    generate_emergency_scenario,
    inject_emergencies,
)
from quantaslice.ai.data.split import leave_config_out, leave_scheduler_out

__all__ = [
    "BSFrame",
    "iter_frames",
    "generate_synthetic",
    "LabelConfig",
    "label_frame",
    "WindowConfig",
    "build_tabular",
    "build_sequences",
    "leave_scheduler_out",
    "leave_config_out",
    "EmergencyEvent",
    "inject_emergencies",
    "generate_emergency_scenario",
]
