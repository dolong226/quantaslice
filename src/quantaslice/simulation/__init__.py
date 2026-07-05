"""``quantaslice.simulation`` — sinh luồng telemetry giả lập (hoặc đọc
dataset thật cùng schema) cho demo/CI, implement
:class:`~quantaslice.core.protocols.TelemetrySource`.

Ràng buộc kiến trúc: package này CHỈ import từ ``quantaslice.core`` —
KHÔNG import ``ai``/``quantum``/``orchestrator``/``pipeline``.
"""

from quantaslice.simulation.dataset_loader import CDRRecord, ItalianTelecomDatasetLoader
from quantaslice.simulation.stream_simulator import StreamSimulator

__all__ = ["CDRRecord", "ItalianTelecomDatasetLoader", "StreamSimulator"]
