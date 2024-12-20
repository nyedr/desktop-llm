"""Configuration for stream processing."""

from dataclasses import dataclass

@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    buffer_size: int = 1024
    chunk_size: int = 256
    max_wait_time: float = 0.1
    flow_control_threshold: float = 0.8
