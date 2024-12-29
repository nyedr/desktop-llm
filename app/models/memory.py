"""Memory-related models and enums."""
from enum import Enum


class MemoryType(str, Enum):
    """Memory type enum for consistent collection naming."""
    EPHEMERAL = "ephemeral"
    MODEL_MEMORY = "model_memory"
