"""Memory-related models and enums."""
from enum import Enum


class MemoryType(str, Enum):
    """Memory type enum for consistent collection naming.

    EPHEMERAL: Short-term memory for conversation context and temporary storage
    MODEL_MEMORY: Long-term memory for model knowledge and persistent storage
    """
    EPHEMERAL = "ephemeral"
    MODEL_MEMORY = "model_memory"

    @classmethod
    def get_default(cls) -> "MemoryType":
        """Get the default memory type."""
        return cls.EPHEMERAL

    def is_persistent(self) -> bool:
        """Check if this memory type is persistent."""
        return self == self.MODEL_MEMORY
