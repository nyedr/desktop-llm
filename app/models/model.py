"""Model data models."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class ModelDetails(BaseModel):
    """Details about a model."""
    parent_model: str = ""
    format: str = ""
    family: str = ""
    families: List[str] = []
    parameter_size: str = ""
    quantization_level: str = ""


class Model(BaseModel):
    """Model information."""
    model: str
    modified_at: datetime
    digest: str
    size: int
    details: ModelDetails
