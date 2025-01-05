"""Model data models."""

from pydantic import BaseModel
from typing import Optional
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
    modified_at: str
    size: int
    details: ModelDetails
    digest: Optional[str] = None
