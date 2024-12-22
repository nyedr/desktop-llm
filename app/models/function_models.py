from typing import List, Optional
from pydantic import BaseModel, Field


class RegisterFunctionRequest(BaseModel):
    """Request model for registering a function."""
    name: str = Field(..., description="The name of the function")
    module_path: str = Field(
        ..., description="The Python module path where the function is defined")
    function_name: str = Field(...,
                               description="The name of the function class in the module")
    type: str = Field(...,
                      description="The type of function (tool, filter, or pipeline)")
    description: str = Field(...,
                             description="A description of what the function does")
    parameters: dict = Field(...,
                             description="The parameters schema for the function")
    output_schema: Optional[dict] = Field(
        None, description="The output schema for the function")
    enabled: bool = Field(True, description="Whether the function is enabled")
    dependencies: List[str] = Field(
        default_factory=list, description="List of dependencies required by the function")
