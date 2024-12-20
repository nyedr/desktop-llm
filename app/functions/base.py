"""Base classes and types for the function system."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class FunctionType(str, Enum):
    """Type of function in the system."""
    FILTER = "filter"
    PIPE = "pipe"
    PIPELINE = "pipeline"

# Error Classes
class FunctionError(Exception):
    """Base class for all function-related errors."""
    pass

class ValidationError(FunctionError):
    """Error in function validation."""
    pass

class TimeoutError(FunctionError):
    """Function execution timeout."""
    pass

class ExecutionError(FunctionError):
    """Error during function execution."""
    pass

class FunctionNotFoundError(FunctionError):
    """Function not found in registry."""
    pass

class ModuleImportError(FunctionError):
    """Error importing function module."""
    pass

class InputValidationError(FunctionError):
    """Error validating function input."""
    pass

class OutputValidationError(FunctionError):
    """Error validating function output."""
    pass

# Base Classes
class BaseFunction(BaseModel, ABC):
    """Base class for all functions."""
    name: str = Field(default="", description="Unique identifier for the function")
    description: str = Field(default="", description="Brief description of the function's purpose")
    type: FunctionType = Field(default=FunctionType.PIPE, description="Type of function (filter, pipe, pipeline)")
    priority: Optional[int] = Field(default=None, description="Execution priority (lower = higher priority)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Optional configuration parameters")

    class Config:
        arbitrary_types_allowed = True

class Filter(BaseFunction, ABC):
    """Base class for filter functions.
    
    Filters can modify both incoming requests (inlet) and outgoing responses (outlet).
    They are executed in priority order for inlet, and reverse priority order for outlet.
    """
    type: FunctionType = FunctionType.FILTER

    @abstractmethod
    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data before it reaches the LLM."""
        pass

    @abstractmethod
    async def outlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing data after LLM response."""
        pass

class Tool(BaseFunction, ABC):
    """Base class for tool functions.
    
    Tools are executed when called by the LLM through function calling.
    They extend the LLM's capabilities with external actions.
    """
    type: FunctionType = FunctionType.PIPE
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters accepted by the tool"
    )

    @abstractmethod
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments."""
        pass

class Pipeline(BaseFunction, ABC):
    """Base class for pipeline functions.
    
    Pipelines provide full control over the interaction flow,
    allowing custom processing sequences and complex workflows.
    """
    type: FunctionType = FunctionType.PIPELINE

    @abstractmethod
    async def pipe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the pipeline."""
        pass

def register_function(
    func_type: FunctionType,
    name: str,
    description: str,
    priority: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
):
    """Decorator to register functions with the system."""
    def decorator(cls):
        cls.type = func_type
        cls.name = name
        cls.description = description
        cls.priority = priority
        cls.config = config or {}
        return cls
    return decorator
