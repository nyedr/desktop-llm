"""Base classes and types for the function system."""

from enum import Enum
from typing import Dict, Any, Optional, List, TypeVar
import logging
from pydantic import BaseModel, Field
from app.models.agent import AgentState

logger = logging.getLogger(__name__)


class FunctionType(str, Enum):
    """Type of function."""
    TOOL = "tool"
    FILTER = "filter"
    PIPELINE = "pipeline"


T = TypeVar('T')


class FunctionError(Exception):
    """Base class for function errors."""

    def __init__(self, message: str, function_name: str = None, details: dict = None):
        self.function_name = function_name
        self.details = details or {}
        super().__init__(message)


class ValidationError(FunctionError):
    """Base class for validation errors."""
    pass


class InputValidationError(ValidationError):
    """Raised when function input validation fails."""

    def __init__(self, message: str, invalid_params: list = None, **kwargs):
        details = {"invalid_params": invalid_params} if invalid_params else {}
        super().__init__(message, details=details, **kwargs)


class OutputValidationError(ValidationError):
    """Raised when function output validation fails."""
    pass


class ToolValidationError(ValidationError):
    """Error raised when tool validation fails.

    This is a specialized validation error for tool-specific validation failures,
    such as invalid tool configurations, policies, or capabilities.

    Examples:
        - Invalid tool policy configuration
        - Missing required tool capabilities
        - Tool dependency conflicts
        - Invalid tool parameters
    """
    pass


class TimeoutError(FunctionError):
    """Error raised when function execution times out."""

    def __init__(self, message: str, timeout_seconds: float = None, **kwargs):
        details = {"timeout_seconds": timeout_seconds} if timeout_seconds else {}
        super().__init__(message, details=details, **kwargs)


class ExecutionError(FunctionError):
    """Error raised when function execution fails."""

    def __init__(self, message: str, original_error: Exception = None, **kwargs):
        details = {"error_type": type(
            original_error).__name__} if original_error else {}
        super().__init__(message, details=details, **kwargs)


class FunctionNotFoundError(FunctionError):
    """Error raised when a function is not found."""
    pass


class FunctionValidationError(FunctionError):
    """Error raised when function validation fails."""
    pass


class ModuleImportError(FunctionError):
    """Error raised when a function module cannot be imported."""
    pass


class SecurityError(FunctionError):
    """Error raised when a security violation is detected."""
    pass


class FunctionConfig(BaseModel):
    """Base configuration for functions."""
    name: str
    description: str
    type: FunctionType
    priority: Optional[int] = None
    config: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}


class FunctionParameters(BaseModel):
    """Base parameters model for functions."""
    type: str = "object"
    properties: Dict[str, Any]
    required: List[str] = []


class FunctionResponse(BaseModel):
    """Base response model for all functions."""
    success: bool = Field(
        default=True, description="Whether the function executed successfully")
    error: Optional[str] = Field(
        None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata")


class ToolResponse(FunctionResponse):
    """Response from tool execution."""
    result: Any = Field(..., description="The result of the tool execution")
    tool_name: str = Field(...,
                           description="Name of the tool that was executed")
    execution_time: float = Field(
        default=0.0, description="Time taken to execute the tool in seconds")


class FilterResponse(FunctionResponse):
    """Response from filter execution."""
    modified_data: Dict[str, Any] = Field(
        ..., description="The modified data after filtering")
    filter_name: str = Field(...,
                             description="Name of the filter that was executed")
    changes_made: bool = Field(
        default=False, description="Whether any changes were made to the data")


class PipelineResponse(FunctionResponse):
    """Response from pipeline execution."""
    results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Results from each step in the pipeline")
    pipeline_name: str = Field(
        ..., description="Name of the pipeline that was executed")
    steps_completed: int = Field(
        default=0, description="Number of steps completed in the pipeline")
    total_steps: int = Field(
        default=0, description="Total number of steps in the pipeline")


class AgentResponse(FunctionResponse):
    """Response from agent execution."""
    agent_name: str = Field(..., description="Name of the agent that executed")
    state: AgentState = Field(..., description="Final state of the agent")
    thoughts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Thoughts generated during execution")
    decisions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Decisions made during execution")
    actions_taken: List[Dict[str, Any]] = Field(
        default_factory=list, description="Actions taken during execution")
    final_output: Dict[str, Any] = Field(
        default_factory=dict, description="Final output from the agent")
