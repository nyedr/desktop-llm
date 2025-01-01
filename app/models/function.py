"""Base classes and types for the function system."""

from enum import Enum
from typing import Dict, Any, Optional, List, Callable, TypeVar, Literal, Union
import logging
from pydantic import BaseModel, Field
import time

logger = logging.getLogger(__name__)


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
    """Base class for function responses."""
    success: bool = Field(
        default=True, description="Whether the function executed successfully")
    error: Optional[str] = Field(
        default=None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the execution")


class ToolResponse(FunctionResponse):
    """Response from tool execution."""
    result: Any = Field(None, description="The result of the tool execution")
    tool_name: str = Field(...,
                           description="Name of the tool that was executed")
    execution_time: float = Field(
        default=0.0, description="Time taken to execute the tool in seconds")


class FilterResponse(FunctionResponse):
    """Response from filter execution."""
    modified_data: Dict[str, Any] = Field(...,
                                          description="The modified data after filtering")
    filter_name: str = Field(...,
                             description="Name of the filter that was executed")
    changes_made: bool = Field(
        default=False, description="Whether any changes were made to the data")


class PipelineResponse(FunctionResponse):
    """Response from pipeline execution."""
    results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Results from each step in the pipeline")
    pipeline_name: str = Field(...,
                               description="Name of the pipeline that was executed")
    steps_completed: int = Field(
        default=0, description="Number of steps completed in the pipeline")
    total_steps: int = Field(
        default=0, description="Total number of steps in the pipeline")


FunctionResult = Union[ToolResponse, FilterResponse, PipelineResponse]


class BaseFunction(BaseModel):
    """Base class for all functions."""
    name: str = Field(default="", description="Name of the function")
    description: str = Field(
        default="", description="Description of the function")
    type: FunctionType = Field(
        default=FunctionType.TOOL, description="Type of the function")
    parameters: Dict[str, Any] = Field(
        default={}, description="Parameters schema for the function")
    config: Dict[str, Any] = Field(
        default={}, description="Configuration for the function")

    async def execute(self, args: Dict[str, Any]) -> FunctionResult:
        """Execute the function with the given arguments."""
        raise NotImplementedError


class Tool(BaseFunction):
    """Base class for tools with retry and parameter normalization capabilities."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")

    async def execute(self, args: Dict[str, Any]) -> ToolResponse:
        """Execute the tool with retry logic."""
        start_time = time.time()
        try:
            result = await self._execute(self.normalize_parameters(args))
            return ToolResponse(
                success=True,
                result=result,
                tool_name=self.name,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResponse(
                success=False,
                error=str(e),
                tool_name=self.name,
                execution_time=time.time() - start_time,
                result=None
            )

    async def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Actual tool implementation to be overridden by subclasses.

        Args:
            args: Normalized tool arguments

        Returns:
            Tool execution results
        """
        raise NotImplementedError

    def normalize_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tool parameters. Can be overridden by subclasses.

        Args:
            args: Original tool arguments

        Returns:
            Normalized arguments
        """
        return args


class Filter(BaseFunction):
    """Base class for filters that modify messages.

    Filters can modify both incoming requests (inlet) and outgoing responses (outlet).
    They are executed in priority order for inlet, and reverse priority order for outlet.
    """
    type: Literal[FunctionType.FILTER] = Field(
        default=FunctionType.FILTER, description="Filter type")
    priority: Optional[int] = Field(
        default=None, description="Filter priority (lower = higher priority)")

    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data before it reaches the LLM.

        Args:
            data: Dictionary containing messages and request info

        Returns:
            Modified request data
        """
        raise NotImplementedError

    async def outlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing data after LLM response.

        Args:
            data: Dictionary containing response data

        Returns:
            Modified response data
        """
        raise NotImplementedError

    async def execute(self, args: Dict[str, Any]) -> FilterResponse:
        """Execute the filter."""
        try:
            modified_data = await self.inlet(args)
            return FilterResponse(
                success=True,
                modified_data=modified_data,
                filter_name=self.name,
                changes_made=modified_data != args
            )
        except Exception as e:
            return FilterResponse(
                success=False,
                error=str(e),
                filter_name=self.name,
                modified_data=args,
                changes_made=False
            )


class Pipeline(BaseFunction):
    """Base class for pipelines that process data through multiple steps."""
    type: Literal[FunctionType.PIPELINE] = Field(
        default=FunctionType.PIPELINE, description="Pipeline type")

    async def pipe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the pipeline steps."""
        raise NotImplementedError

    async def execute(self, args: Dict[str, Any]) -> PipelineResponse:
        """Execute the pipeline."""
        try:
            results = []
            step_count = 0
            total_steps = len(self._get_pipeline_steps())

            async for step_result in self.pipe(args):
                results.append(step_result)
                step_count += 1

            return PipelineResponse(
                success=True,
                results=results,
                pipeline_name=self.name,
                steps_completed=step_count,
                total_steps=total_steps
            )
        except Exception as e:
            return PipelineResponse(
                success=False,
                error=str(e),
                pipeline_name=self.name,
                steps_completed=step_count,
                total_steps=total_steps,
                results=results
            )

    def _get_pipeline_steps(self) -> List[str]:
        """Get list of pipeline steps. Override in subclass."""
        return []


def register_function(
    func_type: FunctionType,
    name: str,
    description: str,
    priority: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> Callable[[type], type]:
    """Decorator to register a function with the registry.

    Args:
        func_type: Type of the function (tool, filter, or pipeline)
        name: Name of the function
        description: Description of the function
        priority: Priority of the function (optional)
        config: Configuration for the function (optional)

    Returns:
        Decorated class
    """
    def decorator(cls: type) -> type:
        """Register the function class."""
        # Store function configuration
        cls._function_config = FunctionConfig(
            name=name,
            description=description,
            type=func_type,
            priority=priority,
            config=config or {},
            parameters=parameters or {}
        )

        # Set the default values for the class fields
        if hasattr(cls, 'model_fields'):  # Pydantic v2
            if 'name' in cls.model_fields:
                cls.model_fields['name'].default = name
            if 'description' in cls.model_fields:
                cls.model_fields['description'].default = description
            if 'type' in cls.model_fields:
                cls.model_fields['type'].default = func_type
            if 'parameters' in cls.model_fields:
                cls.model_fields['parameters'].default = parameters or {}
        else:  # Pydantic v1
            if hasattr(cls, '__fields__'):
                if 'name' in cls.__fields__:
                    cls.__fields__['name'].default = name
                    cls.__fields__['name'].field_info.default = name
                if 'description' in cls.__fields__:
                    cls.__fields__['description'].default = description
                    cls.__fields__[
                        'description'].field_info.default = description
                if 'type' in cls.__fields__:
                    cls.__fields__['type'].default = func_type
                    cls.__fields__['type'].field_info.default = func_type
                if 'parameters' in cls.__fields__:
                    cls.__fields__['parameters'].default = parameters or {}
                    cls.__fields__[
                        'parameters'].field_info.default = parameters or {}

        # Set class-level attributes
        cls.name = name
        cls.description = description
        cls.type = func_type

        # Register the function with the registry
        from app.functions.registry import function_registry
        function_registry.register(cls)

        return cls
    return decorator
