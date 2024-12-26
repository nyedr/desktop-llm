"""Base classes and types for the function system."""

from enum import Enum
from typing import Dict, Any, Optional, List, Callable, TypeVar, Literal
import asyncio
import logging
from pydantic import BaseModel, Field

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


class FunctionParameters(BaseModel):
    """Base parameters model for functions."""
    type: str = "object"
    properties: Dict[str, Any]
    required: List[str] = []


class BaseFunction(BaseModel):
    """Base class for all functions."""
    name: str = Field(..., description="Name of the function")
    description: str = Field(..., description="Description of the function")
    type: FunctionType = Field(..., description="Type of the function")
    parameters: Dict[str, Any] = Field(
        default={}, description="Parameters schema for the function")
    config: Dict[str, Any] = Field(
        default={}, description="Configuration for the function")

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the function with the given arguments."""
        raise NotImplementedError


class Tool(BaseFunction):
    """Base class for tools with retry and parameter normalization capabilities."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")

    # Default retry configuration
    retry_config: Dict[str, Any] = {
        "max_retries": 3,
        "retry_delay": 1.0,  # seconds
        "retry_on_errors": [InputValidationError, TimeoutError]
    }

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with retry logic.

        Args:
            args: Tool arguments

        Returns:
            Tool execution results

        Raises:
            ExecutionError: If all retries fail
        """
        retries = 0
        last_error = None
        normalized_args = self.normalize_parameters(args)

        while retries <= self.retry_config["max_retries"]:
            try:
                if retries > 0:
                    logger.info(
                        f"Retrying {self.name} (attempt {retries}/{self.retry_config['max_retries']})")

                return await self._execute(normalized_args)

            except tuple(self.retry_config["retry_on_errors"]) as e:
                last_error = e
                retries += 1

                if retries <= self.retry_config["max_retries"]:
                    # Try to fix the error if possible
                    normalized_args = await self._handle_error(e, normalized_args)
                    await asyncio.sleep(self.retry_config["retry_delay"])

                continue

            except Exception as e:
                # Don't retry on unexpected errors
                raise ExecutionError(
                    f"Unexpected error in {self.name}: {str(e)}",
                    function_name=self.name,
                    original_error=e
                )

        # If we get here, all retries failed
        raise ExecutionError(
            f"All retries failed for {self.name}. Last error: {str(last_error)}",
            function_name=self.name,
            original_error=last_error
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

    async def _handle_error(self, error: Exception, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors and attempt to fix parameters. Can be overridden by subclasses.

        Args:
            error: The error that occurred
            args: Current arguments

        Returns:
            Modified arguments for retry
        """
        if isinstance(error, InputValidationError):
            # Try to fix validation errors
            return self._fix_validation_error(error, args)

        # Return original args if we don't know how to fix the error
        return args

    def _fix_validation_error(self, error: InputValidationError, args: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to fix validation errors. Can be overridden by subclasses.

        Args:
            error: The validation error
            args: Current arguments

        Returns:
            Fixed arguments
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

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the filter with the given arguments.

        This is a compatibility method that uses inlet by default.
        """
        return await self.inlet(args)


class Pipeline(BaseFunction):
    """Base class for pipelines that process data through multiple steps."""
    type: Literal[FunctionType.PIPELINE] = Field(
        default=FunctionType.PIPELINE, description="Pipeline type")

    async def pipe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the pipeline steps."""
        raise NotImplementedError

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline with the given arguments."""
        return await self.pipe(args)


def register_function(
    func_type: FunctionType,
    name: str,
    description: str,
    priority: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
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
        cls._function_config = FunctionConfig(
            name=name,
            description=description,
            type=func_type,
            priority=priority,
            config=config or {}
        )
        return cls
    return decorator
