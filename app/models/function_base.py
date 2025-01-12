"""Base classes for the function system."""

import time
from typing import Callable, Dict, Any, List, Optional, AsyncGenerator
from pydantic import BaseModel, Field

from app.models.function import (
    FunctionConfig,
    FunctionType,
    FunctionResponse,
    ToolResponse,
    FilterResponse,
    PipelineResponse
)


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

    async def execute(self, args: Dict[str, Any]) -> FunctionResponse:
        """Execute the function with the given arguments."""
        raise NotImplementedError


class Tool(BaseFunction):
    """Base class for tools with retry and parameter normalization capabilities."""
    type: FunctionType = Field(
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
    type: FunctionType = Field(
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
    type: FunctionType = Field(
        default=FunctionType.PIPELINE, description="Pipeline type")

    async def pipe(self, data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
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
    """Decorator to register a function with the registry."""
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
