# Function Development Guide

## Overview

This guide provides comprehensive documentation for developing functions in the Desktop LLM system. The function system is designed to be modular, extensible, and type-safe, with clear separation of concerns between different function types.

## System Architecture

### Core Components

1. **Base Classes** (`app/functions/base.py`):

   - Defines the foundational types and interfaces
   - Provides error handling classes and retry mechanisms
   - Implements parameter normalization
   - Implements the function registration decorator

2. **Registry** (`app/functions/registry.py`):

   - Manages function registration and discovery
   - Handles dynamic loading of functions
   - Maintains function metadata
   - Provides Pydantic v1/v2 compatibility layer

3. **Executor** (`app/functions/executor.py`):

   - Executes functions with validation
   - Handles tool calls from the LLM
   - Provides error handling and logging

4. **Utilities** (`app/functions/utils.py`):

   - Common utilities for function development
   - Message handling helpers
   - Model interaction utilities
   - Application constants and settings
   - Message type conversion and validation

5. **Chat Helper** (`app/functions/chat.py`):
   - Streamlined interface for LLM interactions
   - Model management utilities
   - Chat completion generation
   - Proper error handling and logging

### Important Implementation Notes

#### Function Registration and Parameters

When creating new functions, parameters are now defined in the `@register_function` decorator rather than as class variables:

```python
from typing import Dict, Any, Literal
from app.functions.base import Tool, FunctionType
from pydantic import Field, ConfigDict

@register_function(
    func_type=FunctionType.TOOL,
    name="my_tool",
    description="Description of what the tool does",
    parameters={  # Parameters defined here
        "type": "object",
        "properties": {
            "param_name": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param_name"]
    }
)
class MyTool(Tool):
    """Tool documentation."""

    # Required: Allow arbitrary types if using custom service classes
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required: Define type, name, and description using Field
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL,
        description="Tool type"
    )
    name: str = Field(
        default="my_tool",
        description="Name of the tool"
    )
    description: str = Field(
        default="Description of what the tool does",
        description="Description of what the tool does"
    )

    # Important: Any instance variables must be declared as Fields
    my_service: MyService = Field(
        default_factory=MyService,  # Use default_factory for class instances
        exclude=True  # Exclude from schema if not needed in API
    )
```

**Common Mistakes to Avoid:**

- ❌ Don't define parameters as a class variable
- ❌ Don't use ClassVar for parameters
- ❌ Don't initialize instance variables without Field declarations
- ❌ Don't forget model_config for custom service types

**Correct Pattern:**

- ✅ Define parameters in the register_function decorator
- ✅ Use Pydantic's Field for all attributes
- ✅ Include proper type annotations
- ✅ Provide default values and descriptions
- ✅ Declare all instance variables as Fields

### Function Types

#### 1. Tools (`class Tool(BaseFunction)`)

Tools extend the LLM's capabilities through function calling:

```python
@register_function(
    func_type=FunctionType.TOOL,
    name="web_scrape",
    description="Scrape and process web pages",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to scrape"
            },
            "raw": {
                "type": "boolean",
                "description": "Get raw content",
                "default": False
            }
        },
        "required": ["url"]
    }
)
class WebScrapeTool(Tool):
    """Tool for web scraping with retry and error handling."""

    async def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Implement the actual tool logic."""
        try:
            return await self.scrape_url(args["url"], args.get("raw", False))
        except Exception as e:
            raise ExecutionError(f"Scraping failed: {str(e)}")

    def normalize_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Optional: Normalize parameters before execution."""
        return args

    async def _handle_error(self, error: Exception, args: Dict[str, Any]) -> Dict[str, Any]:
        """Optional: Custom error handling logic."""
        return args
```

#### 2. Filters (`class Filter(BaseFunction)`)

Filters modify data flow with priority-based execution:

```python
@register_function(
    func_type=FunctionType.FILTER,
    name="text_modifier",
    description="Modifies text content",
    priority=1,  # Lower number = higher priority
    config={
        "prefix": "[Modified] ",
        "suffix": " [End]"
    }
)
class TextModifierFilter(Filter):
    """Filter that processes both incoming and outgoing data."""

    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data (high to low priority)."""
        messages = data.get("messages", [])
        if not messages:
            return data

        modified_messages = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                message = message.copy()
                message["content"] = self._modify_content(message["content"])
            modified_messages.append(message)

        data["messages"] = modified_messages
        return data

    async def outlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing data (low to high priority)."""
        if data.get("role") == "assistant":
            data = data.copy()
            if isinstance(data.get("content"), str):
                data["content"] = self._modify_content(data["content"])
        return data
```

#### 3. Pipelines (`class Pipeline(BaseFunction)`)

Pipelines orchestrate complex multi-step workflows:

```python
@register_function(
    func_type=FunctionType.PIPELINE,
    name="multi_step_processor",
    description="Processes data through multiple steps",
    config={
        "max_steps": 3,
        "timeout_per_step": 30
    }
)
class MultiStepPipeline(Pipeline):
    """Pipeline with multiple processing steps."""

    async def pipe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through defined steps."""
        result = data

        # Step 1: Message normalization
        messages = [ensure_strict_message(msg) for msg in result.get("messages", [])]

        # Step 2: Content processing
        processed_messages = []
        for msg in messages:
            if msg.role == "user":
                processed_content = await self._process_user_message(msg.content)
                processed_messages.append(
                    UserMessage(
                        role="user",
                        content=processed_content
                    )
                )
            else:
                processed_messages.append(msg)

        # Step 3: Generate summary
        summary = await self._generate_summary(processed_messages)

        return {
            "messages": processed_messages,
            "summary": summary
        }

    async def _process_user_message(self, content: str) -> str:
        """Custom processing for user messages."""
        pass

    async def _generate_summary(self, messages: List[StrictChatMessage]) -> Dict[str, Any]:
        """Generate summary of processed messages."""
        pass
```

### Error Handling System

The system provides a comprehensive error hierarchy:

```python
class FunctionError(Exception):
    """Base class for all function errors."""
    pass

class ValidationError(FunctionError):
    """Base for validation errors."""
    pass

class InputValidationError(ValidationError):
    """Input validation failure."""
    pass

class OutputValidationError(ValidationError):
    """Output validation failure."""
    pass

class TimeoutError(FunctionError):
    """Execution timeout."""
    pass

class ExecutionError(FunctionError):
    """General execution failure."""
    pass

class SecurityError(FunctionError):
    """Security violation."""
    pass

class ModuleImportError(FunctionError):
    """Module import failure."""
    pass
```

### Parameter Normalization

The system includes a high-priority filter for parameter normalization:

```python
@register_function(
    func_type=FunctionType.FILTER,
    name="parameter_normalizer",
    description="Normalizes function parameters",
    priority=1
)
class ParameterNormalizerFilter(Filter):
    """System-wide parameter normalization."""

    COMMON_NORMALIZATIONS = {
        "temperature_units": {
            "celsius": ["Celsius", "CELSIUS", "C", "c"],
            "fahrenheit": ["Fahrenheit", "FAHRENHEIT", "F", "f"]
        },
        "boolean_values": {
            True: ["true", "True", "TRUE", "1", "yes", "Yes", "YES"],
            False: ["false", "False", "FALSE", "0", "no", "No", "NO"]
        }
    }

    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters in tool calls."""
        if "tool_calls" in data:
            for tool_call in data["tool_calls"]:
                if "function" in tool_call:
                    args = tool_call["function"].get("arguments", {})
                    tool_call["function"]["arguments"] = self._normalize_parameters(
                        args,
                        tool_call["function"].get("name")
                    )
        return data
```

### Utilities

The system provides several utility functions:

```python
# Message handling
def get_last_user_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the last user message."""
    pass

def get_last_assistant_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the last assistant message."""
    pass

def get_system_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the system message."""
    pass

def ensure_strict_message(msg: Any) -> StrictChatMessage:
    """Convert to StrictChatMessage with validation."""
    pass

# Response validation
def validate_function_response(response: FunctionResult) -> bool:
    """Validate a function response."""
    pass

def validate_tool_response(response: ToolResponse) -> bool:
    """Validate a tool response."""
    pass

def validate_filter_response(response: FilterResponse) -> bool:
    """Validate a filter response."""
    pass

def validate_pipeline_response(response: PipelineResponse) -> bool:
    """Validate a pipeline response."""
    pass

def ensure_response_type(response: Any, expected_type: Type[FunctionResult]) -> FunctionResult:
    """Ensure a response matches the expected type."""
    pass

def create_error_response(error: Exception, function_type: str, function_name: str, **kwargs) -> FunctionResult:
    """Create an error response of the appropriate type."""
    pass

# Model interaction
async def generate_chat_completion(
    messages: List[StrictChatMessage],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    enable_tools: bool = True,
    function_service=None
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """Generate chat completions with streaming support."""
    pass

async def get_all_models() -> List[str]:
    """Get available models."""
    pass
```

### Using the Chat Helper

The `ChatHelper` class provides a streamlined interface for functions that need to interact with language models:

```python
from app.functions.chat import ChatHelper
from app.models.chat import UserMessage, SystemMessage

class MyLLMTool(Tool):
    async def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        chat_helper = ChatHelper()

        # Create messages
        messages = [
            SystemMessage(content="System prompt"),
            UserMessage(content=args["user_input"])
        ]

        # Generate completion
        responses = []
        async for response in chat_helper.generate_completion(
            messages=messages,
            model="llama2",  # Optional, uses default if not specified
            temperature=0.7,  # Optional
            stream=True      # Optional
        ):
            responses.append(response)

        return {"responses": responses}
```

Key features of the ChatHelper:

1. **Model Management**:

   - Get available models with `get_available_models()`
   - Automatic fallback to default model
   - Proper error handling for model issues

2. **Chat Generation**:

   - Streaming support with async iteration
   - Configurable parameters (temperature, max_tokens)
   - Tool integration capabilities
   - Proper error handling and logging

3. **Integration with Function System**:
   - Compatible with Tool, Filter, and Pipeline types
   - Supports function calling in chat completions
   - Handles both streaming and non-streaming responses

### Response Types

The system defines standard response types for all functions:

```python
class FunctionResponse(BaseModel):
    """Base class for function responses."""
    success: bool = Field(default=True, description="Whether the function executed successfully")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the execution")

class ToolResponse(FunctionResponse):
    """Response from tool execution."""
    result: Any = Field(..., description="The result of the tool execution")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    execution_time: float = Field(default=0.0, description="Time taken to execute the tool in seconds")

class FilterResponse(FunctionResponse):
    """Response from filter execution."""
    modified_data: Dict[str, Any] = Field(..., description="The modified data after filtering")
    filter_name: str = Field(..., description="Name of the filter that was executed")
    changes_made: bool = Field(default=False, description="Whether any changes were made to the data")

class PipelineResponse(FunctionResponse):
    """Response from pipeline execution."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from each step in the pipeline")
    pipeline_name: str = Field(..., description="Name of the pipeline that was executed")
    steps_completed: int = Field(default=0, description="Number of steps completed in the pipeline")
    total_steps: int = Field(default=0, description="Total number of steps in the pipeline")
```

### Response Validation

The system provides comprehensive validation for function responses:

```python
# Validate a tool response
try:
    result = await tool.execute(args)
    validate_tool_response(result)
except ValidationError as e:
    logger.error(f"Invalid tool response: {e}")
    result = create_error_response(e, "tool", tool.name)

# Validate a filter response
try:
    result = await filter.execute(args)
    validate_filter_response(result)
except ValidationError as e:
    logger.error(f"Invalid filter response: {e}")
    result = create_error_response(e, "filter", filter.name)

# Validate a pipeline response
try:
    result = await pipeline.execute(args)
    validate_pipeline_response(result)
except ValidationError as e:
    logger.error(f"Invalid pipeline response: {e}")
    result = create_error_response(e, "pipeline", pipeline.name)

# Ensure response type matches expectation
try:
    result = ensure_response_type(response, ToolResponse)
except ValidationError as e:
    logger.error(f"Response type mismatch: {e}")
    result = create_error_response(e, "tool", "unknown")
```

### Response Validation Rules

1. **Common Rules for All Responses**:

   - Must be an instance of the correct response type
   - Failed responses must include an error message
   - Success flag must be boolean
   - Metadata must be a dictionary

2. **Tool Response Rules**:

   - Successful responses must include a result
   - Execution time cannot be negative
   - Tool name must be provided

3. **Filter Response Rules**:

   - Modified data must be a dictionary
   - Changes made flag must be boolean
   - Filter name must be provided

4. **Pipeline Response Rules**:
   - Completed steps cannot exceed total steps
   - Step counts cannot be negative
   - Results must be a list
   - Pipeline name must be provided

### Error Response Creation

The system provides a utility to create appropriate error responses:

```python
# Create a tool error response
error_response = create_error_response(
    error=ValueError("Invalid input"),
    function_type="tool",
    function_name="my_tool",
    execution_time=1.5
)

# Create a filter error response
error_response = create_error_response(
    error=RuntimeError("Processing failed"),
    function_type="filter",
    function_name="my_filter"
)

# Create a pipeline error response
error_response = create_error_response(
    error=Exception("Step 2 failed"),
    function_type="pipeline",
    function_name="my_pipeline",
    steps_completed=1,
    total_steps=3
)
```

### Pydantic Compatibility

The system supports both Pydantic v1 and v2:

```python
def pydantic_field_exists(func_cls, field_name: str) -> bool:
    """Check field existence in v1/v2."""
    if hasattr(func_cls, "model_fields"):  # v2
        return field_name in func_cls.model_fields
    else:  # v1
        return field_name in func_cls.__fields__

def get_field_default(func_cls, field_name: str):
    """Get field default in v1/v2."""
    if hasattr(func_cls, "model_fields"):  # v2
        return func_cls.model_fields[field_name].default
    else:  # v1
        return func_cls.__fields__[field_name].default

def set_field_default(func_cls, field_name: str, value):
    """Set field default in v1/v2."""
    if hasattr(func_cls, "model_fields"):  # v2
        func_cls.model_fields[field_name].default = value
    else:  # v1
        func_cls.__fields__[field_name].default = value
        func_cls.__fields__[field_name].field_info.default = value
```

## Version Compatibility

### Python Version Requirements

- Minimum Python version: 3.9
- Required for:
  - Type hints
  - Async/await syntax
  - Dict union operations

### Dependencies

Specify dependencies in `requirements.txt`:

```txt
pydantic>=2.0.0
jsonschema>=4.0.0
asyncio>=3.4.3
```

### Breaking Changes

Document any breaking changes in your function:

```python
class VersionedTool(Tool):
    """A tool with version compatibility notes.

    Version History:
    - 1.0.0: Initial release
    - 1.1.0: Added streaming support
    - 2.0.0: Breaking change - New parameter format

    Minimum Requirements:
    - Python 3.9+
    - Pydantic 2.0+
    """
    pass
```

## Security Considerations

### 1. Input Validation

Always validate inputs thoroughly:

```python
class SecureTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Validate input types
        if not isinstance(args.get("input"), str):
            raise InputValidationError("Input must be a string")

        # Sanitize inputs
        sanitized_input = sanitize_input(args["input"])

        # Check for malicious content
        if contains_malicious_content(sanitized_input):
            raise SecurityError("Malicious content detected")
```

### 2. Resource Management

Implement proper resource management:

```python
class ResourceAwareTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Set timeouts
        timeout = self.config.get("timeout", 30)

        try:
            async with asyncio.timeout(timeout):
                # Resource-intensive operation
                result = await self.process(args)

        except asyncio.TimeoutError:
            # Clean up resources
            await self.cleanup()
            raise TimeoutError(f"Operation timed out after {timeout}s")
```

## Testing

Create comprehensive tests for your functions:

```python
# test_calculator_tool.py
import pytest
from app.functions.types.tools.calculator import CalculatorTool

@pytest.mark.asyncio
async def test_calculator_addition():
    tool = CalculatorTool()
    result = await tool.execute({
        "operation": "add",
        "a": 5,
        "b": 3
    })
    assert result["result"] == 8

@pytest.mark.asyncio
async def test_calculator_invalid_input():
    tool = CalculatorTool()
    with pytest.raises(InputValidationError):
        await tool.execute({
            "operation": "invalid",
            "a": 5,
            "b": 3
        })
```

### Testing Error Handling

Create comprehensive tests for your error handling:

```python
@pytest.mark.asyncio
async def test_weather_tool_parameter_normalization():
    tool = WeatherTool()

    # Test unit normalization
    args = {"location": "New York", "unit": "Celsius"}
    normalized = tool.normalize_parameters(args)
    assert normalized["unit"] == "celsius"

    # Test error handling
    with pytest.raises(InputValidationError):
        await tool.execute({"location": "New York", "unit": "invalid"})

    # Test retry mechanism
    result = await tool.execute({"location": "New York", "unit": "C"})
    assert result["unit"] == "celsius"
```

## Debugging and Monitoring

### 1. Logging

Implement proper logging:

```python
import logging

logger = logging.getLogger(__name__)

class LoggingTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting execution with args: {args}")

        try:
            result = await self.process(args)
            logger.debug(f"Processing result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            raise
```

### 2. Performance Monitoring

Monitor execution times and resource usage:

```python
import time
import psutil

class MonitoredTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            result = await self.process(args)

            # Log performance metrics
            execution_time = time.time() - start_time
            memory_used = psutil.Process().memory_info().rss - start_memory

            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info(f"Memory used: {memory_used / 1024 / 1024:.2f}MB")

            return result

        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            raise
```

## Best Practices for Error Handling

1. **Layer Your Defenses**:

   - Use the parameter normalizer filter for common cases
   - Implement function-specific normalization
   - Add retry logic for transient failures

2. **Normalize Early**:

   ```python
   def normalize_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
       normalized = args.copy()
       # Normalize at the start of execution
       normalized["param"] = self._normalize_param(normalized.get("param"))
       return normalized
   ```

3. **Handle Errors Gracefully**:

   ```python
   async def _handle_error(self, error: Exception, args: Dict[str, Any]) -> Dict[str, Any]:
       if isinstance(error, InputValidationError):
           # Try to fix the input
           return self._fix_validation_error(error, args)
       elif isinstance(error, TimeoutError):
           # Maybe reduce the scope of the request
           return self._reduce_request_scope(args)
       return args
   ```

4. **Provide Helpful Error Messages**:

   ```python
   def _fix_validation_error(self, error: InputValidationError, args: Dict[str, Any]) -> Dict[str, Any]:
       if "unit" in error.details.get("invalid_params", []):
           logger.info(f"Converting invalid unit '{args['unit']}' to 'fahrenheit'")
           return {"unit": "fahrenheit", **{k:v for k,v in args.items() if k != "unit"}}
       return args
   ```

5. **Log Extensively**:
   ```python
   async def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
       try:
           logger.debug(f"Executing {self.name} with args: {args}")
           result = await self._do_execute(args)
           logger.debug(f"Execution result: {result}")
           return result
       except Exception as e:
           logger.error(f"Error in {self.name}: {e}", exc_info=True)
           raise
   ```

## Function-Specific Parameter Handling

Individual functions can implement custom parameter normalization and error handling:

```python
class WeatherTool(Tool):
    # Parameter normalization mappings
    UNIT_MAPPINGS = {
        "celsius": ["Celsius", "CELSIUS", "C", "c", "centigrade"],
        "fahrenheit": ["Fahrenheit", "FAHRENHEIT", "F", "f"]
    }

    def normalize_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        normalized = args.copy()

        # Normalize temperature unit
        if "unit" in normalized:
            unit = str(normalized["unit"]).lower()
            for standard, variants in self.UNIT_MAPPINGS.items():
                if unit in [v.lower() for v in variants]:
                    normalized["unit"] = standard
                    break

        return normalized

    def _fix_validation_error(self, error: InputValidationError, args: Dict[str, Any]) -> Dict[str, Any]:
        fixed = args.copy()
        if "unit" in error.details.get("invalid_params", []):
            fixed["unit"] = "fahrenheit"  # Default to fahrenheit
        return fixed
```
