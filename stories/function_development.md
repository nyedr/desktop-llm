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

3. **Executor** (`app/functions/executor.py`):

   - Executes functions with validation
   - Handles tool calls from the LLM
   - Provides error handling and logging

4. **Utilities** (`app/functions/utils.py`):
   - Common utilities for function development
   - Message handling helpers
   - Model interaction utilities
   - Application constants and settings

### Important Implementation Notes

#### Tool Parameters Definition

When creating new tools, you must use Pydantic's `Field` to properly define tool parameters. This ensures they are correctly exposed in the function registry and API.

```python
from typing import Dict, Any, Literal
from app.functions.base import Tool, FunctionType
from pydantic import Field, ConfigDict

@register_function(
    func_type=FunctionType.TOOL,
    name="my_tool",
    description="Description of what the tool does"
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

    # Required: Define parameters schema using Field
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Description of param1"
                }
            },
            "required": ["param1"]
        },
        description="Parameters schema for the tool"
    )

    # Important: Any instance variables must also be declared as Fields
    my_service: MyService = Field(
        default_factory=MyService,  # Use default_factory for class instances
        exclude=True  # Exclude from schema if not needed in API
    )
```

**Common Mistakes to Avoid:**

- ❌ Don't define parameters as a class variable without Field:
  ```python
  parameters = {  # Wrong: won't be properly exposed
      "type": "object",
      "properties": {...}
  }
  ```
- ❌ Don't use ClassVar for parameters:
  ```python
  parameters: ClassVar[Dict[str, Any]] = {...}  # Wrong: won't work with Pydantic
  ```
- ❌ Don't put parameters in the register_function decorator:
  ```python
  @register_function(
      parameters={...}  # Wrong: not supported by decorator
  )
  ```
- ❌ Don't initialize instance variables in **init** without declaring them as Fields:
  ```python
  def __init__(self):
      self.my_service = MyService()  # Wrong: will raise "object has no field" error
  ```
- ❌ Don't forget to allow arbitrary types when using custom services:
  ```python
  class MyTool(Tool):  # Wrong: will fail with custom service types
      my_service: MyService = Field(default_factory=MyService)
  ```

**Correct Pattern:**

- ✅ Use Pydantic's Field for all tool attributes
- ✅ Include proper type annotations
- ✅ Provide default values and descriptions
- ✅ Keep registration separate from parameter definition
- ✅ Declare all instance variables as Fields with proper defaults
- ✅ Add model_config when using custom service types:
  ```python
  class MyTool(Tool):
      model_config = ConfigDict(arbitrary_types_allowed=True)
      my_service: MyService = Field(default_factory=MyService)
  ```

### Function Types

The system supports three main types of functions, defined in `app/functions/base.py`:

#### 1. Tools (`class Tool(BaseFunction)`)

Tools extend the LLM's capabilities through function calling. They are executed when explicitly called by the LLM during its processing.

**Key Characteristics:**

- Executed during LLM interaction
- Perform specific, atomic operations
- Have well-defined input parameters and output schemas
- Can interact with external services or perform computations

**Implementation Details:**

```python
from app.functions.base import Tool, FunctionType
from typing import Dict, Any

@register_function(
    func_type=FunctionType.TOOL,
    name="calculator",
    description="Performs basic arithmetic calculations"
)
class CalculatorTool(Tool):
    # Define parameter schema
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform"
            },
            "a": {"type": "number", "description": "First operand"},
            "b": {"type": "number", "description": "Second operand"}
        },
        "required": ["operation", "a", "b"]
    }

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with validated arguments."""
        operation = args["operation"]
        a = args["a"]
        b = args["b"]

        # Implement calculation logic
        result = perform_calculation(operation, a, b)

        # Return structured response
        return {
            "result": result,
            "operation": operation,
            "operands": {"a": a, "b": b}
        }
```

#### 2. Filters (`class Filter(BaseFunction)`)

Filters intercept and modify data flow before it reaches the LLM (inlet) or after the LLM response (outlet).

**Key Characteristics:**

- Can modify both incoming and outgoing data
- Execute in priority order (lower number = higher priority)
- Support bidirectional processing
- Ideal for content modification, validation, and routing

**Implementation Details:**

```python
from app.functions.base import Filter, FunctionType
from typing import Dict, Any

@register_function(
    func_type=FunctionType.FILTER,
    name="text_modifier",
    description="Modifies text content in both inlet and outlet",
    priority=1  # Lower number = higher priority
)
class TextModifierFilter(Filter):
    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data before LLM.

        Args:
            data: Dictionary containing:
                - messages: List of chat messages
                - metadata: Request metadata

        Returns:
            Modified request data
        """
        messages = data.get("messages", [])

        # Modify messages as needed
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                message["content"] = preprocess_text(content)

        return data

    async def outlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing data after LLM response.

        Args:
            data: Dictionary containing:
                - response: LLM response
                - metadata: Response metadata

        Returns:
            Modified response data
        """
        response = data.get("response", {})

        # Modify response as needed
        if "content" in response:
            response["content"] = postprocess_text(response["content"])

        return data
```

#### 3. Pipelines (`class Pipeline(BaseFunction)`)

Pipelines define custom processing sequences with full control over the interaction flow.

**Key Characteristics:**

- Handle complex multi-step workflows
- Can orchestrate multiple tools and filters
- Provide custom LLM provider integration
- Support advanced data processing flows

**Implementation Details:**

```python
from app.functions.base import Pipeline, FunctionType
from typing import Dict, Any

@register_function(
    func_type=FunctionType.PIPELINE,
    name="multi_step_processor",
    description="Processes data through multiple steps"
)
class MultiStepPipeline(Pipeline):
    # Optional configuration
    config = {
        "max_steps": 3,
        "timeout_per_step": 30
    }

    async def pipe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through multiple steps.

        Args:
            data: Input data dictionary

        Returns:
            Processed data
        """
        result = data

        # Step 1: Initial processing
        result = await self.step_one(result)

        # Step 2: Secondary processing
        result = await self.step_two(result)

        # Step 3: Final processing
        result = await self.step_three(result)

        return result

    async def step_one(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """First processing step."""
        # Implementation
        return processed_data

    async def step_two(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Second processing step."""
        # Implementation
        return processed_data

    async def step_three(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Third processing step."""
        # Implementation
        return processed_data
```

## Development Utilities

The system provides several utilities to help with function development, available in `app.functions.utils`:

### 1. Message Handling

These utilities help you work with conversation history:

```python
from app.functions.utils import (
    get_last_user_message,
    get_last_assistant_message,
    get_system_message
)

# Get messages by role
last_user_msg = get_last_user_message(messages)      # Latest user message
last_assistant_msg = get_last_assistant_message(messages)  # Latest assistant message
system_msg = get_system_message(messages)            # System message if present

# Example usage in a tool
async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
    messages = args.get("messages", [])

    # Get the last user message for context
    last_msg = get_last_user_message(messages)
    if last_msg:
        # Process based on user's last message
        content = last_msg.get("content", "")
        # ... processing logic ...
```

### 2. Model Interaction

Utilities for interacting with language models:

```python
from app.functions.utils import generate_chat_completion, get_all_models

class LanguageModelTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Get available models
        models = await get_all_models()

        # Generate completions with streaming
        messages = [{"role": "user", "content": "Your prompt here"}]

        responses = []
        async for response in generate_chat_completion(
            messages=messages,
            model="llama2",          # Optional: defaults to DEFAULT_MODEL
            temperature=0.7,         # Optional: defaults to MODEL_TEMPERATURE
            max_tokens=1000,         # Optional: defaults to MAX_TOKENS
            stream=True,             # Enable streaming
            tools=available_tools,   # Optional: tools for function calling
            enable_tools=True        # Enable tool usage
        ):
            responses.append(response)

        return {"responses": responses}
```

### 3. Application Constants

Access system-wide constants and configurations:

```python
from app.functions.utils import APP_CONSTANTS

class ConfigAwareTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Access configuration
        model = APP_CONSTANTS["DEFAULT_MODEL"]
        max_tokens = APP_CONSTANTS["MAX_TOKENS"]

        # Use timeouts
        request_timeout = APP_CONSTANTS["MODEL_REQUEST_TIMEOUT"]
        generation_timeout = APP_CONSTANTS["GENERATION_REQUEST_TIMEOUT"]

        # Check feature flags
        if APP_CONSTANTS["FUNCTION_CALLS_ENABLED"]:
            # Implement function calling logic
            pass
```

### 4. User Settings

Access and use user-specific settings:

```python
from app.functions.utils import USER_SETTINGS

class UserAwareTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Use user preferences
        model = USER_SETTINGS["model"]
        temperature = USER_SETTINGS["temperature"]

        # Check user features
        if USER_SETTINGS["function_calls_enabled"]:
            # Implement user-specific logic
            pass
```

## Error Handling

The system provides several error types in `app/functions/base.py`:

```python
from app.functions.base import (
    FunctionError,
    ValidationError,
    InputValidationError,
    OutputValidationError,
    TimeoutError,
    ExecutionError
)

class RobustTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Validate input
            if not self.validate_input(args):
                raise InputValidationError("Invalid input parameters")

            # Execute with timeout
            try:
                async with asyncio.timeout(30):
                    result = await self.process(args)
            except asyncio.TimeoutError:
                raise TimeoutError("Operation timed out", timeout_seconds=30)

            # Validate output
            if not self.validate_output(result):
                raise OutputValidationError("Invalid output format")

            return result

        except Exception as e:
            raise ExecutionError("Tool execution failed", original_error=e)
```

## Best Practices

### 1. Function Development

#### Naming and Structure

```python
class WellStructuredTool(Tool):
    """A well-structured tool following best practices.

    This tool demonstrates proper organization, error handling,
    and documentation practices.
    """

    # Clear parameter schema
    parameters = {
        "type": "object",
        "properties": {
            "input_field": {
                "type": "string",
                "description": "Clear description of the input"
            }
        },
        "required": ["input_field"]
    }

    # Optional configuration
    config = {
        "timeout": 30,
        "max_retries": 3
    }

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with proper error handling.

        Args:
            args: Validated input arguments

        Returns:
            Processed results

        Raises:
            InputValidationError: If input validation fails
            TimeoutError: If processing exceeds timeout
            ExecutionError: If processing fails
        """
        try:
            # Implementation
            pass
        except Exception as e:
            raise ExecutionError("Processing failed", original_error=e)
```

### 2. Testing

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

### Error Handling and Parameter Normalization

The system provides robust error handling and parameter normalization at multiple levels:

#### 1. Parameter Normalization Filter (`app/functions/types/filters/parameter_normalizer.py`)

A high-priority filter that normalizes parameters before they reach functions:

```python
@register_function(
    func_type=FunctionType.FILTER,
    name="parameter_normalizer",
    description="Normalizes function parameters before execution",
    priority=1
)
class ParameterNormalizerFilter(Filter):
    # Common parameter normalizations
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
        # Normalize parameters in tool calls
        if "tool_calls" in data:
            for tool_call in data["tool_calls"]:
                if "function" in tool_call:
                    args = tool_call["function"].get("arguments", {})
                    normalized_args = self._normalize_parameters(args)
                    tool_call["function"]["arguments"] = normalized_args
        return data
```

#### 2. Base Tool Class Retry Mechanism

The base `Tool` class includes built-in retry logic and parameter normalization:

```python
class Tool(BaseFunction):
    # Default retry configuration
    retry_config: Dict[str, Any] = {
        "max_retries": 3,
        "retry_delay": 1.0,  # seconds
        "retry_on_errors": [InputValidationError, TimeoutError]
    }

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        retries = 0
        last_error = None
        normalized_args = self.normalize_parameters(args)

        while retries <= self.retry_config["max_retries"]:
            try:
                return await self._execute(normalized_args)
            except tuple(self.retry_config["retry_on_errors"]) as e:
                last_error = e
                retries += 1
                if retries <= self.retry_config["max_retries"]:
                    normalized_args = await self._handle_error(e, normalized_args)
                    await asyncio.sleep(self.retry_config["retry_delay"])
                continue
            except Exception as e:
                raise ExecutionError(f"Unexpected error: {str(e)}")

        raise ExecutionError(f"All retries failed. Last error: {str(last_error)}")
```

#### 3. Function-Specific Parameter Handling

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

### Best Practices for Error Handling

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
