"""Utility functions for the function system."""

import logging
from typing import Dict, Any, List, Optional, Type, Tuple
from app.core.config import config
from app.models.function import (
    FilterResponse,
    FunctionResponse,
    FunctionType,
    PipelineResponse,
    ToolResponse,
    ToolValidationError,
    ValidationError
)

logger = logging.getLogger(__name__)


def get_last_user_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the last user message from the conversation history."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message
    return None


def get_last_assistant_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the last assistant message from the conversation history.

    Args:
        messages: List of message dictionaries

    Returns:
        The last assistant message or None if not found
    """
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message
    return None


def get_system_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the system message from the conversation history.

    Args:
        messages: List of message dictionaries

    Returns:
        The system message or None if not found
    """
    for message in messages:
        if message.get("role") == "system":
            return message
    return None


def validate_function_response(response: FunctionResponse) -> bool:
    """Validate a function response.

    Args:
        response: The function response to validate

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ValidationError: If response is invalid
    """
    if not response.success and not response.error:
        raise ValidationError("Failed responses must include an error message")

    return True


def validate_tool_response(response: ToolResponse) -> bool:
    """Validate a tool response.

    Args:
        response: The tool response to validate

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ValidationError: If response is invalid
    """
    validate_function_response(response)

    if response.success and response.result is None:
        raise ValidationError(
            "Successful tool responses must include a result")

    if response.execution_time < 0:
        raise ValidationError("Execution time cannot be negative")

    return True


def validate_filter_response(response: FilterResponse) -> bool:
    """Validate a filter response.

    Args:
        response: The filter response to validate

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ValidationError: If response is invalid
    """
    validate_function_response(response)

    if response.success and not isinstance(response.modified_data, dict):
        raise ValidationError("Modified data must be a dictionary")

    return True


def validate_pipeline_response(response: PipelineResponse) -> bool:
    """Validate a pipeline response.

    Args:
        response: The pipeline response to validate

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ValidationError: If response is invalid
    """
    validate_function_response(response)

    if response.steps_completed > response.total_steps:
        raise ValidationError("Completed steps cannot exceed total steps")

    if response.steps_completed < 0 or response.total_steps < 0:
        raise ValidationError("Step counts cannot be negative")

    return True


def ensure_response_type(response: Any, expected_type: Type[FunctionResponse]) -> FunctionResponse:
    """Ensure a response matches the expected type.

    Args:
        response: The response to validate
        expected_type: The expected response type

    Returns:
        FunctionResponse: The validated response

    Raises:
        ValidationError: If response is invalid
    """
    if not isinstance(response, expected_type):
        raise ValidationError(
            f"Expected {expected_type.__name__} but got {type(response).__name__}")

    if isinstance(response, ToolResponse):
        validate_tool_response(response)
    elif isinstance(response, FilterResponse):
        validate_filter_response(response)
    elif isinstance(response, PipelineResponse):
        validate_pipeline_response(response)

    return response


def create_error_response(
    error: Exception,
    function_type: str,
    function_name: str,
    **kwargs
) -> FunctionResponse:
    """Create an error response of the appropriate type.

    Args:
        error: The error that occurred
        function_type: The type of function ("tool", "filter", or "pipeline")
        function_name: The name of the function
        **kwargs: Additional response-specific fields

    Returns:
        FunctionResponse: An appropriate error response
    """
    base_args = {
        "success": False,
        "error": str(error)
    }

    if function_type == "tool":
        return ToolResponse(
            **base_args,
            tool_name=function_name,
            result=None,
            **kwargs
        )
    elif function_type == "filter":
        return FilterResponse(
            **base_args,
            filter_name=function_name,
            modified_data={},
            changes_made=False,
            **kwargs
        )
    else:  # pipeline
        return PipelineResponse(
            **base_args,
            pipeline_name=function_name,
            results=[],
            steps_completed=0,
            total_steps=0,
            **kwargs
        )


# Constants from config
APP_CONSTANTS = {
    "DEFAULT_MODEL": config.llm.model,
    "MODEL_TEMPERATURE": config.llm.temperature,
    "MAX_TOKENS": config.llm.max_tokens,
    "FUNCTION_CALLS_ENABLED": config.llm.enable_tools,
    "ENABLE_MODEL_FILTER": config.functions.enable_model_filter,
    "MODEL_FILTER_LIST": config.functions.model_filter_list,
    "BASE_URL": config.llm.base_url,
    "MODEL_REQUEST_TIMEOUT": config.llm.timeout,
    "GENERATION_REQUEST_TIMEOUT": config.llm.timeout
}


def get_registered_tools() -> List[Dict[str, Any]]:
    """Get all registered tools from the function service.

    Returns:
        List of registered tool schemas
    """
    # Import here to avoid circular dependency
    from app.dependencies.providers import Providers
    function_service = Providers.get_function_service()
    return function_service.get_function_schemas()


def verify_registered_tool(tool_name: str) -> Optional[Dict[str, Any]]:
    """Verify if a tool is registered and get its schema.

    Args:
        tool_name: Name of the tool to verify

    Returns:
        Tool schema if registered, None otherwise
    """
    # Import here to avoid circular dependency
    from app.dependencies.providers import Providers
    function_service = Providers.get_function_service()
    tool_class = function_service.get_function(tool_name)

    if not tool_class:
        return None

    # Verify it's actually a tool type
    if tool_class.model_fields['type'].default != FunctionType.TOOL:
        return None

    # Get the schema for this tool
    schemas = function_service.get_function_schemas()
    return next(
        (schema for schema in schemas
         if schema.get("function", {}).get("name") == tool_name),
        None
    )


def validate_tool_names(tool_names: List[str]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Validate tool names against registered tools.

    Args:
        tool_names: List of tool names to validate

    Returns:
        Tuple of (valid_tools, invalid_tools, tool_schemas)
    """
    valid_tools = []
    invalid_tools = []
    tool_schemas = []

    for name in tool_names:
        schema = verify_registered_tool(name)
        if schema:
            valid_tools.append(name)
            tool_schemas.append(schema)
        else:
            invalid_tools.append(name)
            logger.warning(
                f"Tool '{name}' is not registered or is not a valid tool")

    return valid_tools, invalid_tools, tool_schemas


def validate_tool_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a tool policy configuration.

    Args:
        policy: Tool policy to validate

    Returns:
        Validated policy dictionary

    Raises:
        ToolValidationError: If policy is invalid
    """
    valid_keys = {
        "rate_limit", "max_retries", "timeout", "cache_results",
        "max_tokens", "temperature", "require_confirmation"
    }

    invalid_keys = set(policy.keys()) - valid_keys
    if invalid_keys:
        raise ToolValidationError(f"Invalid policy keys: {invalid_keys}")

    validated = {}

    # Validate rate limit
    if "rate_limit" in policy:
        rate_limit = policy["rate_limit"]
        if not isinstance(rate_limit, (int, float)) or rate_limit <= 0:
            raise ToolValidationError("rate_limit must be a positive number")
        validated["rate_limit"] = rate_limit

    # Validate max retries
    if "max_retries" in policy:
        max_retries = policy["max_retries"]
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ToolValidationError(
                "max_retries must be a non-negative integer")
        validated["max_retries"] = max_retries

    # Validate timeout
    if "timeout" in policy:
        timeout = policy["timeout"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ToolValidationError("timeout must be a positive number")
        validated["timeout"] = timeout

    # Validate cache_results
    if "cache_results" in policy:
        if not isinstance(policy["cache_results"], bool):
            raise ToolValidationError("cache_results must be a boolean")
        validated["cache_results"] = policy["cache_results"]

    # Validate max_tokens
    if "max_tokens" in policy:
        max_tokens = policy["max_tokens"]
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ToolValidationError("max_tokens must be a positive integer")
        validated["max_tokens"] = max_tokens

    # Validate temperature
    if "temperature" in policy:
        temperature = policy["temperature"]
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 1:
            raise ToolValidationError(
                "temperature must be a number between 0 and 1")
        validated["temperature"] = temperature

    # Validate require_confirmation
    if "require_confirmation" in policy:
        if not isinstance(policy["require_confirmation"], bool):
            raise ToolValidationError("require_confirmation must be a boolean")
        validated["require_confirmation"] = policy["require_confirmation"]

    # If no validation errors, include any default values for missing optional fields
    defaults = {
        "max_retries": 3,
        "timeout": 30.0,
        "cache_results": False,
        "require_confirmation": False,
        "temperature": 0.7
    }

    # Apply defaults for any missing optional fields
    for key, default_value in defaults.items():
        if key not in validated:
            validated[key] = default_value

    return validated


def get_safe_tool_list(
    tool_names: Optional[List[str]] = None,
    required_capabilities: Optional[List[str]] = None,
    validate_policies: bool = True
) -> List[Dict[str, Any]]:
    """Get a filtered and validated list of registered tools.

    Args:
        tool_names: Optional list of specific tool names to include
        required_capabilities: Optional list of required capabilities
        validate_policies: Whether to validate tool policies

    Returns:
        List of validated tool schemas
    """
    # Import here to avoid circular dependency
    from app.dependencies.providers import Providers
    function_service = Providers.get_function_service()
    all_tools = function_service.get_function_schemas()
    filtered_tools = []

    for tool in all_tools:
        tool_name = tool.get("function", {}).get("name")
        if not tool_name:
            continue

        # Skip if not in specified tools (if tool_names provided)
        if tool_names and tool_name not in tool_names:
            continue

        # Check capabilities
        if required_capabilities:
            tool_capabilities = tool.get("function", {}).get(
                "metadata", {}).get("capabilities", [])
            if not all(cap in tool_capabilities for cap in required_capabilities):
                continue

        # Validate tool schema
        try:
            # Validate policy if present and requested
            if validate_policies and "metadata" in tool["function"] and "policy" in tool["function"]["metadata"]:
                try:
                    validated_policy = validate_tool_policy(
                        tool["function"]["metadata"]["policy"])
                    tool["function"]["metadata"]["policy"] = validated_policy
                except ToolValidationError as e:
                    logger.warning(f"Invalid tool policy for {tool_name}: {e}")
                    continue

            filtered_tools.append(tool)
        except Exception as e:
            logger.error(f"Error validating tool {tool_name}: {e}")
            continue

    return filtered_tools


__all__ = [
    'get_last_user_message',
    'get_last_assistant_message',
    'get_system_message',
    'validate_function_response',
    'validate_tool_response',
    'validate_filter_response',
    'validate_pipeline_response',
    'ensure_response_type',
    'create_error_response',
    'APP_CONSTANTS',
    'get_registered_tools',
    'verify_registered_tool',
    'validate_tool_names',
    'validate_tool_policy',
    'get_safe_tool_list',
]
