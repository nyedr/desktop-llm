"""Utility functions for the function system."""

import logging
from typing import Dict, Any, List, Optional, Type
from app.core.config import config
from app.models.chat import AssistantMessage, StrictChatMessage, SystemMessage, ToolMessage, UserMessage
from app.models.function import (
    ToolResponse, FilterResponse, PipelineResponse,
    FunctionResult, ValidationError
)

logger = logging.getLogger(__name__)


def get_last_user_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the last user message from the conversation history.

    Args:
        messages: List of message dictionaries

    Returns:
        The last user message or None if not found
    """
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


def ensure_strict_message(msg: Any) -> StrictChatMessage:
    """Convert input to StrictChatMessage, filtering invalid fields.

    Args:
        msg: Input message (dict or StrictChatMessage)

    Returns:
        Properly typed StrictChatMessage instance
    """
    if isinstance(msg, StrictChatMessage):
        return msg

    if not isinstance(msg, dict):
        raise ValueError(f"Invalid message type: {type(msg)}")

    role = msg.get('role')
    if role == 'user':
        # Filter for UserMessage fields
        msg_data = {k: v for k, v in msg.items()
                    if k in UserMessage.model_fields}
        return UserMessage(**msg_data)
    elif role == 'assistant':
        # Filter for AssistantMessage fields
        msg_data = {k: v for k, v in msg.items()
                    if k in AssistantMessage.model_fields}
        return AssistantMessage(**msg_data)
    elif role == 'system':
        # Filter for SystemMessage fields
        msg_data = {k: v for k, v in msg.items()
                    if k in SystemMessage.model_fields}
        return SystemMessage(**msg_data)
    elif role == 'tool':
        # Filter for ToolMessage fields
        msg_data = {k: v for k, v in msg.items()
                    if k in ToolMessage.model_fields}
        return ToolMessage(**msg_data)
    else:
        raise ValueError(f"Invalid message role: {role}")


def validate_function_response(response: FunctionResult) -> bool:
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


def ensure_response_type(response: Any, expected_type: Type[FunctionResult]) -> FunctionResult:
    """Ensure a response matches the expected type.

    Args:
        response: The response to validate
        expected_type: The expected response type

    Returns:
        FunctionResult: The validated response

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
) -> FunctionResult:
    """Create an error response of the appropriate type.

    Args:
        error: The error that occurred
        function_type: The type of function ("tool", "filter", or "pipeline")
        function_name: The name of the function
        **kwargs: Additional response-specific fields

    Returns:
        FunctionResult: An appropriate error response
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
    "DEFAULT_MODEL": config.DEFAULT_MODEL,
    "MODEL_TEMPERATURE": config.MODEL_TEMPERATURE,
    "MAX_TOKENS": config.MAX_TOKENS,
    "FUNCTION_CALLS_ENABLED": config.FUNCTION_CALLS_ENABLED,
    "ENABLE_MODEL_FILTER": config.ENABLE_MODEL_FILTER,
    "MODEL_FILTER_LIST": config.MODEL_FILTER_LIST,
    "OLLAMA_BASE_URLS": config.OLLAMA_BASE_URLS,
    "MODEL_REQUEST_TIMEOUT": config.MODEL_REQUEST_TIMEOUT,
    "GENERATION_REQUEST_TIMEOUT": config.GENERATION_REQUEST_TIMEOUT
}

__all__ = [
    'get_last_user_message',
    'get_last_assistant_message',
    'get_system_message',
    'ensure_strict_message',
    'validate_function_response',
    'validate_tool_response',
    'validate_filter_response',
    'validate_pipeline_response',
    'ensure_response_type',
    'create_error_response',
    'APP_CONSTANTS'
]
