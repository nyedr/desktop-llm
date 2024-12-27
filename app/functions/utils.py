"""Utility functions for the function system."""

import logging
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from app.core.config import config
from app.models.chat import ChatMessage
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

# Initialize services
model_service = ModelService()


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


async def get_all_models() -> List[str]:
    """Get a list of all available models.

    Returns:
        List of model names
    """
    try:
        return await model_service.get_models()
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return []


async def generate_chat_completion(
    messages: List[Union[Dict[str, Any], ChatMessage]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    enable_tools: bool = False,
    function_service=None
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """Generate a chat completion using the model service.

    Args:
        messages: List of messages in the conversation
        model: Model to use (defaults to config.DEFAULT_MODEL)
        temperature: Temperature for generation (defaults to config.MODEL_TEMPERATURE)
        max_tokens: Maximum tokens to generate (defaults to config.MAX_TOKENS)
        stream: Whether to stream the response
        tools: List of available tools
        enable_tools: Whether to enable tool usage
        function_service: Service for executing functions

    Returns:
        AsyncGenerator yielding response chunks
    """
    try:
        async for response in model_service.chat(
            messages=messages,
            model=model or config.DEFAULT_MODEL,
            temperature=temperature or config.MODEL_TEMPERATURE,
            max_tokens=max_tokens or config.MAX_TOKENS,
            stream=stream,
            tools=tools,
            enable_tools=enable_tools,
            function_service=function_service
        ):
            yield response
    except Exception as e:
        logger.error(f"Error generating chat completion: {e}")
        yield {"error": str(e)}

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
    'get_all_models',
    'generate_chat_completion',
    'APP_CONSTANTS',
    'USER_SETTINGS'
]
