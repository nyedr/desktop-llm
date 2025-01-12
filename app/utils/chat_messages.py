"""Utility functions for working with chat messages."""

from datetime import datetime
import json
import logging
from typing import Dict, Any, List, Optional, Union
from app.models.function_base import Filter
from app.models.chat import (
    ChatRequest,
    ChatStreamEvent,
    AssistantMessage,
    StrictChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage
)
from app.utils.filters import apply_filters, get_filter
from app.core.config import config

logger = logging.getLogger(__name__)


def ensure_strict_message(msg: Any) -> StrictChatMessage:
    """Ensure a message is a StrictChatMessage instance.

    Args:
        msg: Message to convert

    Returns:
        StrictChatMessage instance
    """
    if isinstance(msg, StrictChatMessage):
        return msg

    if isinstance(msg, dict):
        role = msg.get("role", "").lower()
        content = msg.get("content", "")

        if role == "user":
            return UserMessage(content=content)
        elif role == "assistant":
            return AssistantMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        elif role == "tool":
            return ToolMessage(
                content=content,
                tool_name=msg.get("name", "unknown_tool"),
                tool_args=msg.get("arguments", {})
            )
        else:
            raise ValueError(f"Unknown message role: {role}")

    raise ValueError(f"Cannot convert {type(msg)} to StrictChatMessage")


async def handle_assistant_message(response: Union[str, Dict[str, Any]], filters: List[Dict[str, Any]], request_id: str) -> Optional[Dict[str, Any]]:
    """Handle assistant message and apply filters."""
    try:
        # Handle string responses (direct content)
        if isinstance(response, str):
            message = {
                "role": "assistant",
                "content": response
            }
        # Handle dict responses (tool calls or structured content)
        elif isinstance(response, dict):
            message = {
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": response.get("tool_calls", [])
            }
        else:
            logger.error(
                f"[{request_id}] Invalid response type: {type(response)}")
            return None

        # Apply filters
        if filters:
            filtered_message = message
            for filter_config in filters:
                filter_instance = get_filter(filter_config)
                if filter_instance:
                    try:
                        filtered_message = await filter_instance.process(filtered_message)
                    except Exception as e:
                        logger.error(
                            f"[{request_id}] Filter processing error: {str(e)}")
                else:
                    logger.warning(
                        f"[{request_id}] Failed to create filter from config: {filter_config}")
            message = filtered_message

        return message

    except Exception as e:
        logger.error(
            f"[{request_id}] Error handling assistant message: {str(e)}", exc_info=True)
        return None


def format_conversation_metadata(
    request_id: str,
    model: str,
    final_messages: List[Dict[str, Any]],
    tool_response: Any,
    chat_request: ChatRequest,
    current_message: Dict[str, Any],
    last_user_message: str
) -> Dict[str, str]:
    return {
        "request_id": request_id,
        "model": model,
        "message_count": len(final_messages),
        "has_tool_calls": bool(tool_response),
        "enable_tools": chat_request.enable_tools,
        "timestamp": datetime.now().isoformat(),
        "temperature": chat_request.temperature or config.llm.temperature,
        "max_tokens": chat_request.max_tokens or config.llm.max_tokens,
        "user_message": last_user_message,
        "assistant_response": current_message.get("content", "") if isinstance(current_message, dict) else getattr(current_message, "content", ""),
        "tool_response": tool_response.dict() if hasattr(tool_response, "dict") else tool_response
    }


async def handle_string_chunk(
    request_id: str,
    response: str,
    filters: List[Filter]
) -> ChatStreamEvent:
    """Handle string chunk with outlet filtering."""
    chunk_message = {
        "role": "assistant",
        "content": str(response)
    }

    if not filters:
        print(chunk_message['content'], end="", flush=True)
        return ChatStreamEvent(event="message", data=json.dumps(chunk_message))

    return await apply_filters(
        filters=filters,
        data=chunk_message,
        request_id=request_id,
        direction="outlet",
        as_event=True,
        filter_name="outlet_chunk_filters"
    )


def convert_to_dict(obj: Any) -> Union[Dict[str, Any], List[Any], Any]:
    """Recursively convert Message objects to dictionaries.

    Args:
        obj: Object to convert (Message, dict, list, or other)

    Returns:
        Converted object in dictionary form
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    return obj
