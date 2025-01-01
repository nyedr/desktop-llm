"""Message handling operations for chat router."""

import json
import logging
from typing import Dict, Any, List, Union
from app.models.chat import ChatStreamEvent
from app.models.function import Filter
from app.utils.filters import apply_filters

logger = logging.getLogger(__name__)


async def handle_assistant_message(
    request_id: str,
    response: Dict[str, Any],
    filters: List[Filter]
) -> ChatStreamEvent:
    """Handle assistant message with outlet filtering."""
    assistant_message = {
        "role": "assistant",
        "content": response.get("content", "")
    }

    if not filters:
        print(assistant_message['content'], end="", flush=True)
        return ChatStreamEvent(event="message", data=json.dumps(assistant_message))

    return await apply_filters(
        filters=filters,
        data=assistant_message,
        request_id=request_id,
        direction="outlet",
        as_event=True,
        filter_name="outlet_message_filters"
    )


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
