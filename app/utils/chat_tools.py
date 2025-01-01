"""Tool-related operations for chat router."""

import json
import logging
from typing import Dict, Any, Union, List
from app.models.chat import ChatStreamEvent
from app.models.function import ToolResponse
from app.models.function import ValidationError
from app.functions.utils import validate_tool_response, create_error_response
from app.services.function_service import FunctionService

logger = logging.getLogger(__name__)


async def handle_tool_response(
    request_id: str,
    response: Union[Dict[str, Any], ToolResponse]
) -> ChatStreamEvent:
    """Handle tool/function response.

    Args:
        request_id: ID of the current request
        response: Tool response data (dict or ToolResponse)

    Returns:
        event to send
    """
    logger.info(f"[{request_id}] Processing tool/function response")
    logger.debug(
        f"[{request_id}] Raw tool/function response: {json.dumps(response, indent=2) if isinstance(response, dict) else str(response)}")

    try:
        # Convert dict to ToolResponse if needed
        if isinstance(response, dict):
            tool_message = {
                "role": "tool",
                "content": response.get("content", ""),
                "name": response.get("name", ""),
                "tool_call_id": response.get("tool_call_id")
            }
        else:
            # Handle ToolResponse object
            validate_tool_response(response)
            tool_message = {
                "role": "tool",
                "content": response.result if response.success else response.error,
                "name": response.tool_name,
                "tool_call_id": response.metadata.get("tool_call_id") if response.metadata else None
            }

        return ChatStreamEvent(event="message", data=json.dumps(tool_message))

    except ValidationError as e:
        logger.error(f"[{request_id}] Invalid tool response: {e}")
        error_response = create_error_response(
            error=e,
            function_type="tool",
            function_name=getattr(response, "tool_name", "unknown"),
            tool_call_id=getattr(response, "metadata", {}).get("tool_call_id")
        )
        return ChatStreamEvent(
            event="error",
            data=json.dumps({"error": error_response.error})
        )


async def handle_tool_calls(
    request_id: str,
    response: Dict[str, Any],
    function_service: FunctionService
) -> List[ChatStreamEvent]:
    """Handle tool calls from assistant.

    Args:
        request_id: ID of the current request
        response: Assistant response containing tool calls
        function_service: Service for handling function calls

    Returns:
        List of events to send
    """
    logger.info(
        f"[{request_id}] Tool calls detected: {json.dumps(response['tool_calls'], indent=2)}")
    events = []

    # Send raw tool call message
    events.append(ChatStreamEvent(event="message", data=json.dumps({
        "role": "assistant",
        "content": str(response)
    })))

    try:
        tool_responses = await function_service.handle_tool_calls(response["tool_calls"])
        for tool_response in tool_responses:
            event = await handle_tool_response(request_id, tool_response)
            events.append(event)

    except Exception as e:
        logger.error(f"[{request_id}] Error executing tool calls: {e}")
        error_response = create_error_response(
            error=e,
            function_type="tool",
            function_name="unknown"
        )
        events.append(ChatStreamEvent(
            event="error",
            data=json.dumps({"error": error_response.error})
        ))

    return events
