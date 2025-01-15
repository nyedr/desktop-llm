"""Utility functions for working with chat tools."""

import json
import logging
from typing import Dict, Any, List, Union, Optional, Tuple
from app.models.chat import ChatStreamEvent
from app.services.function_service import FunctionService

logger = logging.getLogger(__name__)


async def process_tool_stream(
    request_id: str,
    chunk: Union[str, Dict[str, Any]],
    function_service: Optional[FunctionService] = None,
    current_tool_call: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[ChatStreamEvent], Optional[Dict[str, Any]], bool]:
    """Process streaming tool calls from model response.

    Args:
        request_id: Request ID for logging
        chunk: Response chunk from model
        function_service: Optional function service for executing tools
        current_tool_call: Current accumulated tool call

    Returns:
        Tuple of (event to yield, updated tool call state, is_complete flag)
    """
    if not isinstance(chunk, dict) or "tool_calls" not in chunk:
        return None, current_tool_call, False

    tool_calls = chunk["tool_calls"]
    if not tool_calls:
        return None, current_tool_call, False

    tool_call = tool_calls[0]  # Handle one tool call at a time

    # Initialize current tool call if needed
    if not current_tool_call:
        current_tool_call = {
            "id": tool_call.get("id"),
            "type": "function",
            "function": {
                "name": tool_call["function"].get("name"),
                "arguments": ""
            }
        }

    # Accumulate arguments if present
    if tool_call["function"].get("arguments"):
        current_tool_call["function"]["arguments"] += tool_call["function"]["arguments"]

    # Check if tool call is complete
    is_complete = False
    try:
        if current_tool_call["function"]["arguments"]:
            # Try to parse accumulated arguments as JSON
            json.loads(current_tool_call["function"]["arguments"])
            is_complete = True
    except json.JSONDecodeError:
        pass

    # If complete, execute the tool and return all events
    if is_complete and function_service:
        try:
            # Execute complete tool call
            result = await function_service.execute_function(
                current_tool_call["function"]["name"],
                json.loads(current_tool_call["function"]["arguments"]),
                request_id=request_id  # Pass request_id for profiling
            )

            # Handle ToolResponse object
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            else:
                result_dict = {
                    "success": getattr(result, "success", True),
                    "result": getattr(result, "result", str(result)),
                    "error": getattr(result, "error", None),
                    "metadata": getattr(result, "metadata", {})
                }

            return ChatStreamEvent(
                event="message",
                data=json.dumps({
                    "role": "tool",
                    "name": current_tool_call["function"]["name"],
                    "content": json.dumps(result_dict),
                    "tool_call_id": current_tool_call["id"]
                })
            ), None, True
        except Exception as e:
            logger.error(f"[{request_id}] Error executing tool: {str(e)}")
            return ChatStreamEvent(
                event="error",
                data=json.dumps({"error": f"Tool execution failed: {str(e)}"})
            ), None, True

    # Return accumulated state for incomplete tool calls
    return ChatStreamEvent(
        event="message",
        data=json.dumps({
            "role": "assistant",
            "content": "",
            "tool_calls": [current_tool_call]
        })
    ), current_tool_call, is_complete


def format_tools_for_chat(tools: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
    """Format tools for chat API request.

    Args:
        tools: List of tool configurations
        request_id: Request ID for logging

    Returns:
        List of formatted tools ready for API request
    """
    formatted_tools = []
    for tool in tools:
        # Handle case where tool is already formatted
        if isinstance(tool, dict) and tool.get("type") == "function":
            formatted_tools.append(tool)
            continue

        # Handle case where tool is a function object
        if hasattr(tool, "name"):
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {})
                }
            }
        # Handle case where tool is a dict with function info
        elif isinstance(tool, dict) and "function" in tool:
            formatted_tool = {
                "type": "function",
                "function": tool["function"]
            }
        # Handle case where tool is a dict with direct properties
        elif isinstance(tool, dict):
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                }
            }
        else:
            logger.warning(
                f"[{request_id}] Skipping invalid tool format: {tool}")
            continue

        if not formatted_tool["function"]["name"]:
            logger.warning(
                f"[{request_id}] Skipping tool without name: {tool}")
            continue

        formatted_tools.append(formatted_tool)

    return formatted_tools
