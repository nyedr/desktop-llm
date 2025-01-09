"""Tool-related operations for chat router."""

import json
import logging
from typing import Dict, Any, Union, List, Optional, AsyncGenerator, Tuple
from app.models.chat import ChatStreamEvent
from app.models.function import ToolResponse
from app.models.function import ValidationError
from app.functions.utils import validate_tool_response, create_error_response
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


async def handle_tool_response(
    request_id: str,
    response: Union[Dict[str, Any], ToolResponse]
) -> ChatStreamEvent:
    """Handle tool/function response."""
    logger.info(f"[{request_id}] Processing tool response")

    try:
        # Convert dict to tool message format
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

    # Send raw tool call message first
    events.append(ChatStreamEvent(
        event="message",
        data=json.dumps({
            "role": "assistant",
            "content": "",
            "tool_calls": response["tool_calls"]
        })
    ))

    try:
        tool_responses = await function_service.handle_tool_calls(response["tool_calls"])
        for tool_response in tool_responses:
            if hasattr(tool_response, 'to_dict'):
                result_dict = tool_response.to_dict()
            else:
                result_dict = {
                    "success": getattr(tool_response, "success", True),
                    "result": getattr(tool_response, "result", str(tool_response)),
                    "error": getattr(tool_response, "error", None),
                    "metadata": getattr(tool_response, "metadata", {})
                }

            # Send tool response
            events.append(ChatStreamEvent(
                event="message",
                data=json.dumps({
                    "role": "tool",
                    "name": tool_response.tool_name if hasattr(tool_response, 'tool_name') else response["tool_calls"][0]["function"]["name"],
                    "content": json.dumps(result_dict),
                    "tool_call_id": response["tool_calls"][0]["id"]
                })
            ))

    except Exception as e:
        logger.error(f"[{request_id}] Error executing tool calls: {e}")
        events.append(ChatStreamEvent(
            event="error",
            data=json.dumps({"error": f"Tool execution failed: {str(e)}"})
        ))

    return events


async def process_streaming_tool_calls(
    request_id: str,
    response,
    function_service: Optional[FunctionService] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Process streaming tool calls from model response."""
    current_tool_call = None
    async for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield {'content': delta.content}
            elif hasattr(delta, 'tool_calls') and delta.tool_calls:
                # Start or update tool call
                tool_call = delta.tool_calls[0]
                logger.debug(f"[{request_id}] Tool call in progress")

                if not current_tool_call:
                    current_tool_call = {
                        'id': tool_call.id,
                        'type': 'function',
                        'function': {
                            'name': getattr(tool_call.function, 'name', None),
                            'arguments': getattr(tool_call.function, 'arguments', '')
                        }
                    }
                else:
                    # Accumulate function arguments
                    if hasattr(tool_call.function, 'arguments'):
                        current_tool_call['function']['arguments'] += tool_call.function.arguments
                    if hasattr(tool_call.function, 'name') and tool_call.function.name:
                        current_tool_call['function']['name'] = tool_call.function.name

                # If we have a complete tool call, process it
                if current_tool_call['function']['name'] and current_tool_call['function']['arguments'].endswith('}'):
                    logger.info(
                        f"[{request_id}] Processing tool call: {current_tool_call['function']['name']}")
                    if function_service:
                        events = await handle_tool_calls(
                            request_id,
                            {'tool_calls': [current_tool_call]},
                            function_service
                        )
                        for event in events:
                            if isinstance(event, ChatStreamEvent):
                                result = json.loads(event.data)
                                if result.get('content'):
                                    yield {'content': result['content']}
                    current_tool_call = None
            # Handle end of response
            elif not delta.content and not hasattr(delta, 'tool_calls'):
                if current_tool_call:
                    logger.warning(
                        f"[{request_id}] Incomplete tool call at end of response")


async def process_non_streaming_tool_calls(
    request_id: str,
    response,
    function_service: Optional[FunctionService] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Process non-streaming tool calls from model response."""
    if response.choices:
        choice = response.choices[0]
        if choice.message.content is not None:
            yield {'content': choice.message.content}
        elif hasattr(choice.message, 'tool_calls') and choice.message.tool_calls and function_service:
            logger.info(f"[{request_id}] Processing non-stream tool calls")
            events = await handle_tool_calls(
                request_id,
                {'tool_calls': choice.message.tool_calls},
                function_service
            )
            for event in events:
                if isinstance(event, ChatStreamEvent):
                    result = json.loads(event.data)
                    if result.get('content'):
                        yield {'content': result['content']}


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
