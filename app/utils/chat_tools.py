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
    chunk: Union[Dict[str, Any], str],
    function_service: Optional[FunctionService],
    current_tool_call: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[ChatStreamEvent], Dict[str, Any], bool]:
    """Process a streaming chunk for tool calls.

    Args:
        request_id: Request identifier
        chunk: Response chunk from model
        function_service: Function service for executing tools
        current_tool_call: Current tool call being processed

    Returns:
        Tuple of (event to yield, updated tool call state, whether tool call is complete)
    """
    if not isinstance(chunk, dict) or "function_call" not in chunk:
        return None, current_tool_call, False

    try:
        tool_call = chunk["function_call"]
        logger.debug(f"[{request_id}] Processing tool call chunk")

        # Initialize or update tool call
        if not current_tool_call:
            current_tool_call = {
                'id': tool_call.get('id'),
                'type': 'function',
                'function': {
                    'name': tool_call.get('name'),
                    'arguments': tool_call.get('arguments', '')
                }
            }
        else:
            # Accumulate function arguments
            if 'arguments' in tool_call:
                current_tool_call['function']['arguments'] += tool_call['arguments']
            if 'name' in tool_call:
                current_tool_call['function']['name'] = tool_call['name']

        # Check if tool call is complete
        if current_tool_call['function'].get('name') and current_tool_call['function']['arguments'].endswith('}'):
            logger.info(
                f"[{request_id}] Executing tool: {current_tool_call['function']['name']}")

            if function_service:
                events = await handle_tool_calls(
                    request_id,
                    {'tool_calls': [current_tool_call]},
                    function_service
                )

                # Process tool response
                for event in events:
                    if isinstance(event, ChatStreamEvent):
                        return event, None, True  # Tool call complete

            return None, None, True  # Tool call complete but no event to yield

        return None, current_tool_call, False  # Tool call still in progress

    except Exception as e:
        logger.error(
            f"[{request_id}] Error processing tool call: {e}", exc_info=True)
        return ChatStreamEvent(
            event="error",
            data=json.dumps({"error": f"Tool processing error: {str(e)}"})
        ), None, True


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
    """Handle tool calls from the model response."""
    logger.info(f"[{request_id}] Processing tool calls")

    if not response.get('tool_calls'):
        logger.warning(f"[{request_id}] No tool calls found in response")
        return []

    try:
        # Execute tool calls
        events = await function_service.handle_tool_calls(response['tool_calls'])

        # Process and validate responses
        processed_events = []
        for event in events:
            if isinstance(event, ChatStreamEvent):
                try:
                    result = json.loads(event.data)
                    if result.get('content'):  # Only include non-empty responses
                        processed_events.append(event)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"[{request_id}] Error decoding tool response: {e}")

        return processed_events

    except Exception as e:
        logger.error(
            f"[{request_id}] Error handling tool calls: {e}", exc_info=True)
        raise


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
