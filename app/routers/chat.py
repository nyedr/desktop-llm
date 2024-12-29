"""Chat router for handling chat-related endpoints and streaming responses."""
import json
import logging
from typing import List, Optional, AsyncGenerator, Any, Dict, Union
from fastapi import APIRouter, Request, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse
from app.context.llm_context import MemoryType
from app.core.config import config
from app.dependencies.providers import (
    get_agent,
    get_model_service,
    get_function_service,
    get_langchain_service,
    get_chroma_service
)
from app.services.agent import Agent
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.services.langchain_service import LangChainService
from app.services.chroma_service import ChromaService
from app.models.chat import (
    ChatRequest, ChatStreamEvent, StrictChatMessage
)
from app.functions.base import FunctionType, Filter
import asyncio
from pydantic import BaseModel
from app.models.memory import MemoryType

router = APIRouter()
logger = logging.getLogger(__name__)


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


async def apply_filters(
    filters: List[Filter],
    data: Dict[str, Any],
    request_id: str,
    direction: str = "outlet"
) -> Dict[str, Any]:
    """Apply filters to data in a consistent way.

    Args:
        filters: List of filters to apply
        data: Data to filter
        request_id: ID of the current request for logging
        direction: Direction of filtering ("inlet" or "outlet")

    Returns:
        Filtered data

    Raises:
        Exception: If filter application fails
    """
    if not filters:
        return data

    # Ensure we're working with a dictionary
    if hasattr(data, "model_dump"):
        data = data.model_dump()

    sorted_filters = sorted(filters, key=lambda f: f.priority or 0)
    filtered_data = data

    for filter_func in sorted_filters:
        try:
            if direction == "inlet":
                filtered_data = await filter_func.inlet(filtered_data)
            else:
                filtered_data = await filter_func.outlet(filtered_data)
        except Exception as e:
            logger.error(
                f"[{request_id}] Error in filter {filter_func.__class__.__name__} {direction}: {e}",
                exc_info=True
            )
            continue

    return filtered_data


async def setup_chat_components(
    request_id: str,
    chat_request: ChatRequest,
    function_service: FunctionService
) -> tuple[Optional[List[Dict]], List[Filter], Optional[Any]]:
    """Setup function schemas, filters, and pipeline for chat."""
    # Get available functions
    function_schemas = None
    if chat_request.enable_tools:
        logger.info(
            f"[{request_id}] Tools are enabled, getting function schemas")
        function_schemas = function_service.get_function_schemas()
        if function_schemas:
            logger.info(
                f"[{request_id}] Available tools: {[f['function']['name'] for f in function_schemas]}")
        else:
            logger.warning(f"[{request_id}] No tool schemas available")

    # Get requested filters
    filters = []
    if chat_request.filters:
        logger.info(f"[{request_id}] Getting filters: {chat_request.filters}")
        for filter_name in chat_request.filters:
            filter_class = function_service.get_function(filter_name)
            if filter_class and filter_class.model_fields['type'].default == FunctionType.FILTER:
                logger.info(
                    f"[{request_id}] Instantiating filter: {filter_name}")
                try:
                    filter_instance = filter_class()
                    filters.append(filter_instance)
                    logger.info(f"[{request_id}] Added filter: {filter_name}")
                except Exception as e:
                    logger.error(
                        f"[{request_id}] Error instantiating filter {filter_name}: {e}")
            else:
                logger.warning(
                    f"[{request_id}] Filter not found or invalid type: {filter_name}")

    # Get requested pipeline
    pipeline = None
    if chat_request.pipeline:
        logger.info(
            f"[{request_id}] Getting pipeline: {chat_request.pipeline}")
        pipeline_class = function_service.get_function(chat_request.pipeline)
        if pipeline_class and pipeline_class.model_fields['type'].default == FunctionType.PIPELINE:
            logger.info(
                f"[{request_id}] Instantiating pipeline: {chat_request.pipeline}")
            try:
                pipeline = pipeline_class()
                logger.info(
                    f"[{request_id}] Added pipeline: {chat_request.pipeline}")
            except Exception as e:
                logger.error(
                    f"[{request_id}] Error instantiating pipeline {chat_request.pipeline}: {e}")
        else:
            logger.warning(
                f"[{request_id}] Pipeline not found or invalid type: {chat_request.pipeline}")

    return function_schemas, filters, pipeline


async def handle_tool_response(
    request_id: str,
    response: Dict[str, Any],
    tool_call_in_progress: bool
) -> tuple[ChatStreamEvent, bool]:
    """Handle tool/function response.

    Args:
        request_id: ID of the current request
        response: Tool response data
        tool_call_in_progress: Whether a tool call is in progress

    Returns:
        Tuple of (event to send, updated tool_call_in_progress flag)
    """
    logger.info(f"[{request_id}] Processing tool/function response")
    logger.debug(
        f"[{request_id}] Raw tool/function response: {json.dumps(response, indent=2)}")

    tool_message = {
        "role": "tool",
        "content": response.get("content", ""),
        "name": response.get("name", ""),
        "tool_call_id": response.get("tool_call_id")
    }
    return ChatStreamEvent(event="message", data=json.dumps(tool_message)), False


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
            logger.info(
                f"[{request_id}] Processing tool response: {json.dumps(tool_response, indent=2)}")
            tool_message = {
                "role": "tool",
                "content": tool_response["content"],
                "name": tool_response["name"]
            }
            events.append(ChatStreamEvent(
                event="message", data=json.dumps(tool_message)))
    except Exception as e:
        logger.error(f"[{request_id}] Error executing tool calls: {e}")
        events.append(ChatStreamEvent(
            event="error",
            data=json.dumps({"error": f"Error executing tool calls: {str(e)}"})
        ))

    return events


async def handle_assistant_message(
    request_id: str,
    response: Dict[str, Any],
    filters: List[Filter]
) -> ChatStreamEvent:
    """Handle assistant message."""
    assistant_message = {
        "role": "assistant",
        "content": response.get("content", "")
    }
    filtered_message = await apply_filters(filters, assistant_message, request_id, "outlet")
    print(filtered_message['content'], end="", flush=True)
    return ChatStreamEvent(event="message", data=json.dumps(filtered_message))


async def handle_string_chunk(
    request_id: str,
    response: str,
    filters: List[Filter]
) -> ChatStreamEvent:
    """Handle string chunk from model."""
    chunk_message = {
        "role": "assistant",
        "content": str(response)
    }
    filtered_message = await apply_filters(filters, chunk_message, request_id, "outlet")
    print(filtered_message['content'], end="", flush=True)
    return ChatStreamEvent(event="message", data=json.dumps(filtered_message))


async def verify_model_availability(
    request_id: str,
    model: str,
    model_service: ModelService
) -> Optional[ChatStreamEvent]:
    """Verify model availability.

    Args:
        request_id: ID of the current request
        model: Model name to verify
        model_service: Service for model operations

    Returns:
        Error event if model not available, None if available
    """
    try:
        models = await model_service.get_all_models(request_id)
    except Exception as model_error:
        logger.error(
            f"[{request_id}] Error fetching models: {model_error}", exc_info=True)
        return ChatStreamEvent(
            event="error",
            data=json.dumps({"error": "Failed to fetch available models"})
        )

    if model not in models:
        logger.error(f"[{request_id}] Model {model} not available")
        return ChatStreamEvent(
            event="error",
            data=json.dumps({"error": f"Model {model} not available"})
        )

    return None


async def stream_chat_response(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent = Depends(get_agent),
    model_service: ModelService = Depends(get_model_service),
    function_service: FunctionService = Depends(get_function_service),
    langchain_service: LangChainService = Depends(get_langchain_service),
    is_test: bool = False
) -> AsyncGenerator[ChatStreamEvent, None]:
    """Generate streaming chat response."""
    request_id = str(id(request))
    logger.info(f"[{request_id}] Starting chat stream")
    tool_call_in_progress = False

    try:
        # Setup components
        function_schemas, filters, pipeline = await setup_chat_components(
            request_id, chat_request, function_service)

        # Apply inlet filters
        data = await apply_filters(
            filters=filters,
            data={"messages": chat_request.messages},
            request_id=request_id,
            direction="inlet"
        )
        chat_request.messages = data["messages"]

        # Process conversation with LangChain using new context management
        if chat_request.enable_memory and langchain_service:
            try:
                chat_request.messages = await langchain_service.process_conversation(
                    messages=chat_request.messages,
                    memory_type=chat_request.memory_type,
                    conversation_id=chat_request.conversation_id,
                    enable_summarization=chat_request.enable_summarization,
                    metadata_filter=chat_request.memory_filter,
                    top_k=chat_request.top_k_memories
                )
                logger.info(
                    f"[{request_id}] Added {chat_request.memory_type} memory context to conversation")
            except Exception as e:
                logger.warning(
                    f"[{request_id}] Failed to add memory context: {e}")

        # Apply pipeline
        processed_messages = chat_request.messages
        if pipeline:
            logger.debug(f"[{request_id}] Applying pipeline: {pipeline.name}")
            try:
                pipeline_data = await pipeline.pipe({"messages": chat_request.messages})
                logger.debug(
                    f"[{request_id}] Pipeline result: {pipeline_data}")

                if "messages" in pipeline_data and pipeline_data["messages"]:
                    processed_messages = pipeline_data["messages"]
                else:
                    logger.warning(
                        f"[{request_id}] Pipeline returned empty messages, using original messages")

                # Handle pipeline summary
                if "summary" in pipeline_data:
                    pipeline_summary = pipeline_data["summary"]
                    # Send initial summary event
                    yield ChatStreamEvent(
                        event="pipeline",
                        data=json.dumps({
                            "summary": pipeline_summary,
                            "status": "processing"
                        })
                    )

                    # Process detailed results if available
                    if isinstance(pipeline_summary, dict):
                        for key, value in pipeline_summary.items():
                            if isinstance(value, list):
                                for item in value:
                                    if item:  # Only send non-empty items
                                        yield ChatStreamEvent(
                                            event="pipeline",
                                            data=json.dumps({
                                                "content_type": key,
                                                "content": item,
                                                "status": "processing"
                                            })
                                        )
                                        await asyncio.sleep(0.1)

                    # Send completion event
                    yield ChatStreamEvent(
                        event="pipeline",
                        data=json.dumps({
                            "status": "complete",
                            "summary": pipeline_summary
                        })
                    )

            except Exception as e:
                logger.error(
                    f"[{request_id}] Error in pipeline execution: {e}")
                yield ChatStreamEvent(
                    event="pipeline_error",
                    data=json.dumps({
                        "error": str(e),
                        "status": "failed"
                    })
                )

        # Verify model
        model = chat_request.model or config.DEFAULT_MODEL
        error_event = await verify_model_availability(request_id, model, model_service)
        if error_event:
            yield error_event
            return

        # Stream chat completion
        first_response = True

        # Log the context window before streaming response
        if chat_request.messages:
            print(f"\nContext window for request {request_id}:", flush=True)
            for msg in chat_request.messages:
                role = msg.get("role") if isinstance(msg, dict) else msg.role
                content = msg.get("content") if isinstance(
                    msg, dict) else msg.content
                print(f"{role}: {content}", flush=True)
            print("\nResponse:", flush=True)

        async for response in agent.chat(
            messages=processed_messages,
            model=model,
            temperature=chat_request.temperature or config.MODEL_TEMPERATURE,
            max_tokens=chat_request.max_tokens or config.MAX_TOKENS,
            stream=True,
            tools=function_schemas,
            enable_tools=chat_request.enable_tools,
            enable_memory=chat_request.enable_memory,
            memory_filter=chat_request.memory_filter,
            top_k_memories=chat_request.top_k_memories
        ):
            if first_response:
                yield ChatStreamEvent(event="start", data=json.dumps({"status": "streaming"}))
                first_response = False

            if not is_test and await request.is_disconnected():
                logger.info(f"[{request_id}] Client disconnected")
                if tool_call_in_progress:
                    logger.info(
                        f"[{request_id}] Waiting for tool call to complete")
                    continue
                break

            if response:
                if isinstance(response, dict):
                    if response.get("role") in ["tool", "function"]:
                        event, tool_call_in_progress = await handle_tool_response(
                            request_id, response, tool_call_in_progress)
                        yield event

                    elif response.get("role") == "assistant":
                        if "tool_calls" in response:
                            tool_call_in_progress = True
                            events = await handle_tool_calls(
                                request_id, response, function_service)
                            for event in events:
                                yield event
                            continue

                        event = await handle_assistant_message(
                            request_id, response, filters)
                        yield event

                else:
                    event = await handle_string_chunk(request_id, response, filters)
                    yield event

    except Exception as e:
        logger.error(
            f"[{request_id}] Error in chat stream: {e}", exc_info=True)
        yield ChatStreamEvent(event="error", data=json.dumps({"error": str(e)}))


@router.post("/chat/stream", response_model=None)
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent = Depends(get_agent),
    model_service: ModelService = Depends(get_model_service),
    function_service: FunctionService = Depends(get_function_service),
    langchain_service: LangChainService = Depends(get_langchain_service),
    chroma_service: ChromaService = Depends(get_chroma_service)
) -> EventSourceResponse:
    """Stream chat completions."""
    return EventSourceResponse(
        stream_chat_response(
            request,
            chat_request,
            agent,
            model_service,
            function_service,
            langchain_service,
            chroma_service
        )
    )


class ChatRequest(BaseModel):
    """Chat request model."""
    messages: List[StrictChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = True
    enable_tools: bool = True
    filters: Optional[List[str]] = None
    pipeline: Optional[str] = None
    memory_type: Optional[str] = MemoryType.EPHEMERAL
    conversation_id: Optional[str] = None
    enable_summarization: Optional[bool] = False
    memory_filter: Optional[Dict[str, Any]] = None
    top_k_memories: Optional[int] = 5


@router.post("/chat/memory/add")
async def add_memory(
    request: Request,
    memory_text: str,
    memory_type: MemoryType = MemoryType.EPHEMERAL,
    metadata: Optional[Dict[str, Any]] = None,
    chroma_service: ChromaService = Depends(get_chroma_service)
):
    """Add a memory to the specified collection."""
    try:
        await chroma_service.add_memory(
            text=memory_text,
            collection=memory_type,
            metadata=metadata
        )
        return {"status": "success", "message": "Memory added successfully."}
    except Exception as e:
        logger.error(f"Error adding memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to add memory: {e}")
