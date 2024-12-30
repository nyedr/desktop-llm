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
    get_lightrag_manager
)
from app.memory.lightrag.manager import EnhancedLightRAGManager
from app.functions.filters import apply_filters
from app.functions.utils import (
    validate_tool_response,
    create_error_response
)
from app.services.agent import Agent
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.services.langchain_service import LangChainService
from app.models.chat import (
    ChatRequest, ChatStreamEvent, StrictChatMessage
)
from app.functions.base import (
    FunctionType,
    Filter,
    ToolResponse,
    ValidationError
)
import asyncio
from pydantic import BaseModel
from app.models.memory import MemoryType
from datetime import datetime

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


async def process_memory_context(
    messages: List[StrictChatMessage],
    langchain_service: LangChainService,
    chat_request: ChatRequest,
    request_id: str
) -> List[StrictChatMessage]:
    """Process memory context in the background without blocking.
    Returns the processed messages with memory context if successful."""
    try:
        # Only process user and assistant messages
        filtered_messages = [
            msg for msg in messages
            if (isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]) or
               (hasattr(msg, "role") and msg.role in ["user", "assistant"])
        ]

        if not filtered_messages:
            logger.info(f"[{request_id}] No messages to process")
            return messages

        # Ensure conversation_id is set to prevent null entries
        if not chat_request.conversation_id:
            chat_request.conversation_id = request_id

        # Format messages for storage
        formatted_messages = []
        for msg in filtered_messages:
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            content = msg.get("content") if isinstance(
                msg, dict) else msg.content
            formatted_messages.append({
                "role": role,
                "content": content,
                "conversation_id": chat_request.conversation_id,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Process conversation with proper metadata
        processed_messages = await langchain_service.process_conversation(
            messages=formatted_messages,
            memory_type=chat_request.memory_type,
            conversation_id=chat_request.conversation_id,
            enable_summarization=chat_request.enable_summarization,
            metadata_filter=chat_request.memory_filter,
            top_k=chat_request.top_k_memories
        )
        logger.info(
            f"[{request_id}] Added {chat_request.memory_type} memory context")

        # Preserve system messages in the final output
        system_messages = [
            msg for msg in messages
            if (isinstance(msg, dict) and msg.get("role") == "system") or
               (hasattr(msg, "role") and msg.role == "system")
        ]

        # Combine messages maintaining order
        final_messages = []
        for msg in messages:
            if (isinstance(msg, dict) and msg.get("role") == "system") or \
               (hasattr(msg, "role") and msg.role == "system"):
                final_messages.append(msg)
            else:
                if processed_messages:
                    final_messages.append(processed_messages.pop(0))

        final_messages.extend(processed_messages)
        return final_messages

    except Exception as e:
        logger.warning(f"[{request_id}] Failed to add memory context: {e}")
        return messages


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

        # Verify model availability
        model = chat_request.model or config.DEFAULT_MODEL
        if error_event := await verify_model_availability(request_id, model, model_service):
            yield error_event
            return

        # Apply inlet filters to the entire messages array if filters exist
        if filters:
            data, filter_success = await apply_filters(
                filters=filters,
                data={"messages": chat_request.messages},
                request_id=request_id,
                direction="inlet",
                filter_name="inlet_message_filters"
            )

            if not filter_success:
                yield ChatStreamEvent(
                    event="error",
                    data=json.dumps({"error": "Failed to apply inlet filters"})
                )
                return

            chat_request.messages = data["messages"]

        # Start memory processing in background if enabled
        memory_task = None
        if chat_request.enable_memory and langchain_service:
            memory_task = asyncio.create_task(
                process_memory_context(
                    messages=chat_request.messages.copy(),  # Create a copy to avoid modification
                    langchain_service=langchain_service,
                    chat_request=chat_request,
                    request_id=request_id
                )
            )
            logger.info(
                f"[{request_id}] Memory processing started in background")

        # Apply pipeline if present
        processed_messages = chat_request.messages
        if pipeline:
            logger.debug(f"[{request_id}] Applying pipeline: {pipeline.name}")
            try:
                pipeline_data = await pipeline.pipe({"messages": processed_messages})
                logger.debug(
                    f"[{request_id}] Pipeline result: {pipeline_data}")

                if "messages" in pipeline_data and pipeline_data["messages"]:
                    processed_messages = pipeline_data["messages"]
                else:
                    logger.warning(
                        f"[{request_id}] Pipeline returned empty messages, using original messages")

                if "summary" in pipeline_data:
                    pipeline_summary = pipeline_data["summary"]
                    yield ChatStreamEvent(
                        event="pipeline",
                        data=json.dumps({
                            "summary": pipeline_summary,
                            "status": "processing"
                        })
                    )

                    if isinstance(pipeline_summary, dict):
                        for key, value in pipeline_summary.items():
                            if isinstance(value, list):
                                for item in value:
                                    if item:
                                        yield ChatStreamEvent(
                                            event="pipeline",
                                            data=json.dumps({
                                                "content_type": key,
                                                "content": item,
                                                "status": "processing"
                                            })
                                        )

                    yield ChatStreamEvent(
                        event="pipeline",
                        data=json.dumps({
                            "status": "complete",
                            "summary": pipeline_summary
                        })
                    )

            except Exception as e:
                logger.error(f"[{request_id}] Pipeline error: {e}")
                yield ChatStreamEvent(
                    event="error",
                    data=json.dumps({"error": f"Pipeline error: {str(e)}"})
                )
                return

        # Log the context window before streaming response
        if processed_messages:
            print(f"\nContext window for request {request_id}:", flush=True)
            for msg in processed_messages:
                role = msg.get("role") if isinstance(msg, dict) else msg.role
                content = msg.get("content") if isinstance(
                    msg, dict) else msg.content
                print(f"{role}: {content}", flush=True)
            print("\nResponse:", flush=True)

        # Stream the response immediately without waiting for memory processing
        first_response = True
        async for chunk in agent.chat(
            messages=processed_messages,
            model=model,
            temperature=chat_request.temperature or config.MODEL_TEMPERATURE,
            max_tokens=chat_request.max_tokens or config.MAX_TOKENS,
            stream=True,
            tools=function_schemas if chat_request.enable_tools else None,
            enable_tools=chat_request.enable_tools,
            enable_memory=False,  # Disable memory since we're handling it separately
            memory_filter=None,
            top_k_memories=None
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

            if isinstance(chunk, dict):
                if "tool_calls" in chunk:
                    for event in await handle_tool_calls(request_id, chunk, function_service):
                        yield event
                    tool_call_in_progress = True
                else:
                    yield await handle_assistant_message(request_id, chunk, filters)
                    tool_call_in_progress = False
            else:
                yield await handle_string_chunk(request_id, chunk, filters)

        # Wait for memory processing to complete in the background
        if memory_task:
            try:
                await memory_task
            except Exception as e:
                logger.warning(
                    f"[{request_id}] Background memory processing failed: {e}")

    except Exception as e:
        logger.error(
            f"[{request_id}] Error in chat stream: {e}", exc_info=True)
        yield ChatStreamEvent(
            event="error",
            data=json.dumps({"error": str(e)})
        )


@router.post("/chat/stream", response_model=None)
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent = Depends(get_agent),
    model_service: ModelService = Depends(get_model_service),
    function_service: FunctionService = Depends(get_function_service),
    langchain_service: LangChainService = Depends(get_langchain_service)
) -> EventSourceResponse:
    """Stream chat completions."""
    return EventSourceResponse(
        stream_chat_response(
            request,
            chat_request,
            agent,
            model_service,
            function_service,
            langchain_service
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
    manager: EnhancedLightRAGManager = Depends(get_lightrag_manager)
):
    """Add a memory to the specified collection using EnhancedLightRAGManager.
    This is a non-blocking operation that processes memory in the background."""
    try:
        # Create task for background processing
        asyncio.create_task(
            manager.ingestor.ingest_text(
                text=memory_text,
                metadata=metadata,
                parent_id=None
            )
        )
        return {"status": "success", "message": "Memory ingestion started"}
    except Exception as e:
        logger.error(f"Error scheduling memory ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to schedule memory ingestion: {e}")
