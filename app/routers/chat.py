"""Chat router for handling chat-related endpoints and streaming responses."""

import json
import logging
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Request, Depends, BackgroundTasks
from sse_starlette.sse import EventSourceResponse

from app.core.config import config
from app.core.prompts import PROMPTS
from app.models.chat import ChatRequest, ChatStreamEvent, StrictChatMessage
from app.services.assistant import Assistant
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.memory.manager import LightRAGManager
from app.services.context_service import LLMContext
from app.dependencies.providers import Providers
from app.utils.chat_setup import verify_model_availability, setup_chat_components
from app.utils.filters import apply_filters
from app.utils.chat_messages import format_conversation_metadata, handle_string_chunk
from app.utils.chat_tools import process_tool_stream
from app.utils.profiling import profile_request

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={
        400: {"description": "Bad request - Invalid input parameters"},
        500: {"description": "Internal server error"},
        429: {"description": "Too many requests - Rate limit exceeded"},
    }
)

logger = logging.getLogger(__name__)


async def process_chat_context(
    request_id: str,
    messages: list[StrictChatMessage],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    enable_memory: bool = True
) -> list[dict]:
    """Process chat messages through the context service."""
    try:
        async with LLMContext(
            request_id=request_id,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            enable_memory=enable_memory
        ) as context:
            return context.get_context_window()
    except Exception as e:
        logger.error(
            f"[{request_id}] Error processing chat context: {e}", exc_info=True)
        return messages


async def stream_chat_response(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    assistant: Assistant = Depends(Providers.get_assistant),
    model_service: ModelService = Depends(Providers.get_model_service),
    function_service: FunctionService = Depends(
        Providers.get_function_service),
    memory_manager: LightRAGManager = Depends(Providers.get_lightrag_manager)
) -> AsyncGenerator[ChatStreamEvent, None]:
    """Generate streaming chat response."""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Starting chat stream")
    messages = chat_request.messages
    final_messages = []
    tool_response = None
    current_message = {"role": "assistant", "content": ""}

    async with profile_request(request_id) as profiler:
        try:
            # Setup components
            function_schemas, filters, pipeline = await setup_chat_components(
                request_id, chat_request, function_service)

            # Verify model availability
            model = chat_request.model or config.llm.model
            if error_event := await verify_model_availability(request_id, model, model_service):
                yield error_event
                return

            # Apply inlet filters
            if filters:
                data, filter_success = await apply_filters(
                    filters=filters,
                    data={"messages": messages},
                    request_id=request_id,
                    direction="inlet",
                    filter_name="inlet_message_filters"
                )
                if not filter_success:
                    yield ChatStreamEvent(
                        event="error",
                        data=json.dumps(
                            {"error": "Failed to apply inlet filters"})
                    )
                    return
                messages = data["messages"]

            # Process pipeline
            if pipeline:
                logger.debug(
                    f"[{request_id}] Applying pipeline: {pipeline.name}")
                try:
                    pipeline_data = await pipeline.pipe({"messages": messages})
                    if "messages" in pipeline_data and pipeline_data["messages"]:
                        messages = pipeline_data["messages"]
                    if "summary" in pipeline_data:
                        yield ChatStreamEvent(
                            event="pipeline",
                            data=json.dumps({
                                "status": "complete",
                                "summary": pipeline_data["summary"]
                            })
                        )
                except Exception as e:
                    logger.error(
                        f"[{request_id}] Pipeline error: {e}", exc_info=True)
                    yield ChatStreamEvent(
                        event="error",
                        data=json.dumps({"error": f"Pipeline error: {str(e)}"})
                    )
                    return

            # Process context
            processed_messages = await process_chat_context(
                request_id=request_id,
                messages=messages,
                model=model,
                max_tokens=chat_request.max_tokens,
                enable_memory=chat_request.enable_memory
            )

            logger.debug(
                f"[{request_id}] Processed messages: {json.dumps(processed_messages, indent=2)}")

            # Start streaming
            yield ChatStreamEvent(event="start", data=json.dumps({"status": "streaming"}))

            # Stream chat response
            current_tool_call = None

            # Record when we're about to send request to model
            profiler.record_model_request()

            async for chunk in assistant.chat(
                messages=processed_messages,
                model=model,
                temperature=chat_request.temperature or config.llm.temperature,
                max_tokens=chat_request.max_tokens or config.llm.max_tokens,
                stream=chat_request.stream if chat_request.stream is not None else True,
                tools=function_schemas,
                enable_tools=chat_request.enable_tools if chat_request.enable_tools is not None else config.llm.enable_tools
            ):
                if not chunk:
                    continue

                # Process tool calls
                if isinstance(chunk, dict) and "tool_calls" in chunk:
                    tool_event, current_tool_call, is_complete = await process_tool_stream(
                        request_id=request_id,
                        chunk=chunk,
                        function_service=function_service,
                        current_tool_call=current_tool_call
                    )

                    if tool_event:
                        yield tool_event
                        if not profiler.first_response_time:
                            profiler.record_first_response()

                        # If tool call is complete, store the response and add to context
                        if is_complete and tool_event.data:
                            tool_data = json.loads(tool_event.data)
                            if tool_data.get("role") == "tool":
                                # Store tool response as a JSON string
                                tool_response = tool_event.data  # Already a JSON string from event

                                # Process tool response to make it more manageable
                                try:
                                    content = json.loads(
                                        tool_data.get("content", "{}"))
                                    if isinstance(content, dict) and "result" in content:
                                        result = content["result"]
                                        # Ensure proper JSON structure without truncation
                                        if isinstance(result, dict):
                                            content["result"] = result
                                            tool_data["content"] = json.dumps(
                                                content)
                                except Exception as e:
                                    logger.warning(
                                        f"[{request_id}] Error processing tool response: {e}")

                                # Add processed tool response to messages
                                processed_messages.append(tool_data)

                                # Generate final response using the tool results
                                try:
                                    # Add a system message to guide the response
                                    processed_messages.append({
                                        "role": "system",
                                        "content": PROMPTS["tool_response_guidance"]
                                    })

                                    # Record when we're about to send request to model for final response
                                    profiler.record_model_request()
                                    # Reset first response timing for the new generation
                                    profiler.reset_first_response()

                                    async for final_chunk in assistant.chat(
                                        messages=processed_messages,
                                        model=model,
                                        temperature=chat_request.temperature or config.llm.temperature,
                                        max_tokens=chat_request.max_tokens or config.llm.max_tokens,
                                        stream=chat_request.stream if chat_request.stream is not None else True,
                                        tools=None,
                                        enable_tools=False
                                    ):
                                        if isinstance(final_chunk, str):
                                            if string_event := await handle_string_chunk(request_id, final_chunk, filters):
                                                yield string_event
                                                if not profiler.first_response_time:
                                                    profiler.record_first_response()
                                                current_message["content"] += final_chunk
                                except Exception as e:
                                    logger.error(
                                        f"[{request_id}] Error generating final response: {e}")
                                    yield ChatStreamEvent(
                                        event="error",
                                        data=json.dumps(
                                            {"error": f"Error generating response: {str(e)}"})
                                    )
                                    continue

                # Handle string chunks (assistant's final response)
                if isinstance(chunk, str):
                    if string_event := await handle_string_chunk(request_id, chunk, filters):
                        yield string_event
                        if not profiler.first_response_time:
                            profiler.record_first_response()
                        current_message["content"] += chunk

            # After all chunks are processed and before storing memory
            if current_message["content"]:
                final_messages = messages + [current_message]

                # Apply outlet filters
                if filters:
                    try:
                        data, filter_success = await apply_filters(
                            filters=filters,
                            data={"messages": final_messages},
                            request_id=request_id,
                            direction="outlet",
                            filter_name="outlet_message_filters"
                        )
                        if filter_success:
                            final_messages = data["messages"]
                    except Exception as e:
                        logger.error(
                            f"[{request_id}] Error applying outlet filters: {e}", exc_info=True)

                # Get last user message safely
                last_user_message = ""
                for msg in reversed(final_messages):
                    if hasattr(msg, "role") and msg.role == "user" and hasattr(msg, "content"):
                        last_user_message = msg.content
                        break

                # Prepare metadata for memory storage
                conversation_metadata = format_conversation_metadata(
                    request_id=request_id,
                    model=model,
                    final_messages=final_messages,
                    tool_response=json.dumps(tool_response) if isinstance(
                        tool_response, dict) else tool_response,
                    chat_request=chat_request,
                    current_message=current_message,
                    last_user_message=last_user_message
                )

                # Add memory storage to background tasks if enabled
                if chat_request.enable_memory and memory_manager and conversation_metadata:
                    async def store_memory_task():
                        try:
                            await memory_manager.store_memory(
                                text=last_user_message,
                                metadata=conversation_metadata
                            )
                            logger.info(
                                f"[{request_id}] Stored conversation memory with metadata")
                        except Exception as e:
                            # Log error but don't propagate it since this is a background task
                            logger.warning(
                                f"[{request_id}] Non-critical error storing conversation memory: {str(e)}")

                    background_tasks.add_task(store_memory_task)

        except Exception as e:
            logger.error(
                f"[{request_id}] Error in chat stream: {e}", exc_info=True)
            yield ChatStreamEvent(
                event="error",
                data=json.dumps({"error": str(e)})
            )


@router.post("/stream",
             response_class=EventSourceResponse,
             summary="Stream Chat Completion",
             description="""
    Generate a streaming chat completion with optional function calling and memory.
    
    Key Features:
    - Streaming response using Server-Sent Events (SSE)
    - Function/tool calling with automatic execution
    - Context management with memory integration
    - Support for inlet/outlet filters and pipelines
    - Automatic model verification
    
    The response is streamed as a series of events:
    - 'start': Indicates the start of streaming
    - 'message': Contains content chunks or tool calls
    - 'error': Contains error information if something fails
    - 'pipeline': Contains pipeline execution results
    """,
             response_description="Server-Sent Events stream of chat completion chunks",
             responses={
                 200: {
                     "description": "Successful response",
                     "content": {
                         "text/event-stream": {
                             "example": "event: message\ndata: {\"content\": \"Hello!\"}\n\n"
                         }
                     }
                 }
             }
             )
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    assistant: Assistant = Depends(Providers.get_assistant),
    model_service: ModelService = Depends(Providers.get_model_service),
    function_service: FunctionService = Depends(
        Providers.get_function_service),
    memory_manager: LightRAGManager = Depends(Providers.get_lightrag_manager)
) -> EventSourceResponse:
    """Stream a chat completion response.

    The response is streamed as Server-Sent Events (SSE) with the following event types:
    - 'start': Indicates the start of streaming
    - 'message': Contains content chunks or tool calls
    - 'error': Contains error information if something fails
    - 'pipeline': Contains pipeline execution results

    Features:
    - Streaming response using SSE
    - Optional function/tool calling with automatic execution
    - Context management with memory integration
    - Support for inlet/outlet filters and pipelines
    - Automatic model verification

    Args:
        request: The FastAPI request object
        chat_request: The chat request parameters
        background_tasks: FastAPI background tasks handler
        assistant: The assistant service for chat completions
        model_service: The model service for LLM operations
        function_service: The function service for tool execution
        memory_manager: The memory manager for context storage

    Returns:
        An EventSourceResponse that streams the chat completion

    Raises:
        HTTPException: If there are errors in request processing
    """
    return EventSourceResponse(
        stream_chat_response(
            request=request,
            chat_request=chat_request,
            background_tasks=background_tasks,
            assistant=assistant,
            model_service=model_service,
            function_service=function_service,
            memory_manager=memory_manager
        ),
        media_type="text/event-stream"
    )
