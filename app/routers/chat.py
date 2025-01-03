"""Chat router for handling chat-related endpoints and streaming responses."""

import json
import logging
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Request, Depends, BackgroundTasks
from sse_starlette.sse import EventSourceResponse

from app.core.config import config
from app.models.chat import ChatRequest, ChatStreamEvent, StrictChatMessage
from app.services.agent import Agent
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.memory.manager import LightRAGManager
from app.services.context_service import LLMContext
from app.dependencies.providers import Providers
from app.utils.chat_setup import verify_model_availability, setup_chat_components
from app.utils.memory_utils import store_conversation_memory
from app.utils.filters import apply_filters
from app.utils.chat_messages import handle_string_chunk
from app.utils.chat_tools import process_tool_stream

router = APIRouter()
logger = logging.getLogger(__name__)


async def process_chat_context(
    request_id: str,
    messages: list[StrictChatMessage],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None
) -> list[dict]:
    """Process chat messages through the context service."""
    try:
        async with LLMContext(
            request_id=request_id,
            messages=messages,
            model=model,
            max_tokens=max_tokens
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
    agent: Agent = Depends(Providers.get_agent),
    model_service: ModelService = Depends(Providers.get_model_service),
    function_service: FunctionService = Depends(
        Providers.get_function_service),
    memory_manager: LightRAGManager = Depends(Providers.get_lightrag_manager)
) -> AsyncGenerator[ChatStreamEvent, None]:
    """Generate streaming chat response."""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Starting chat stream")
    tool_call_in_progress = False
    messages = chat_request.messages

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
                    data=json.dumps({"error": "Failed to apply inlet filters"})
                )
                return
            messages = data["messages"]

        # Process pipeline
        if pipeline:
            logger.debug(f"[{request_id}] Applying pipeline: {pipeline.name}")
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
            max_tokens=chat_request.max_tokens
        )

        # Start streaming
        current_message = {"role": "assistant", "content": ""}
        yield ChatStreamEvent(event="start", data=json.dumps({"status": "streaming"}))

        # Stream chat response
        current_tool_call = None
        tool_response = None

        async for chunk in agent.chat(
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

                    # If tool call is complete, store the response and add to context
                    if is_complete and tool_event.data:
                        tool_data = json.loads(tool_event.data)
                        if tool_data.get("role") == "tool":
                            tool_response = tool_data
                            processed_messages.append(tool_data)
                continue

            # Handle string chunks (assistant's final response)
            if isinstance(chunk, str):
                if string_event := await handle_string_chunk(request_id, chunk, filters):
                    yield string_event
                    current_message["content"] += chunk

        # Store memory if needed
        if current_message["content"] and chat_request.enable_memory and memory_manager:
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

            # Store conversation memory
            await store_conversation_memory(
                request_id=request_id,
                messages=final_messages,
                lightrag_manager=memory_manager,
                conversation_id=chat_request.conversation_id or str(
                    uuid.uuid4()),
                model=model
            )

    except Exception as e:
        logger.error(
            f"[{request_id}] Error in chat stream: {e}", exc_info=True)
        yield ChatStreamEvent(
            event="error",
            data=json.dumps({"error": str(e)})
        )


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    agent: Agent = Depends(Providers.get_agent),
    model_service: ModelService = Depends(Providers.get_model_service),
    function_service: FunctionService = Depends(
        Providers.get_function_service),
    memory_manager: LightRAGManager = Depends(Providers.get_lightrag_manager)
) -> EventSourceResponse:
    """Stream chat response."""
    return EventSourceResponse(
        stream_chat_response(
            request=request,
            chat_request=chat_request,
            background_tasks=background_tasks,
            agent=agent,
            model_service=model_service,
            function_service=function_service,
            memory_manager=memory_manager
        )
    )
