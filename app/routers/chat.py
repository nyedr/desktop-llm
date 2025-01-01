"""Chat router for handling chat-related endpoints and streaming responses."""

import datetime
import json
import logging
from typing import List, Optional, AsyncGenerator, Any, Dict
from fastapi import APIRouter, Request, Depends
from sse_starlette.sse import EventSourceResponse

from app.core.config import config
from app.dependencies.providers import Providers
from app.utils.filters import apply_filters
from app.memory.lightrag.manager import EnhancedLightRAGManager
from app.services.agent import Agent
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.models.chat import ChatRequest, ChatStreamEvent
from app.models.function import Filter, FunctionType
from app.services.context_service import LLMContext
from app.utils.chat_messages import handle_assistant_message, handle_string_chunk
from app.utils.chat_tools import handle_tool_calls

router = APIRouter()
logger = logging.getLogger(__name__)


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


async def stream_chat_response(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent = Depends(Providers.get_agent),
    model_service: ModelService = Depends(Providers.get_model_service),
    function_service: FunctionService = Depends(
        Providers.get_function_service),
    lightrag_manager: EnhancedLightRAGManager = Depends(
        Providers.get_lightrag_manager),
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

        # Process messages with LLMContext
        async with LLMContext(
            request_id=request_id,
            messages=chat_request.messages,
            lightrag_manager=lightrag_manager if chat_request.enable_memory else None,
            memory_filter=chat_request.memory_filter,
            top_k_memories=chat_request.top_k_memories,
            enable_memory=chat_request.enable_memory,
            conversation_id=chat_request.conversation_id,
            model=model,
            max_tokens=chat_request.max_tokens
        ) as context_service:
            processed_messages = context_service.get_context_window()

        # Apply pipeline if present
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
                print(f"{role}: {content[:100]}...", flush=True)

        # Stream the response
        first_response = True
        async for chunk in agent.chat(
            messages=processed_messages,
            model=model,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
            stream=True,
            tools=function_schemas,
            enable_tools=chat_request.enable_tools
        ):
            if first_response:
                yield ChatStreamEvent(event="start", data=json.dumps({"status": "streaming"}))
                first_response = False

            if not chunk:
                continue

            # Handle tool calls
            if isinstance(chunk, dict) and "function_call" in chunk:
                tool_call_in_progress = True
                tool_event = await handle_tool_calls(
                    chunk=chunk,
                    function_service=function_service,
                    request_id=request_id
                )
                if tool_event:
                    yield tool_event
                continue

            # Handle end of tool call
            if tool_call_in_progress and not chunk.get("function_call"):
                tool_call_in_progress = False
                yield ChatStreamEvent(
                    event="function_call",
                    data=json.dumps({"status": "complete"})
                )

            # Handle assistant messages
            if isinstance(chunk, dict):
                assistant_event = await handle_assistant_message(
                    response=chunk,
                    filters=filters,
                    request_id=request_id
                )
                if assistant_event:
                    yield assistant_event
                continue

            # Handle string chunks
            string_event = await handle_string_chunk(
                request_id=request_id,
                response=chunk,
                filters=filters
            )

            if string_event:
                yield string_event

        # Store memory after completion if enabled
        if chat_request.enable_memory and lightrag_manager:
            try:
                await lightrag_manager.ingestor.ingest_text(
                    text="\n".join([
                        f"{msg.get('role') if isinstance(msg, dict) else msg.role}: {msg.get('content') if isinstance(msg, dict) else msg.content}"
                        for msg in processed_messages
                        if (isinstance(msg, dict) and msg.get('role') != 'system') or
                           (hasattr(msg, 'role') and msg.role != 'system')
                    ]),
                    metadata={
                        "conversation_id": chat_request.conversation_id or request_id,
                        "request_id": request_id,
                        "model": model,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                logger.info(f"[{request_id}] Stored conversation memory")
            except Exception as e:
                logger.warning(
                    f"[{request_id}] Error storing conversation memory: {e}")

        # Apply outlet filters if they exist
        if filters and processed_messages:
            data, filter_success = await apply_filters(
                filters=filters,
                data={"messages": processed_messages},
                request_id=request_id,
                direction="outlet",
                filter_name="outlet_message_filters"
            )

            if not filter_success:
                yield ChatStreamEvent(
                    event="error",
                    data=json.dumps(
                        {"error": "Failed to apply outlet filters"})
                )
                return

            # Send filtered messages
            yield ChatStreamEvent(
                event="filtered_messages",
                data=json.dumps({"messages": data["messages"]})
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
    agent: Agent = Depends(Providers.get_agent),
    model_service: ModelService = Depends(Providers.get_model_service),
    function_service: FunctionService = Depends(
        Providers.get_function_service),
    lightrag_manager: EnhancedLightRAGManager = Depends(
        Providers.get_lightrag_manager)
) -> EventSourceResponse:
    """Stream chat response."""
    return EventSourceResponse(
        stream_chat_response(
            request=request,
            chat_request=chat_request,
            agent=agent,
            model_service=model_service,
            function_service=function_service,
            lightrag_manager=lightrag_manager
        )
    )
