"""Chat router."""
import json
import logging
from typing import List, Optional, AsyncGenerator, Any
from fastapi import APIRouter, Request, Depends, Response
from sse_starlette.sse import EventSourceResponse
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
from app.models.chat import ChatRequest, ChatMessage
from app.functions.base import FunctionType
import asyncio
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


async def stream_chat_response(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent,
    model_service: ModelService,
    function_service: FunctionService,
    langchain_service: LangChainService = Depends(get_langchain_service),
    chroma_service: ChromaService = Depends(get_chroma_service),
    is_test: bool = False
) -> AsyncGenerator[dict, None]:
    """Generate streaming chat response."""
    request_id = str(id(request))
    logger.info(f"[{request_id}] Starting chat stream")
    tool_call_in_progress = False

    try:
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
            logger.info(
                f"[{request_id}] Getting filters: {chat_request.filters}")
            for filter_name in chat_request.filters:
                logger.debug(
                    f"[{request_id}] Looking up filter: {filter_name}")
                filter_class = function_service.get_function(filter_name)
                logger.debug(
                    f"[{request_id}] Found filter class: {filter_class}")
                if filter_class and filter_class.model_fields['type'].default == FunctionType.FILTER:
                    logger.info(
                        f"[{request_id}] Instantiating filter: {filter_name}")
                    try:
                        filter_instance = filter_class()
                        filters.append(filter_instance)
                        logger.info(
                            f"[{request_id}] Added filter: {filter_name}")
                        logger.debug(
                            f"[{request_id}] Filter instance: {filter_instance}")
                    except Exception as e:
                        logger.error(
                            f"[{request_id}] Error instantiating filter {filter_name}: {e}")
                else:
                    logger.warning(
                        f"[{request_id}] Filter not found or invalid type: {filter_name}")
                    if filter_class:
                        logger.debug(
                            f"[{request_id}] Found class type: {filter_class.model_fields['type'].default}")

        # Get requested pipeline
        pipeline = None
        if chat_request.pipeline:
            logger.info(
                f"[{request_id}] Getting pipeline: {chat_request.pipeline}")
            pipeline_class = function_service.get_function(
                chat_request.pipeline)
            logger.debug(
                f"[{request_id}] Found pipeline class: {pipeline_class}")
            if pipeline_class and pipeline_class.model_fields['type'].default == FunctionType.PIPELINE:
                logger.info(
                    f"[{request_id}] Instantiating pipeline: {chat_request.pipeline}")
                try:
                    pipeline = pipeline_class()
                    logger.info(
                        f"[{request_id}] Added pipeline: {chat_request.pipeline}")
                    logger.debug(
                        f"[{request_id}] Pipeline instance: {pipeline}")
                except Exception as e:
                    logger.error(
                        f"[{request_id}] Error instantiating pipeline {chat_request.pipeline}: {e}")
            else:
                logger.warning(
                    f"[{request_id}] Pipeline not found or invalid type: {chat_request.pipeline}")
                if pipeline_class:
                    logger.debug(
                        f"[{request_id}] Found class type: {pipeline_class.model_fields['type'].default}")

        # Apply inlet filters in priority order
        data = {"messages": chat_request.messages}
        sorted_filters = sorted(filters, key=lambda f: f.priority or 0)
        for filter_func in sorted_filters:
            logger.debug(
                f"[{request_id}] Applying inlet filter: {filter_func.name} (priority: {filter_func.priority})")
            try:
                data = await filter_func.inlet(data)
                logger.debug(
                    f"[{request_id}] Filter {filter_func.name} inlet result: {data}")
            except Exception as e:
                logger.error(
                    f"[{request_id}] Error in filter {filter_func.name} inlet: {e}")
        chat_request.messages = data["messages"]

        # Process conversation with LangChain service to add memory context
        if chat_request.enable_memory and langchain_service:
            try:
                chat_request.messages = await langchain_service.process_conversation(
                    chat_request.messages,
                    metadata_filter=chat_request.memory_filter,
                    top_k=chat_request.top_k_memories
                )
                logger.info(
                    f"[{request_id}] Added memory context to conversation")
            except Exception as e:
                logger.warning(
                    f"[{request_id}] Failed to add memory context: {e}")

        # Apply pipeline
        if pipeline:
            logger.debug(f"[{request_id}] Applying pipeline: {pipeline.name}")
            try:
                data = await pipeline.pipe({"messages": chat_request.messages})
                logger.debug(f"[{request_id}] Pipeline result: {data}")

                # Update messages if pipeline modified them
                if "messages" in data:
                    chat_request.messages = data["messages"]

                # Stream pipeline results
                if "summary" in data:
                    # Send initial summary event
                    yield {
                        "event": "pipeline",
                        "data": json.dumps({
                            "summary": data["summary"],
                            "status": "processing"
                        })
                    }

                    # Process detailed results if available
                    if isinstance(data["summary"], dict):
                        for key, value in data["summary"].items():
                            if isinstance(value, list):
                                for item in value:
                                    if item:  # Only send non-empty items
                                        yield {
                                            "event": "pipeline",
                                            "data": json.dumps({
                                                "type": key,
                                                "content": item,
                                                "status": "processing"
                                            })
                                        }
                                        await asyncio.sleep(0.1)

                    # Send completion event
                    yield {
                        "event": "pipeline",
                        "data": json.dumps({
                            "status": "complete",
                            "summary": data["summary"]
                        })
                    }
            except Exception as e:
                logger.error(
                    f"[{request_id}] Error in pipeline execution: {e}")
                yield {
                    "event": "pipeline_error",
                    "data": json.dumps({
                        "error": str(e),
                        "status": "failed"
                    })
                }

        # Verify model availability
        try:
            models = await model_service.get_all_models(request_id)
        except Exception as model_error:
            logger.error(
                f"[{request_id}] Error fetching models: {model_error}", exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": "Failed to fetch available models"})}
            return

        model = chat_request.model or config.DEFAULT_MODEL
        if model not in models:
            logger.error(f"[{request_id}] Model {model} not available")
            yield {"event": "error", "data": json.dumps({"error": f"Model {model} not available"})}
            return

        # Stream chat completion
        logger.info(
            f"[{request_id}] Starting chat with model {model}, tools enabled: {chat_request.enable_tools}")
        if function_schemas:
            logger.info(
                f"[{request_id}] Passing {len(function_schemas)} tools to model")

        first_response = True
        async for response in agent.chat(
            messages=chat_request.messages,
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
                yield {"event": "start", "data": json.dumps({"status": "streaming"})}
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
                        logger.info(
                            f"[{request_id}] Received tool/function response: {json.dumps(response, indent=2)}")
                        filtered_response = response.copy()
                        for filter_func in sorted(filters, key=lambda f: f.priority or 0, reverse=True):
                            try:
                                filtered_response = await filter_func.outlet(filtered_response)
                            except Exception as e:
                                logger.error(
                                    f"[{request_id}] Error in filter {filter_func.name} outlet: {e}")
                        yield {
                            "event": "message",
                            "data": json.dumps(filtered_response)
                        }
                        tool_call_in_progress = False

                    elif response.get("role") == "assistant":
                        filtered_chunk = {
                            "role": "assistant",
                            "content": response.get("content", "")
                        }
                        if "tool_calls" in response:
                            filtered_chunk["tool_calls"] = response["tool_calls"]
                            logger.info(
                                f"[{request_id}] Tool calls detected: {json.dumps(response['tool_calls'], indent=2)}")
                            tool_call_in_progress = True

                        for filter_func in sorted(filters, key=lambda f: f.priority or 0, reverse=True):
                            try:
                                filtered_chunk = await filter_func.outlet(filtered_chunk)
                            except Exception as e:
                                logger.error(
                                    f"[{request_id}] Error in filter {filter_func.name} outlet: {e}")

                        yield {
                            "event": "message",
                            "data": json.dumps(filtered_chunk)
                        }

                        if "tool_calls" in filtered_chunk:
                            try:
                                logger.info(
                                    f"[{request_id}] Executing tool calls")
                                tool_responses = await function_service.handle_tool_calls(filtered_chunk["tool_calls"])
                                for tool_response in tool_responses:
                                    logger.info(
                                        f"[{request_id}] Tool response: {json.dumps(tool_response, indent=2)}")
                                    yield {
                                        "event": "message",
                                        "data": json.dumps({
                                            "role": "tool",
                                            "content": tool_response["content"],
                                            "name": tool_response["name"]
                                        })
                                    }
                            except Exception as e:
                                logger.error(
                                    f"[{request_id}] Error executing tool calls: {e}")
                                yield {
                                    "event": "error",
                                    "data": json.dumps({"error": f"Error executing tool calls: {str(e)}"})
                                }

    except Exception as e:
        logger.error(
            f"[{request_id}] Error in chat stream: {e}", exc_info=True)
        yield {"event": "error", "data": json.dumps({"error": str(e)})}


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
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = True
    enable_tools: bool = True
    filters: Optional[List[str]] = None
    pipeline: Optional[str] = None
