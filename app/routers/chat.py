"""Chat router."""
import json
import logging
from typing import List, Optional, AsyncGenerator, Any
from fastapi import APIRouter, Request, Depends, Response
from sse_starlette.sse import EventSourceResponse
from app.core.config import config
from app.dependencies.providers import get_agent, get_model_service, get_function_service
from app.services.agent import Agent
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
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
    is_test: bool = False
) -> AsyncGenerator[dict, None]:
    """Generate streaming chat response."""
    request_id = str(id(request))
    logger.info(f"[{request_id}] Starting chat stream")
    
    try:
        # Get available functions
        function_schemas = function_service.get_function_schemas() if chat_request.enable_tools else None
        
        # Get requested filters
        filters = []
        if chat_request.filters:
            logger.info(f"[{request_id}] Getting filters: {chat_request.filters}")
            for filter_name in chat_request.filters:
                filter_class = function_service.get_function(filter_name)
                if filter_class and filter_class.type == FunctionType.FILTER:
                    filter_instance = filter_class()
                    filters.append(filter_instance)
                    logger.info(f"[{request_id}] Added filter: {filter_name}")
                else:
                    logger.warning(f"[{request_id}] Filter not found or invalid type: {filter_name}")
        
        # Get requested pipeline
        pipeline = None
        if chat_request.pipeline:
            logger.info(f"[{request_id}] Getting pipeline: {chat_request.pipeline}")
            pipeline_class = function_service.get_function(chat_request.pipeline)
            if pipeline_class and pipeline_class.type == FunctionType.PIPELINE:
                pipeline = pipeline_class()
                logger.info(f"[{request_id}] Added pipeline: {chat_request.pipeline}")
            else:
                logger.warning(f"[{request_id}] Pipeline not found or invalid type: {chat_request.pipeline}")
        
        # Apply inlet filters
        data = {"messages": chat_request.messages}
        for filter_func in filters:
            logger.debug(f"[{request_id}] Applying inlet filter: {filter_func.name}")
            data = await filter_func.inlet(data)
        chat_request.messages = data["messages"]
        
        # Apply pipeline
        if pipeline:
            logger.debug(f"[{request_id}] Applying pipeline: {pipeline.name}")
            data = await pipeline.pipe(data)
            if "messages" in data:
                chat_request.messages = data["messages"]
            if "summary" in data:
                # Send pipeline summary as a separate event
                yield {
                    "event": "pipeline",
                    "data": json.dumps({"summary": data["summary"]})
                }
                
                # Stream each summary item
                if isinstance(data["summary"], dict):
                    for key, value in data["summary"].items():
                        if isinstance(value, list):
                            for item in value:
                                yield {
                                    "event": "pipeline",
                                    "data": json.dumps({"type": key, "content": item})
                                }
                                await asyncio.sleep(0.1)  # Add a small delay between items
        
        # Verify model availability
        try:
            models = await model_service.get_all_models(request_id)
        except Exception as model_error:
            logger.error(f"[{request_id}] Error fetching models: {model_error}", exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": "Failed to fetch available models"})}
            return
            
        model = chat_request.model or config.DEFAULT_MODEL
        if model not in models:
            logger.error(f"[{request_id}] Model {model} not available")
            yield {"event": "error", "data": json.dumps({"error": f"Model {model} not available"})}
            return
            
        # Stream chat completion
        first_response = True
        async for response in agent.chat(
            messages=chat_request.messages,
            model=model,
            temperature=chat_request.temperature or config.MODEL_TEMPERATURE,
            max_tokens=chat_request.max_tokens or config.MAX_TOKENS,
            stream=True,
            tools=function_schemas,
            enable_tools=chat_request.enable_tools
        ):
            if first_response:
                yield {"event": "start", "data": json.dumps({"status": "streaming"})}
                first_response = False

            if not is_test and await request.is_disconnected():
                logger.info(f"[{request_id}] Client disconnected")
                break
                
            if response:
                if isinstance(response, dict):
                    # Handle tool calls and responses immediately
                    if response.get("role") in ["tool", "function"]:
                        filtered_response = response.copy()
                        for filter_func in filters:
                            filtered_response = await filter_func.outlet(filtered_response)
                        yield {
                            "event": "message",
                            "data": json.dumps(filtered_response)
                        }
                    # Stream assistant messages chunk by chunk
                    elif response.get("role") == "assistant":
                        # Create filtered chunk with required fields
                        filtered_chunk = {
                            "role": "assistant",
                            "content": response.get("content", "")
                        }
                        # Include tool_calls if present
                        if "tool_calls" in response:
                            filtered_chunk["tool_calls"] = response["tool_calls"]
                        
                        # Apply outlet filters
                        for filter_func in filters:
                            filtered_chunk = await filter_func.outlet(filtered_chunk)
                            
                        yield {
                            "event": "message",
                            "data": json.dumps(filtered_chunk)
                        }
                else:
                    logger.warning(f"[{request_id}] Non-dict response: {response}")
                
        # End streaming response
        yield {"event": "end", "data": json.dumps({"status": "complete"})}
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in event generator: {e}", exc_info=True)
        yield {"event": "error", "data": json.dumps({"error": str(e)})}
        
@router.post("/chat/stream", response_model=None)
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent = Depends(get_agent),
    model_service: ModelService = Depends(get_model_service),
    function_service: FunctionService = Depends(get_function_service)
) -> Any:
    """Stream chat completions using Server-Sent Events."""
    is_test = request.headers.get("x-test-request") == "true"
    
    generator = stream_chat_response(
        request=request,
        chat_request=chat_request,
        agent=agent,
        model_service=model_service,
        function_service=function_service,
        is_test=is_test
    )

    # Always return SSE response, even in test mode
    return EventSourceResponse(generator)

class ChatRequest(BaseModel):
    """Chat request model."""
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = True
    enable_tools: bool = False
    filters: Optional[List[str]] = None
    pipeline: Optional[str] = None
