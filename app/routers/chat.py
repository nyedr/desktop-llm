"""Chat router."""
import json
import time
import logging
import asyncio
from typing import List, Optional, AsyncGenerator
from fastapi import APIRouter, Request, Depends, HTTPException, status
from pydantic import BaseModel, Field, ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse
from starlette.background import BackgroundTask

from app.core.config import config
from app.dependencies.providers import get_agent, get_model_service, get_function_service
from app.services.agent import Agent
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.models.chat import ChatRequest, ChatMessage
from app.functions.base import FunctionType

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent = Depends(get_agent),
    model_service: ModelService = Depends(get_model_service),
    function_service: FunctionService = Depends(get_function_service)
) -> EventSourceResponse:
    """Stream chat completions using Server-Sent Events."""
    request_id = str(id(request))
    logger.info(f"[{request_id}] Starting chat stream")
    
    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            # Get available functions
            logger.debug(f"[{request_id}] Getting available functions")
            function_schemas = function_service.get_function_schemas() if chat_request.enable_tools else None
            if function_schemas:
                logger.debug(f"[{request_id}] Available functions: {[f['function']['name'] for f in function_schemas]}")
            
            # Get requested filters
            filters = []
            if chat_request.filters:
                for filter_name in chat_request.filters:
                    filter_func = function_service.get_function(filter_name)
                    if filter_func and filter_func.type == FunctionType.FILTER:
                        filters.append(filter_func())
                    else:
                        logger.warning(f"[{request_id}] Filter not found or invalid type: {filter_name}")
            
            # Get requested pipeline
            pipeline = None
            if chat_request.pipeline:
                pipeline_func = function_service.get_function(chat_request.pipeline)
                if pipeline_func and pipeline_func.type == FunctionType.PIPELINE:
                    pipeline = pipeline_func()
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
            logger.debug(f"[{request_id}] Fetching models")
            try:
                models = await model_service.get_all_models(request_id)
            except Exception as model_error:
                logger.error(f"[{request_id}] Error fetching models: {model_error}", exc_info=True)
                yield {"event": "error", "data": json.dumps({"error": "Failed to fetch available models"})}
                return
                
            model = chat_request.model or config.DEFAULT_MODEL
            logger.debug(f"[{request_id}] Available models: {list(models.keys())}")
            logger.debug(f"[{request_id}] Selected model: {model}")
            
            if model not in models:
                logger.error(f"[{request_id}] Model {model} not available")
                yield {"event": "error", "data": json.dumps({"error": f"Model {model} not available"})}
                return
                
            # Start streaming response
            yield {"event": "start", "data": json.dumps({"status": "streaming"})}
            
            # Stream chat completion
            logger.debug(f"[{request_id}] Starting chat completion stream with model {model}")
            try:
                current_chunk = {"role": "assistant", "content": ""}
                async for response in agent.chat(
                    messages=chat_request.messages,
                    model=model,
                    temperature=chat_request.temperature or config.MODEL_TEMPERATURE,
                    max_tokens=chat_request.max_tokens or config.MAX_TOKENS,
                    stream=True,
                    tools=function_schemas,
                    enable_tools=chat_request.enable_tools
                ):
                    if await request.is_disconnected():
                        logger.info(f"[{request_id}] Client disconnected")
                        break
                        
                    logger.debug(f"[{request_id}] Received response: {response}")
                    if response:  # Only yield non-empty responses
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
                                chunk = response.copy()
                                if "content" in chunk:
                                    # Apply outlet filters to each chunk
                                    filtered_chunk = {"role": "assistant", "content": chunk["content"]}
                                    for filter_func in filters:
                                        filtered_chunk = await filter_func.outlet(filtered_chunk)
                                    yield {
                                        "event": "message",
                                        "data": json.dumps(filtered_chunk)
                                    }
                        else:
                            logger.warning(f"[{request_id}] Non-dict response: {response}")
                        
            except Exception as e:
                logger.error(f"[{request_id}] Error in chat completion: {e}", exc_info=True)
                yield {"event": "error", "data": json.dumps({"error": str(e)})}
                
            # End streaming response
            yield {"event": "end", "data": json.dumps({"status": "complete"})}
            
        except Exception as e:
            logger.error(f"[{request_id}] Error in event generator: {e}", exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
    
    return EventSourceResponse(event_generator())

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
