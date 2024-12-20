"""Model service module."""

import logging
import json
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from datetime import datetime
import uuid
import asyncio

from ollama import AsyncClient
from app.core.config import config
from app.models.chat import ChatMessage

logger = logging.getLogger(__name__)

class ModelService:
    """Service for managing model interactions."""
    
    def __init__(self):
        """Initialize the model service."""
        self.models = {}
        self.models_loaded = False
        self.last_models_fetch = None
        self.ollama_client = AsyncClient(host=config.OLLAMA_BASE_URLS[0])
        
    async def fetch_ollama_models(self, request_id: str) -> List[Dict[str, Any]]:
        """Fetch models from Ollama server."""
        try:
            # Use the ollama.list() function through AsyncClient
            models_response = await self.ollama_client.list()
            if models_response:
                logger.info(f"[{request_id}] Successfully fetched models from Ollama")
                return models_response['models'] if 'models' in models_response else []
            return []
        except Exception as e:
            logger.error(f"[{request_id}] Failed to fetch models from Ollama: {e}")
            return []

    async def fetch_openai_models(self, request_id: str) -> List[Dict[str, Any]]:
        """Fetch models from OpenAI API."""
        if not config.OPENAI_API_KEY:
            logger.debug(f"[{request_id}] No OpenAI API key configured, skipping model fetch")
            return []
            
        try:
            headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
            async with self.ollama_client.session.get("https://api.openai.com/v1/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('data', [])
                    logger.info(f"[{request_id}] Successfully fetched OpenAI models")
                    return models
                else:
                    error_text = await response.text()
                    logger.error(f"[{request_id}] Error fetching OpenAI models: {error_text}")
                    return []
        except Exception as e:
            logger.error(f"[{request_id}] Failed to fetch OpenAI models: {e}")
            return []

    async def get_all_models(self, request_id: str) -> Dict[str, Any]:
        """Fetch and cache available models from multiple providers."""
        try:
            # Check cache if it's still valid (less than 5 minutes old)
            if self.models_loaded and self.last_models_fetch:
                age = (datetime.now() - self.last_models_fetch).total_seconds()
                if age < 300:  # 5 minutes
                    logger.debug(f"[{request_id}] Using cached models")
                    return self.models

            # Fetch Ollama models
            ollama_models = await self.fetch_ollama_models(request_id)
            
            # Fetch OpenAI models if configured
            openai_models = await self.fetch_openai_models(request_id) if config.OPENAI_API_KEY else []

            # Update models cache
            self.models = {}
            
            # Process Ollama models
            for model in ollama_models:
                model_name = model.get('name', '')
                if model_name:
                    self.models[model_name] = {
                        'model': model_name,
                        'provider': 'ollama',
                        'size': model.get('size'),
                        'modified_at': model.get('modified_at'),
                        'details': model
                    }

            # Process OpenAI models
            for model in openai_models:
                model_id = model.get('id', '')
                if model_id:
                    self.models[model_id] = {
                        'model': model_id,
                        'provider': 'openai',
                        'details': model
                    }

            self.models_loaded = True
            self.last_models_fetch = datetime.now()
            logger.info(f"[{request_id}] Successfully fetched all models")
            logger.debug(f"[{request_id}] Available models: {list(self.models.keys())}")
            return self.models

        except Exception as e:
            logger.error(f"[{request_id}] Error fetching models: {e}", exc_info=True)
            raise  # Re-raise to handle in caller
            
    async def handle_tool_call(self, tool_call, function_service):
        """Handle a tool call, execute the function, and return the result."""
        function_name = tool_call['function']['name']
        function_args = tool_call['function']['arguments']  # Arguments are already a dict

        if function_service:
            try:
                logger.info(f"Calling function: {function_name} with arguments: {function_args}")
                function_response = await function_service.execute_function(function_name, function_args)
                logger.info(f"Function {function_name} returned: {function_response}")
                return json.dumps(function_response) if not isinstance(function_response, str) else function_response
            except Exception as e:
                logger.error(f"Error executing function {function_name}: {e}", exc_info=True)
                return json.dumps({"error": str(e)})
        else:
            logger.error(f"Function service not available to handle tool call {function_name}.")
            return json.dumps({"error": "Function service not available."})

    async def chat(
        self,
        messages: List[Union[Dict[str, Any], ChatMessage]],
        model: str = config.DEFAULT_MODEL,
        temperature: float = config.MODEL_TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_tools: bool = False,
        function_service = None
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Chat with a model asynchronously with single continuous stream."""
        try:
            if enable_tools and tools:
                logger.info(f"Adding {len(tools)} tools to chat payload")

            # Convert messages to the expected format
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    formatted_messages.append(msg.model_dump(exclude_none=True))
                else:
                    formatted_messages.append(msg)

            if stream:
                stream_response = await self.ollama_client.chat(
                    model=model,
                    messages=formatted_messages,
                    tools=tools if enable_tools and tools else None,
                    stream=True,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens
                    }
                )
                
                async for chunk in stream_response:
                    if 'message' in chunk:
                        message = chunk['message']
                        
                        # Handle tool calls
                        if 'tool_calls' in message:
                            logger.info(f"Tool call detected: {message}")
                            yield message  # Yield tool call first
                            
                            if function_service:
                                tool_calls = message.get('tool_calls', [])
                                
                                for tool_call in tool_calls:
                                    # Execute function
                                    tool_response = await self.handle_tool_call(tool_call, function_service)
                                    
                                    # Generate tool call ID
                                    tool_call_id = str(uuid.uuid4())
                                    
                                    # Add tool call to conversation
                                    formatted_messages.append({
                                        'role': 'assistant',
                                        'content': '',
                                        'tool_calls': [{
                                            'id': tool_call_id,
                                            'function': tool_call['function']
                                        }]
                                    })
                                    
                                    # Add tool response to conversation
                                    tool_response_msg = {
                                        'role': 'tool',
                                        'content': tool_response,
                                        'name': tool_call['function']['name'],
                                        'tool_call_id': tool_call_id
                                    }
                                    formatted_messages.append(tool_response_msg)
                                    yield tool_response_msg
                                
                                # Continue conversation with updated context
                                continuation_stream = await self.ollama_client.chat(
                                    model=model,
                                    messages=formatted_messages,
                                    stream=True,
                                    options={
                                        'temperature': temperature,
                                        'num_predict': max_tokens
                                    }
                                )
                                
                                current_content = ""
                                async for continuation in continuation_stream:
                                    if 'message' in continuation:
                                        cont_message = continuation['message']
                                        if 'content' in cont_message:
                                            content = cont_message.get('content', '')
                                            if content:
                                                # Split content into words and stream each
                                                words = content.split()
                                                for word in words:
                                                    yield {
                                                        'role': 'assistant',
                                                        'content': word + " "
                                                    }
                                                    await asyncio.sleep(0.05)  # Add a small delay between words
                                                
                        # Handle content
                        elif 'content' in message:
                            content = message.get('content', '')
                            if content:
                                # Split content into words and stream each
                                words = content.split()
                                for word in words:
                                    yield {
                                        'role': 'assistant',
                                        'content': word + " "
                                    }
                                    await asyncio.sleep(0.05)  # Add a small delay between words
            else:
                # Non-streaming mode
                response = await self.ollama_client.chat(
                    model=model,
                    messages=formatted_messages,
                    tools=tools if enable_tools and tools else None,
                    stream=False,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens
                    }
                )
                if 'message' in response:
                    yield response['message']
                else:
                    logger.warning(f"Unexpected response format: {response}")
                    
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            raise
