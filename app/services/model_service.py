"""Model service module."""

import logging
import json
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple
import uuid
import asyncio
import httpx

from ollama import AsyncClient
from app.core.config import config
from app.models.chat import ChatMessage
from app.models.model import Model
from app.models.completion import CompletionResponse

logger = logging.getLogger(__name__)


class ModelServiceError(Exception):
    """Base exception for model service errors."""
    pass


class OllamaConnectionError(ModelServiceError):
    """Raised when connection to Ollama server fails."""
    pass


class ModelService:
    """Service for interacting with language models."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.OLLAMA_BASE_URLS[0]
        self.request_timeout = config.MODEL_REQUEST_TIMEOUT
        self.generation_timeout = config.GENERATION_REQUEST_TIMEOUT
        self.default_model = config.DEFAULT_MODEL
        self.default_temperature = config.DEFAULT_TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        self.function_calls_enabled = config.FUNCTION_CALLS_ENABLED
        self.enable_model_filter = config.ENABLE_MODEL_FILTER
        self.model_filter_list = config.MODEL_FILTER_LIST
        self.ollama_health_checked = False
        self.ollama_available = False
        self.ollama_client = AsyncClient(host=self.base_url)
        logger.info("Initialized ModelService")
        logger.debug(f"Using Ollama base URL: {self.base_url}")

    def _get_request_id(self) -> Optional[str]:
        """Get the current request ID from context vars."""
        return str(uuid.uuid4())

    async def fetch_models(self) -> List[Model]:
        """Fetch available models from Ollama."""
        request_id = self._get_request_id()
        logger.info(f"[{request_id}] Fetching models from Ollama")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = [Model(**model)
                          for model in response.json()["models"]]
                logger.info(f"[{request_id}] Found {len(models)} models")
                return models

        except Exception as e:
            logger.error(f"[{request_id}] Error fetching models: {e}")
            raise

    async def get_models(self) -> List[str]:
        """Get list of available model names."""
        request_id = self._get_request_id()
        try:
            models = await self.fetch_models()
            model_names = [model.model for model in models]
            return model_names
        except Exception as e:
            logger.error(f"[{request_id}] Error getting models: {e}")
            raise

    async def get_model_info(self, model_name: str) -> Optional[Model]:
        """Get detailed information about a specific model."""
        request_id = self._get_request_id()
        try:
            models = await self.fetch_models()
            for model in models:
                if model.model == model_name:
                    return model
            return None
        except Exception as e:
            logger.error(f"[{request_id}] Error getting model info: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Generate text from a prompt."""
        request_id = self._get_request_id()
        logger.info(
            f"[{request_id}] Starting generation with model: {model or self.default_model}")

        try:
            # Prepare request data
            data = {
                "model": model or self.default_model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature or self.default_temperature,
                }
            }

            if max_tokens:
                data["options"]["num_predict"] = max_tokens

            if tools and self.function_calls_enabled:
                data["tools"] = tools
                if tool_choice:
                    data["tool_choice"] = tool_choice

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=data,
                    timeout=self.generation_timeout
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            completion = CompletionResponse.model_validate_json(
                                line)
                            yield completion
                        except Exception as e:
                            logger.error(
                                f"[{request_id}] Error parsing completion response: {e}")
                            continue

        except Exception as e:
            logger.error(f"[{request_id}] Error during generation: {e}")
            raise

    async def handle_tool_call(self, tool_call, function_service):
        """Handle a tool call, execute the function, and return the result."""
        function_name = tool_call['function']['name']
        # Arguments are already a dict
        function_args = tool_call['function']['arguments']

        if function_service:
            try:
                logger.info(
                    f"Calling function: {function_name} with arguments: {function_args}")
                function_response = await function_service.execute_function(function_name, function_args)
                logger.info(
                    f"Function {function_name} returned: {function_response}")
                return json.dumps(function_response) if not isinstance(function_response, str) else function_response
            except Exception as e:
                logger.error(
                    f"Error executing function {function_name}: {e}", exc_info=True)
                return json.dumps({"error": str(e)})
        else:
            logger.error(
                f"Function service not available to handle tool call {function_name}.")
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
        function_service=None
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Chat with a model asynchronously with single continuous stream."""
        request_id = self._get_request_id()
        try:
            if enable_tools:
                if tools:
                    logger.info(
                        f"[{request_id}] Tools enabled with {len(tools)} available tools")
                    logger.debug(
                        f"[{request_id}] Available tools: {[t['function']['name'] for t in (tools or [])]}")
                else:
                    logger.warning(
                        f"[{request_id}] Tools enabled but no tools provided")
            else:
                logger.info(f"[{request_id}] Tools disabled for this request")

            # Convert messages to the expected format
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    formatted_messages.append(
                        msg.model_dump(exclude_none=True))
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
                                    formatted_messages.append(
                                        tool_response_msg)
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
                                            content = cont_message.get(
                                                'content', '')
                                            if content:
                                                # Split content into words and stream each
                                                words = content.split()
                                                for word in words:
                                                    yield {
                                                        'role': 'assistant',
                                                        'content': word + " "
                                                    }
                                                    # Add a small delay between words
                                                    await asyncio.sleep(0.05)

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
                                    # Add a small delay between words
                                    await asyncio.sleep(0.05)
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

    async def check_ollama_health(self, request_id: str) -> Tuple[bool, str]:
        """Check if Ollama server is available and responding.

        Returns:
            Tuple[bool, str]: (is_healthy, status_message)
        """
        try:
            # Try to connect to Ollama server
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/version")
                if response.status_code == 200:
                    version_info = response.json()
                    logger.info(
                        f"[{request_id}] Connected to Ollama server version: {version_info.get('version')}")
                    self.ollama_health_checked = True
                    self.ollama_available = True
                    return True, "Ollama server is available"
                else:
                    error_text = response.text
                    logger.error(
                        f"[{request_id}] Ollama server returned error: {error_text}")
                    return False, f"Ollama server returned status {response.status_code}"

        except Exception as e:
            logger.error(
                f"[{request_id}] Failed to connect to Ollama server at {self.base_url}: {str(e)}")
            return False, f"Failed to connect to Ollama server: {str(e)}"

    async def get_all_models(self, request_id: str) -> Dict[str, Any]:
        """Fetch and cache available models from multiple providers."""
        try:
            models = await self.fetch_models()
            result = {}

            # Process models
            for model in models:
                result[model.model] = {
                    'model': model.model,
                    'provider': 'ollama',
                    'size': model.size,
                    'modified_at': model.modified_at,
                    'details': {
                        'parent_model': model.details.parent_model,
                        'format': model.details.format,
                        'family': model.details.family,
                        'families': model.details.families,
                        'parameter_size': model.details.parameter_size,
                        'quantization_level': model.details.quantization_level
                    }
                }

            logger.info(f"[{request_id}] Successfully fetched all models")
            logger.debug(
                f"[{request_id}] Available models: {list(result.keys())}")
            return result

        except Exception as e:
            logger.error(f"[{request_id}] Error fetching models: {e}")
            return {}
