"""Model service module."""

import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple
import uuid
from openai import AsyncOpenAI
import openai
import asyncio

from app.core.config import config
from app.models.chat import StrictChatMessage
from app.models.model import Model
from lightrag.llm import ollama_embedding

from app.utils.chat_tools import format_tools_for_chat

logger = logging.getLogger(__name__)


class ModelServiceError(Exception):
    """Base exception for model service errors."""
    pass


class EmbeddingProviderError(ModelServiceError):
    """Raised when embedding provider fails."""
    pass


class CompletionProviderError(ModelServiceError):
    """Raised when completion provider fails."""
    pass


class ModelService:
    """Service for interacting with language models through OpenAI-compatible endpoints."""

    def __init__(self):
        """Initialize model service with configuration."""
        try:
            self._init_config()
            if not config.llm.api_key:
                logger.warning(
                    "No API key provided - some features may be limited")
            self._init_client()
            self._init_embeddings()
            if config.llm.api_key:
                self._verify_connection()
            logger.info("Initialized ModelService")
        except Exception as e:
            logger.error(f"Failed to initialize ModelService: {e}")
            raise

    def _verify_connection(self):
        """Verify connection to OpenAI endpoint works."""
        try:
            # Try to make a simple request to verify connection
            response = self.client.models.list()
            if not response:
                raise CompletionProviderError(
                    "No models available from provider")
        except Exception as e:
            raise CompletionProviderError(
                f"Failed to connect to OpenAI endpoint: {str(e)}")

    def _prepare_messages(self, messages: List[Union[Dict[str, Any], StrictChatMessage]]) -> List[Dict[str, Any]]:
        """Prepare messages for API request.

        Args:
            messages: List of messages to prepare

        Returns:
            List of formatted message dictionaries
        """
        formatted = []
        for msg in messages:
            if hasattr(msg, 'model_dump'):
                # Handle Pydantic models
                msg_dict = msg.model_dump()
            elif isinstance(msg, dict):
                # Handle dictionaries
                msg_dict = msg
            else:
                # Handle unexpected types
                logger.warning(f"Unexpected message type: {type(msg)}")
                msg_dict = {"role": "user", "content": str(msg)}

            # Ensure required fields are present and properly formatted
            formatted_msg = {
                "role": msg_dict.get("role", "user"),
                "content": str(msg_dict.get("content", "")).strip()
            }

            # Only add optional fields if they exist and are not None
            if msg_dict.get("name"):
                formatted_msg["name"] = msg_dict["name"]
            if msg_dict.get("function_call"):
                formatted_msg["function_call"] = msg_dict["function_call"]
            if msg_dict.get("tool_calls"):
                formatted_msg["tool_calls"] = msg_dict["tool_calls"]

            formatted.append(formatted_msg)

        return formatted

    def _init_config(self):
        """Initialize configuration parameters."""
        self.request_timeout = config.llm.timeout
        self.generation_timeout = config.llm.timeout
        self.default_model = "deepseek/deepseek-chat"  # Set default model
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        self.function_calls_enabled = config.llm.enable_tools
        self.enable_model_filter = config.functions.enable_model_filter
        self.model_filter_list = config.functions.model_filter_list

        # Log embedding configuration
        logger.info("Initializing model service with configuration:")
        logger.info(f"Default model: {self.default_model}")
        logger.info(
            f"Default embedding model (Ollama): {config.memory.default_embedding_model}")

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            if not config.llm.api_key:
                logger.warning("Initializing OpenAI client without API key")
            self.client = AsyncOpenAI(
                # OpenAI client requires non-empty string
                api_key=config.llm.api_key,
                base_url=str(config.llm.base_url),
                default_headers={
                    "HTTP-Referer": "http://localhost:8001",
                    "X-Title": "Desktop LLM"
                }
                
            )
        except Exception as e:
            raise CompletionProviderError(
                f"Failed to initialize OpenAI client: {e}")

    def _init_embeddings(self):
        """Initialize embeddings configuration."""
        try:
            logger.info("Using Ollama nomic-embed-text model for embeddings")
        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to initialize embedding configuration: {e}")

    def _get_request_id(self) -> str:
        """Get a unique request ID."""
        return str(uuid.uuid4())

    async def get_embeddings(
        self,
        texts: Union[str, List[str]],
    ) -> List[List[float]]:
        """Get embeddings using Ollama with nomic-embed-text model."""
        try:
            if isinstance(texts, str):
                texts = [texts]

            embeddings = await ollama_embedding(
                texts,
                embed_model="nomic-embed-text",
                host="http://localhost:11434"
            )
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise EmbeddingProviderError(f"Failed to get embeddings: {e}")

    async def chat(
        self,
        messages: List[Union[Dict[str, Any], StrictChatMessage]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_tools: bool = True,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Generate chat completions with optional tool execution.

        Args:
            messages: List of chat messages
            model: Optional override for model
            temperature: Optional override for temperature
            max_tokens: Optional override for max_tokens
            stream: Whether to stream the response
            tools: Optional list of tools to enable
            enable_tools: Whether to enable tool execution

        Yields:
            Response chunks from the model and tool execution
        """
        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] Chat request started")

        try:
            # Format messages for API
            formatted_messages = self._prepare_messages(messages)

            # Prepare request parameters
            params = {
                "model": model or self.default_model,
                "messages": formatted_messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "stream": stream
            }

            # Add tools if enabled
            if enable_tools and tools:
                formatted_tools = format_tools_for_chat(tools, request_id)
                if formatted_tools:
                    params["tools"] = formatted_tools
                    params["tool_choice"] = "auto"

            # Get response from API
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**params),
                    timeout=self.request_timeout
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"[{request_id}] Request timed out after {self.request_timeout}s")
                raise CompletionProviderError("Request timed out")
            except openai.APIError as e:
                logger.error(f"[{request_id}] OpenRouter API error: {str(e)}")
                raise

            if stream:
                chunk_count = 0
                try:
                    async for chunk in response:
                        try:
                            chunk_count += 1
                            if not chunk or not chunk.choices:
                                continue

                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                yield delta.content
                            elif hasattr(delta, 'tool_calls') and delta.tool_calls:
                                yield {"tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments
                                        }
                                    } for tool_call in delta.tool_calls
                                ]}

                        except Exception as e:
                            logger.error(
                                f"[{request_id}] Error processing chunk: {str(e)}")
                            continue

                except Exception as e:
                    logger.error(
                        f"[{request_id}] Error in stream processing: {str(e)}")
                    raise

            else:
                if not response or not response.choices:
                    raise CompletionProviderError(
                        "Empty response from provider")

                message = response.choices[0].message
                if message.content:
                    yield message.content
                elif hasattr(message, 'tool_calls') and message.tool_calls:
                    yield {"tool_calls": message.tool_calls}

        except Exception as e:
            logger.error(f"[{request_id}] Chat error: {str(e)}")
            raise CompletionProviderError(f"Chat failed: {e}")

    async def fetch_models(self) -> List[Model]:
        """Fetch available models from the OpenAI-compatible endpoint."""
        request_id = self._get_request_id()
        logger.info(f"[{request_id}] Fetching models")

        try:
            response = await self.client.models.list()
            models = []
            for model_data in response.data:
                models.append(Model(
                    model=model_data.id,
                    modified_at=str(model_data.created),
                    size=0,
                    details={
                        "parent_model": model_data.id,
                        "format": "openai",
                        "family": "openai",
                        "families": ["openai"],
                        "parameter_size": "unknown",
                        "quantization_level": "unknown"
                    }
                ))

            logger.info(f"[{request_id}] Found {len(models)} total models")
            return models

        except Exception as e:
            logger.error(f"[{request_id}] Error fetching models: {e}")
            raise CompletionProviderError(f"Failed to fetch models: {e}")

    async def get_models(self) -> List[str]:
        """Get list of available model names."""
        request_id = self._get_request_id()
        try:
            models = await self.fetch_models()
            model_names = [model.model for model in models]
            return model_names
        except Exception as e:
            logger.error(f"[{request_id}] Error getting models: {e}")
            raise CompletionProviderError(f"Failed to get model list: {e}")

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
            raise CompletionProviderError(f"Failed to get model info: {e}")

    async def check_health(self, request_id: str) -> Dict[str, Tuple[bool, str]]:
        """Check health of all configured endpoints.

        Returns:
            Dict[str, Tuple[bool, str]]: Health status for each endpoint
        """
        health_status = {}

        # Check OpenAI-compatible endpoint
        try:
            await self.client.models.list()
            health_status["openai"] = (
                True, "OpenAI-compatible endpoint is available")
        except Exception as e:
            health_status["openai"] = (
                False, f"Failed to connect to OpenAI endpoint: {str(e)}")

        return health_status
