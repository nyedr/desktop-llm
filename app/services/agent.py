"""Agent service for managing model interactions and function execution."""
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import logging
from pydantic import BaseModel

from app.services.function_service import FunctionService
from app.services.model_service import ModelService
from app.services.langchain_service import LangChainService
from app.core.config import config
from app.models.chat import StrictChatMessage, ChatMessage

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Chat message model."""
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class Agent:
    """Agent for managing model interactions and function execution."""

    def __init__(
        self,
        model_service: ModelService,
        function_service: FunctionService,
        langchain_service: Optional[LangChainService] = None,
        model: str = config.DEFAULT_MODEL,
        temperature: float = config.MODEL_TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS
    ):
        self.model_service = model_service
        self.function_service = function_service
        self.langchain_service = langchain_service
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model_cache = {}

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> AsyncGenerator[str, None]:
        """Generate a completion for the given prompt."""
        try:
            completion_stream = self.model_service.generate(
                prompt=prompt,
                model=model or self.model,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream
            )

            async for response in completion_stream:
                if response:  # Only yield non-empty responses
                    yield response

        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise

    async def chat(
        self,
        messages: List[Union[Dict[str, Any], ChatMessage, StrictChatMessage]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_tools: bool = False,
        enable_memory: bool = True,
        memory_filter: Optional[Dict[str, Any]] = None,
        top_k_memories: int = 5
    ) -> AsyncGenerator[str, None]:
        """Generate chat completions with optional memory context."""
        try:
            if enable_tools and tools:
                logger.debug(
                    f"Tools enabled for chat. Available tools: {[t['function']['name'] for t in tools]}")
            else:
                logger.debug("No tools enabled for chat")

            # Convert messages to list if needed
            if not isinstance(messages, list):
                messages = [messages]

            # Keep track of messages for the conversation
            conversation = list(messages)

            # Add memory context if enabled and LangChain service is available
            if enable_memory and self.langchain_service:
                try:
                    conversation = await self.langchain_service.process_conversation(
                        conversation,
                        metadata_filter=memory_filter,
                        top_k=top_k_memories
                    )
                    logger.debug("Added memory context to conversation")
                except Exception as e:
                    logger.warning(f"Failed to add memory context: {e}")

            # Get response from model service
            async for response in self.model_service.chat(
                messages=conversation,
                model=model or self.model,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream,
                tools=tools,
                enable_tools=enable_tools,
                function_service=self.function_service
            ):
                if response:
                    yield response

        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            raise

    async def execute_function(
        self,
        function_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute a function with the given arguments."""
        try:
            result = await self.function_service.execute_function(
                function_name=function_name,
                arguments=arguments
            )
            return result
        except Exception as e:
            logger.error(f"Error executing function: {e}")
            raise
