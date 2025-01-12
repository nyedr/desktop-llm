"""Assistant service module."""

import logging
from typing import AsyncGenerator, Dict, Any, List, Optional, Union

from app.core.config import config
from app.models.chat import StrictChatMessage
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)


class Assistant:
    """Assistant for handling chat interactions."""

    def __init__(self):
        """Initialize Assistant with default configuration."""
        self.model = config.llm.model
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens

        # Services will be initialized later
        self.model_service = None
        self._initialized = False

    @classmethod
    async def create(cls, model_service: ModelService) -> 'Assistant':
        """Create and initialize a new Assistant instance."""
        Assistant = cls()
        await Assistant.initialize(model_service)
        return Assistant

    async def initialize(self, model_service: ModelService):
        """Initialize the Assistant with required services."""
        if self._initialized:
            return

        self.model_service = model_service
        self._initialized = True

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> AsyncGenerator[str, None]:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt
            model: Optional override for model
            temperature: Optional override for temperature
            max_tokens: Optional override for max_tokens
            stream: Whether to stream the response

        Yields:
            Generated text chunks
        """
        if not self._initialized:
            raise RuntimeError("Assistant not initialized")

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
        if not self._initialized:
            raise RuntimeError("Assistant not initialized")

        try:
            async for chunk in self.model_service.chat(
                messages=messages,
                model=model or self.model,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream,
                tools=tools,
                enable_tools=enable_tools
            ):
                if chunk:  # Only yield non-empty responses
                    yield chunk

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise

    async def cleanup(self):
        """Cleanup Assistant resources."""
        try:
            logger.info("Cleaning up Assistant...")
            self._initialized = False
            self.model_service = None
            logger.info("Assistant cleanup complete")
        except Exception as e:
            logger.error(f"Error during Assistant cleanup: {e}")
            raise
