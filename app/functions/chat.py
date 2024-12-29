"""Chat utilities for custom functions."""

import logging
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from app.core.config import config
from app.models.chat import StrictChatMessage
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)


class ChatHelper:
    """Helper class for chat operations in custom functions."""

    def __init__(self):
        self.model_service = ModelService()

    async def get_available_models(self) -> List[str]:
        """Get a list of available models.

        Returns:
            List of model names
        """
        try:
            return await self.model_service.get_models()
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []

    async def generate_completion(
        self,
        messages: List[StrictChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_tools: bool = True,
        function_service=None
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Generate a chat completion for use in custom functions.

        Args:
            messages: List of messages in the conversation
            model: Model to use (defaults to config.DEFAULT_MODEL)
            temperature: Temperature for generation (defaults to config.MODEL_TEMPERATURE)
            max_tokens: Maximum tokens to generate (defaults to config.MAX_TOKENS)
            stream: Whether to stream the response
            tools: List of available tools
            enable_tools: Whether to enable tool usage
            function_service: Service for executing functions

        Returns:
            AsyncGenerator yielding response chunks
        """
        try:
            async for response in self.model_service.chat(
                messages=messages,
                model=model or config.DEFAULT_MODEL,
                temperature=temperature or config.MODEL_TEMPERATURE,
                max_tokens=max_tokens or config.MAX_TOKENS,
                stream=stream,
                tools=tools,
                enable_tools=enable_tools,
                function_service=function_service
            ):
                yield response
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            yield {"error": str(e)}
