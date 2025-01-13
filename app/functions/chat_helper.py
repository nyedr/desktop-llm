"""Chat utilities for custom functions."""

import logging
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from app.core.config import config
from app.models.chat import StrictChatMessage

logger = logging.getLogger(__name__)


class ChatHelper:
    """Helper class for chat operations in custom functions."""

    def __init__(self, model_service=None, function_service=None):
        """Initialize with optional service dependencies.

        Args:
            model_service: Optional ModelService instance
            function_service: Optional FunctionService instance
        """
        from app.services.model_service import ModelService
        from app.services.function_service import FunctionService

        self.model_service = model_service or ModelService()
        self.function_service = function_service or FunctionService()

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
        tool_filter: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_tools: bool = True,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Generate a chat completion for use in custom functions.

        Args:
            messages: List of messages in the conversation
            model: Model to use (defaults to config.DEFAULT_MODEL)
            temperature: Temperature for generation (defaults to config.MODEL_TEMPERATURE)
            max_tokens: Maximum tokens to generate (defaults to config.MAX_TOKENS)
            stream: Whether to stream the response
            tool_filter: Optional list of tool names to include (if None, includes all registered tools)
            tools: Additional tool schemas to include (in addition to registered tools)
            enable_tools: Whether to enable tool usage

        Returns:
            AsyncGenerator yielding response chunks
        """
        try:
            # Get registered tools from function service
            registered_tools = []
            if enable_tools:
                # Get all registered tools
                all_tools = self.function_service.get_function_schemas()

                # Filter tools if tool_filter is provided
                if tool_filter:
                    registered_tools = [
                        tool for tool in all_tools
                        if tool.get("function", {}).get("name") in tool_filter
                    ]
                else:
                    registered_tools = all_tools

                # Add additional tools if provided
                if tools:
                    registered_tools.extend([
                        {"type": "function", "function": tool}
                        for tool in tools
                    ])

            # Log which model we're using
            used_model = model or config.llm.model
            logger.info(f"Using model: {used_model} for chat completion")

            async for response in self.model_service.chat(
                messages=messages,
                model=used_model,
                temperature=temperature or config.llm.temperature,
                max_tokens=max_tokens or config.llm.max_tokens,
                stream=stream,
                tools=registered_tools if enable_tools else None,
                enable_tools=enable_tools
            ):
                yield response
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            yield {"error": str(e)}
