"""Example filter that modifies text content."""

from typing import Dict, Any
from app.models.function import Filter, FunctionType, register_function
from app.models.chat import StrictChatMessage
import logging

logger = logging.getLogger(__name__)


@register_function(
    func_type=FunctionType.FILTER,
    name="text_modifier",
    description="Modifies text content of messages",
    priority=1,
    config={
        "prefix": "[Modified] ",
        "suffix": " [End]"
    }
)
class TextModifierFilter(Filter):
    """Filter that modifies text content in both inlet and outlet."""

    def _modify_content(self, content: str) -> str:
        """Helper to modify content with prefix and suffix."""
        prefix = self.config.get("prefix", "[Modified] ")
        suffix = self.config.get("suffix", " [End]")
        modified = f"{prefix}{content}{suffix}"
        logger.debug(f"Content modified: '{content}' -> '{modified}'")
        return modified

    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages.

        Args:
            data: Dictionary containing messages array

        Returns:
            Modified request data
        """
        messages = data.get("messages", [])
        if not messages:
            return data

        modified_messages = []
        for message in messages:
            # Handle both dictionary and StrictChatMessage objects
            role = message.role if isinstance(
                message, StrictChatMessage) else message["role"]

            if role == "user":
                if isinstance(message, StrictChatMessage):
                    message = message.model_dump()
                else:
                    message = message.copy()
                message["content"] = self._modify_content(message["content"])
            modified_messages.append(message)

        data["messages"] = modified_messages
        return data

    async def outlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing messages.

        Args:
            data: Dictionary containing a single message or messages array

        Returns:
            Modified message data
        """
        try:
            # Handle messages array
            if "messages" in data:
                data["messages"] = [
                    await self._process_message(msg) for msg in data["messages"]
                ]
                return data

            # Handle single message
            return await self._process_message(data)

        except Exception as e:
            logger.error(f"Error in outlet filter: {e}")
            return data

    async def _process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single message.

        Args:
            message: Message dictionary to process

        Returns:
            Processed message
        """
        # Skip if message doesn't have required fields
        if not isinstance(message, dict) or "role" not in message:
            return message

        # Don't modify tool/function responses
        if message["role"] in ["tool", "function"]:
            return message

        # Only modify content for assistant messages
        if message["role"] == "assistant":
            message = message.copy()
            content = message.get("content")

            if isinstance(content, str):
                content = content.rstrip()  # Remove trailing spaces
                if content:  # Check if there's actual content after stripping
                    logger.info(f"Modifying content: '{content}'")
                    message["content"] = self._modify_content(content)
                    logger.info(f"Modified to: '{message['content']}'")
            elif isinstance(content, dict) and "content" in content:
                content = content.copy()
                # Remove trailing spaces
                stripped_content = content["content"].rstrip()
                if stripped_content:  # Check if there's actual content after stripping
                    content["content"] = self._modify_content(stripped_content)
                    message["content"] = content

        return message
