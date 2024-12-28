"""Example filter that modifies text content."""

from typing import Dict, Any
from app.functions.base import Filter, FunctionType, register_function
from app.models.chat import ChatMessage
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
            # Handle both dictionary and ChatMessage objects
            role = message.role if isinstance(
                message, ChatMessage) else message["role"]

            if role == "user":
                if isinstance(message, ChatMessage):
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
            data: Dictionary containing a single message

        Returns:
            Modified message data
        """
        # Don't modify tool responses
        if data["role"] in ["tool", "function"]:
            return data

        # Only modify content for assistant messages
        if data["role"] == "assistant":
            data = data.copy()
            if isinstance(data["content"], str):
                content = data["content"].rstrip()  # Remove trailing spaces
                if content:  # Check if there's actual content after stripping
                    logger.info(f"Modifying content: '{content}'")
                    data["content"] = self._modify_content(content)
                    logger.info(f"Modified to: '{data['content']}'")
            elif isinstance(data["content"], dict) and "content" in data["content"]:
                content = data["content"].copy()
                # Remove trailing spaces
                stripped_content = content["content"].rstrip()
                if stripped_content:  # Check if there's actual content after stripping
                    content["content"] = self._modify_content(stripped_content)
                    data["content"] = content

        return data
