"""Example filter that modifies text content."""

from typing import Dict, Any, List, Literal
from app.functions.base import Filter, FunctionType, register_function
from pydantic import Field


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

    name: str = Field(default="text_modifier",
                      description="Name of the filter")
    description: str = Field(
        default="Modifies text content in both inlet and outlet",
        description="Description of the filter"
    )
    type: Literal[FunctionType.FILTER] = Field(
        default=FunctionType.FILTER, description="Filter type")
    priority: int = Field(default=1, description="Filter priority")
    config: Dict[str, Any] = Field(
        default={
            "prefix": "[Modified] ",
            "suffix": " [End]"
        },
        description="Configuration for the filter"
    )

    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages.

        Args:
            data: Dictionary containing messages and request info

        Returns:
            Modified request data
        """
        messages = data.get("messages", [])
        if not messages:
            return data

        prefix = self.config.get("prefix", "")
        suffix = self.config.get("suffix", "")

        modified_messages = []
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                message = message.copy()
                message["content"] = f"{prefix}{content}{suffix}"
            modified_messages.append(message)

        data["messages"] = modified_messages
        return data

    async def outlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing messages.

        Args:
            data: Dictionary containing response data

        Returns:
            Modified response data
        """
        # Don't modify tool responses as they contain JSON data
        if data.get("role") in ["tool", "function"]:
            return data

        # Only modify content for assistant messages
        if "content" in data and data.get("role") == "assistant":
            prefix = self.config.get("prefix", "")
            suffix = self.config.get("suffix", "")
            data = data.copy()
            data["content"] = f"{prefix}{data['content']}{suffix}"
        return data
