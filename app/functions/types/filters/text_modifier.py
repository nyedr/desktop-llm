"""Example filter that modifies text in both directions."""

from typing import Dict, Any
from app.functions.base import Filter, FunctionType, register_function
from app.models.chat import ChatMessage

@register_function(
    func_type=FunctionType.FILTER,
    name="text_modifier",
    description="Modifies text content in both inlet and outlet",
    priority=1
)
class TextModifierFilter(Filter):
    """Filter that adds context to incoming messages and formatting to outgoing ones."""
    
    name: str = "text_modifier"
    description: str = "Modifies text content in both inlet and outlet"
    type: FunctionType = FunctionType.FILTER
    priority: int = 1

    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add context to incoming messages.
        
        Args:
            data: Request data containing messages
            
        Returns:
            Modified request data
        """
        messages = data.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, ChatMessage) and last_message.role == "user":
                last_message.content = f"Context: This is a user message. Content: {last_message.content}"
        return data

    async def outlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format outgoing messages.
        
        Args:
            data: Response data containing message
            
        Returns:
            Modified response data
        """
        if "content" in data:
            data["content"] = f"[Processed Response] {data['content']}"
        return data
