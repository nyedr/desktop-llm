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
            new_messages = []
            for msg in messages[:-1]:
                new_messages.append(msg)
            
            last_message = messages[-1]
            if isinstance(last_message, ChatMessage) and last_message.role == "user":
                # Create a new message with modified content
                new_message = {
                    "role": last_message.role,
                    "content": f"Context: This is a user message. Content: {last_message.content}"
                }
                if last_message.name:
                    new_message["name"] = last_message.name
                if last_message.tool_calls:
                    new_message["tool_calls"] = last_message.tool_calls
                if last_message.tool_call_id:
                    new_message["tool_call_id"] = last_message.tool_call_id
                if last_message.images:
                    new_message["images"] = last_message.images
                new_messages.append(ChatMessage(**new_message))
            else:
                new_messages.append(last_message)
            
            data["messages"] = new_messages
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
