"""Example pipeline that performs multiple processing steps."""

from typing import Dict, Any, List
from app.functions.base import Pipeline, FunctionType
from app.models.chat import ChatMessage
from app.functions import register_function

@register_function(
    func_type=FunctionType.PIPELINE,
    name="multi_step_processor",
    description="Processes data through multiple steps",
    config={
        "max_steps": 3,
        "timeout_per_step": 30
    }
)
class MultiStepPipeline(Pipeline):
    """Pipeline that processes data through multiple defined steps."""
    
    name: str = "multi_step_processor"
    description: str = "Processes data through multiple steps"
    type: FunctionType = FunctionType.PIPELINE
    config: Dict[str, Any] = {
        "max_steps": 3,
        "timeout_per_step": 30
    }

    async def pipe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through multiple steps.
        
        Args:
            data: Input data containing messages and request info
            
        Returns:
            Processed data after all steps
        """
        messages = data.get("messages", [])
        if not messages:
            return data

        # Step 1: Collect all user messages
        user_messages = [
            msg for msg in messages 
            if isinstance(msg, ChatMessage) and msg.role == "user"
        ]

        # Step 2: Process each message
        processed_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                if msg.role == "user":
                    # Process user messages
                    new_message = {
                        "role": msg.role,
                        "content": f"{msg.content} (Length: {len(msg.content)})"
                    }
                    if msg.name:
                        new_message["name"] = msg.name
                    if msg.tool_calls:
                        new_message["tool_calls"] = msg.tool_calls
                    if msg.tool_call_id:
                        new_message["tool_call_id"] = msg.tool_call_id
                    if msg.images:
                        new_message["images"] = msg.images
                    processed_messages.append(ChatMessage(**new_message))
                else:
                    # Keep non-user messages unchanged
                    processed_messages.append(msg)

        # Step 3: Combine results
        summary = {
            "original_count": len(messages),
            "user_messages": len(user_messages),
            "processed": [msg.content for msg in processed_messages if msg.role == "user"],
            "total_length": sum(len(msg.content) for msg in processed_messages if msg.role == "user")
        }
        
        return {
            "messages": processed_messages,
            "summary": summary
        }
