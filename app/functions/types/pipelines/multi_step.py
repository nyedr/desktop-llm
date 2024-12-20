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
        for msg in user_messages:
            content = msg.content
            # Example processing: Add length info
            processed_content = f"{content} (Length: {len(content)})"
            processed_messages.append(processed_content)

        # Step 3: Combine results
        summary = {
            "original_count": len(messages),
            "user_messages": len(user_messages),
            "processed": processed_messages,
            "total_length": sum(len(m) for m in processed_messages)
        }
        
        return {
            "messages": messages,
            "summary": summary
        }
