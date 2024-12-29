"""Example pipeline that performs multiple processing steps."""

from typing import Dict, Any
from app.functions.base import Pipeline, FunctionType, register_function
from app.functions.utils import ensure_strict_message


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

        # Convert all messages to StrictChatMessage objects
        messages = [ensure_strict_message(msg) for msg in messages]

        # Step 1: Collect all user messages
        user_messages = [
            msg for msg in messages
            if msg.role == "user"
        ]

        # Step 2: Process each message
        processed_messages = []
        for msg in messages:
            if msg.role == "user":
                # Process user messages
                processed_content = f"{msg.content} (Length: {len(msg.content)})"
                # Create specific message type based on role
                if msg.role == "user":
                    from app.models.chat import UserMessage
                    new_message = UserMessage(
                        role=msg.role,
                        content=processed_content,
                        images=msg.images
                    )
                else:
                    # For non-user messages, keep original
                    new_message = msg
                processed_messages.append(new_message)
            else:
                # Keep non-user messages unchanged
                processed_messages.append(msg)

        # Step 3: Combine results
        summary = {
            "original_count": len(messages),
            "user_messages": len(user_messages),
            "processed": [msg.content for msg in processed_messages if msg.role == "user"],
            "total_length": sum(len(msg.content.split(" (Length: ")[0]) for msg in processed_messages if msg.role == "user")
        }

        return {
            "messages": processed_messages,
            "summary": summary
        }
