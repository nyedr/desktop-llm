"""Example pipeline that performs multiple processing steps."""

from typing import Dict, Any
from app.functions.base import Pipeline, FunctionType, register_function
from app.models.chat import ChatMessage


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

    def _ensure_chat_message(self, msg: Any) -> ChatMessage:
        """Convert dictionary to ChatMessage if needed."""
        if isinstance(msg, dict):
            # Create a copy of the message dict without metadata
            msg_data = {k: v for k, v in msg.items()
                        if k in ChatMessage.model_fields}
            return ChatMessage(**msg_data)
        return msg

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

        # Convert all messages to ChatMessage objects
        messages = [self._ensure_chat_message(msg) for msg in messages]

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
                new_message = ChatMessage(
                    role=msg.role,
                    content=processed_content,
                    name=msg.name,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                    images=msg.images
                )
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
