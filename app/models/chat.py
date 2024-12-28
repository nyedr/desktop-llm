"""Chat request/response models."""

# TODO: Future Refactoring
# The codebase should be refactored to use the stricter message types (StrictChatMessage)
# throughout all components. This includes:
# 1. Updating all services to use StrictChatMessage instead of ChatMessage
# 2. Updating all function implementations to handle strict types
# 3. Updating tests to use strict types
# 4. Maintaining backward compatibility during transition using to_strict() and from_strict()
# 5. Eventually deprecating the ChatMessage class once migration is complete

from typing import Dict, List, Optional, Union, Literal, Any
from pydantic import BaseModel, Field, field_validator


class Function(BaseModel):
    """Function data for tool calls."""
    name: str
    arguments: Dict[str, str]


class ToolCall(BaseModel):
    """Tool call data."""
    function: Function


class BaseMessage(BaseModel):
    """Base message with common fields."""
    role: str
    content: Union[str, Dict[str, Any]]
    images: Optional[List[str]] = None


class AssistantMessage(BaseMessage):
    """Assistant message data."""
    role: Literal["assistant"] = "assistant"
    content: Union[str, Dict[str, Optional[Union[str, List[ToolCall]]]]]
    type: Optional[Literal["chunk"]] = None
    tool_calls: Optional[List[ToolCall]] = None


class ToolMessage(BaseMessage):
    """Tool message data."""
    role: Literal["tool"] = "tool"
    content: str
    name: str
    tool_call_id: Optional[str] = None


class UserMessage(BaseMessage):
    """User message data."""
    role: Literal["user"] = "user"
    content: str


class SystemMessage(BaseMessage):
    """System message data."""
    role: Literal["system"] = "system"
    content: str


StrictChatMessage = Union[AssistantMessage,
                          ToolMessage, UserMessage, SystemMessage]


class ChatMessage(BaseModel):
    """A chat message with backward compatibility."""
    role: str = Field(..., description="The role of the message sender",
                      pattern="^(user|assistant|system|tool)$")
    content: str = Field(...,
                         description="The content of the message", min_length=1)
    name: Optional[str] = Field(
        None, description="Name of the sender (optional)")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool calls made by the assistant")
    tool_call_id: Optional[str] = Field(
        None, description="ID of the tool call this message is responding to")
    images: Optional[List[str]] = Field(
        None, description="Base64-encoded images attached to the message")
    type: Optional[Literal["chunk"]] = Field(
        None, description="Type of message (e.g., chunk for streaming)")

    def to_strict(self) -> StrictChatMessage:
        """Convert to a strict message type."""
        base_data = {
            "role": self.role,
            "content": self.content,
            "images": self.images
        }

        if self.role == "assistant":
            return AssistantMessage(
                **base_data,
                type=self.type,
                tool_calls=[ToolCall(**tc) for tc in (self.tool_calls or [])]
            )
        elif self.role == "tool":
            return ToolMessage(
                **base_data,
                name=self.name or "",
                tool_call_id=self.tool_call_id
            )
        elif self.role == "user":
            return UserMessage(**base_data)
        else:  # system
            return SystemMessage(**base_data)

    @classmethod
    def from_strict(cls, message: StrictChatMessage) -> "ChatMessage":
        """Create from a strict message type."""
        data = message.model_dump()
        if isinstance(message, AssistantMessage):
            data["tool_calls"] = [tc.model_dump()
                                  for tc in (message.tool_calls or [])]
        return cls(**data)


class ChatRequest(BaseModel):
    """Chat request data."""
    messages: List[Union[ChatMessage, StrictChatMessage]]
    model: Optional[str] = Field(None, description="The model to use for chat")
    temperature: Optional[float] = Field(
        None, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        None, description="Maximum tokens to generate")
    stream: bool = Field(True, description="Whether to stream the response")
    enable_tools: bool = Field(
        True, description="Whether to enable tool usage")
    enable_memory: bool = Field(
        True, description="Whether to enable memory usage")
    memory_filter: Optional[Dict[str, Any]] = Field(
        None, description="Filter for memory usage")
    top_k_memories: Optional[int] = Field(
        5, description="Top k memories to consider")
    filters: Optional[List[str]] = Field(
        None, description="List of filters to apply")
    pipeline: Optional[str] = Field(
        None, description="Pipeline to process the request")

    @field_validator("messages")
    def validate_messages(cls, v):
        """Validate messages."""
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class ChatStreamEvent(BaseModel):
    """Chat stream event data."""
    event: Literal["start", "message", "error", "pipeline", "pipeline_error"]
    data: str  # JSON string of the actual data
