"""Chat request/response models."""


from typing import Dict, List, Optional, Union, Literal, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from enum import Enum


class ChatRole(str, Enum):
    """Possible roles for a chat message."""
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"

    def __str__(self):
        """Return the string value of the enum."""
        return self.value


class Function(BaseModel):
    """Function data for tool calls."""
    name: str
    arguments: Dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class ToolCall(BaseModel):
    """Tool call data for assistant messages.

    Example:
        {
            "function": {
                "name": "get_weather",
                "arguments": {"location": "San Francisco", "unit": "celsius"}
            }
        }
    """
    function: Function

    model_config = ConfigDict(extra="forbid")


class BaseMessage(BaseModel):
    """Base fields for all messages."""
    role: ChatRole
    content: str

    model_config = ConfigDict(extra="forbid")


class AssistantMessage(BaseMessage):
    """Assistant message data."""
    role: Literal[ChatRole.ASSISTANT] = ChatRole.ASSISTANT
    type: Optional[Literal["chunk"]] = None
    tool_calls: Optional[List[ToolCall]] = None

    @model_validator(mode='after')
    def validate_tool_calls(self) -> 'AssistantMessage':
        """Validate tool calls are only present for assistant messages."""
        if self.role != ChatRole.ASSISTANT and self.tool_calls:
            raise ValueError(
                "tool_calls can only be present for assistant messages")
        return self


class UserMessage(BaseMessage):
    """User message data."""
    role: Literal[ChatRole.USER] = ChatRole.USER
    images: Optional[List[str]] = Field(
        None, description="Base64-encoded images attached to the message"
    )

    @model_validator(mode='after')
    def validate_images(self) -> 'UserMessage':
        """Validate images are only present for user messages."""
        if self.role != ChatRole.USER and self.images:
            raise ValueError("Images are only allowed for user messages")
        return self


class SystemMessage(BaseMessage):
    """System message data."""
    role: Literal[ChatRole.SYSTEM] = ChatRole.SYSTEM


class ToolMessage(BaseMessage):
    """Tool message data (response from a tool)."""
    role: Literal[ChatRole.TOOL] = ChatRole.TOOL
    name: str = Field(..., description="Name of the tool responding")
    tool_call_id: Optional[str] = Field(
        None, description="ID linking to the original tool call"
    )


StrictChatMessage = Union[AssistantMessage,
                          UserMessage, SystemMessage, ToolMessage]


class ChatRequest(BaseModel):
    """Chat request data."""
    messages: List[StrictChatMessage]
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
    memory_type: Optional[str] = Field(
        "ephemeral", description="Type of memory to use (ephemeral or persistent)")
    memory_filter: Optional[Dict[str, Any]] = Field(
        None, description="Filter for memory usage")
    top_k_memories: Optional[int] = Field(
        5, description="Top k memories to consider")
    filters: Optional[List[str]] = Field(
        None, description="List of filters to apply")
    pipeline: Optional[str] = Field(
        None, description="Pipeline to process the request")
    conversation_id: Optional[str] = Field(
        None, description="ID of the conversation for memory context")
    enable_summarization: Optional[bool] = Field(
        False, description="Whether to enable conversation summarization")

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
