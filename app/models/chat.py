"""Chat request/response models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """A chat message."""
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


class ChatRequest(BaseModel):
    """A chat request."""
    messages: List[ChatMessage] = Field(...,
                                        description="The conversation history")
    model: Optional[str] = Field(None, description="The model to use for chat")
    temperature: Optional[float] = Field(
        None, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        None, description="Maximum tokens to generate")
    stream: bool = Field(True, description="Whether to stream the response")
    timeout: Optional[int] = Field(
        None, description="Request timeout in seconds")
    tools: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of tools available to the model")
    enable_tools: bool = Field(
        True, description="Whether to enable tool usage")
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


class ChatResponse(BaseModel):
    """A chat response."""
    model: str = Field(..., description="The model used for chat")
    choices: List[Dict[str, Any]
                  ] = Field(..., description="List of completion choices")
    usage: Optional[Dict[str, int]] = Field(
        None, description="Token usage statistics")
