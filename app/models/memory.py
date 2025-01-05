"""Models for memory operations."""
from datetime import datetime
import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class MemoryContent(BaseModel):
    """Content structure for memory responses."""
    user_message: str = Field(default="")
    assistant_response: str = Field(default="")
    tool_response: Optional[str] = Field(default=None)


class MemoryMetadata(BaseModel):
    """Metadata structure for memory responses."""
    memory_id: str
    timestamp: datetime
    content_type: str = Field(default="chat_memory")
    chunk_size: int = Field(default=512)
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    embedding_model: str
    request_id: Optional[str] = None
    model: Optional[str] = None
    message_count: Optional[int] = None
    has_tool_calls: bool = Field(default=False)
    enable_tools: bool = Field(default=True)

    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        """Parse timestamp from string if needed."""
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v


class MemoryResponse(BaseModel):
    """Standardized memory response structure."""
    metadata: MemoryMetadata
    content: MemoryContent

    @classmethod
    def from_raw_response(cls, response: Dict[str, Any]) -> 'MemoryResponse':
        """Create a MemoryResponse from a raw response dictionary.

        Args:
            response: Raw response dictionary from LightRAG

        Returns:
            MemoryResponse: Properly structured memory response
        """
        if isinstance(response, dict) and "content" in response:
            try:
                # Parse the content field if it's a JSON string
                if isinstance(response["content"], str):
                    content_data = json.loads(response["content"])
                else:
                    content_data = response["content"]

                # Extract metadata and content
                metadata = {
                    key: value for key, value in content_data.items()
                    if key not in ["content", "user_message", "assistant_response", "tool_response"]
                }

                content = content_data.get("content", {})
                if not isinstance(content, dict):
                    content = {
                        "user_message": str(content),
                        "assistant_response": "",
                        "tool_response": None
                    }

                return cls(
                    metadata=MemoryMetadata(**metadata),
                    content=MemoryContent(**content)
                )
            except Exception as e:
                raise ValueError(f"Failed to parse memory response: {e}")
        raise ValueError("Invalid memory response format")
