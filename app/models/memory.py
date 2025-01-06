"""Models for memory operations."""
from datetime import datetime
import json
from typing import Any, Dict, Optional, Union
import uuid

from pydantic import BaseModel, Field, field_validator


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

    @field_validator('timestamp', mode='before')
    @classmethod
    def parse_timestamp(cls, v: Any) -> datetime:
        """Parse timestamp from string if needed.

        Args:
            v: The value to parse

        Returns:
            datetime: Parsed datetime object
        """
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

    @classmethod
    def from_lightrag_response(cls, response: Union[str, Dict[str, Any]], default_metadata: Optional[Dict[str, Any]] = None) -> Optional['MemoryResponse']:
        """Create a MemoryResponse from a LightRAG response.

        Args:
            response: Raw response from LightRAG (string or dict format)
            default_metadata: Default metadata to use if not present in response

        Returns:
            Optional[MemoryResponse]: Structured memory response if valid, None otherwise
        """
        if not response:
            return None

        try:
            # Handle string response format (naive mode)
            if isinstance(response, str):
                chunks = response.split("--New Chunk--")
                for chunk in chunks:
                    try:
                        chunk_data = json.loads(chunk.strip())
                        content = chunk_data.get("content", "").strip()
                        metadata = chunk_data.get("metadata", {})

                        if content:
                            return cls._create_response(content, metadata, default_metadata)
                    except json.JSONDecodeError:
                        continue

            # Handle dictionary response format
            elif isinstance(response, dict) and "sources" in response:
                sources = response["sources"]
                for source in sources:
                    content = source.get("content", "").strip()
                    metadata = source.get("metadata", {})

                    if content:
                        return cls._create_response(content, metadata, default_metadata)

            return None

        except Exception as e:
            raise ValueError(f"Failed to parse LightRAG response: {e}")

    @classmethod
    def _create_response(cls, content: str, metadata: Dict[str, Any], default_metadata: Optional[Dict[str, Any]] = None) -> 'MemoryResponse':
        """Create a standardized MemoryResponse with complete metadata.

        Args:
            content: Content string
            metadata: Existing metadata
            default_metadata: Default metadata to use if not present

        Returns:
            MemoryResponse: Properly structured memory response
        """
        # Ensure required metadata fields exist
        complete_metadata = {
            "memory_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "embedding_model": "minilm",
            **(default_metadata or {}),
            **metadata
        }

        return cls(
            metadata=MemoryMetadata(**complete_metadata),
            content=MemoryContent(
                user_message=metadata.get("user_message", content),
                assistant_response=metadata.get("assistant_response", ""),
                tool_response=metadata.get("tool_response")
            )
        )
