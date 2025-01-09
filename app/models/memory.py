"""Models for memory operations."""
from datetime import datetime
import json
import logging
from typing import Any, Dict, Optional, Union, List

from pydantic import BaseModel, Field, field_validator

from app.models.chat import ChatRole
from app.utils.utils import format_timestamp, format_timestamp_date

logger = logging.getLogger(__name__)


class MemoryContent(BaseModel):
    """Content structure for memory responses."""
    user_message: str = Field(default="")
    assistant_response: str = Field(default="")
    tool_response: Optional[str] = Field(default=None)

    def format_conversation(self) -> str:
        """Format the conversation content."""
        conversation = [f"User: {self.user_message}"]
        if self.assistant_response:
            conversation.append(f"Assistant: {self.assistant_response}")
        if self.tool_response:
            conversation.append(f"Tool Response: {self.tool_response}")
        return "\n".join(conversation)


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
        """Parse timestamp from string if needed."""
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    def format_metadata(self, exclude_internal: bool = True) -> str:
        """Format metadata for LLM context.

        Args:
            exclude_internal: Whether to exclude internal fields

        Returns:
            str: Formatted metadata string
        """
        internal_fields = {
            "memory_id", "timestamp", "request_id", "content_type",
            "content", "source", "chunk_index", "token_count",
            "user_message", "assistant_response", "tool_response"
        } if exclude_internal else set()

        metadata_dict = self.model_dump()
        metadata_items = [
            f"- {key}: {value}"
            for key, value in metadata_dict.items()
            if key not in internal_fields and value is not None
        ]
        return "\n".join(metadata_items)


class MemoryResponse(BaseModel):
    """Standardized memory response structure."""
    metadata: MemoryMetadata
    content: MemoryContent

    def to_context_message(self) -> Dict[str, Any]:
        """Convert memory response to a context message format.

        Returns:
            Dict[str, Any]: Formatted context message
        """
        try:
            logger.debug(
                f"Converting memory to context message - Content: {self.content.model_dump_json()}")

            # Format timestamp
            formatted_time = format_timestamp_date(self.metadata.timestamp)
            time_from_now = datetime.now() - self.metadata.timestamp
            time_from_now_str = format_timestamp(time_from_now)

            # Format metadata and conversation
            metadata_str = self.metadata.format_metadata()
            conversation_str = self.content.format_conversation()

            logger.debug(f"Formatted conversation: {conversation_str}")
            logger.debug(f"Formatted metadata: {metadata_str}")

            # Create memory message with metadata context
            message = {
                "role": ChatRole.SYSTEM,
                "content": (
                    f"[Memory from {time_from_now_str} ({formatted_time})]\n"
                    f"Context:\n{metadata_str}\n\n"
                    f"Conversation:\n{conversation_str}"
                ),
                "metadata": {"type": "memory"}
            }
            logger.debug(f"Created context message: {message}")
            return message
        except Exception as e:
            logger.error(
                f"Error formatting memory to context message: {e}", exc_info=True)
            # Fallback to simple format if detailed formatting fails
            return {
                "role": ChatRole.SYSTEM,
                "content": f"[Memory] User: {self.content.user_message}",
                "metadata": {"type": "memory"}
            }

    @classmethod
    def format_memory_context(cls, memories: List[Union['MemoryResponse', Dict[str, Any]]]) -> Dict[str, Any]:
        """Format multiple memories into a single context message."""
        if not memories:
            logger.debug("No memories to format")
            return None

        memory_contents = []
        for i, memory in enumerate(memories, 1):
            try:
                logger.debug(f"Processing memory {i} of type: {type(memory)}")
                if isinstance(memory, dict):
                    logger.debug(
                        f"Memory {i} content: {json.dumps(memory, default=str)}")
                else:
                    logger.debug(
                        f"Memory {i} content: {memory.model_dump_json()}")

                # If it's already a MemoryResponse, use it directly
                if isinstance(memory, MemoryResponse):
                    formatted = memory.to_context_message()
                    memory_contents.append(
                        f"[Memory {i}]: {formatted['content']}")
                    logger.debug(
                        f"Added formatted memory {i} from MemoryResponse")
                    continue

                # If it's a dict, try to format it directly
                if isinstance(memory, dict):
                    if "content" in memory:
                        formatted = {
                            "role": ChatRole.SYSTEM,
                            "content": f"[Memory] {memory['content']}",
                            "metadata": {"type": "memory"}
                        }
                        memory_contents.append(
                            f"[Memory {i}]: {formatted['content']}")
                        logger.debug(f"Added formatted memory {i} from dict")
                        continue

                logger.warning(
                    f"Memory {i} could not be formatted: invalid type or structure")

            except Exception as e:
                logger.error(
                    f"Failed to format memory {i}: {str(e)}", exc_info=True)
                continue

        if not memory_contents:
            logger.warning("No memories were successfully formatted")
            return None

        context_message = {
            "role": ChatRole.SYSTEM,
            "content": "Relevant context from memory:\n" + "\n\n".join(memory_contents),
            "metadata": {"type": "memory_context"}
        }
        logger.debug(
            f"Created final context message with {len(memory_contents)} memories")
        return context_message

    @classmethod
    def from_lightrag_response(cls, response: Union[str, Dict[str, Any]], default_metadata: Optional[Dict[str, Any]] = None) -> Optional[List['MemoryResponse']]:
        """Create MemoryResponses from a LightRAG response.

        Args:
            response: Raw response from LightRAG (string or dict format)
            default_metadata: Default metadata to use if not present in response

        Returns:
            Optional[List[MemoryResponse]]: List of structured memory responses if valid, None otherwise
        """
        if not response:
            logger.warning("Empty response received from LightRAG")
            return None

        try:
            memories = []
            logger.debug(
                f"Processing LightRAG response of type: {type(response)}")

            # Handle dictionary response format
            if isinstance(response, dict):
                # Handle vector_context if present (new LightRAG format)
                if "vector_context" in response:
                    try:
                        vector_context = response["vector_context"]
                        if isinstance(vector_context, str):
                            chunks = vector_context.split("--New Chunk--")
                            logger.debug(
                                f"Processing {len(chunks)} chunks from vector_context")

                            for chunk in chunks:
                                chunk = chunk.strip()
                                if not chunk:
                                    continue

                                try:
                                    chunk_data = json.loads(chunk)
                                    content = chunk_data.get(
                                        "content", "").strip()
                                    metadata = chunk_data.get("metadata", {})

                                    if content and metadata:
                                        try:
                                            memory = MemoryResponse(
                                                metadata=MemoryMetadata(**{
                                                    **(default_metadata or {}),
                                                    **metadata
                                                }),
                                                content=MemoryContent(
                                                    user_message=metadata.get(
                                                        "user_message", content),
                                                    assistant_response=metadata.get(
                                                        "assistant_response", ""),
                                                    tool_response=json.dumps(metadata.get("tool_response")) if isinstance(
                                                        metadata.get("tool_response"), dict) else metadata.get("tool_response")
                                                )
                                            )
                                            memories.append(memory)
                                        except Exception as e:
                                            logger.error(
                                                f"Failed to create memory: {str(e)}")
                                except json.JSONDecodeError as e:
                                    logger.error(
                                        f"Failed to parse chunk as JSON: {str(e)}")
                                    continue
                                except Exception as e:
                                    logger.error(
                                        f"Error processing chunk: {str(e)}")
                                    continue

                    except Exception as e:
                        logger.error(
                            f"Error processing vector_context: {str(e)}")

                # Handle sources if present (old format)
                elif "sources" in response:
                    sources = response["sources"]
                    logger.debug(f"Processing {len(sources)} sources")

                    for source in sources:
                        try:
                            content = source.get("content", "").strip()
                            metadata = source.get("metadata", {})

                            if content and metadata:
                                try:
                                    memory = MemoryResponse(
                                        metadata=MemoryMetadata(**{
                                            **(default_metadata or {}),
                                            **metadata
                                        }),
                                        content=MemoryContent(
                                            user_message=metadata.get(
                                                "user_message", content),
                                            assistant_response=metadata.get(
                                                "assistant_response", ""),
                                            tool_response=json.dumps(metadata.get("tool_response")) if isinstance(
                                                metadata.get("tool_response"), dict) else metadata.get("tool_response")
                                        )
                                    )
                                    memories.append(memory)
                                except Exception as e:
                                    logger.error(
                                        f"Failed to create memory from source: {str(e)}")
                            else:
                                logger.debug(
                                    "Skipping source with missing content or metadata")
                        except Exception as e:
                            logger.error(f"Error processing source: {str(e)}")
                            continue
                else:
                    logger.warning(
                        "No recognized memory format found in response")

            # Handle string response format (naive mode)
            elif isinstance(response, str):
                try:
                    # Try to parse as JSON first
                    data = json.loads(response)
                    return cls.from_lightrag_response(data, default_metadata)
                except json.JSONDecodeError:
                    # Fall back to chunk processing
                    chunks = response.split("--New Chunk--")
                    logger.debug(
                        f"Processing {len(chunks)} chunks from string response")

                    for chunk in chunks:
                        if not chunk.strip():
                            continue
                        try:
                            chunk_data = json.loads(chunk)
                            return cls.from_lightrag_response(chunk_data, default_metadata)
                        except json.JSONDecodeError:
                            logger.debug("Failed to parse chunk as JSON")
                            continue

            logger.info(f"Created {len(memories)} memories")
            return memories if memories else None

        except Exception as e:
            logger.error(f"Failed to parse LightRAG response: {str(e)}")
            return None
