"""Context manager for LLM interactions."""

import logging
from typing import Dict, Any, List, Optional, Union
from transformers import AutoTokenizer
from collections import defaultdict
from datetime import datetime

from app.core.config import config
from app.models.chat import StrictChatMessage, ChatRole
from app.models.memory import MemoryResponse
from app.dependencies.providers import Providers
from app.utils.utils import format_timestamp

logger = logging.getLogger(__name__)


class LLMContext:
    """Context manager for LLM interactions.

    Handles:
    1. System prompt and context management
    2. Token counting and context size management
    3. Message processing (text, images, files)
    4. Memory retrieval and integration
    5. Context source labeling and organization
    """

    def __init__(
        self,
        request_id: str,
        messages: List[StrictChatMessage],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        enable_memory: bool = True,
    ):
        """Initialize LLM context.

        Args:
            request_id: Unique identifier for the request
            messages: List of chat messages
            model: LLM model to use
            max_tokens: Maximum tokens for response
            enable_memory: Whether to use memory retrieval (default: True)
        """
        self.request_id = request_id
        self.messages = messages
        self.model = model or config.llm.model
        self.max_tokens = max_tokens or config.llm.max_tokens
        self.enable_memory = enable_memory

        # Reserve tokens for system message and memory context
        # Reserve 20% for system and memory
        self.reserved_tokens = int(self.max_tokens * 0.2)
        self.available_tokens = self.max_tokens - self.reserved_tokens

        # Runtime state
        self.token_count = 0
        self.error = None
        self.processed_messages = []
        self.memory_manager = None
        self.context_sources = defaultdict(list)

        # Initialize tokenizer for context management
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.llm.tokenizer_model or "gpt2"
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
            self.tokenizer = None

    async def __aenter__(self) -> 'LLMContext':
        """Set up LLM context.

        - Initialize memory manager
        - Process input messages
        - Retrieve and integrate relevant memories
        - Set up token tracking
        - Prepare context window
        """
        try:
            # Initialize memory manager
            self.memory_manager = await Providers.get_lightrag_manager()
            if not self.memory_manager:
                logger.warning(
                    f"[{self.request_id}] Memory manager not available - context will not include memories")

            # Process messages and retrieve memories
            await self._process_messages()

            # Initialize token tracking
            await self._setup_token_tracking()

            # Manage context size
            await self._manage_context_size()

            logger.info(f"[{self.request_id}] LLM context initialized")
            return self

        except Exception as e:
            self.error = e
            logger.error(
                f"[{self.request_id}] Error initializing LLM context: {e}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up LLM context."""
        try:
            await self._log_token_usage()
            logger.info(f"[{self.request_id}] LLM context cleaned up")

        except Exception as e:
            logger.error(
                f"[{self.request_id}] Error cleaning up LLM context: {e}")
            if not self.error:  # Don't override original error
                self.error = e

    async def _process_messages(self) -> None:
        """Process input messages and prepare them for the model."""
        try:
            processed = []
            total_tokens = 0

            # Add base system prompt if no system message exists
            if not any(self._get_message_value(msg, "role") == ChatRole.SYSTEM for msg in self.messages):
                system_msg = self._create_system_message()
                system_tokens = self.count_message_tokens(system_msg)
                total_tokens += system_tokens
                processed.append(system_msg)

            # Find the most recent user message
            last_user_msg_index = -1
            for i, msg in enumerate(reversed(self.messages)):
                if self._get_message_value(msg, "role") == ChatRole.USER:
                    last_user_msg_index = len(self.messages) - 1 - i
                    break

            # Process each message
            for i, msg in enumerate(self.messages):
                # Skip if we'd exceed token limit
                msg_tokens = self.count_message_tokens(msg)
                if total_tokens + msg_tokens > self.available_tokens:
                    logger.warning(
                        f"[{self.request_id}] Skipping message due to token limit")
                    continue

                # Process message based on type
                if self._get_message_value(msg, "images"):
                    processed_msg = await self._process_image_message(msg)
                elif self._get_message_value(msg, "file_path"):
                    processed_msg = await self._process_file_message(msg)
                else:
                    processed_msg = await self._process_text_message(msg)

                # Retrieve relevant memories only for the most recent user message if memory is enabled
                if self.enable_memory and i == last_user_msg_index:
                    query = self._get_message_value(
                        processed_msg, "content", "")
                    memories = await self._retrieve_relevant_memories(query)

                    # Add memories to context sources
                    for memory in memories:
                        memory_tokens = self.count_message_tokens(memory)
                        if total_tokens + memory_tokens <= self.available_tokens:
                            self.context_sources["memory"].append(memory)
                            total_tokens += memory_tokens
                        else:
                            logger.debug(
                                f"[{self.request_id}] Skipping memory due to token limit")
                            break

                processed.append(processed_msg)
                total_tokens += msg_tokens

            self.processed_messages = processed
            logger.debug(
                f"[{self.request_id}] Processed {len(processed)} messages with {len(self.context_sources['memory'])} memories"
                f" (memory {'enabled' if self.enable_memory else 'disabled'})")

        except Exception as e:
            logger.error(
                f"[{self.request_id}] Error processing messages: {e}", exc_info=True)
            raise

    def _create_system_message(self) -> Dict[str, Any]:
        """Create the base system message with context awareness."""
        from app.core.prompts import get_system_prompt

        return {
            "role": ChatRole.SYSTEM,
            "content": get_system_prompt(self.enable_memory),
            "metadata": {"type": "system"}
        }

    async def _retrieve_relevant_memories(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query.

        Args:
            query: Query string to search memories

        Returns:
            List of formatted memory messages with metadata
        """
        try:
            if not self.memory_manager or not self.memory_manager._initialized:
                logger.warning(
                    f"[{self.request_id}] Memory manager not available or not initialized")
                return []

            logger.debug(
                f"[{self.request_id}] Retrieving memories for query: {query}")
            memories = []

            # Get memory response with metadata
            try:
                memory_response: Optional[MemoryResponse] = await self.memory_manager.query_memory(query)
            except Exception as e:
                logger.error(f"[{self.request_id}] Error querying memory: {e}")
                return []

            # Format memory if we got a response
            if memory_response:
                try:
                    # Format timestamp
                    formatted_time = memory_response.metadata.timestamp.strftime(
                        "%Y-%m-%d %H:%M:%S")
                    time_from_now = datetime.now() - memory_response.metadata.timestamp
                    time_from_now_str = format_timestamp(time_from_now)

                    # Format metadata for LLM, excluding internal fields
                    internal_fields = {
                        "memory_id", "timestamp", "request_id", "content_type",
                        "content", "source", "chunk_index", "token_count",
                        "user_message", "assistant_response", "tool_response"
                    }
                    metadata_dict = memory_response.metadata.model_dump()
                    metadata_str = "\n".join([
                        f"- {key}: {value}"
                        for key, value in metadata_dict.items()
                        if key not in internal_fields and value is not None
                    ])

                    # Format conversation content
                    conversation_str = (
                        f"User: {memory_response.content.user_message}\n"
                        f"Assistant: {memory_response.content.assistant_response}"
                    )
                    if memory_response.content.tool_response:
                        conversation_str += f"\nTool Response: {memory_response.content.tool_response}"

                    # Create memory message with metadata context
                    memory_message = {
                        "role": ChatRole.SYSTEM,
                        "content": (
                            f"[Memory from {time_from_now_str} ({formatted_time})]\n"
                            f"Context:\n{metadata_str}\n\n"
                            f"Conversation:\n{conversation_str}"
                        ),
                        "metadata": {"type": "memory"}
                    }
                    memories.append(memory_message)

                except Exception as e:
                    logger.error(
                        f"[{self.request_id}] Error formatting memory response: {e}", exc_info=True)
                    # Try to create a simple memory message if formatting fails
                    try:
                        memories.append({
                            "role": ChatRole.SYSTEM,
                            "content": f"[Memory] User: {memory_response.content.user_message}",
                            "metadata": {"type": "memory"}
                        })
                    except:
                        pass

            logger.debug(
                f"[{self.request_id}] Retrieved {len(memories)} relevant memories")
            return memories

        except Exception as e:
            logger.error(
                f"[{self.request_id}] Error retrieving memories: {e}", exc_info=True)
            return []

    async def _setup_token_tracking(self) -> None:
        """Initialize token tracking."""
        if self.tokenizer:
            self.token_count = sum(
                self.count_message_tokens(msg)
                for msg in self.processed_messages
            )
            logger.debug(f"Initial token count: {self.token_count}")

    async def _manage_context_size(self) -> None:
        """Ensure context stays within token limits."""
        if not self.tokenizer:
            return

        while self.token_count > self.max_tokens:
            # Remove oldest non-system message
            for i, msg in enumerate(self.processed_messages):
                if self._get_message_value(msg, "role") != ChatRole.SYSTEM:
                    self.token_count -= self.count_message_tokens(msg)
                    self.processed_messages.pop(i)
                    break

    async def _log_token_usage(self) -> None:
        """Log token usage statistics."""
        if self.token_count > 0:
            logger.info(
                f"[{self.request_id}] Token usage: {self.token_count}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not self.tokenizer:
            # Fallback to approximate token count
            return len(text.split())
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, message: Union[Dict[str, Any], StrictChatMessage]) -> int:
        """Count tokens in a message."""
        total = 0
        # Count role tokens
        total += self.count_tokens(
            str(self._get_message_value(message, "role", "")))
        # Count content tokens
        total += self.count_tokens(
            str(self._get_message_value(message, "content", "")))
        # Count name tokens if present
        name = self._get_message_value(message, "name")
        if name:
            total += self.count_tokens(str(name))
        return total

    def _get_message_value(self, message: Union[Dict[str, Any], StrictChatMessage], key: str, default: Any = None) -> Any:
        """Safely get a value from either a dict or StrictChatMessage object."""
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    async def _process_image_message(self, msg: Union[Dict[str, Any], StrictChatMessage]) -> Dict[str, Any]:
        """Process a message containing images."""
        try:
            content = str(self._get_message_value(msg, "content", ""))
            role = self._get_message_value(msg, "role")
            images = self._get_message_value(msg, "images", [])

            # Add image placeholders to content
            if images:
                image_descriptions = []
                for i, image in enumerate(images, 1):
                    image_descriptions.append(f"[Image {i}]")
                content = f"{content}\n{' '.join(image_descriptions)}"

            return {
                "role": role,
                "content": content,
                "metadata": {
                    "type": "image",
                    "has_image": True,
                    "image_count": len(images),
                    "images": images
                }
            }
        except Exception as e:
            logger.error(
                f"[{self.request_id}] Error processing image message: {e}", exc_info=True)
            return self._create_fallback_message(msg)

    async def _process_file_message(self, msg: Union[Dict[str, Any], StrictChatMessage]) -> Dict[str, Any]:
        """Process a message containing file references."""
        try:
            content = str(self._get_message_value(msg, "content", ""))
            role = self._get_message_value(msg, "role")
            file_path = self._get_message_value(msg, "file_path")

            return {
                "role": role,
                "content": content,
                "metadata": {
                    "type": "file",
                    "has_file": True,
                    "file_path": file_path
                }
            }
        except Exception as e:
            logger.error(
                f"[{self.request_id}] Error processing file message: {e}", exc_info=True)
            return self._create_fallback_message(msg)

    async def _process_text_message(self, msg: Union[Dict[str, Any], StrictChatMessage]) -> Dict[str, Any]:
        """Process a regular text message."""
        try:
            return {
                "role": self._get_message_value(msg, "role"),
                "content": str(self._get_message_value(msg, "content", "")).strip(),
                "name": self._get_message_value(msg, "name"),
                "metadata": {
                    "type": "text"
                }
            }
        except Exception as e:
            logger.error(
                f"[{self.request_id}] Error processing text message: {e}", exc_info=True)
            return self._create_fallback_message(msg)

    def _create_fallback_message(self, msg: Union[Dict[str, Any], StrictChatMessage]) -> Dict[str, Any]:
        """Create a fallback message when processing fails."""
        return {
            "role": self._get_message_value(msg, "role", ChatRole.USER),
            "content": str(self._get_message_value(msg, "content", "Error processing message")),
            "metadata": {"type": "fallback"}
        }

    def get_context_window(self) -> List[Dict[str, Any]]:
        """Get the optimized context window with all sources properly labeled.

        Returns a list of messages with the following structure:
        1. System message (with context awareness)
        2. Retrieved memories (if any)
        3. Processed conversation messages

        The context is optimized to:
        - Stay within token limits
        - Prioritize system message and recent context
        - Properly label and organize different sources
        """
        context_window = []

        # Add system message
        system_msg = next((msg for msg in self.processed_messages
                          if self._get_message_value(msg, "role") == ChatRole.SYSTEM),
                          self._create_system_message())
        context_window.append(system_msg)

        # Add memory context if available
        if self.context_sources["memory"]:
            memory_context = "\n\n".join([
                f"[Memory {i+1}]: {mem['content']}"
                for i, mem in enumerate(self.context_sources["memory"])
            ])
            context_window.append({
                "role": ChatRole.SYSTEM,
                "content": f"Relevant context from memory:\n{memory_context}",
                "metadata": {"type": "memory_context"}
            })

        # Add conversation messages with source labels
        for msg in self.processed_messages:
            if self._get_message_value(msg, "role") != ChatRole.SYSTEM:
                content = self._get_message_value(msg, "content", "")
                metadata = self._get_message_value(msg, "metadata", {})

                # Add source labels if needed
                if metadata.get("has_file"):
                    content = f"[File Content] {content}"
                elif metadata.get("has_image"):
                    content = f"[Image Content] {content}"

                context_window.append({
                    "role": self._get_message_value(msg, "role"),
                    "content": content,
                    "name": self._get_message_value(msg, "name"),
                    "metadata": metadata
                })

        return context_window
