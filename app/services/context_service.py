"""Context manager for LLM interactions."""

import logging
from typing import Dict, Any, List, Optional, Union
import uuid
from transformers import AutoTokenizer
from datetime import datetime

from app.core.config import config
from app.models.chat import StrictChatMessage, ChatRole
from app.memory.lightrag.manager import EnhancedLightRAGManager

logger = logging.getLogger(__name__)


class LLMContext:
    """Context manager for LLM interactions.

    Handles:
    1. System prompt and context management
    2. Memory retrieval and integration
    3. Token counting and context size management
    4. Message processing (text, images, files)
    """

    def __init__(
        self,
        request_id: str,
        messages: List[StrictChatMessage],
        lightrag_manager: Optional[EnhancedLightRAGManager] = None,
        memory_filter: Optional[Dict[str, Any]] = None,
        top_k_memories: int = 5,
        enable_memory: bool = True,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """Initialize LLM context.

        Args:
            request_id: Unique identifier for the request
            messages: List of chat messages
            lightrag_manager: Manager for LightRAG memory operations
            memory_filter: Filter for memory retrieval
            top_k_memories: Number of memories to retrieve
            enable_memory: Whether to enable memory operations
            conversation_id: ID for conversation context
            model: LLM model to use
            max_tokens: Maximum tokens for response
        """
        self.request_id = request_id
        self.messages = messages
        self.lightrag_manager = lightrag_manager
        self.memory_filter = memory_filter or {}
        self.top_k_memories = top_k_memories
        self.enable_memory = enable_memory
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.model = model or config.DEFAULT_MODEL
        self.max_tokens = max_tokens or config.MAX_TOKENS

        # Runtime state
        self.token_count = 0
        self.memory_context = []
        self.error = None
        self.processed_messages = []

        # Initialize tokenizer for context management
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.TOKENIZER_MODEL or "gpt2"
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
            self.tokenizer = None

    async def __aenter__(self) -> 'LLMContext':
        """Set up LLM context.

        - Process input messages
        - Initialize memory context if enabled
        - Set up token tracking
        - Prepare context window
        """
        try:
            # Process messages
            await self._process_messages()

            # Setup memory if enabled
            if self.enable_memory and self.lightrag_manager:
                await self._setup_memory_context()

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
        """Clean up LLM context.

        - Store conversation memory if enabled
        - Log token usage
        - Handle any errors
        """
        try:
            if exc_type is None and self.enable_memory and self.lightrag_manager:
                await self._store_conversation_memory()

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

            # Add base system prompt if no system message exists
            has_system_message = any(
                self._get_message_value(msg, "role") == ChatRole.SYSTEM
                for msg in self.messages
            )

            if not has_system_message:
                processed.append({
                    "role": ChatRole.SYSTEM,
                    "content": (
                        "You are a highly capable AI assistant designed to help users with a wide range of tasks. "
                        "You have access to tools and memory to provide accurate, helpful, and informed responses. "
                        "Follow these guidelines:\n\n"
                        "1. Be concise and clear in your responses\n"
                        "2. Use available tools when appropriate\n"
                        "3. Reference relevant memories when helpful\n"
                        "4. Maintain a professional and friendly tone\n"
                        "5. Process and respond to all types of inputs (text, images, files)\n"
                    )
                })

            # Process each message
            for msg in self.messages:
                if self._get_message_value(msg, "images"):
                    # TODO: Implement image message processing
                    processed.append(await self._process_image_message(msg))
                elif self._get_message_value(msg, "file_path"):
                    # TODO: Implement file message processing
                    processed.append(await self._process_file_message(msg))
                else:
                    processed.append(await self._process_text_message(msg))

            self.processed_messages = processed
            logger.debug(f"Processed {len(processed)} messages")

        except Exception as e:
            logger.error(f"Error processing messages: {e}", exc_info=True)
            self.processed_messages = self.messages

    async def _setup_memory_context(self) -> None:
        """Set up memory context using LightRAG."""
        try:
            # Get the last user message as query
            last_user_message = None
            for msg in reversed(self.processed_messages):
                if self._get_message_value(msg, "role") == ChatRole.USER:
                    last_user_message = self._get_message_value(msg, "content")
                    break

            if last_user_message:
                memories = await self.lightrag_manager.retrieve_memories(
                    query=last_user_message,
                    metadata_filter=self.memory_filter,
                    top_k=self.top_k_memories
                )

                if memories:
                    self.memory_context = memories
                    logger.info(
                        f"[{self.request_id}] Retrieved {len(memories)} memories")

        except Exception as e:
            logger.warning(
                f"[{self.request_id}] Error setting up memory context: {e}")
            # Non-critical error, continue without memory context

    async def _setup_token_tracking(self) -> None:
        """Initialize token tracking."""
        if self.tokenizer:
            self.token_count = sum(
                self.count_message_tokens(msg)
                for msg in self.processed_messages
            )
            logger.debug(f"Initial token count: {self.token_count}")

    async def _store_conversation_memory(self) -> None:
        """Store conversation memory in LightRAG."""
        try:
            # Filter out system messages and concatenate conversation
            conversation = "\n".join([
                f"{self._get_message_value(msg, 'role')}: {self._get_message_value(msg, 'content')}"
                for msg in self.processed_messages
                if self._get_message_value(msg, "role") != ChatRole.SYSTEM
            ])

            if conversation:
                await self.lightrag_manager.ingestor.ingest_text(
                    text=conversation,
                    metadata={
                        "conversation_id": self.conversation_id,
                        "request_id": self.request_id,
                        "model": self.model,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                logger.info(f"[{self.request_id}] Stored conversation memory")

        except Exception as e:
            logger.warning(
                f"[{self.request_id}] Error storing conversation memory: {e}")
            # Non-critical error, continue without storing memory

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

    def get_context_window(self) -> List[StrictChatMessage]:
        """Get the current context window with memory context."""
        if not self.memory_context:
            return self.processed_messages

        # Create system message with memory context
        memory_msg = {
            "role": ChatRole.SYSTEM,
            "content": "Relevant context from memory:\n" + "\n".join(
                [f"- {memory.get('content', '')}" for memory in self.memory_context]
            )
        }

        # Add memory context at the start, preserving other system messages
        context_window = [
            msg for msg in self.processed_messages
            if self._get_message_value(msg, "role") == ChatRole.SYSTEM
        ]
        context_window.append(memory_msg)
        context_window.extend([
            msg for msg in self.processed_messages
            if self._get_message_value(msg, "role") != ChatRole.SYSTEM
        ])

        return context_window

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
        """Process a message containing images.
        TODO: Implement image processing (OCR, description)
        """
        content = str(self._get_message_value(msg, "content", ""))
        role = self._get_message_value(msg, "role")
        images = self._get_message_value(msg, "images", [])

        if images:
            content += " ".join(["\n[Image attached]" for _ in images])

        return {
            "role": role,
            "content": content,
            "metadata": {
                "has_image": True,
                "image_count": len(images)
            }
        }

    async def _process_file_message(self, msg: Union[Dict[str, Any], StrictChatMessage]) -> Dict[str, Any]:
        """Process a message containing file references.
        TODO: Implement file content extraction
        """
        content = str(self._get_message_value(msg, "content", ""))
        role = self._get_message_value(msg, "role")
        file_path = self._get_message_value(msg, "file_path")

        if file_path:
            content += f"\n[File: {file_path}]"

        return {
            "role": role,
            "content": content,
            "metadata": {
                "has_file": True,
                "file_path": file_path
            }
        }

    async def _process_text_message(self, msg: Union[Dict[str, Any], StrictChatMessage]) -> Dict[str, Any]:
        """Process a regular text message."""
        return {
            "role": self._get_message_value(msg, "role"),
            "content": str(self._get_message_value(msg, "content", "")).strip(),
            "name": self._get_message_value(msg, "name"),
            "metadata": {
                "type": "text",
                "conversation_id": self.conversation_id
            }
        }
