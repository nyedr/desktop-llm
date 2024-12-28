"""LLM Context Manager for handling model context and memory retrieval."""
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING, Union
from transformers import AutoTokenizer
from datetime import datetime

from app.services.chroma_service import ChromaService
from app.core.config import config

# Forward references for type checking
if TYPE_CHECKING:
    from app.services.langchain_service import LangChainService
    from app.models.chat import ChatMessage

logger = logging.getLogger(__name__)


class LLMContextManager:
    """Context manager for orchestrating LLM interactions.

    Handles:
    1. User input processing (text, images, files)
    2. Memory retrieval from Chroma
    3. Prompt engineering with systematic labeling
    4. Context-size management with token counting
    5. Teardown logic with summarization
    """

    def __init__(
        self,
        chroma_service: ChromaService,
        langchain_service: 'LangChainService',
        conversation_history: List[Dict[str, Any]],
        max_context_tokens: int = config.MAX_TOKENS,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        model_name: str = config.DEFAULT_MODEL
    ):
        """Initialize the LLM context manager.

        Args:
            chroma_service: Service for interacting with ChromaDB
            langchain_service: Service for LangChain operations
            conversation_history: List of conversation messages
            max_context_tokens: Maximum tokens allowed in context
            metadata_filter: Optional filter for memory retrieval
            top_k: Number of memories to retrieve
            model_name: Name of the model for tokenizer selection
        """
        self.chroma_service = chroma_service
        self.langchain_service = langchain_service
        self.conversation_history = conversation_history
        self.metadata_filter = metadata_filter
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.model_name = model_name

        # Storage for processed data
        self.final_context_messages: List[Dict[str, Any]] = []
        self.processed_inputs: List[Dict[str, Any]] = []
        self.retrieved_memories: List[Dict[str, Any]] = []

        # Initialize tokenizer for context management
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.TOKENIZER_MODEL or "gpt2"
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
            self.tokenizer = None

    async def __aenter__(self):
        """Prepare context data by processing inputs, retrieving memories, and engineering prompt."""
        try:
            # 1. Process user inputs
            await self._process_user_inputs()

            # 2. Retrieve relevant memories
            await self._retrieve_relevant_memories()

            # 3. Engineer the prompt with systematic labeling
            await self._engineer_prompt()

            # 4. Manage context size with token counting
            await self._manage_context_size()

            return self

        except Exception as e:
            logger.error(f"Error initializing LLM context: {e}", exc_info=True)
            # Initialize with empty context on error
            self.final_context_messages = []
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Handle teardown logic and cleanup with summarization."""
        if exc_type:
            logger.error(
                f"Error in LLM context manager: {exc_type.__name__}: {exc_val}",
                exc_info=True
            )
        await self._teardown()

    def get_context_messages(self) -> List[Dict[str, Any]]:
        """Get the final processed messages for the model.

        Returns:
            List of messages with proper context and structure
        """
        return self.final_context_messages

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        if not self.tokenizer:
            # Fallback to approximate token count
            return len(text.split())
        return len(self.tokenizer.encode(text))

    def _get_message_value(self, message: Union[Dict[str, Any], 'ChatMessage'], key: str, default: Any = None) -> Any:
        """Safely get a value from either a dict or ChatMessage object.

        Args:
            message: The message object (dict or ChatMessage)
            key: The key to retrieve
            default: Default value if key not found

        Returns:
            The value or default
        """
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    def count_message_tokens(self, message: Union[Dict[str, Any], 'ChatMessage']) -> int:
        """Count tokens in a message dictionary or ChatMessage.

        Args:
            message: Message dictionary or ChatMessage object

        Returns:
            Total token count for the message
        """
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

    async def _process_user_inputs(self):
        """Process and validate user inputs with type-specific handling."""
        try:
            processed = []
            for i, msg in enumerate(self.conversation_history):
                # Check if this is a function result that should be attributed to the assistant
                if (self._get_message_value(msg, "role") == "function" or
                    (self._get_message_value(msg, "role") == "system" and
                     self._get_message_value(msg, "name") == "function")):

                    # Look for the preceding assistant message that made the call
                    if i > 0 and self._get_message_value(self.conversation_history[i-1], "role") == "assistant":
                        # Append function result to the assistant's message
                        function_name = self._get_message_value(
                            msg, "name", "function")
                        function_content = self._get_message_value(
                            msg, "content", "")

                        # Format the function result as part of the assistant's response
                        processed[-1]["content"] += f"\n\nI called the {function_name} function, which returned:\n{function_content}"
                        processed[-1]["metadata"]["has_function_call"] = True
                        continue
                    else:
                        # If no preceding assistant message, create a new one
                        processed.append({
                            "role": "assistant",
                            "content": f"I received this information from the {self._get_message_value(msg, 'name', 'function')} function:\n{self._get_message_value(msg, 'content', '')}",
                            "metadata": {
                                "type": "function_result",
                                "has_function_call": True
                            }
                        })
                        continue

                # Handle different message types
                if self._get_message_value(msg, "images"):
                    # Process images if present
                    processed.append(await self._process_image_message(msg))
                elif self._get_message_value(msg, "file_path"):
                    # Process file references if present
                    processed.append(await self._process_file_message(msg))
                else:
                    # Regular text message
                    processed.append(self._process_text_message(msg))

            self.processed_inputs = processed
            logger.debug(f"Processed {len(processed)} input messages")

        except Exception as e:
            logger.error(f"Error processing user inputs: {e}", exc_info=True)
            self.processed_inputs = self.conversation_history

    async def _process_image_message(self, msg: Union[Dict[str, Any], 'ChatMessage']) -> Dict[str, Any]:
        """Process a message containing images with potential OCR or description."""
        content = str(self._get_message_value(msg, "content", ""))
        if self._get_message_value(msg, "images"):
            # TODO: Implement image processing (OCR, description)
            content += "\n[Image attached]"
        return {
            "role": str(self._get_message_value(msg, "role")),
            "content": content,
            "metadata": {"has_image": True}
        }

    async def _process_file_message(self, msg: Union[Dict[str, Any], 'ChatMessage']) -> Dict[str, Any]:
        """Process a message containing file references with content extraction."""
        content = str(self._get_message_value(msg, "content", ""))
        if self._get_message_value(msg, "file_path"):
            # TODO: Implement file content extraction
            content += f"\n[File: {self._get_message_value(msg, 'file_path')}]"
        return {
            "role": str(self._get_message_value(msg, "role")),
            "content": content,
            "metadata": {"has_file": True}
        }

    def _process_text_message(self, msg: Union[Dict[str, Any], 'ChatMessage']) -> Dict[str, Any]:
        """Process a regular text message with normalization."""
        return {
            "role": str(self._get_message_value(msg, "role")),
            "content": str(self._get_message_value(msg, "content", "")).strip(),
            "name": self._get_message_value(msg, "name"),
            "metadata": {"type": "text"}
        }

    async def _retrieve_relevant_memories(self):
        """Retrieve relevant memories from Chroma with semantic search."""
        logger.debug("Starting _retrieve_relevant_memories method")
        try:
            # Combine processed inputs into a query
            query = " ".join([
                str(self._get_message_value(msg, "content", ""))
                for msg in self.processed_inputs
                # Only use user messages for query
                if self._get_message_value(msg, "role") == "user"
            ])

            if not query.strip():
                logger.debug("No content available to query for memories")
                return

            # Retrieve memories with metadata if filter is provided
            if self.metadata_filter:
                memories = await self.chroma_service.retrieve_with_metadata(
                    query, self.metadata_filter, self.top_k
                )
            else:
                memories = await self.chroma_service.retrieve_memories(
                    query, self.top_k
                )

            # Process retrieved memories with metadata
            self.retrieved_memories = [
                {
                    "content": memory["document"],
                    "metadata": memory["metadata"],
                    "relevance_score": memory.get("relevance", 0.0)
                }
                for memory in (memories or [])
            ]

            # Sort by relevance score
            self.retrieved_memories.sort(
                key=lambda x: x["relevance_score"],
                reverse=True
            )

            logger.debug(
                f"Retrieved {len(self.retrieved_memories)} relevant memories"
            )

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            self.retrieved_memories = []

    async def _engineer_prompt(self):
        """Construct the final prompt with systematic labeling and structure."""
        try:
            # Start with system context from memories
            context_messages = []

            # Add base system prompt
            context_messages.append({
                "role": "system",
                "content": (
                    "You are a highly capable AI assistant designed to help users with a wide range of tasks. "
                    "You have access to tools and long-term memory to provide accurate, helpful, and informed responses. "
                    "Follow these operational guidelines to ensure optimal performance:\n\n"

                    "1. **Concise Responses**: Keep your responses clear and focused, addressing the user's query efficiently.\n"
                    "2. **Tool Usage**: Utilize the tools available to you whenever appropriate to enhance your responses.\n"
                    "3. **Memory Reference**: Refer to relevant stored memories when it improves the quality of your response.\n"
                    "4. **Professional Tone**: Maintain a professional, friendly, and helpful tone in all interactions.\n"
                    "5. **Memory Management**: Use the `add_memory` tool to store information that could enhance the user experience, such as:\n"
                    "   - User-provided personal details (e.g., birthdays, preferences, likes, dislikes).\n"
                    "   - Information that might be useful for future interactions (e.g., frequent topics of interest).\n\n"
                    "   Always store such information upon encountering it unless explicitly instructed otherwise by the user.\n"
                    "6. **Seamless Tool Integration**: Do not explicitly state when you are using a tool; seamlessly integrate its output into your response.\n\n"

                    "Your primary goal is to provide an exceptional and personalized user experience by leveraging your tools and memory effectively."
                )
            })

            # Add memory context if available
            if self.retrieved_memories:
                memory_content = []
                for memory in self.retrieved_memories:
                    # Safely get metadata with defaults
                    metadata = memory.get("metadata", {}) or {}
                    source = metadata.get("source", "unknown")
                    timestamp = metadata.get("timestamp", "unknown")
                    score = memory.get("relevance_score", 0.0)

                    # Format memory with metadata
                    memory_content.append(
                        f"[Memory from {source}]\n"
                        f"Timestamp: {timestamp}\n"
                        f"Relevance: {score:.2f}\n"
                        f"Content: {memory.get('content', '')}"
                    )

                # Combine memories into a single system message
                if memory_content:
                    context_messages.append({
                        "role": "system",
                        "content": (
                            "Here is relevant context from previous interactions:\n\n" +
                            "\n\n".join(memory_content)
                        )
                    })

            # Add processed conversation with proper structure
            context_messages.extend(self.processed_inputs)

            self.final_context_messages = context_messages
            logger.debug(
                f"Engineered prompt with {len(context_messages)} messages"
            )

        except Exception as e:
            logger.error(f"Error engineering prompt: {e}", exc_info=True)
            # Fallback to processed inputs on error
            self.final_context_messages = self.processed_inputs

    async def _manage_context_size(self):
        """Ensure context stays within token limits with smart truncation."""
        try:
            # Calculate current token count
            total_tokens = sum(
                self.count_message_tokens(msg)
                for msg in self.final_context_messages
            )

            if total_tokens > self.max_context_tokens:
                logger.info(
                    f"Context size ({total_tokens} tokens) exceeds limit "
                    f"({self.max_context_tokens} tokens). Truncating..."
                )

                # Separate system and conversation messages
                system_messages = [
                    msg for msg in self.final_context_messages
                    if msg.get("role") == "system"
                ]
                conversation = [
                    msg for msg in self.final_context_messages
                    if msg.get("role") != "system"
                ]

                # Calculate tokens used by system messages
                system_tokens = sum(
                    self.count_message_tokens(msg)
                    for msg in system_messages
                )

                # Calculate remaining tokens for conversation
                remaining_tokens = self.max_context_tokens - system_tokens

                # Keep most recent conversation messages that fit
                truncated_conversation = []
                current_tokens = 0

                for msg in reversed(conversation):
                    msg_tokens = self.count_message_tokens(msg)
                    if current_tokens + msg_tokens <= remaining_tokens:
                        truncated_conversation.insert(0, msg)
                        current_tokens += msg_tokens
                    else:
                        break

                self.final_context_messages = system_messages + truncated_conversation
                logger.debug(
                    f"Truncated context to {len(self.final_context_messages)} messages"
                )

        except Exception as e:
            logger.error(f"Error managing context size: {e}", exc_info=True)

    async def _summarize_conversation(self) -> Optional[str]:
        """Summarize the current conversation for storage.

        Returns:
            Optional summary of the conversation
        """
        try:
            if not self.final_context_messages:
                return None

            # Use LangChain for summarization
            summary = await self.langchain_service.query_memory(
                "Summarize this conversation",
                context=self.final_context_messages
            )
            return summary.get("result")

        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}", exc_info=True)
            return None

    async def _teardown(self):
        """Clean up resources and store conversation summary."""
        logger.debug("Starting _teardown method")
        try:
            # Only store summaries for conversations with user messages
            has_user_messages = any(
                self._get_message_value(msg, "role") == "user"
                for msg in self.final_context_messages
            )

            if not has_user_messages:
                logger.debug(
                    "No user messages found, skipping summary storage")
                return

            # Only summarize if we have actual content to store
            has_assistant_response = any(
                self._get_message_value(msg, "role") == "assistant" and
                self._get_message_value(msg, "content", "").strip() and
                not self._get_message_value(msg, "metadata", {}).get(
                    "type") == "function_result"
                for msg in self.final_context_messages
            )

            if not has_assistant_response:
                logger.debug(
                    "No complete assistant response found, skipping summary storage")
                return

            # Get the last complete exchange
            last_exchange = []
            for msg in reversed(self.final_context_messages):
                role = self._get_message_value(msg, "role")
                if role in ["user", "assistant"]:
                    last_exchange.insert(0, msg)
                if len(last_exchange) >= 2:  # We have a complete exchange
                    break

            if len(last_exchange) < 2:
                logger.debug(
                    "No complete exchange found, skipping summary storage")
                return

            # Summarize only the last complete exchange
            summary = await self._summarize_conversation()

            if summary:
                # Enhanced duplicate detection with semantic similarity
                existing_memories = await self.chroma_service.retrieve_memories(
                    summary,
                    top_k=5,
                    score_threshold=0.95  # Require 95% similarity
                )

                # Check for duplicates using multiple criteria
                if existing_memories:
                    for memory in existing_memories:
                        existing_content = memory.get(
                            "document", "").strip().lower()
                        new_content = summary.strip().lower()

                        # Check for exact match
                        if existing_content == new_content:
                            logger.debug(
                                "Found exact duplicate memory, skipping storage")
                            return

                        # Check for semantic similarity
                        if memory.get("relevance_score", 0) >= 0.95:
                            logger.debug(
                                "Found highly similar memory (95%+), skipping storage")
                            return

                        # Check for significant overlap in key phrases
                        existing_phrases = set(existing_content.split())
                        new_phrases = set(new_content.split())
                        overlap = len(existing_phrases &
                                      new_phrases) / len(new_phrases)
                        if overlap > 0.9:
                            logger.debug(
                                "Found memory with 90%+ phrase overlap, skipping storage")
                            return

                # Store summary in Chroma with proper metadata
                await self.chroma_service.add_memory(
                    summary,
                    metadata={
                        "type": "conversation_summary",
                        "model": self.model_name,
                        "timestamp": datetime.now().isoformat(),
                        # Only count the actual exchange
                        "message_count": len(last_exchange),
                        "has_system_context": any(
                            self._get_message_value(msg, "role") == "system"
                            for msg in self.final_context_messages
                        ),
                        "has_function_calls": any(
                            self._get_message_value(msg, "name") == "function"
                            for msg in self.final_context_messages
                        ),
                        "user_query": next(
                            (self._get_message_value(msg, "content")
                             for msg in last_exchange
                             if self._get_message_value(msg, "role") == "user"),
                            ""
                        )
                    }
                )
                logger.debug(
                    f"Stored summary of exchange with {len(last_exchange)} messages")

        except Exception as e:
            logger.error(f"Error in teardown: {e}", exc_info=True)
