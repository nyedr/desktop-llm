"""LLM Context Manager for handling model context and memory retrieval."""
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING, Union, Tuple
import uuid
from transformers import AutoTokenizer
from datetime import datetime

from app.services.chroma_service import ChromaService
from app.core.config import config
from app.models.memory import MemoryType

# Forward references for type checking
if TYPE_CHECKING:
    from app.services.langchain_service import LangChainService
    from app.models.chat import StrictChatMessage

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
        model_name: str = config.DEFAULT_MODEL,
        memory_type: MemoryType = MemoryType.EPHEMERAL,
        conversation_id: Optional[str] = None,
        enable_summarization: bool = False,
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
            memory_type: Type of memory to use (ephemeral or model)
            conversation_id: Optional conversation ID for this context
            enable_summarization: Whether to enable automatic summarization
        """
        self.chroma_service = chroma_service
        self.langchain_service = langchain_service
        self.conversation_history = conversation_history
        self.metadata_filter = metadata_filter
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.model_name = model_name
        self.memory_type = memory_type

        # Handle conversation ID
        self.conversation_id = conversation_id or str(uuid.uuid4())

        self.enable_summarization = enable_summarization

        # Storage for processed data
        self.final_context_messages: List[Dict[str, Any]] = []
        self.processed_inputs: List[Dict[str, Any]] = []
        self.retrieved_memories: List[Dict[str, Any]] = []
        self.message_chunks: Dict[str, List[Dict[str, Any]]] = {}

        # Initialize tokenizer for context management
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.TOKENIZER_MODEL or "gpt2"
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
            self.tokenizer = None

    async def add_to_memory(
        self,
        text: str,
        memory_type: MemoryType = MemoryType.EPHEMERAL,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        conversation_ids: Optional[List[str]] = None
    ) -> Optional[List[str]]:
        """Add text to memory with appropriate chunking and metadata.

        Args:
            text: Text to store in memory
            memory_type: Type of memory (ephemeral or model)
            metadata: Optional metadata for the memory
            chunk_size: Optional maximum chunk size in tokens
            conversation_ids: Optional list of conversation IDs

        Returns:
            List of memory IDs if successful, None if failed or duplicate
        """
        try:
            if not text:
                return None

            # Generate a single response_id for all chunks
            response_id = str(uuid.uuid4())

            # Prepare base metadata
            base_metadata = {
                **(metadata or {}),
                # Use single conversation_id for consistency
                "conversation_id": self.conversation_id,
                "response_id": response_id,  # Use same response_id for all chunks
                "timestamp": datetime.now().isoformat()
            }

            # Add memory type specific metadata
            if memory_type == MemoryType.MODEL_MEMORY:
                base_metadata.update({
                    "type": "model_memory",
                    "persistent": True,
                    "last_accessed": datetime.now().isoformat()
                })
            else:
                base_metadata.update({
                    "type": "message",
                    "persistent": False
                })

            # Let ChromaService handle chunking with consistent response_id
            return await self.chroma_service.add_memory(
                text=text,
                collection=memory_type,
                metadata=base_metadata,
                max_chunk_tokens=chunk_size if chunk_size else config.MAX_CHUNK_TOKENS,
                conversation_id=self.conversation_id
            )

        except Exception as e:
            logger.error(f"Failed to add memory: {e}", exc_info=True)
            return None

    async def update_model_memory(self, memory_id: str, new_content: str) -> bool:
        """Update an existing model memory with new content.

        Args:
            memory_id: The ID of the memory to update
            new_content: The new content to store

        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify this is a model memory
            existing_memory = await self.chroma_service.get_memory(memory_id, "model_memory")
            if not existing_memory or existing_memory.get("metadata", {}).get("memory_type") != "model":
                logger.warning(
                    f"Attempted to update non-model memory: {memory_id}")
                return False

            # Update the memory with new content and updated timestamp
            return await self.chroma_service.update_memory(
                memory_id,
                new_content,
                "model_memory",
                {
                    **existing_memory.get("metadata", {}),
                    "last_accessed": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to update model memory: {e}", exc_info=True)
            return False

    async def delete_model_memory(self, memory_id: str) -> bool:
        """Delete a model memory.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify this is a model memory
            existing_memory = await self.chroma_service.get_memory(memory_id, "model_memory")
            if not existing_memory or existing_memory.get("metadata", {}).get("memory_type") != "model":
                logger.warning(
                    f"Attempted to delete non-model memory: {memory_id}")
                return False

            return await self.chroma_service.delete_memory(memory_id, "model_memory")
        except Exception as e:
            logger.error(f"Failed to delete model memory: {e}", exc_info=True)
            return False

    async def get_model_memories(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all model memories for a user.

        Args:
            user_id: Optional user ID to filter memories

        Returns:
            List of model memories with metadata
        """
        try:
            metadata_filter = {"memory_type": "model"}
            if user_id:
                metadata_filter["user_id"] = user_id

            memories = await self.chroma_service.retrieve_memories(
                query="*",
                collection="model_memory",
                metadata_filter=metadata_filter
            )

            # Reassemble any chunked memories
            chunk_groups = {}
            for memory in memories:
                metadata = memory.get("metadata", {})
                if metadata.get("is_chunk"):
                    original_id = metadata.get("original_message_id")
                    if original_id not in chunk_groups:
                        chunk_groups[original_id] = []
                    chunk_groups[original_id].append(memory)

            # Process chunked memories
            processed_memories = []
            for original_id, chunks in chunk_groups.items():
                if len(chunks) > 1:
                    # Sort chunks by their index
                    chunks.sort(key=lambda x: x.get(
                        "metadata", {}).get("chunk_index", 0))
                    # Reassemble the message
                    reassembled = {
                        "content": " ".join(chunk["document"] for chunk in chunks),
                        "metadata": chunks[0]["metadata"].copy(),
                        "id": original_id
                    }
                    # Remove chunk-specific metadata
                    reassembled["metadata"].pop("is_chunk", None)
                    reassembled["metadata"].pop("chunk_index", None)
                    reassembled["metadata"].pop("total_chunks", None)
                    processed_memories.append(reassembled)
                else:
                    # Single chunk or non-chunked memory
                    processed_memories.append({
                        "content": chunks[0]["document"],
                        "metadata": chunks[0]["metadata"],
                        "id": chunks[0]["id"]
                    })

            return processed_memories
        except Exception as e:
            logger.error(
                f"Failed to retrieve model memories: {e}", exc_info=True)
            return []

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

    def _get_message_value(self, message: Union[Dict[str, Any], 'StrictChatMessage'], key: str, default: Any = None) -> Any:
        """Safely get a value from either a dict or StrictChatMessage object.

        Args:
            message: The message object (dict or StrictChatMessage)
            key: The key to retrieve
            default: Default value if key not found

        Returns:
            The value or default
        """
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    def count_message_tokens(self, message: Union[Dict[str, Any], 'StrictChatMessage']) -> int:
        """Count tokens in a message dictionary or StrictChatMessage object.

        Args:
            message: Message dictionary or StrictChatMessage object

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
                    processed.extend(await self._process_function_result(msg, i))
                    continue

                # Handle different message types
                if self._get_message_value(msg, "images"):
                    processed.append(await self._process_image_message(msg))
                elif self._get_message_value(msg, "file_path"):
                    processed.append(await self._process_file_message(msg))
                else:
                    processed.append(await self._process_text_message(msg))

            self.processed_inputs = processed
            logger.debug(f"Processed {len(processed)} input messages")

        except Exception as e:
            logger.error(f"Error processing user inputs: {e}", exc_info=True)
            self.processed_inputs = self.conversation_history

    async def _process_function_result(self, msg: Union[Dict[str, Any], 'StrictChatMessage'], index: int) -> List[Dict[str, Any]]:
        """Process a function result message.

        Args:
            msg: The function result message
            index: Current message index in conversation history

        Returns:
            List of processed messages
        """
        try:
            function_name = self._get_message_value(msg, "name", "function")
            function_content = self._get_message_value(msg, "content", "")

            # Find the last assistant message that might have called this function
            last_assistant_msg = None
            last_assistant_idx = -1

            # Look back through history for the last assistant message
            for i in range(index - 1, -1, -1):
                prev_msg = self.conversation_history[i]
                if self._get_message_value(prev_msg, "role") == "assistant":
                    last_assistant_msg = prev_msg
                    last_assistant_idx = i
                    break

            # If we found a recent assistant message
            if last_assistant_msg and last_assistant_idx >= 0:
                # Check if there are any messages between assistant and function result
                intermediate_messages = self.conversation_history[last_assistant_idx + 1:index]

                # If there are no intermediate user messages, append to assistant
                if not any(self._get_message_value(m, "role") == "user" for m in intermediate_messages):
                    # Get the original assistant content
                    assistant_content = self._get_message_value(
                        last_assistant_msg, "content", "")

                    # Create updated assistant message
                    return [{
                        "role": "assistant",
                        "content": f"{assistant_content}\n\nFunction Result ({function_name}):\n{function_content}",
                        "metadata": {
                            "has_function_call": True,
                            "function_name": function_name,
                            "original_message_index": last_assistant_idx,
                            "conversation_id": self.conversation_id
                        }
                    }]

            # If we couldn't append to an assistant message, create a new one
            return [{
                "role": "assistant",
                "content": f"Function Result ({function_name}):\n{function_content}",
                "metadata": {
                    "type": "function_result",
                    "has_function_call": True,
                    "function_name": function_name,
                    "conversation_id": self.conversation_id
                }
            }]

        except Exception as e:
            logger.error(
                f"Error processing function result: {e}", exc_info=True)
            # Return a safe fallback
            return [{
                "role": "system",
                "content": f"Error processing function result: {str(e)}",
                "metadata": {"type": "error", "error": str(e)}
            }]

    async def _process_image_message(self, msg: Union[Dict[str, Any], 'StrictChatMessage']) -> Dict[str, Any]:
        """Process a message containing images with potential OCR or description.

        Args:
            msg: Message containing image data

        Returns:
            Processed message dictionary
        """
        content = str(self._get_message_value(msg, "content", ""))
        role = str(self._get_message_value(msg, "role"))
        name = self._get_message_value(msg, "name")
        images = self._get_message_value(msg, "images", [])

        # Process image content
        if images:
            # TODO: Implement image processing (OCR, description)
            image_descriptions = ["\n[Image attached]" for _ in images]
            content += " ".join(image_descriptions)

        # Always store in ephemeral collection
        await self.chroma_service.add_memory(
            text=content,
            collection=MemoryType.EPHEMERAL,
            metadata={
                "role": role,
                "name": name,
                "type": "image_message",
                "has_image": True,
                "image_count": len(images),
                "conversation_id": self.conversation_id,
                "timestamp": datetime.now().isoformat()
            }
        )

        return {
            "role": role,
            "content": content,
            "name": name,
            "metadata": {
                "type": "image_message",
                "has_image": True,
                "image_count": len(images),
                "conversation_id": self.conversation_id
            }
        }

    async def _process_file_message(self, msg: Union[Dict[str, Any], 'StrictChatMessage']) -> Dict[str, Any]:
        """Process a message containing file references with content extraction.

        Args:
            msg: Message containing file reference

        Returns:
            Processed message dictionary
        """
        content = str(self._get_message_value(msg, "content", ""))
        role = str(self._get_message_value(msg, "role"))
        name = self._get_message_value(msg, "name")
        file_path = self._get_message_value(msg, "file_path")

        if file_path:
            # TODO: Implement file content extraction
            content += f"\n[File: {file_path}]"

        # Store in ephemeral collection
        await self.chroma_service.add_memory(
            text=content,
            collection=MemoryType.EPHEMERAL,
            metadata={
                "role": role,
                "name": name,
                "type": "file_message",
                "has_file": True,
                "file_path": file_path,
                "conversation_id": self.conversation_id,
                "timestamp": datetime.now().isoformat()
            }
        )

        return {
            "role": role,
            "content": content,
            "name": name,
            "metadata": {
                "type": "file_message",
                "has_file": True,
                "file_path": file_path,
                "conversation_id": self.conversation_id
            }
        }

    async def _process_text_message(self, msg: Union[Dict[str, Any], 'StrictChatMessage']) -> Dict[str, Any]:
        """Process a regular text message with normalization.

        Args:
            msg: Message to process (dict or StrictChatMessage)

        Returns:
            Processed message dictionary

        Note:
            All messages are stored in ephemeral collection as they represent
            the actual conversation flow. Model memory is only modified through
            explicit user or model actions.
        """
        content = str(self._get_message_value(msg, "content", "")).strip()
        role = str(self._get_message_value(msg, "role"))
        name = self._get_message_value(msg, "name")

        # Store in ephemeral collection for conversation history
        await self.chroma_service.add_memory(
            text=content,
            collection=MemoryType.EPHEMERAL,  # Use enum value
            metadata={
                "role": role,
                "name": name,
                "type": "text",
                "conversation_id": self.conversation_id,
                "timestamp": datetime.now().isoformat()
            },
            max_chunk_tokens=config.MAX_CHUNK_TOKENS  # Explicitly pass chunk size
        )

        return {
            "role": role,
            "content": content,
            "name": name,
            "metadata": {
                "type": "text",
                "conversation_id": self.conversation_id
            }
        }

    async def retrieve_memories(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from both ephemeral and model memory collections.

        Args:
            query: The query to search for relevant memories
            top_k: Number of memories to retrieve
            metadata_filter: Optional metadata filter for the search

        Returns:
            List of retrieved memories with metadata
        """
        try:
            # Retrieve from both collections
            ephemeral_memories = await self.chroma_service.retrieve_memories(
                query=query,
                collection=MemoryType.EPHEMERAL,
                top_k=top_k,
                metadata_filter=metadata_filter
            )
            model_memories = await self.chroma_service.retrieve_memories(
                query=query,
                collection=MemoryType.MODEL_MEMORY,
                top_k=top_k,
                metadata_filter=metadata_filter
            )

            # Combine and process results
            all_memories = (ephemeral_memories or []) + (model_memories or [])

            # Group chunks by their original_message_id
            chunk_groups = {}
            for memory in all_memories:
                metadata = memory.get("metadata", {})
                if metadata.get("is_chunk"):
                    original_id = metadata.get("original_message_id")
                    if original_id not in chunk_groups:
                        chunk_groups[original_id] = []
                    chunk_groups[original_id].append(memory)

            # Process each group using _reassemble_chunks
            processed_memories = []
            for chunks in chunk_groups.values():
                if len(chunks) > 1:
                    reassembled = await self._reassemble_chunks(chunks)
                    if reassembled:
                        processed_memories.append(reassembled)
                else:
                    # Single chunk or non-chunked memory
                    processed_memories.append({
                        "content": chunks[0]["document"],
                        "metadata": chunks[0]["metadata"],
                        "relevance_score": chunks[0].get("relevance_score", 0),
                        "collection": chunks[0].get("collection", MemoryType.EPHEMERAL)
                    })

            # Add non-chunked memories
            for memory in all_memories:
                if not memory.get("metadata", {}).get("is_chunk"):
                    processed_memories.append(memory)

            # Sort by relevance score and prioritize model memory
            processed_memories.sort(
                key=lambda x: (
                    x["relevance_score"],
                    1 if x.get("collection") == MemoryType.MODEL_MEMORY else 0
                ),
                reverse=True
            )

            return processed_memories[:top_k]

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return []

    async def _reassemble_chunks(self, chunks: List[Dict[str, Any]], sort_key: str = "relevance_score") -> Optional[Dict[str, Any]]:
        """Reassemble chunks into a complete message.

        Args:
            chunks: List of chunks to reassemble
            sort_key: Key to use for sorting when selecting best chunk (for relevance)

        Returns:
            Reassembled message with metadata, or None if no chunks
        """
        if not chunks:
            return None

        # Sort chunks by their index
        chunks.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))

        # Get the highest score if using relevance scoring
        max_score = max((chunk.get(sort_key, 0)
                        for chunk in chunks), default=0)

        # Combine chunks
        reassembled = {
            "content": " ".join(chunk.get("document", chunk.get("content", "")) for chunk in chunks),
            "metadata": chunks[0].get("metadata", {}).copy(),
            "relevance_score": max_score,
            "collection": chunks[0].get("collection", MemoryType.EPHEMERAL)
        }

        # Remove chunk-specific metadata
        reassembled["metadata"].pop("is_chunk", None)
        reassembled["metadata"].pop("chunk_index", None)
        reassembled["metadata"].pop("total_chunks", None)
        reassembled["metadata"].pop("original_message_id", None)

        return reassembled

    async def _retrieve_relevant_memories(self):
        """Retrieve relevant memories from Chroma with semantic search."""
        logger.debug("Starting _retrieve_relevant_memories method")
        try:
            # Combine processed inputs into a query
            query = " ".join([
                str(self._get_message_value(msg, "content", ""))
                for msg in self.processed_inputs
                if self._get_message_value(msg, "role") == "user"
            ])

            if not query.strip():
                logger.debug("No content available to query for memories")
                return

            # Build metadata filter to include all messages and memories
            metadata_filter = {
                "$or": [
                    {"type": "message"},
                    {"type": "memory"},
                    {"type": "text"}
                ]
            }

            # Retrieve from both collections
            ephemeral_memories = await self.chroma_service.retrieve_memories(
                query=query,
                collection=MemoryType.EPHEMERAL,
                top_k=self.top_k,
                metadata_filter=metadata_filter
            )
            model_memories = await self.chroma_service.retrieve_memories(
                query=query,
                collection=MemoryType.MODEL_MEMORY,
                top_k=self.top_k,
                metadata_filter=metadata_filter
            )

            # Combine and process retrieved memories
            all_memories = (ephemeral_memories or []) + (model_memories or [])

            # Group chunks by their response_id
            chunk_groups = {}
            for memory in all_memories:
                metadata = memory.get("metadata", {})
                response_id = metadata.get("response_id")
                if response_id:
                    if response_id not in chunk_groups:
                        chunk_groups[response_id] = []
                    chunk_groups[response_id].append(memory)

            # Reassemble chunks into complete messages
            self.retrieved_memories = []
            for response_id, chunks in chunk_groups.items():
                if len(chunks) > 1:
                    reassembled = await self._reassemble_chunks(chunks)
                    if reassembled:
                        self.retrieved_memories.append(reassembled)
                else:
                    # Single chunk or non-chunked memory
                    self.retrieved_memories.append({
                        "content": chunks[0]["document"],
                        "metadata": chunks[0]["metadata"],
                        "relevance_score": chunks[0].get("relevance_score", 0),
                        "collection": chunks[0].get("collection", MemoryType.EPHEMERAL)
                    })

            # Sort by relevance score and prioritize model memory
            self.retrieved_memories.sort(
                key=lambda x: (
                    x["relevance_score"],
                    1 if x.get("collection") == MemoryType.MODEL_MEMORY else 0
                ),
                reverse=True
            )

            # Limit to top_k results
            self.retrieved_memories = self.retrieved_memories[:self.top_k]

            logger.debug(
                f"Retrieved {len(self.retrieved_memories)} relevant memories "
                f"({len(ephemeral_memories or [])} ephemeral, "
                f"{len(model_memories or [])} model)"
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

                    # Determine the source based on metadata
                    source = "unknown"
                    if metadata.get("type") == "message":
                        source = metadata.get("role", "unknown")
                    elif metadata.get("type") == "memory":
                        source = "memory"
                    elif metadata.get("type") == "text":
                        source = metadata.get("source", "text")

                    # Get timestamp and format it nicely
                    timestamp = metadata.get("timestamp", "unknown")
                    if isinstance(timestamp, str) and timestamp != "unknown":
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            pass

                    score = memory.get("relevance_score", 0.0)
                    content = memory.get("content", memory.get("document", ""))

                    # Format memory with metadata
                    memory_content.append(
                        f"[Memory from {source}]\n"
                        f"Timestamp: {timestamp}\n"
                        f"Relevance: {score:.2f}\n"
                        f"Content: {content}"
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

    async def _summarize_overflowing_messages(
        self,
        messages: List[Dict[str, Any]],
        excess_tokens: int
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Summarize messages that exceed the token limit.

        Args:
            messages: List of messages to potentially summarize
            excess_tokens: Number of tokens to reduce by

        Returns:
            Tuple of (summary message if created, remaining messages)
        """
        if not messages or not self.enable_summarization:
            return None, messages

        # Take the oldest messages that exceed our limit
        messages_to_summarize = []
        current_tokens = 0

        for msg in messages:
            msg_tokens = self.count_message_tokens(msg)
            if current_tokens + msg_tokens <= excess_tokens:
                messages_to_summarize.append(msg)
                current_tokens += msg_tokens
            else:
                break

        if not messages_to_summarize:
            return None, messages

        # Generate summary
        summary = await self._summarize_conversation(messages_to_summarize)
        if not summary:
            return None, messages

        # Store summary in ephemeral collection
        await self.chroma_service.add_memory(
            text=summary,
            collection=MemoryType.EPHEMERAL,  # Force ephemeral storage for summaries
            metadata={
                "type": "conversation_summary",
                "conversation_id": self.conversation_id,
                "summarized_messages": len(messages_to_summarize),
                "timestamp": datetime.now().isoformat()
            }
        )

        # Create summary message
        summary_msg = {
            "role": "system",
            "content": f"Previous conversation summary: {summary}",
            "metadata": {
                "type": "summary",
                "conversation_id": self.conversation_id
            }
        }

        # Return summary and remaining messages
        return summary_msg, messages[len(messages_to_summarize):]

    async def _manage_context_size(self):
        """Ensure context stays within token limits with smart truncation and summarization."""
        try:
            while True:
                # Calculate current token count
                total_tokens = sum(
                    self.count_message_tokens(msg)
                    for msg in self.final_context_messages
                )

                if total_tokens <= self.max_context_tokens:
                    break

                logger.info(
                    f"Context size ({total_tokens} tokens) exceeds limit "
                    f"({self.max_context_tokens} tokens). Managing size..."
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

                # Try to summarize conversation messages
                excess_tokens = total_tokens - self.max_context_tokens
                summary_msg, remaining_messages = await self._summarize_overflowing_messages(
                    conversation, excess_tokens
                )

                if summary_msg:
                    # Update conversation with summary and remaining messages
                    conversation = [summary_msg] + remaining_messages
                    self.final_context_messages = system_messages + conversation
                    continue

                # If we can't summarize or summarization didn't help enough,
                # calculate remaining tokens and truncate
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
                    f"Managed context to {len(self.final_context_messages)} messages"
                )

        except Exception as e:
            logger.error(f"Error managing context size: {e}", exc_info=True)

    async def _teardown(self):
        """Clean up resources."""
        logger.debug("Starting _teardown method")
        # No specific cleanup needed, as summarization is now handled in _manage_context_size
        pass

    async def _summarize_conversation(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Summarize a set of conversation messages.

        Args:
            messages: List of messages to summarize

        Returns:
            Optional summary of the conversation
        """
        try:
            if not messages:
                return None

            # Format messages for summarization
            conversation_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in messages
            ])

            # Use LangChain for summarization with specific instructions
            summary = await self.langchain_service.query_memory(
                "Create a concise summary of this conversation exchange. "
                "Focus on key points, decisions made, and important information. "
                "Maintain context for future interactions.",
                context={"conversation": conversation_text}
            )
            return summary.get("result")

        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}", exc_info=True)
            return None
