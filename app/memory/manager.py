"""Memory manager for LightRAG integration and memory operations."""

import logging
import asyncio
from typing import Optional, Dict, Union, Any
from pathlib import Path
import uuid
from datetime import datetime
import json

from app.core.config import config
from .datastore import MemoryDatastore
from .ingestion import MemoryIngestor
from app.services.model_service import ModelService
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.base import QueryParam

logger = logging.getLogger(__name__)


class LightRAGManager:
    """Memory manager for LightRAG integration and memory operations.

    Public Interface:
    - query_memory: Query the memory system
    - store_memory: Store new memories
    - store_file: Store file contents as memory
    """

    def __init__(self, working_dir: Optional[Union[str, Path]] = None):
        """Initialize the memory manager.

        Args:
            working_dir: Working directory for LightRAG storage.
                If None, uses config.memory.data_dir.
        """
        self.working_dir = Path(working_dir or config.memory.data_dir)
        self._initialized = False

    async def initialize(self, datastore: Optional[MemoryDatastore] = None):
        """Initialize the memory system and all components."""
        if self._initialized:
            return

        try:
            # Set up datastore and model service
            self.datastore = datastore or MemoryDatastore(
                str(self.working_dir / "memory.db"))
            self.model_service = ModelService()

            # Initialize working directory
            self.working_dir.mkdir(parents=True, exist_ok=True)

            # Define async LLM function that uses model service
            async def llm_model_func(prompt: str, system_prompt: str = None, history_messages: list = None, **kwargs):
                try:
                    messages = []
                    if system_prompt:
                        messages.append(
                            {"role": "system", "content": system_prompt})
                    if history_messages:
                        messages.extend(history_messages)
                    messages.append({"role": "user", "content": prompt})

                    llm_params = kwargs.get("llm_params", {})
                    response_text = ""

                    # Stream response chunks and accumulate
                    async for chunk in self.model_service.chat(
                        messages=messages,
                        stream=True,
                        # model="meta-llama/llama-3.2-3b-instruct",
                        model="deepseek/deepseek-chat",
                        temperature=llm_params.get(
                            "temperature", config.llm.temperature),
                        max_tokens=llm_params.get(
                            "max_tokens", config.llm.max_tokens),
                        enable_tools=False
                    ):
                        if isinstance(chunk, dict):
                            response_text += chunk.get("content", "")
                        elif isinstance(chunk, str):
                            response_text += chunk

                    return response_text

                except Exception as e:
                    logger.error(
                        f"Error in LLM function: {str(e)}", exc_info=True)
                    raise

            # Create a sync wrapper for the async embedding function
            async def embedding_func(texts):
                try:
                    embeddings = await self.model_service.get_embeddings(texts)
                    return embeddings
                except Exception as e:
                    logger.error(
                        f"Error in embedding function: {str(e)}", exc_info=True)
                    raise

            self.rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=768,
                    max_token_size=config.memory.max_chunk_tokens,
                    func=embedding_func
                ),
                kv_storage="JsonKVStorage",
                vector_storage="NanoVectorDBStorage",
                graph_storage="NetworkXStorage",
                addon_params={
                    "example_number": 3,
                    "language": "English",
                    "mode": "hybrid",
                }
            )

            # Initialize memory-specific components
            self.ingestor = MemoryIngestor(self, self.datastore)
            self._initialized = True

        except Exception as e:
            logger.error(
                f"Failed to initialize LightRAGManager: {str(e)}", exc_info=True)
            raise

    async def query_memory(self, query: str, only_need_context: bool = True) -> Optional[Dict[str, Any]]:
        """Query memory for relevant information.

        Args:
            query: Query string to search memories
            only_need_context: If True, only return the memory content without LLM processing

        Returns:
            Dictionary containing memory content and metadata if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.debug(f"Querying memory with: {query}")

            # Create query parameters
            query_param = QueryParam(
                mode="hybrid",
                stream=False,
                response_type="natural",
                top_k=5,
                only_need_context=only_need_context
            )

            # Get memory response from RAG
            memory_response = await self.rag.aquery(query=query, param=query_param)

            if not memory_response:
                logger.debug("No memory found")
                return None

            # Parse the response
            try:
                if isinstance(memory_response, str):
                    try:
                        # Try to parse the JSON response
                        metadata = json.loads(memory_response)
                        # Extract content from metadata
                        content = metadata.pop("content", "")
                        return {
                            "content": content,
                            "metadata": metadata
                        }
                    except json.JSONDecodeError:
                        # If not JSON, treat as raw content
                        return {
                            "content": memory_response,
                            "metadata": {
                                "timestamp": datetime.now().isoformat(),
                                "content_type": "text",
                                "source": "raw_response"
                            }
                        }
                else:
                    logger.warning(
                        f"Unexpected response type: {type(memory_response)}")
                    return {
                        "content": str(memory_response),
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "content_type": "text",
                            "source": "unknown_format"
                        }
                    }

            except Exception as parse_error:
                logger.error(
                    f"Error parsing memory response: {parse_error}", exc_info=True)
                return {
                    "content": str(memory_response),
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "content_type": "text",
                        "source": "parse_error"
                    }
                }

        except Exception as e:
            logger.error(f"Error querying memory: {e}", exc_info=True)
            return None

    async def store_memory(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Store a new memory with metadata.

        This method is designed to be run in the background. It handles:
        1. Storing in SQL database
        2. Storing in LightRAG with proper formatting
        3. Processing and embedding content

        Args:
            text: The text content to store (user's message)
            metadata: Optional metadata about the memory including:
                - request_id: Unique request identifier
                - model: The model used
                - message_count: Number of messages in conversation
                - has_tool_calls: Whether tool calls were made
                - enable_tools: Whether tools were enabled
                - timestamp: ISO format timestamp
                - temperature: Model temperature
                - max_tokens: Model max tokens
                - user_message: Last user message
                - assistant_response: Assistant's response
                - tool_response: Tool response if any

        Returns:
            str: The unique memory ID
        """
        if not self._initialized:
            await self.initialize()

        memory_id = str(uuid.uuid4())
        try:
            # Ensure required metadata fields
            base_metadata = {
                "memory_id": memory_id,
                "timestamp": datetime.now().isoformat(),
                "content_type": "chat_memory"
            }

            # Merge with provided metadata, ensuring no None values
            full_metadata = {
                **base_metadata,
                **{k: v for k, v in (metadata or {}).items() if v is not None}
            }

            # Store in datastore for SQL-based querying (sync operation)
            self.datastore.store_entity(
                entity_id=memory_id,
                text=text,  # This is the user's message
                metadata=full_metadata
            )

            # Format the conversation content
            conversation_content = {
                "user_message": full_metadata.get("user_message", text),
                "assistant_response": full_metadata.get("assistant_response", ""),
                "tool_response": full_metadata.get("tool_response")
            }

            # Store in LightRAG with conversation content and metadata
            memory_data = {
                **full_metadata,
                "content": conversation_content  # Store structured conversation content
            }

            # Convert to JSON for storage and store in LightRAG (async operation)
            memory_json = json.dumps(memory_data, indent=2)
            await self.rag.ainsert([memory_json])

            logger.info(f"Memory stored with ID: {memory_id} and metadata")
            return memory_id

        except Exception as e:
            logger.error(f"Error storing memory: {e}", exc_info=True)
            raise

    async def store_file(self, file_path: Union[str, Path]) -> bool:
        """Store a file as memory.

        Args:
            file_path: Path to the file to store

        Returns:
            bool: True if successful, False otherwise
        """
        return await self.ingestor.ingest_file(file_path)

    async def start(self):
        """Start the memory system.
        Ensures the system is initialized and ready for use."""
        if not self._initialized:
            await self.initialize()
        logger.info("Memory system started")

    async def stop(self):
        """Stop the memory system.
        Performs any necessary cleanup."""
        if self._initialized:
            logger.info("Memory system stopped")
            self._initialized = False
