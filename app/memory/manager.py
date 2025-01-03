"""Memory manager for LightRAG integration and memory operations."""

import logging
import asyncio
from typing import Optional, Dict, Union, AsyncGenerator
from pathlib import Path
from collections import deque
import uuid

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

    Internal methods handle LightRAG initialization, memory processing,
    and background tasks.
    """

    def __init__(self, working_dir: Optional[Union[str, Path]] = None):
        """Initialize the memory manager.

        Args:
            working_dir: Working directory for LightRAG storage.
                If None, uses config.memory.data_dir.
        """
        self.working_dir = Path(working_dir or config.memory.data_dir)
        self.memory_queue = deque()
        self.processing = False
        self._tasks = []
        self._initialized = False

    async def initialize(self, datastore: Optional[MemoryDatastore] = None):
        """Initialize the memory system and all components.

        Args:
            datastore: Optional datastore for memory persistence
        """
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

    async def query_memory(
        self,
        query: str,
        query_param: Optional[QueryParam] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Query the memory system.

        Args:
            query: The query string
            query_param: Optional query parameters

        Returns:
            Memory response string or an async generator for streamed responses
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use default query params if none provided
            if not query_param:
                query_param = QueryParam(
                    mode="hybrid",
                    stream=False,
                    response_type="natural",
                    top_k=5,
                    only_need_context=True
                )

            # Validate streaming parameters
            if query_param.stream and query_param.only_need_context:
                logger.warning(
                    "Streaming is not supported with only_need_context=True. Setting stream=False.")
                query_param.stream = False

            # Get the query response
            response = await self.rag.aquery(query=query, param=query_param)

            if query_param.stream:
                # For streaming responses, create an async generator
                async def stream_generator():
                    try:
                        async for chunk in response:
                            if isinstance(chunk, dict):
                                yield chunk.get("content", "")
                            elif isinstance(chunk, str):
                                yield chunk
                    except Exception as e:
                        logger.error(
                            f"Error in stream generator: {str(e)}", exc_info=True)
                        raise
                return stream_generator()
            else:
                # For non-streaming responses, return directly
                return response

        except asyncio.CancelledError:
            logger.warning("Query memory operation cancelled")
            raise
        except Exception as e:
            logger.error(f"Error querying memory: {str(e)}", exc_info=True)
            raise

    async def store_memory(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Store a new memory with metadata.

        Args:
            text: The text content to store
            metadata: Optional metadata about the memory

        Returns:
            str: The unique memory ID
        """
        if not self._initialized:
            await self.initialize()

        memory_id = str(uuid.uuid4())
        try:
            # Store in datastore first
            await self.datastore.store_memory(
                memory_id=memory_id,
                text=text,
                metadata=metadata or {}
            )

            # Queue for processing instead of direct insertion
            await self.queue_memory(text, metadata)

            return memory_id

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise

    async def store_file(self, file_path: Union[str, Path]) -> bool:
        """Store a file as memory.

        Args:
            file_path: Path to the file to store

        Returns:
            bool: True if successful, False otherwise
        """
        return await self.ingestor.ingest_file(file_path)

    # Internal methods
    async def queue_memory(self, text: str, metadata: Optional[Dict] = None):
        """Queue memory for processing."""
        if not text:
            return

        try:
            # Add to processing queue
            self.memory_queue.append({
                'text': text,
                'metadata': metadata or {},
                'retries': 0
            })
            logger.debug(f"Queued memory for processing")
        except Exception as e:
            logger.error(f"Error queueing memory: {e}")
            raise

    async def _process_memory_queue(self):
        """Process queued memory items."""
        while self.processing:
            try:
                if self.memory_queue:
                    memory = self.memory_queue.popleft()
                    try:
                        # Insert into LightRAG asynchronously
                        logger.info(
                            f"Processing memory text: {memory['text'][:100]}...")
                        await self.rag.ainsert(memory['text'])
                        logger.info(
                            "Memory processed and embedded successfully")

                    except Exception as e:
                        logger.error(f"Error processing memory: {e}")
                        # Retry with backoff
                        memory['retries'] += 1
                        if memory['retries'] < 3:
                            self.memory_queue.append(memory)
                            await asyncio.sleep(memory['retries'] * config.memory.queue_error_retry_delay)

                await asyncio.sleep(config.memory.queue_process_delay)
            except Exception as e:
                logger.error(f"Error processing memory: {e}")
                await asyncio.sleep(config.memory.queue_error_retry_delay)

    async def start(self):
        """Start the memory system and background tasks."""
        if not self._initialized:
            await self.initialize()

        if not self.processing:
            self.processing = True
            self._tasks = [
                asyncio.create_task(self._process_memory_queue())
            ]
        logger.info("Started memory processing")

    async def stop(self):
        """Stop the memory system and background tasks."""
        if self._initialized and self.processing:
            logger.info("Stopping memory system")
            self.processing = False
            for task in self._tasks:
                task.cancel()
            self._tasks = []
