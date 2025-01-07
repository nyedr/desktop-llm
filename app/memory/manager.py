"""Memory manager for LightRAG integration and memory operations."""

import logging
from typing import Optional, Dict, Union, List
from pathlib import Path
import uuid
from datetime import datetime
import asyncio
import time
from contextlib import contextmanager

from app.core.config import config
from .datastore import MemoryDatastore
from .ingestion import MemoryIngestor
from .embeddings import EmbeddingService, MINILM_DIM, BATCH_SIZE
from app.models.memory import MemoryResponse
from app.services.model_service import ModelService
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.base import QueryParam

logger = logging.getLogger(__name__)

# Constants for optimization
EMBEDDING_CACHE_SIZE = 1000
DEFAULT_MAX_TOKENS = 4096
TOP_K_MEMORY_RESULTS = 5

# Model configurations
OLLAMA_EMBED_MODEL = "nomic-embed-text"
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXTRACTION_MODEL_NAME = "meta-llama/llama-3.2-3b-instruct"
# EXTRACTION_MODEL_NAME = "deepseek/deepseek-chat"


class ProfilingStats:
    """Container for profiling statistics."""

    def __init__(self):
        self.embedding_times = []
        self.query_times = []
        self.store_times = []
        self.total_embedding_calls = 0
        self.total_query_calls = 0
        self.total_store_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0


@contextmanager
def profile_operation(stats_list: List[float], operation_name: str):
    """Context manager for profiling operations."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        stats_list.append(duration)
        logger.debug(f"{operation_name} took {duration:.3f} seconds")


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
        self.profiling_stats = ProfilingStats()

        # These will be initialized during initialize()
        self._llm_semaphore = None
        self.datastore = None
        self.model_service = None
        self.embedding_service = None
        self.rag = None
        self.ingestor = None

    async def initialize(self, datastore: Optional[MemoryDatastore] = None):
        """Initialize the memory system and all components."""
        if self._initialized:
            return

        try:
            # Initialize semaphores
            self._llm_semaphore = asyncio.Semaphore(5)

            # Set up services
            if datastore:
                self.datastore = datastore
            else:
                self.datastore = await MemoryDatastore(str(self.working_dir / "memory.db")).initialize()

            if not self.datastore:
                raise ValueError("Failed to initialize datastore")

            self.model_service = ModelService()
            if not self.model_service:
                raise ValueError("Failed to initialize model service")

            self.embedding_service = EmbeddingService()
            if not self.embedding_service:
                raise ValueError("Failed to initialize embedding service")

            # Initialize working directory
            self.working_dir.mkdir(parents=True, exist_ok=True)

            # Initialize LightRAG with MiniLM for queries
            self.rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=self._get_llm_func(),
                llm_model_name=EXTRACTION_MODEL_NAME,
                embedding_func=EmbeddingFunc(
                    embedding_dim=MINILM_DIM,
                    max_token_size=config.memory.max_chunk_tokens,
                    func=lambda texts: self.embedding_service.get_embeddings(
                        texts, force_model="minilm")
                ),
                enable_llm_cache=True,
                embedding_cache_config={
                    "enabled": True,
                    "similarity_threshold": 0.95,
                    "max_size": EMBEDDING_CACHE_SIZE
                },
                chunk_token_size=256,
                chunk_overlap_token_size=32,
                kv_storage="JsonKVStorage",
                vector_storage="NanoVectorDBStorage",
                graph_storage="NetworkXStorage",
                embedding_batch_num=BATCH_SIZE,
                embedding_func_max_async=10,
                llm_model_max_async=5,
                addon_params={
                    "example_number": 3,
                    "language": "English",
                    "mode": "local",
                }
            )

            # Initialize memory components
            self.ingestor = MemoryIngestor(self, self.datastore)
            self._initialized = True
            logger.info("LightRAGManager initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize LightRAGManager: {str(e)}", exc_info=True)
            # Clean up any partially initialized components
            self._llm_semaphore = None
            self.datastore = None
            self.model_service = None
            self.embedding_service = None
            self.rag = None
            self.ingestor = None
            self._initialized = False
            raise

    async def _wrapped_llm_func(self, prompt: str, system_prompt: Optional[str] = None, history_messages: Optional[List[Dict]] = None, **kwargs) -> str:
        """Wrapped LLM function with proper error handling and streaming support."""
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

            max_tokens = llm_params.get("max_tokens", DEFAULT_MAX_TOKENS)
            temperature = llm_params.get("temperature", 0.7)

            async for chunk in self.model_service.chat(
                messages=messages,
                stream=True,
                model=EXTRACTION_MODEL_NAME,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_tools=False
            ):
                if isinstance(chunk, dict):
                    response_text += chunk.get("content", "")
                elif isinstance(chunk, str):
                    response_text += chunk

            return response_text

        except Exception as e:
            logger.error(f"Error in LLM function: {str(e)}", exc_info=True)
            raise

    def _get_llm_func(self):
        """Get a picklable LLM function for LightRAG."""
        async def llm_func(prompt: str, system_prompt: Optional[str] = None, history_messages: Optional[List[Dict]] = None, **kwargs) -> str:
            return await self._wrapped_llm_func(prompt, system_prompt, history_messages, **kwargs)
        return llm_func

    async def query_memory(self, query: str, only_need_context: bool = True) -> Optional[List[MemoryResponse]]:
        """Query memory for relevant information.

        Args:
            query: Query string to search memories
            only_need_context: Whether to only return context without LLM processing

        Returns:
            Optional[List[MemoryResponse]]: List of structured memory responses if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        self.profiling_stats.total_query_calls += 1
        with profile_operation(self.profiling_stats.query_times, "memory_query"):
            try:
                logger.info(f"Starting memory query with: {query}")
                logger.info(f"Current working directory: {self.working_dir}")

                # Create query parameters with local mode for entity-based search
                query_param = QueryParam(
                    mode="mix",
                    stream=False,
                    top_k=TOP_K_MEMORY_RESULTS,
                    only_need_context=only_need_context,
                    max_token_for_local_context=3000,
                    max_token_for_global_context=3000,
                    max_token_for_text_unit=3000,
                )
                logger.info(f"Query parameters: {query_param}")

                # Get memory response from RAG with timeout
                try:
                    logger.info("Executing RAG query...")
                    memory_response = await asyncio.wait_for(
                        self.rag.aquery(query=query, param=query_param),
                        timeout=30
                    )
                    logger.info(
                        f"Raw memory response type: {type(memory_response)}")
                    logger.debug(f"Raw memory response: {memory_response}")

                except asyncio.TimeoutError:
                    logger.error("Memory query timed out after 30 seconds")
                    return None
                except Exception as e:
                    logger.error(
                        f"Error during RAG query: {str(e)}", exc_info=True)
                    return None

                if not memory_response:
                    logger.warning("No memory response received")
                    return None

                # Create default metadata for response
                default_metadata = {
                    "content_type": "chat_memory",
                    "chunk_size": 512,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "temperature": 0.7,
                    "embedding_model": "minilm"
                }

                # Use the MemoryResponse factory method to handle the response
                memories = MemoryResponse.from_lightrag_response(
                    response=memory_response,
                    default_metadata=default_metadata
                )

                if memories:
                    logger.info(f"Found {len(memories)} memories")
                    for memory in memories:
                        logger.debug(
                            f"Memory content: {memory.content.model_dump_json()}")
                else:
                    logger.warning("No valid memories found in response")

                return memories

            except Exception as e:
                logger.error(
                    f"Unexpected error in query_memory: {str(e)}", exc_info=True)
                return None

    async def store_memory(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Store a new memory with metadata.

        Args:
            text: The user's message or file content to store
            metadata: Additional metadata for the memory

        Returns:
            str: The memory ID
        """
        if not self._initialized:
            await self.initialize()

        self.profiling_stats.total_store_calls += 1
        with profile_operation(self.profiling_stats.store_times, "memory_store"):
            if not self.datastore:
                raise ValueError("Datastore not initialized")

            memory_id = str(uuid.uuid4())
            try:
                # Create metadata dictionary
                base_metadata = {
                    "memory_id": memory_id,
                    "timestamp": datetime.now().isoformat(),
                    "content_type": "chat_memory",
                    "chunk_size": 512,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "temperature": 0.7,
                    "embedding_model": "minilm"
                }

                # Merge metadata
                full_metadata = {
                    **base_metadata,
                    **{k: v for k, v in (metadata or {}).items() if v is not None}
                }

                # Store metadata in SQL database
                await self.datastore.store_entity(
                    entity_id=memory_id,
                    text=text,
                    metadata=full_metadata
                )

                # Store memory in LightRAG
                await self.rag.ainsert([text], metadata=[full_metadata])

                logger.info(f"Memory stored with ID: {memory_id}")
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
