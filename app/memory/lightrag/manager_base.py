"""Base manager class for LightRAG memory system."""

from typing import Dict, Optional, Union, List
from pathlib import Path
from lightrag import LightRAG, QueryParam
from .datastore import MemoryDatastore
from .tasks_base import MemoryTaskManager
from .config import EMBEDDING_MODEL
import logging
from sentence_transformers import SentenceTransformer
from app.services.model_service import ModelService
from lightrag.llm import ollama_model_complete

logger = logging.getLogger(__name__)


class EmbeddingFunction:
    """Wrapper for embedding function with required attributes."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.embedding_dim = model.get_sentence_embedding_dimension()

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        return self.model.encode(texts).tolist()


class LightRAGManager:
    """Main manager class for LightRAG memory operations."""

    def __init__(self, config: Union[str, Path, Dict]):
        # Convert string/Path to config dict if needed
        if isinstance(config, (str, Path)):
            self.working_dir = str(config)
        elif isinstance(config, dict):
            self.working_dir = config.get("working_dir")
            if not self.working_dir:
                raise ValueError("working_dir must be provided in config")

        # Store the original config for other settings
        self.config = config if isinstance(config, dict) else {
            "working_dir": self.working_dir}

        self.rag = None
        self.datastore = None
        self.task_manager = None
        self._initialized = False
        self._embedding_model = None
        self._model_service = ModelService()  # Initialize ModelService

    @property
    def initialized(self) -> bool:
        """Check if the manager is initialized."""
        return self._initialized

    async def initialize(self, datastore: Optional[MemoryDatastore] = None):
        """Initialize the LightRAG manager and its components."""
        if self._initialized:
            return

        try:
            # Ensure working directory exists
            working_dir = Path(self.working_dir)
            working_dir.mkdir(parents=True, exist_ok=True)

            # Clear existing vector database files
            vdb_files = [
                working_dir / "vdb_entities.json",
                working_dir / "vdb_chunks.json",
                working_dir / "vdb_relationships.json",
                working_dir / "graph_chunk_entity_relation.graphml",
                working_dir / "kv_store_text_chunks.json",
                working_dir / "kv_store_llm_response_cache.json",
                working_dir / "kv_store_full_docs.json"
            ]
            for file in vdb_files:
                if file.exists():
                    logger.info(
                        f"Removing existing vector database file: {file}")
                    file.unlink()

            # Initialize the embedding model
            logger.debug(f"Initializing embedding model: {EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)

            # Create wrapped embedding function with required attributes
            embedding_func = EmbeddingFunction(self._embedding_model)
            logger.debug(
                f"Created embedding function with dimension: {embedding_func.embedding_dim}")

            # Initialize LightRAG with working directory and embedding function
            self.rag = LightRAG(
                working_dir=str(working_dir),
                embedding_func=embedding_func,
                llm_model_name="granite3.1-dense:2b-instruct-q4_K_M",
                llm_model_max_async=4,
                llm_model_max_token_size=26000,
                llm_model_kwargs={
                    "host": "http://localhost:11434", "options": {"num_ctx": 26000}},
                enable_llm_cache=True,
                llm_model_func=ollama_model_complete,
            )

            self.datastore = datastore or MemoryDatastore()
            self.task_manager = MemoryTaskManager()
            self._initialized = True
            logger.debug(
                f"LightRAG manager initialized with working directory: {working_dir}")
        except Exception as e:
            logger.error(
                f"Failed to initialize LightRAG manager: {str(e)}", exc_info=True)
            raise

    async def query(self, query: QueryParam) -> Dict:
        """Execute a query against the memory system."""
        if not self._initialized:
            await self.initialize()
        return await self.rag.query(query)

    async def ingest(self, data: Dict) -> Dict:
        """Ingest new data into the memory system."""
        if not self._initialized:
            await self.initialize()
        return await self.rag.ingest(data)

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by its ID."""
        if not self._initialized:
            await self.initialize()
        return await self.rag.delete(memory_id)

    async def update(self, memory_id: str, data: Dict) -> Dict:
        """Update an existing memory."""
        if not self._initialized:
            await self.initialize()
        return await self.rag.update(memory_id, data)
