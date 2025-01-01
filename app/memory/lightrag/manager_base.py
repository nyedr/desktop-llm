"""Base manager class for LightRAG memory system."""

from typing import Dict, Optional, Union
from pathlib import Path
from lightrag import QueryParam
from app.services.rag_service import create_lightrag
from .datastore import MemoryDatastore
from .tasks_base import MemoryTaskManager
import logging

logger = logging.getLogger(__name__)


class LightRAGManager:
    """Main manager class for LightRAG memory operations."""

    def __init__(self, config: Union[str, Path, Dict]):
        # Convert string/Path to config dict if needed
        if isinstance(config, (str, Path)):
            self.working_dir = str(config)
            self.config = {"working_dir": self.working_dir}
        elif isinstance(config, dict):
            self.working_dir = config.get("working_dir")
            if not self.working_dir:
                raise ValueError("working_dir must be provided in config")
            self.config = config

        self.rag = None
        self.datastore = None
        self.task_manager = None
        self._initialized = False

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

            # Initialize LightRAG using create_lightrag from rag_service
            self.rag = create_lightrag(
                working_dir=str(working_dir),
                # Pass through any additional config from self.config
                **{k: v for k, v in self.config.items() if k != 'working_dir'}
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
