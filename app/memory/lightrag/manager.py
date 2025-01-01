"""Enhanced memory manager combining LightRAG and auxiliary storage."""

import hashlib
import logging
from datetime import datetime
from typing import List, Optional, Dict, Union
from pathlib import Path
import json

from lightrag import QueryParam
from .config import LIGHTRAG_DATA_DIR
from .datastore import MemoryDatastore
from .ingestion import MemoryIngestor
from .tasks import MemoryTasks
from .manager_base import LightRAGManager
from app.models.memory import MemoryType

logger = logging.getLogger(__name__)


class EnhancedLightRAGManager(LightRAGManager):
    """Enhanced memory manager combining LightRAG with auxiliary storage."""

    def __init__(self, working_dir: Union[str, Path, Dict] = LIGHTRAG_DATA_DIR):
        """Initialize the enhanced memory manager."""
        # Convert to config dict if needed
        config = working_dir if isinstance(working_dir, dict) else {
            "working_dir": str(working_dir)}

        # Initialize parent class
        super().__init__(config)

        # Initialize enhanced components
        self.datastore = None
        self.ingestor = None
        self.tasks = None
        self._initialized = False

    async def initialize(self):
        """Initialize the memory system and all components."""
        if self._initialized:
            return

        try:
            # Initialize datastore first
            self.datastore = MemoryDatastore()

            # Initialize parent class with our datastore
            await super().initialize(self.datastore)

            # Initialize remaining components
            self.ingestor = MemoryIngestor(self, self.datastore)
            self.tasks = MemoryTasks(self, self.datastore)

            # Initialize tasks
            await self.tasks.initialize()

            # Mark as initialized
            self._initialized = True

            logger.info("Enhanced LightRAGManager initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize Enhanced LightRAGManager: {str(e)}")
            raise

    async def retrieve_memories(
        self,
        query: str,
        collection: Optional[MemoryType] = None,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Retrieve relevant memories based on query and metadata filter."""
        try:
            if not self._initialized:
                await self.initialize()

            # Create query parameters
            param = QueryParam()
            param.mode = "hybrid"
            param.k = top_k

            # Add filters
            if metadata_filter or collection:
                param.filter = metadata_filter or {}
                if collection:
                    param.filter["collection"] = collection.value

            # Check cache for query results
            cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
            cached_result = self.datastore.get_cache(cache_key)
            if cached_result:
                logger.info("Using cached query results")
                return cached_result["matches"]

            # Execute query
            result = await self.rag.aquery(query, param)

            # Process results
            matches = []
            seen_hashes = set()

            if result and hasattr(result, 'matches'):
                for match in result.matches:
                    try:
                        content = match.get("content", "")
                        if not content:
                            continue

                        # Split into metadata and content sections
                        parts = content.split("\nCONTENT: ", 1)
                        if len(parts) == 2:
                            metadata_str = parts[0].replace("METADATA: ", "")
                            try:
                                metadata = json.loads(metadata_str)
                                content = parts[1]

                                # Generate content hash for deduplication
                                content_hash = hashlib.md5(
                                    content.encode()).hexdigest()

                                # Skip if we've seen this content before
                                if content_hash in seen_hashes:
                                    continue

                                # Enrich with stored metadata
                                if memory_id := metadata.get("memory_id"):
                                    stored_metadata = self.datastore.get_metadata(
                                        memory_id)
                                    metadata.update(stored_metadata)

                                matches.append({
                                    "content": content,
                                    "metadata": metadata,
                                    "score": match.get("score", 1.0),
                                    "hash": content_hash
                                })

                                seen_hashes.add(content_hash)

                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Failed to parse metadata: {metadata_str}")
                                continue
                    except Exception as e:
                        logger.warning(f"Error processing match: {str(e)}")
                        continue

            # Cache results for future queries
            self.datastore.set_cache(cache_key, {
                "matches": matches[:top_k],
                "timestamp": datetime.now().isoformat()
            })

            return matches[:top_k]

        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}", exc_info=True)
            return []

    async def start(self):
        """Start the memory system and background tasks."""
        logger.info("Starting enhanced memory system")
        await self.tasks.start()

    async def stop(self):
        """Stop the memory system and background tasks."""
        logger.info("Stopping enhanced memory system")
        await self.tasks.stop()
