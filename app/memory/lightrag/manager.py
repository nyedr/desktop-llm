"""Enhanced memory manager combining LightRAG and auxiliary storage."""

import hashlib
import logging
from datetime import datetime, timedelta
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

    async def start(self):
        """Start the memory system and background tasks."""
        logger.info("Starting enhanced memory system")
        await self.tasks.start()

    async def stop(self):
        """Stop the memory system and background tasks."""
        logger.info("Stopping enhanced memory system")
        await self.tasks.stop()

    async def add_memory(
        self,
        text: str,
        collection: MemoryType = MemoryType.EPHEMERAL,
        metadata: Optional[Dict] = None,
        max_chunk_tokens: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> str:
        """Add a new memory with background processing."""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

            # Check cache for existing memory with same content
            cache_key = f"content_hash_{content_hash}"
            existing = self.datastore.get_cache(cache_key)
            if existing:
                logger.info(
                    f"Found existing memory for content hash {content_hash}")
                return existing["memory_id"]

            # Generate memory ID
            memory_id = hashlib.sha256(
                f"{content_hash}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:32]

            # Format metadata
            full_metadata = metadata or {}
            full_metadata.update({
                "timestamp": datetime.now().isoformat(),
                "collection": collection.value,
                "memory_id": memory_id,
                "content_hash": content_hash,
                "storage_type": "chunk" if chunk_size else "full"
            })

            # Store metadata
            for key, value in full_metadata.items():
                self.datastore.set_metadata(memory_id, key, str(value))

            # Cache the content hash to memory ID mapping
            self.datastore.set_cache(cache_key, {
                "memory_id": memory_id,
                "timestamp": full_metadata["timestamp"]
            }, expiration=None)  # No expiration for content hash cache

            # Queue for processing
            await self.tasks.queue_memory_processing(text, {
                **full_metadata,
                "memory_id": memory_id,
                "chunk_size": chunk_size,
                "max_chunk_tokens": max_chunk_tokens
            })

            logger.info(
                f"Added memory {memory_id} to collection {collection.value}")
            return memory_id

        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}", exc_info=True)
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
            if isinstance(result, str):
                matches = [{"content": result, "metadata": {}, "score": 1.0}]
            elif isinstance(result, dict) and "matches" in result:
                processed_matches = []
                seen_hashes = set()

                for match in result["matches"]:
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

                                processed_matches.append({
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

                matches = processed_matches[:top_k]
            else:
                matches = []

            # Cache results for future queries
            self.datastore.set_cache(cache_key, {
                "matches": matches,
                "timestamp": datetime.now().isoformat()
            }, expiration=timedelta(minutes=30))  # Cache for 30 minutes

            return matches

        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}", exc_info=True)
            return []
