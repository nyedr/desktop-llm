"""Background tasks and maintenance for LightRAG memory system."""

import logging
import asyncio
from typing import Dict, Any
from collections import deque
from .manager_base import LightRAGManager
from .datastore import MemoryDatastore
from .config import (
    MEMORY_QUEUE_PROCESS_DELAY,
    MEMORY_QUEUE_ERROR_RETRY_DELAY,
    CLEANUP_INTERVAL,
    MONITORING_INTERVAL
)
import json

logger = logging.getLogger(__name__)


class MemoryTasks:
    """Handles background tasks and maintenance operations."""

    def __init__(self, manager: LightRAGManager, datastore: MemoryDatastore):
        self.manager = manager
        self.datastore = datastore
        self._running = False
        self._tasks = []
        self._memory_queue = deque()
        self._processing = False

    async def initialize(self):
        """Initialize the task system."""
        if not self._running:
            await self.start()
        logger.info("Memory tasks initialized")

    async def start(self):
        """Start background tasks."""
        if self._running:
            return

        self._running = True
        logger.info("Starting memory background tasks")

        # Start scheduled tasks
        self._tasks = [
            asyncio.create_task(self._cleanup_task()),
            asyncio.create_task(self._monitoring_task()),
            asyncio.create_task(self._process_memory_queue())
        ]

    async def stop(self):
        """Stop background tasks."""
        self._running = False
        logger.info("Stopping memory background tasks")
        for task in self._tasks:
            task.cancel()
        self._tasks = []

    async def queue_memory_processing(self, text: str, metadata: Dict[str, Any]) -> None:
        """Queue memory for background processing."""
        try:
            # Check required metadata
            memory_id = metadata.get('memory_id')
            conversation_id = metadata.get('conversation_id')

            if not memory_id or not conversation_id:
                logger.warning(
                    "Missing memory_id or conversation_id in metadata")
                return

            # Handle chunking if specified
            chunk_size = metadata.get('chunk_size')
            if chunk_size:
                # Split text into chunks
                chunks = self.manager._split_text_into_chunks(text, chunk_size)
                for i, chunk in enumerate(chunks):
                    # Format chunk metadata
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                    await self.manager.ingestor.ingest_text(
                        text=chunk,
                        metadata=chunk_metadata
                    )
            else:
                # Store full document
                await self.manager.ingestor.ingest_text(
                    text=text,
                    metadata=metadata
                )

            logger.info(f"Stored memory text with ID: {memory_id}")

        except Exception as e:
            logger.error(
                f"Error in background memory processing: {e}", exc_info=True)
            raise

    async def _process_memory_queue(self):
        """Process queued memories in the background."""
        while self._running:
            try:
                if self._memory_queue and not self._processing:
                    self._processing = True
                    memory_text, metadata = self._memory_queue.popleft()

                    try:
                        await self.queue_memory_processing(memory_text, metadata)
                    except Exception as e:
                        logger.error(
                            f"Error processing memory in background: {str(e)}")
                    finally:
                        self._processing = False

                await asyncio.sleep(MEMORY_QUEUE_PROCESS_DELAY)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory processing task failed: {str(e)}")
                await asyncio.sleep(MEMORY_QUEUE_ERROR_RETRY_DELAY)

    async def _cleanup_task(self):
        """Periodic cleanup of expired cache entries."""
        while self._running:
            try:
                logger.info("Running memory cleanup task")
                self.datastore.cleanup_expired_cache()
                await asyncio.sleep(CLEANUP_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task failed: {str(e)}")
                await asyncio.sleep(MEMORY_QUEUE_ERROR_RETRY_DELAY)

    async def _monitoring_task(self):
        """Monitor memory system health and performance."""
        while self._running:
            try:
                logger.info("Running memory monitoring task")
                # TODO: Implement monitoring logic
                await asyncio.sleep(MONITORING_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring task failed: {str(e)}")
                await asyncio.sleep(MEMORY_QUEUE_ERROR_RETRY_DELAY)
