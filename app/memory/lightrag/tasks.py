"""Background tasks and maintenance for LightRAG memory system."""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict
from collections import deque
from .manager_base import LightRAGManager
from .datastore import MemoryDatastore
from .config import (
    MEMORY_QUEUE_PROCESS_DELAY,
    MEMORY_QUEUE_ERROR_RETRY_DELAY,
    CLEANUP_INTERVAL,
    OPTIMIZATION_INTERVAL,
    MONITORING_INTERVAL,
    DEFAULT_RETENTION_DAYS
)

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

    async def start(self):
        """Start background tasks."""
        if self._running:
            return

        self._running = True
        logger.info("Starting memory background tasks")

        # Start scheduled tasks
        self._tasks = [
            asyncio.create_task(self._cleanup_task()),
            asyncio.create_task(self._optimization_task()),
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

    async def queue_memory_processing(self, memory_text: str, metadata: Dict = None):
        """Queue a memory for background processing."""
        self._memory_queue.append((memory_text, metadata))
        logger.debug(
            f"Queued memory for processing. Queue size: {len(self._memory_queue)}")

    async def _process_memory_queue(self):
        """Process queued memories in the background."""
        while self._running:
            try:
                if self._memory_queue and not self._processing:
                    self._processing = True
                    memory_text, metadata = self._memory_queue.popleft()

                    try:
                        # Extract entities using embeddings
                        entities = await self.manager.extract_entities(memory_text)

                        # Build custom knowledge graph
                        custom_kg = {
                            "entities": [],
                            "relationships": []
                        }

                        # Process entities and relationships
                        for entity in entities:
                            sanitized_text = self.manager._sanitize_id(
                                entity["text"])
                            custom_kg["entities"].append({
                                "entity_name": sanitized_text,
                                "entity_type": entity["label"],
                                "description": f"Entity extracted from text: {entity['text']} ({entity['label']})",
                                "source_id": "memory"
                            })

                            # Add relationships if appropriate
                            relation_type = self.manager.interpret_relation(
                                entity["label"], memory_text)
                            if relation_type:
                                custom_kg["relationships"].append({
                                    "src_id": "USER",
                                    "tgt_id": sanitized_text,
                                    "description": f"User {relation_type} {entity['text']}",
                                    "keywords": f"{relation_type} {entity['text']}",
                                    "source_id": "memory"
                                })

                        # Update the knowledge graph
                        if custom_kg["entities"] or custom_kg["relationships"]:
                            await self.manager.rag.ainsert_custom_kg(custom_kg)
                            logger.info(
                                "Added entities and relationships to knowledge graph in background")

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
        """Periodic cleanup of old or unused memory entries."""
        while self._running:
            try:
                logger.info("Running memory cleanup task")

                # Clean up old entities and relationships
                cutoff = datetime.now() - timedelta(days=DEFAULT_RETENTION_DAYS)
                await self._cleanup_old_entities(cutoff)
                await self._cleanup_unused_relationships()

                await asyncio.sleep(CLEANUP_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task failed: {str(e)}")
                await asyncio.sleep(MEMORY_QUEUE_ERROR_RETRY_DELAY)

    async def _cleanup_old_entities(self, cutoff: datetime):
        """Remove entities older than the specified cutoff while maintaining hierarchy."""
        logger.info(f"Cleaning up entities older than {cutoff}")

        # Get old entities
        old_entities = self.datastore.get_entities_older_than(cutoff)

        # Process in hierarchy-aware manner
        for entity in old_entities:
            # Check if entity has children
            children = self.datastore.get_child_entities(entity['id'])
            if children:
                # Move children to parent if exists
                parent = self.datastore.get_entity_parent(entity['id'])
                if parent:
                    for child in children:
                        self.datastore.set_entity_hierarchy(
                            child['id'],
                            child['hierarchy_level'],
                            parent['id']
                        )

            # Remove entity and its metadata
            self.datastore.delete_entity(entity['id'])
            self.datastore.delete_entity_metadata(entity['id'])

        logger.info(f"Cleaned up {len(old_entities)} old entities")

    async def _cleanup_unused_relationships(self):
        """Remove relationships that are no longer valid, maintaining hierarchy integrity."""
        logger.info("Cleaning up unused relationships")

        # Get all relationships
        relationships = self.datastore.get_all_relationships()

        # Check each relationship
        for rel in relationships:
            src_exists = self.datastore.entity_exists(rel['src_entity'])
            dst_exists = self.datastore.entity_exists(rel['dst_entity'])

            if not src_exists or not dst_exists:
                # Remove invalid relationship
                self.datastore.delete_relationship(rel['id'])

        logger.info("Completed relationship cleanup")

    async def _optimization_task(self):
        """Periodic optimization of memory storage and retrieval."""
        while self._running:
            try:
                logger.info("Running memory optimization task")
                await self.optimize_memory()
                await asyncio.sleep(OPTIMIZATION_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization task failed: {str(e)}")
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

    async def summarize_memory(self, entity_id: Optional[str] = None) -> Dict:
        """
        Generate summaries for memory content.

        Args:
            entity_id: Optional specific entity to summarize

        Returns:
            Dict: Summary information
        """
        logger.info(f"Generating memory summary for entity: {entity_id}")

        summary = {
            'entity_id': entity_id,
            'summary': "Summary not implemented yet",
            'related_entities': [],
            'key_points': []
        }

        # TODO: Implement actual summarization logic
        return summary

    async def optimize_memory(self) -> Dict:
        """
        Optimize memory storage and retrieval performance.

        Returns:
            Dict: Optimization results
        """
        logger.info("Optimizing memory storage")

        results = {
            'status': 'success',
            'actions_taken': [],
            'performance_improvement': 0.0
        }

        # TODO: Implement actual optimization logic
        return results
