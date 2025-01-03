"""Enhanced document and file ingestion for LightRAG memory system."""

import hashlib
import logging
from pathlib import Path
from typing import Union, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryIngestor:
    """Handles ingestion of content into the enhanced memory system."""

    def __init__(self, manager, datastore):
        """Initialize the memory ingestor."""
        self.manager = manager
        self.datastore = datastore

    def _generate_content_hash(self, text: str, metadata: Dict) -> str:
        """Generate a content hash that considers both content and context."""
        conversation_id = metadata.get('conversation_id', '')
        memory_id = metadata.get('memory_id', '')
        dedup_content = f"{text}|{conversation_id}|{memory_id}"
        return hashlib.md5(dedup_content.encode('utf-8')).hexdigest()

    async def ingest_text(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Ingest plain text content into the memory system."""
        if not text.strip():
            return None

        logger.info(f"Ingesting text content (length: {len(text)})")

        # Generate content hash that includes context
        content_hash = self._generate_content_hash(text, metadata or {})

        # Check for existing document with same content in same context
        existing = self.manager.datastore.search_entities(
            content_hash, limit=1)
        if existing:
            logger.debug(
                f"Skipping duplicate content with hash {content_hash}")
            return existing[0]["id"]

        # Create document metadata
        doc_metadata = {
            'is_document': 'true',
            'length': len(text),
            'content_hash': content_hash,
            'timestamp': datetime.now().isoformat()
        }

        # Add any additional metadata
        if metadata:
            doc_metadata.update(metadata)

        # Format text with metadata for LightRAG
        memory_text = f"METADATA: {doc_metadata}\nCONTENT: {text}"

        # Store in LightRAG using the manager's instance
        await self.manager.ainsert(memory_text)
        logger.info(f"Stored document with hash: {content_hash}")

        return content_hash

    async def ingest_file(self, file_path: Union[str, Path]) -> bool:
        """Ingest content from a file into the memory system."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {path}")
                return False

            # Read file content
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Create file metadata
            metadata = {
                'source': 'file',
                'filename': path.name,
                'path': str(path.resolve()),
                'ingested_at': datetime.now().isoformat()
            }

            # Ingest the text with metadata
            await self.ingest_text(text, metadata)
            logger.info(f"Successfully ingested file {path.name}")
            return True

        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {str(e)}")
            return False
