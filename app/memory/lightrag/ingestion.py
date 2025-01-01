"""Enhanced document and file ingestion for LightRAG memory system."""

import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Union, Dict, Optional, List, Tuple
import asyncio
from datetime import datetime
import uuid
from .manager_base import LightRAGManager
from .datastore import MemoryDatastore

logger = logging.getLogger(__name__)

# Constants
MAX_CHUNK_SIZE = 2000  # Characters
OVERLAP_SIZE = 200     # Characters
MIN_CHUNK_SIZE = 100   # Characters
BATCH_SIZE = 10        # Files per batch
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class ContentChunker:
    """Handles text chunking with overlap."""

    @staticmethod
    def chunk_text(text: str) -> List[Tuple[str, Dict]]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        chunks = []
        start = 0
        previous_hash = None

        while start < len(text):
            # Calculate end position
            end = min(start + MAX_CHUNK_SIZE, len(text))

            # Adjust end to avoid splitting words
            if end < len(text):
                while end > start and not text[end].isspace():
                    end -= 1
                if end == start:
                    end = min(start + MAX_CHUNK_SIZE, len(text))

            # Extract chunk
            chunk = text[start:end].strip()
            if not chunk:
                break

            # Generate chunk hash
            chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()

            # Create chunk metadata
            chunk_meta = {
                'start_pos': start,
                'end_pos': end,
                'chunk_hash': chunk_hash,
                'previous_chunk_hash': previous_hash
            }

            chunks.append((chunk, chunk_meta))
            previous_hash = chunk_hash

            # Move start position for next chunk
            start = max(start + MIN_CHUNK_SIZE, end - OVERLAP_SIZE)

        return chunks


class TextProcessor:
    """Processes plain text content."""

    @staticmethod
    def supported_types() -> List[str]:
        """Return supported MIME types."""
        return ['text/plain']

    @staticmethod
    async def process(path: Path) -> List[Tuple[str, Dict]]:
        """Process text file into chunks."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return ContentChunker.chunk_text(text)
        except Exception as e:
            logger.error(f"Error processing text file {path}: {e}")
            return []


class PDFProcessor:
    """Processes PDF documents."""

    @staticmethod
    def supported_types() -> List[str]:
        """Return supported MIME types."""
        return ['application/pdf']

    @staticmethod
    async def process(path: Path) -> List[Tuple[str, Dict]]:
        """Process PDF file into chunks."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return ContentChunker.chunk_text(text)
        except ImportError:
            logger.error("PyMuPDF not installed. Cannot process PDF files.")
            return []
        except Exception as e:
            logger.error(f"Error processing PDF file {path}: {e}")
            return []


class DOCXProcessor:
    """Processes DOCX documents."""

    @staticmethod
    def supported_types() -> List[str]:
        """Return supported MIME types."""
        return ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']

    @staticmethod
    async def process(path: Path) -> List[Tuple[str, Dict]]:
        """Process DOCX file into chunks."""
        try:
            from docx import Document
            doc = Document(path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return ContentChunker.chunk_text(text)
        except ImportError:
            logger.error(
                "python-docx not installed. Cannot process DOCX files.")
            return []
        except Exception as e:
            logger.error(f"Error processing DOCX file {path}: {e}")
            return []


class MemoryIngestor:
    """Handles ingestion of various content types into the enhanced memory system."""

    def __init__(self, manager: LightRAGManager, datastore: MemoryDatastore):
        self.manager = manager
        self.datastore = datastore
        self.processors = {
            'text': TextProcessor,
            'pdf': PDFProcessor,
            'docx': DOCXProcessor
        }

    async def ingest_text(self, text: str, metadata: Optional[Dict] = None, parent_id: Optional[str] = None):
        """
        Ingest plain text content into the memory system as chunks with metadata.

        Args:
            text: The text content to ingest
            metadata: Optional metadata to associate with the content
            parent_id: Optional parent entity ID for hierarchical organization

        Returns:
            str: The document entity ID
        """
        if not text.strip():
            return None

        logger.info(f"Ingesting text content (length: {len(text)})")

        # Generate content hash for deduplication
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

        # Check for existing document with same content
        existing = self.manager.datastore.search_entities(
            content_hash, limit=1)
        if existing:
            logger.debug(
                f"Skipping duplicate content with hash {content_hash}")
            return existing[0]["id"]

        # Create document-level entity with proper hierarchy
        doc_entity_id = str(uuid.uuid4())
        doc_title = metadata.get(
            'title', 'Untitled Document') if metadata else 'Untitled Document'

        # Create document entity with minimal metadata
        doc_metadata = {
            'is_document': 'true',
            'title': doc_title,
            'length': len(text),
            'content_hash': content_hash,
            'timestamp': datetime.now().isoformat(),
            'hierarchy_level': 'document',
            'parent_id': parent_id if parent_id else None,
            'entity_id': doc_entity_id
        }

        # Format text with metadata for LightRAG
        memory_text = f"METADATA: {doc_metadata}\nCONTENT: {text}"

        # Store in LightRAG
        await self.manager.rag.ainsert(memory_text)
        logger.info(f"Stored document with ID: {doc_entity_id}")

        return doc_entity_id

    async def ingest_file(self, file_path: Union[str, Path]) -> bool:
        """
        Ingest content from a file into the memory system as chunks with metadata.

        Args:
            file_path: Path to the file to ingest

        Returns:
            bool: True if ingestion was successful, False otherwise
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return False

        # Check file size
        if path.stat().st_size > MAX_FILE_SIZE:
            logger.error(
                f"File too large: {path} ({path.stat().st_size} bytes)")
            return False

        logger.info(f"Ingesting file: {path.name}")

        # Determine file type
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            logger.error(f"Could not determine MIME type for {path}")
            return False

        # Find appropriate processor
        processor = None
        for p in self.processors.values():
            if mime_type in p.supported_types():
                processor = p
                break

        if not processor:
            logger.error(f"No processor found for MIME type {mime_type}")
            return False

        # Process file
        chunks = await processor.process(path)
        if not chunks:
            return False

        # Create file metadata
        metadata = {
            'source': 'file',
            'filename': path.name,
            'filetype': mime_type,
            'path': str(path.resolve()),
            'ingested_at': datetime.now().isoformat()
        }

        # Ingest chunks with metadata
        for chunk, chunk_meta in chunks:
            # Create chunk metadata
            chunk_metadata = {
                'start_pos': str(chunk_meta['start_pos']),
                'end_pos': str(chunk_meta['end_pos']),
                'is_chunk': 'true',
                'file_name': path.name
            }

            # Add file metadata to chunk
            chunk_metadata.update(metadata)

            # Format text with metadata for LightRAG
            memory_text = f"METADATA: {chunk_metadata}\nCONTENT: {chunk}"

            # Store in LightRAG
            await self.manager.rag.ainsert(memory_text)

        return True

    async def ingest_directory(self, dir_path: Union[str, Path], batch_size: int = BATCH_SIZE) -> int:
        """
        Ingest all supported files from a directory with optimized batch processing.

        Args:
            dir_path: Path to the directory to ingest
            batch_size: Number of files to process concurrently

        Returns:
            int: Number of successfully ingested files
        """
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            logger.error(f"Directory not found: {path}")
            return 0

        logger.info(f"Ingesting directory: {path}")
        success_count = 0

        # Process files in batches
        files = [f for f in path.iterdir() if f.is_file()]
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            tasks = [self.ingest_file(f) for f in batch]
            results = await asyncio.gather(*tasks)
            success_count += sum(1 for r in results if r)

            # Commit after each batch
            await asyncio.sleep(0)  # Yield control

        logger.info(
            f"Successfully ingested {success_count}/{len(files)} files")
        return success_count
