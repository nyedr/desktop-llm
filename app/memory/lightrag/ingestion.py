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
    """Handles content chunking and hierarchical organization."""

    @staticmethod
    def chunk_text(text: str) -> List[Tuple[str, Dict]]:
        """Split text into meaningful chunks with metadata and deduplication."""
        chunks = []
        start = 0
        text_length = len(text)
        previous_chunk_hash = None

        while start < text_length:
            end = min(start + MAX_CHUNK_SIZE, text_length)

            # Try to find a natural break point
            break_point = ContentChunker._find_break_point(text, start, end)
            if break_point > start + MIN_CHUNK_SIZE:
                end = break_point

            chunk = text[start:end].strip()
            if chunk:
                # Generate content hash that includes overlap context
                chunk_hash = ContentChunker._generate_chunk_hash(
                    text, start, end)

                # Skip if this chunk is too similar to previous chunk
                if previous_chunk_hash and ContentChunker._chunk_similarity(chunk_hash, previous_chunk_hash) > 0.8:
                    start = end - OVERLAP_SIZE if end - OVERLAP_SIZE > start else end
                    continue

                chunks.append((chunk, {
                    'start_pos': start,
                    'end_pos': end,
                    'length': end - start,
                    'chunk_hash': chunk_hash,
                    'previous_chunk_hash': previous_chunk_hash
                }))
                previous_chunk_hash = chunk_hash

            start = end - OVERLAP_SIZE if end - OVERLAP_SIZE > start else end

        return chunks

    @staticmethod
    def _generate_chunk_hash(text: str, start: int, end: int) -> str:
        """Generate a unique hash for a chunk that includes context."""
        import hashlib
        # Include surrounding text for better uniqueness
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end]
        return hashlib.sha256(context.encode('utf-8')).hexdigest()

    @staticmethod
    def _chunk_similarity(hash1: str, hash2: str) -> float:
        """Calculate similarity between two chunk hashes."""
        # Simple similarity based on hash prefix match
        match_length = 0
        for c1, c2 in zip(hash1, hash2):
            if c1 == c2:
                match_length += 1
            else:
                break
        return match_length / len(hash1)

    @staticmethod
    def _find_break_point(text: str, start: int, end: int) -> int:
        """Find a natural break point in the text."""
        # Look for paragraph breaks first
        para_break = text.rfind('\n\n', start, end)
        if para_break != -1:
            return para_break + 2

        # Look for sentence breaks
        sentence_break = max(
            text.rfind('. ', start, end),
            text.rfind('! ', start, end),
            text.rfind('? ', start, end)
        )
        if sentence_break != -1:
            return sentence_break + 2

        # Fall back to word breaks
        word_break = text.rfind(' ', start, end)
        return word_break if word_break != -1 else end


class FileProcessor:
    """Base class for file type specific processors."""

    @classmethod
    def supported_types(cls) -> List[str]:
        """Return list of supported MIME types."""
        return []

    @classmethod
    async def process(cls, file_path: Path) -> Optional[List[Tuple[str, Dict]]]:
        """Process file and return chunked content with metadata."""
        return None


class TextProcessor(FileProcessor):
    """Processor for plain text files."""

    @classmethod
    def supported_types(cls) -> List[str]:
        return ['text/plain']

    @classmethod
    async def process(cls, file_path: Path) -> Optional[List[Tuple[str, Dict]]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return ContentChunker.chunk_text(content)
        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {str(e)}")
            return None


class PDFProcessor(FileProcessor):
    """Processor for PDF files."""

    @classmethod
    def supported_types(cls) -> List[str]:
        return ['application/pdf']

    @classmethod
    async def process(cls, file_path: Path) -> Optional[List[Tuple[str, Dict]]]:
        try:
            from PyPDF2 import PdfReader
            import re

            # Initialize PDF reader
            reader = PdfReader(file_path)
            full_text = ""

            # Extract text from each page
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Clean up text
                    page_text = re.sub(r'\s+', ' ', page_text).strip()
                    full_text += page_text + "\n"

            if not full_text.strip():
                logger.warning(f"No text extracted from PDF: {file_path}")
                return None

            return ContentChunker.chunk_text(full_text)
        except Exception as e:
            logger.error(f"Failed to process PDF file {file_path}: {str(e)}")
            return None


class DOCXProcessor(FileProcessor):
    """Processor for DOCX files."""

    @classmethod
    def supported_types(cls) -> List[str]:
        return ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']

    @classmethod
    async def process(cls, file_path: Path) -> Optional[List[Tuple[str, Dict]]]:
        try:
            from docx import Document
            import re

            # Load DOCX document
            doc = Document(file_path)
            full_text = ""

            # Extract text from paragraphs
            for para in doc.paragraphs:
                para_text = para.text.strip()
                if para_text:
                    # Clean up text
                    para_text = re.sub(r'\s+', ' ', para_text)
                    full_text += para_text + "\n"

            if not full_text.strip():
                logger.warning(f"No text extracted from DOCX: {file_path}")
                return None

            return ContentChunker.chunk_text(full_text)
        except Exception as e:
            logger.error(f"Failed to process DOCX file {file_path}: {str(e)}")
            return None


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

        # Create document entity metadata
        doc_metadata = {
            'is_document': 'true',
            'title': doc_title,
            'length': len(text),
            'chunk_count': 0,  # Will be updated as chunks are added
            'content_hash': content_hash,
            'timestamp': datetime.now().isoformat()
        }
        if metadata:
            doc_metadata.update(metadata)

        # Store document metadata without the full text
        await self.manager.insert_entity({
            **doc_metadata,
            'hierarchy_level': 'document',
            'parent_id': parent_id if parent_id else None,
            'entity_id': doc_entity_id
        })

        # Chunk the content
        chunks = ContentChunker.chunk_text(text)
        doc_metadata['chunk_count'] = len(chunks)

        # Ingest chunks with proper parent reference
        for chunk, chunk_meta in chunks:
            # Check for duplicate chunk using chunk_hash
            if await self.manager.chunk_exists(chunk_meta['chunk_hash']):
                logger.debug(
                    f"Skipping duplicate chunk: {chunk_meta['chunk_hash']}")
                continue

            # Create chunk metadata with parent reference
            chunk_metadata = {
                'start_pos': str(chunk_meta['start_pos']),
                'end_pos': str(chunk_meta['end_pos']),
                'is_chunk': 'true',
                'doc_title': doc_title,
                'parent_id': doc_entity_id,  # Reference parent document
                'chunk_hash': chunk_meta['chunk_hash'],
                'previous_chunk_hash': chunk_meta['previous_chunk_hash'],
                'hierarchy_level': 'chunk'
            }

            # Add document metadata to chunk
            if metadata:
                chunk_metadata.update(metadata)

            # Store chunk in LightRAG with combined metadata
            await self.manager.insert_text(chunk, chunk_metadata)

        # Update document entity with final chunk count and chunk references
        await self.manager.update_entity(doc_entity_id, {
            'chunk_count': len(chunks),
            'chunk_references': [c[1]['chunk_hash'] for c in chunks]
        })

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

            # Store chunk in LightRAG with combined metadata
            await self.manager.insert_text(chunk, chunk_metadata)

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
