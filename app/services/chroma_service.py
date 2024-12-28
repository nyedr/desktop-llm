"""Service for managing vector-based memory storage using ChromaDB."""
import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.core.config import config

logger = logging.getLogger(__name__)


class ChromaService:
    """Service for managing vector-based memory storage using ChromaDB."""

    def __init__(self):
        """Initialize the Chroma service."""
        self.client = None
        self.collection = None
        self.embeddings = None

    async def initialize(self):
        """Initialize the Chroma client and collection."""
        try:
            logger.info("Initializing Chroma Service...")

            # Initialize Chroma client with persistence
            self.client = chromadb.PersistentClient(
                path=config.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Initialize embeddings model
            self.embeddings = SentenceTransformer(
                config.CHROMA_EMBEDDING_MODEL
            )

            # Get or create collection with proper metadata
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    "description": "Long-term memory storage for conversations and context",
                    "created_at": datetime.now().isoformat()
                }
            )

            logger.info("Chroma Service initialized successfully")
            return self.collection

        except Exception as e:
            logger.error(
                f"Failed to initialize Chroma Service: {e}", exc_info=True)
            raise

    async def _is_duplicate_memory(self, text: str, threshold: float = 0.95) -> bool:
        """Check if a memory already exists with high similarity.

        Args:
            text: Text to check for duplicates
            threshold: Similarity threshold (0-1)

        Returns:
            True if duplicate exists, False otherwise
        """
        try:
            # First check for exact matches
            results = self.collection.get(include=["documents"])
            if results["documents"]:
                for existing_text in results["documents"]:
                    if existing_text == text:
                        return True

            # For conversation summaries, use stricter threshold
            if "conversation_summary" in text.lower():
                threshold = 0.98

            # Generate embedding for the new text
            new_embedding = self.embeddings.encode(text).reshape(1, -1)

            # Get existing embeddings
            results = self.collection.get(include=["embeddings"])
            if not results:
                return False

            # Convert embeddings to numpy array if they exist
            embeddings = results.get("embeddings")
            if embeddings is None:
                return False

            # Convert to numpy array if not already
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Check if embeddings array is empty
            if embeddings.size == 0:
                return False

            # Calculate similarity scores
            existing_embeddings = np.array(results["embeddings"])
            similarities = cosine_similarity(
                new_embedding, existing_embeddings)

            # Check if any similarity exceeds threshold
            return np.any(similarities > threshold)

        except Exception as e:
            logger.error(
                f"Failed to check for duplicate memory: {e}", exc_info=True)
            return False

    async def add_memory(
        self,
        text: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> Union[str, List[str], None]:
        """Add one or more memories to the vector store.

        Args:
            text: Single text entry or list of text entries
            metadata: Optional metadata for the entries

        Returns:
            Single memory ID, list of memory IDs, or None if duplicate
        """
        try:
            if not self.collection:
                await self.initialize()

            # Handle single entry case
            if isinstance(text, str):
                # Validate input
                if not text or not isinstance(text, str):
                    raise ValueError("Text must be a non-empty string")

                # Check for duplicates before adding
                if await self._is_duplicate_memory(text):
                    logger.debug(f"Skipping duplicate memory: {text[:50]}...")
                    return None
                return await self._add_single_memory(text, metadata)

            # Handle batch case
            if not text:
                raise ValueError("Text list cannot be empty")

            unique_texts = []
            unique_metadatas = []
            for i, t in enumerate(text):
                if not t or not isinstance(t, str):
                    raise ValueError(f"Invalid text at index {i}")
                if not await self._is_duplicate_memory(t):
                    unique_texts.append(t)
                    if metadata and i < len(metadata):
                        unique_metadatas.append(metadata[i])

            if not unique_texts:
                logger.debug("All memories in batch were duplicates")
                return []

            return await self._add_batch_memories(unique_texts, unique_metadatas)

        except Exception as e:
            logger.error(f"Failed to add memory: {e}", exc_info=True)
            raise

    async def _add_single_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a single memory entry.

        Args:
            text: Text content to store
            metadata: Optional metadata for the entry

        Returns:
            Memory ID
        """
        # Remove special characters except basic punctuation
        sanitized_text = re.sub(r"[^\w\s.,!?']", "", text).strip()
        if not sanitized_text:
            raise ValueError(
                "Text contains no valid content after sanitization")

        # Generate embedding
        embedding = self.embeddings.encode(sanitized_text).tolist()

        # Generate a unique ID
        memory_id = str(uuid.uuid4())

        # Prepare metadata, converting None values to empty strings
        full_metadata = {
            "timestamp": datetime.now().isoformat(),
            "type": "memory",
            **{
                k: v if v is not None else ""
                for k, v in (metadata or {}).items()
            }
        }

        # Add to collection
        self.collection.add(
            ids=[memory_id],
            documents=[sanitized_text],
            embeddings=[embedding],
            metadatas=[full_metadata]
        )
        logger.debug(f"Added memory with ID: {memory_id}")
        return memory_id

    async def _add_batch_memories(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add multiple memories in batch.

        Args:
            texts: List of text entries
            metadatas: Optional list of metadata dictionaries

        Returns:
            List of memory IDs
        """
        # Sanitize texts
        sanitized_texts = []
        for text in texts:
            sanitized = re.sub(r"[^\w\s.,!?']", "", text).strip()
            if not sanitized:
                raise ValueError(
                    "Text contains no valid content after sanitization")
            sanitized_texts.append(sanitized)

        # Generate embeddings
        embeddings = [
            self.embeddings.encode(text).tolist()
            for text in sanitized_texts
        ]

        # Generate IDs
        memory_ids = [str(uuid.uuid4()) for _ in texts]

        # Prepare metadata, converting None values to empty strings
        timestamp = datetime.now().isoformat()
        full_metadatas = [
            {
                "timestamp": timestamp,
                "type": "memory",
                "batch_index": i,
                **{
                    k: v if v is not None else ""
                    for k, v in (metadatas[i] if metadatas and i < len(metadatas) else {}).items()
                }
            }
            for i in range(len(texts))
        ]

        # Add to collection
        self.collection.add(
            ids=memory_ids,
            documents=sanitized_texts,
            embeddings=embeddings,
            metadatas=full_metadatas
        )
        logger.debug(f"Added {len(memory_ids)} memories in batch")
        return memory_ids

    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum relevance score (0-1), default 0.5 for balanced matching

        Returns:
            List of memory dictionaries with content and metadata
        """
        try:
            if not self.collection:
                await self.initialize()

            # Generate query embedding
            query_embedding = self.embeddings.encode(query).tolist()

            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:  # Check if we have results
                for i in range(len(results["ids"][0])):
                    # Convert distance to similarity score (cosine distance to similarity)
                    similarity = 1 - results["distances"][0][i]
                    if similarity >= score_threshold:
                        formatted_results.append({
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "relevance_score": similarity
                        })

            return sorted(formatted_results, key=lambda x: x["relevance_score"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            raise

    async def retrieve_with_metadata(
        self,
        query: str,
        metadata_filter: Dict[str, Any],
        top_k: int = 4,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with metadata filtering.

        Args:
            query: Search query
            metadata_filter: Filter conditions for metadata
            top_k: Number of results to return
            score_threshold: Minimum relevance score (0-1)

        Returns:
            List of memory dictionaries with content and metadata
        """
        try:
            if not self.collection:
                await self.initialize()

            # Generate query embedding
            query_embedding = self.embeddings.encode(query).tolist()

            # Query collection with filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Format and filter results
            memories = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    relevance_score = 1 - results["distances"][0][i]
                    if relevance_score >= score_threshold:
                        memories.append({
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "relevance_score": relevance_score
                        })

            logger.debug(
                f"Retrieved {len(memories)} memories for query: '{query}' "
                f"with metadata filter: {metadata_filter}"
            )
            return sorted(memories, key=lambda x: x["relevance_score"], reverse=True)

        except Exception as e:
            logger.error(
                f"Failed to retrieve memories with metadata: {e}", exc_info=True)
            raise

    async def update_memory(
        self,
        memory_id: str,
        new_text: str,
        new_metadata: Optional[Dict[str, Any]] = None
    ):
        """Update an existing memory entry.

        Args:
            memory_id: ID of the memory to update
            new_text: New text content
            new_metadata: Optional new metadata
        """
        try:
            if not self.collection:
                await self.initialize()

            # Validate and sanitize input
            if not new_text or not isinstance(new_text, str):
                raise ValueError("Text must be a non-empty string")

            # Remove special characters except basic punctuation
            sanitized_text = re.sub(r"[^\w\s.,!?']", "", new_text).strip()
            if not sanitized_text:
                raise ValueError(
                    "Text contains no valid content after sanitization")

            # Generate new embedding
            new_embedding = self.embeddings.encode(sanitized_text).tolist()

            # Prepare updated metadata
            full_metadata = {
                "timestamp": datetime.now().isoformat(),
                "type": "memory",
                "updated_at": datetime.now().isoformat(),
                **{
                    k: v if v is not None else ""
                    for k, v in (new_metadata or {}).items()
                }
            }

            # Update in collection
            self.collection.update(
                ids=[memory_id],
                embeddings=[new_embedding],
                documents=[sanitized_text],
                metadatas=[full_metadata]
            )
            logger.debug(f"Updated memory with ID: {memory_id}")

        except Exception as e:
            logger.error(f"Failed to update memory: {e}", exc_info=True)
            raise

    async def delete_memory(self, memory_id: str):
        """Delete a memory entry.

        Args:
            memory_id: ID of the memory to delete
        """
        try:
            if not self.collection:
                await self.initialize()

            self.collection.delete(ids=[memory_id])
            logger.debug(f"Deleted memory with ID: {memory_id}")

        except Exception as e:
            logger.error(f"Failed to delete memory: {e}", exc_info=True)
            raise

    async def clear_collection(self):
        """Clear all memories from the collection."""
        try:
            if not self.collection:
                await self.initialize()
            self.collection.delete(where={})
            logger.info("Cleared all memories from collection")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}", exc_info=True)
            raise
