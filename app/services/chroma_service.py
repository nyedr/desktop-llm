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
from transformers import AutoTokenizer

from app.core.config import config
from app.models.memory import MemoryType

logger = logging.getLogger(__name__)


class ChromaService:
    """Service for managing vector-based memory storage using ChromaDB."""

    def __init__(self):
        """Initialize the Chroma service."""
        self.client = None
        self.ephemeral_collection = None
        self.model_memory_collection = None
        self.embeddings = None
        self.tokenizer = None

    async def initialize(self):
        """Initialize the Chroma client and collections."""
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

            # Initialize tokenizer for chunking
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.TOKENIZER_MODEL or "gpt2"
            )

            # Get or create ephemeral logs collection
            self.ephemeral_collection = self.client.get_or_create_collection(
                name=MemoryType.EPHEMERAL,
                metadata={
                    "hnsw:space": "cosine",
                    "description": "Ephemeral conversation logs storage",
                    "created_at": datetime.now().isoformat()
                }
            )

            # Get or create model memory collection
            self.model_memory_collection = self.client.get_or_create_collection(
                name=MemoryType.MODEL_MEMORY,
                metadata={
                    "hnsw:space": "cosine",
                    "description": "Long-term model memory storage",
                    "created_at": datetime.now().isoformat()
                }
            )

            logger.info("Chroma Service initialized successfully")
            return {
                "ephemeral_collection": self.ephemeral_collection,
                "model_memory_collection": self.model_memory_collection
            }

        except Exception as e:
            logger.error(
                f"Failed to initialize Chroma Service: {e}", exc_info=True)
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using the initialized tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        if not self.tokenizer:
            # Fallback to approximate token count
            return len(text.split())
        return len(self.tokenizer.encode(text))

    async def chunk_text(
        self,
        text: str,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 50
    ) -> List[str]:
        """Chunk text into smaller pieces based on token count.

        Args:
            text: Text to chunk
            max_chunk_tokens: Maximum tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks

        Returns:
            List of text chunks
        """
        if not text or self.count_tokens(text) <= max_chunk_tokens:
            return [text]

        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            # Get chunk of tokens up to max_chunk_tokens
            end = start + max_chunk_tokens
            chunk_tokens = tokens[start:end]

            # Decode chunk back to text
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)

            # Move start position, accounting for overlap
            start = end - overlap_tokens

        return chunks

    async def _is_duplicate_memory(self, text: str, collection: MemoryType, metadata: Optional[Dict[str, Any]] = None, threshold: float = 0.95) -> bool:
        """Check if a memory already exists with high similarity in the specified collection.

        Args:
            text: Text to check for duplicates
            collection: Collection to check in (MemoryType.EPHEMERAL or MemoryType.MODEL_MEMORY)
            metadata: Optional metadata for the new memory
            threshold: Similarity threshold (0-1)

        Returns:
            True if duplicate exists, False otherwise
        """
        try:
            target_collection = self.ephemeral_collection if collection == MemoryType.EPHEMERAL else self.model_memory_collection

            # First check for exact matches
            results = target_collection.get(include=["documents", "metadatas"])
            if results["documents"]:
                for i, existing_text in enumerate(results["documents"]):
                    if existing_text == text:
                        # For summaries, only consider exact matches from same conversation
                        existing_metadata = results["metadatas"][i]
                        if existing_metadata.get("type") == "conversation_summary":
                            # Compare conversation IDs
                            new_conv_id = metadata.get(
                                "conversation_id") if metadata else None
                            existing_conv_id = existing_metadata.get(
                                "conversation_id")
                            if new_conv_id and existing_conv_id and new_conv_id == existing_conv_id:
                                return True
                            continue  # Different conversation, allow similar summaries
                        return True  # For non-summaries, exact match means duplicate

            # For conversation summaries, use stricter threshold and check conversation_id
            is_summary = metadata.get(
                "type") == "conversation_summary" if metadata else False
            if is_summary:
                threshold = 0.98  # Stricter for summaries
                # Get existing summaries for this conversation
                results = target_collection.get(
                    include=["embeddings", "metadatas"],
                    where={"type": "conversation_summary"}
                )
            else:
                results = target_collection.get(include=["embeddings"])

            # Check if results exist and embeddings are not empty
            if not results or len(results["embeddings"]) == 0:
                return False

            # Generate embedding for the new text
            new_embedding = self.embeddings.encode(text).reshape(1, -1)

            # Convert to numpy array if not already
            embeddings = np.array(results["embeddings"])
            if embeddings.size == 0:
                return False

            # Calculate similarity scores
            similarities = cosine_similarity(new_embedding, embeddings)

            # For summaries, only consider duplicates from same conversation
            if is_summary:
                new_conv_id = metadata.get(
                    "conversation_id") if metadata else None
                for i, similarity in enumerate(similarities[0]):
                    if similarity > threshold:
                        existing_metadata = results["metadatas"][i]
                        existing_conv_id = existing_metadata.get(
                            "conversation_id")
                        if new_conv_id and existing_conv_id and new_conv_id == existing_conv_id:
                            return True
                return False

            # For non-summaries, any high similarity is a duplicate
            return np.any(similarities > threshold)

        except Exception as e:
            logger.error(
                f"Failed to check for duplicate memory: {e}", exc_info=True)
            return False

    async def add_memory(
        self,
        text: Union[str, List[str]],
        collection: MemoryType,
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        max_chunk_tokens: Optional[int] = 512,
        conversation_id: Optional[str] = None
    ) -> Optional[Union[str, List[str]]]:
        """Add one or more memories to the specified collection.

        Args:
            text: Single text entry or list of text entries
            collection: Collection to add to (MemoryType.EPHEMERAL or MemoryType.MODEL_MEMORY)
            metadata: Optional metadata for the entries
            max_chunk_tokens: Maximum tokens per chunk (if chunking needed)
            conversation_id: Optional conversation ID this memory belongs to

        Returns:
            Optional[Union[str, List[str]]]: 
            - For single text: str (memory ID) or None if duplicate
            - For list of texts: List[str] (memory IDs) or None if all duplicates
        """
        try:
            if not self.ephemeral_collection or not self.model_memory_collection:
                await self.initialize()

            # Force summaries to ephemeral collection
            if metadata and isinstance(metadata, dict):
                memory_type = metadata.get("type", "")
                if memory_type in ["conversation_summary", "summary"]:
                    collection = MemoryType.EPHEMERAL
                    logger.debug("Forcing summary to ephemeral collection")

            # Handle single entry case
            if isinstance(text, str):
                # Validate input
                if not text or not isinstance(text, str):
                    logger.warning("Invalid text input: empty or not string")
                    return None

                # Check for duplicates before adding
                if await self._is_duplicate_memory(text, collection, metadata):
                    logger.debug(f"Skipping duplicate memory: {text[:50]}...")
                    return None

                # Generate a unique response_id for this memory
                response_id = str(uuid.uuid4())

                # If chunking is enabled and needed, split the text
                if max_chunk_tokens and self.count_tokens(text) > max_chunk_tokens:
                    chunks = await self.chunk_text(text, max_chunk_tokens)
                    chunk_metadata = []

                    for idx, _ in enumerate(chunks):
                        chunk_meta = {
                            **(metadata or {}),
                            "response_id": response_id,
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                            "is_chunk": True,
                            "timestamp": datetime.now().isoformat()
                        }
                        if conversation_id:
                            chunk_meta["conversation_id"] = conversation_id
                        chunk_metadata.append(chunk_meta)

                    memory_ids = await self._add_batch_memories(chunks, collection, chunk_metadata)
                    # Return first ID for single text
                    return memory_ids[0] if memory_ids else None

                # For single memory, prepare metadata
                full_metadata = {
                    **(metadata or {}),
                    "response_id": response_id,
                    "timestamp": datetime.now().isoformat()
                }
                if conversation_id:
                    full_metadata["conversation_id"] = conversation_id

                return await self._add_single_memory(text, collection, full_metadata)

            # Handle batch case
            if not text:
                logger.warning("Empty text list provided")
                return None

            # Process each text entry in the batch
            unique_texts = []
            unique_metadatas = []

            for i, t in enumerate(text):
                if not t or not isinstance(t, str):
                    logger.warning(f"Invalid text at index {i}")
                    continue

                if not await self._is_duplicate_memory(t, collection, metadata[i] if metadata else None):
                    unique_texts.append(t)
                    if metadata and i < len(metadata):
                        meta = metadata[i].copy()
                        if conversation_id:
                            meta["conversation_id"] = conversation_id
                        if "response_id" not in meta:
                            meta["response_id"] = str(uuid.uuid4())
                        meta["timestamp"] = datetime.now().isoformat()
                        unique_metadatas.append(meta)

            if not unique_texts:
                logger.debug("All memories in batch were duplicates")
                return None

            return await self._add_batch_memories(unique_texts, collection, unique_metadatas)

        except Exception as e:
            logger.error(f"Failed to add memory: {e}", exc_info=True)
            return None

    async def _add_single_memory(
        self,
        text: str,
        collection: MemoryType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a single memory entry to the specified collection.

        Args:
            text: Text content to store
            collection: Collection to add to (MemoryType.EPHEMERAL or MemoryType.MODEL_MEMORY)
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

        # Prepare metadata
        base_metadata = {
            "timestamp": datetime.now().isoformat(),
            "type": "memory",
            "response_id": metadata.get("response_id", str(uuid.uuid4())),
            "conversation_id": metadata.get("conversation_id", ""),
            "is_chunk": metadata.get("is_chunk", False),
            "chunk_index": metadata.get("chunk_index", 0),
            "total_chunks": metadata.get("total_chunks", 1)
        }

        # Merge with provided metadata, ensuring no None values
        full_metadata = {
            **base_metadata,
            **{
                k: v if v is not None else ""
                for k, v in (metadata or {}).items()
            }
        }

        # Get target collection
        target_collection = self.ephemeral_collection if collection == MemoryType.EPHEMERAL else self.model_memory_collection

        # Add to collection
        target_collection.add(
            ids=[memory_id],
            documents=[sanitized_text],
            embeddings=[embedding],
            metadatas=[full_metadata]
        )
        logger.debug(
            f"Added memory with ID: {memory_id} to {collection} collection")
        return memory_id

    async def _add_batch_memories(
        self,
        texts: List[str],
        collection: MemoryType,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add multiple memories in batch to the specified collection.

        Args:
            texts: List of text entries
            collection: Collection to add to (MemoryType.EPHEMERAL or MemoryType.MODEL_MEMORY)
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

        # Prepare metadata for each entry
        full_metadatas = []
        timestamp = datetime.now().isoformat()

        for i in range(len(texts)):
            # Start with base metadata
            base_metadata = {
                "timestamp": timestamp,
                "type": "memory",
                "response_id": str(uuid.uuid4()),
                "conversation_id": "",
                "is_chunk": False,
                "chunk_index": 0,
                "total_chunks": 1
            }

            # Update with provided metadata if available
            if metadatas and i < len(metadatas):
                provided_metadata = metadatas[i]
                base_metadata.update({
                    k: v if v is not None else ""
                    for k, v in provided_metadata.items()
                })

            full_metadatas.append(base_metadata)

        # Get target collection
        target_collection = self.ephemeral_collection if collection == MemoryType.EPHEMERAL else self.model_memory_collection

        # Add to collection
        target_collection.add(
            ids=memory_ids,
            documents=sanitized_texts,
            embeddings=embeddings,
            metadatas=full_metadatas
        )
        logger.debug(
            f"Added {len(memory_ids)} memories in batch to {collection} collection")
        return memory_ids

    async def retrieve_memories(
        self,
        query: str,
        collection: MemoryType,
        top_k: int = 4,
        score_threshold: float = 0.5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        reassemble_chunks: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on semantic similarity from the specified collection.

        Args:
            query: Search query
            collection: Collection to query (MemoryType.EPHEMERAL or MemoryType.MODEL_MEMORY)
            top_k: Number of results to return
            score_threshold: Minimum relevance score (0-1)
            metadata_filter: Optional filter conditions for metadata
            conversation_id: Optional conversation ID to filter by
            reassemble_chunks: Whether to reassemble chunked memories

        Returns:
            List of memory dictionaries with content and metadata
        """
        try:
            if not self.ephemeral_collection or not self.model_memory_collection:
                await self.initialize()

            # Validate collection
            if collection not in [MemoryType.EPHEMERAL, MemoryType.MODEL_MEMORY]:
                raise ValueError("Invalid collection specified")

            # Get target collection
            target_collection = self.ephemeral_collection if collection == MemoryType.EPHEMERAL else self.model_memory_collection

            # Generate query embedding
            query_embedding = self.embeddings.encode(query).tolist()

            # Build combined metadata filter
            combined_filter = metadata_filter or {}
            if conversation_id:
                combined_filter["conversation_id"] = conversation_id

            # Query the collection with combined metadata filter
            results = target_collection.query(
                query_embeddings=[query_embedding],
                # Get extra results if we need to reassemble chunks
                n_results=top_k * 2 if reassemble_chunks else top_k,
                include=["documents", "metadatas", "distances"],
                where=combined_filter
            )

            # Format and process results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                # Group chunks by response_id if reassembling
                chunk_groups = {}

                for i in range(len(results["ids"][0])):
                    # Convert distance to similarity score
                    similarity = 1 - results["distances"][0][i]
                    if similarity >= score_threshold:
                        document = results["documents"][0][i]
                        metadata = results["metadatas"][0][i]

                        if reassemble_chunks and metadata.get("is_chunk"):
                            response_id = metadata.get("response_id")
                            if response_id not in chunk_groups:
                                chunk_groups[response_id] = []
                            chunk_groups[response_id].append({
                                "document": document,
                                "metadata": metadata,
                                "relevance_score": similarity
                            })
                        else:
                            formatted_results.append({
                                "document": document,
                                "metadata": metadata,
                                "relevance_score": similarity
                            })

                # Reassemble chunks if needed
                if reassemble_chunks and chunk_groups:
                    for response_id, chunks in chunk_groups.items():
                        if len(chunks) > 1:
                            # Sort chunks by index
                            chunks.sort(
                                key=lambda x: x["metadata"].get("chunk_index", 0))

                            # Combine chunks
                            combined_text = " ".join(
                                chunk["document"] for chunk in chunks)

                            # Use metadata from first chunk but remove chunk-specific fields
                            combined_metadata = chunks[0]["metadata"].copy()
                            combined_metadata.pop("is_chunk", None)
                            combined_metadata.pop("chunk_index", None)
                            combined_metadata.pop("total_chunks", None)

                            # Use highest relevance score from chunks
                            max_score = max(chunk["relevance_score"]
                                            for chunk in chunks)

                            formatted_results.append({
                                "document": combined_text,
                                "metadata": combined_metadata,
                                "relevance_score": max_score
                            })
                        else:
                            # Single chunk, add as is
                            formatted_results.append(chunks[0])

            # Sort by relevance score and limit to top_k
            formatted_results.sort(
                key=lambda x: x["relevance_score"], reverse=True)
            formatted_results = formatted_results[:top_k]

            logger.debug(
                f"Retrieved {len(formatted_results)} memories from {collection} collection")
            return formatted_results

        except Exception as e:
            logger.error(
                f"Failed to retrieve memories from {collection} collection: {e}", exc_info=True)
            raise

    async def retrieve_with_metadata(
        self,
        query: str,
        collection: MemoryType,
        metadata_filter: Dict[str, Any],
        top_k: int = 4,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with metadata filtering from the specified collection.

        Args:
            query: Search query
            collection: Collection to query (MemoryType.EPHEMERAL or MemoryType.MODEL_MEMORY)
            metadata_filter: Filter conditions for metadata
            top_k: Number of results to return
            score_threshold: Minimum relevance score (0-1)

        Returns:
            List of memory dictionaries with content and metadata
        """
        try:
            if not self.ephemeral_collection or not self.model_memory_collection:
                await self.initialize()

            # Validate collection
            if collection not in [MemoryType.EPHEMERAL, MemoryType.MODEL_MEMORY]:
                raise ValueError("Invalid collection specified")

            # Get target collection
            target_collection = self.ephemeral_collection if collection == MemoryType.EPHEMERAL else self.model_memory_collection

            # Generate query embedding
            query_embedding = self.embeddings.encode(query).tolist()

            # Build combined metadata filter
            combined_filter = {**metadata_filter}

            # Query collection with combined filter
            results = target_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=combined_filter,
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

    async def add_conversation_summary(
        self,
        summary: str,
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Add a conversation summary to the ephemeral collection.

        Args:
            summary: The summary text to store
            conversation_id: ID of the conversation being summarized
            metadata: Optional additional metadata

        Returns:
            Memory ID if successful, None if duplicate
        """
        try:
            if not summary or not isinstance(summary, str):
                raise ValueError("Summary must be a non-empty string")

            # Prepare metadata
            full_metadata = {
                "type": "conversation_summary",  # This will force it to ephemeral
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }

            # Add to ephemeral collection
            return await self.add_memory(
                text=summary,
                collection=MemoryType.EPHEMERAL,  # Use enum value
                metadata=full_metadata,
                max_chunk_tokens=config.MAX_CHUNK_TOKENS  # Use consistent chunk size
            )
        except Exception as e:
            logger.error(
                f"Failed to add conversation summary: {e}", exc_info=True)
            raise

    async def get_conversation_summaries(
        self,
        conversation_id: str,
        top_k: int = 5,
        score_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Retrieve summaries for a specific conversation.

        Args:
            conversation_id: ID of the conversation to retrieve summaries for
            top_k: Number of summaries to return
            score_threshold: Minimum relevance score (0-1)

        Returns:
            List of summary dictionaries with content and metadata
        """
        try:
            return await self.retrieve_with_metadata(
                query="conversation summary",
                collection=MemoryType.MODEL_MEMORY,
                metadata_filter={
                    "type": "conversation_summary",
                    "conversation_id": conversation_id
                },
                top_k=top_k,
                score_threshold=score_threshold
            )
        except Exception as e:
            logger.error(
                f"Failed to get conversation summaries: {e}", exc_info=True)
            raise

    async def update_memory(
        self,
        memory_id: str,
        collection: MemoryType,
        new_text: str,
        new_metadata: Optional[Dict[str, Any]] = None
    ):
        """Update an existing memory entry in the specified collection.

        Args:
            memory_id: ID of the memory to update
            collection: Collection containing the memory (MemoryType.EPHEMERAL or MemoryType.MODEL_MEMORY)
            new_text: New text content
            new_metadata: Optional new metadata
        """
        try:
            if not self.ephemeral_collection or not self.model_memory_collection:
                await self.initialize()

            # Validate collection
            if collection not in [MemoryType.EPHEMERAL, MemoryType.MODEL_MEMORY]:
                raise ValueError("Invalid collection specified")

            # Get target collection
            target_collection = self.ephemeral_collection if collection == MemoryType.EPHEMERAL else self.model_memory_collection

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
            target_collection.update(
                ids=[memory_id],
                embeddings=[new_embedding],
                documents=[sanitized_text],
                metadatas=[full_metadata]
            )
            logger.debug(f"Updated memory with ID: {memory_id}")

        except Exception as e:
            logger.error(f"Failed to update memory: {e}", exc_info=True)
            raise

    async def delete_memory(
        self,
        memory_id: str,
        collection: MemoryType
    ):
        """Delete a memory entry from the specified collection.

        Args:
            memory_id: ID of the memory to delete
            collection: Collection containing the memory (MemoryType.EPHEMERAL or MemoryType.MODEL_MEMORY)
        """
        try:
            if not self.ephemeral_collection or not self.model_memory_collection:
                await self.initialize()

            # Validate collection
            if collection not in [MemoryType.EPHEMERAL, MemoryType.MODEL_MEMORY]:
                raise ValueError("Invalid collection specified")

            # Get target collection
            target_collection = self.ephemeral_collection if collection == MemoryType.EPHEMERAL else self.model_memory_collection

            target_collection.delete(ids=[memory_id])
            logger.debug(
                f"Deleted memory with ID: {memory_id} from {collection} collection")

        except Exception as e:
            logger.error(f"Failed to delete memory: {e}", exc_info=True)
            raise

    async def clear_collection(self, collection: MemoryType):
        """Clear all memories from the specified collection.

        Args:
            collection: Collection to clear (MemoryType.EPHEMERAL or MemoryType.MODEL_MEMORY)
        """
        try:
            if not self.ephemeral_collection or not self.model_memory_collection:
                await self.initialize()

            # Validate collection
            if collection not in [MemoryType.EPHEMERAL, MemoryType.MODEL_MEMORY]:
                raise ValueError("Invalid collection specified")

            # Get target collection
            target_collection = self.ephemeral_collection if collection == MemoryType.EPHEMERAL else self.model_memory_collection

            target_collection.delete(where={})
            logger.info(f"Cleared all memories from {collection} collection")
        except Exception as e:
            logger.error(
                f"Failed to clear {collection} collection: {e}", exc_info=True)
            raise
