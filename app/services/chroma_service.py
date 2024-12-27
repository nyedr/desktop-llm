import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import uuid

from app.core.config import config

logger = logging.getLogger(__name__)


class ChromaService:
    def __init__(self):
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
                config.CHROMA_EMBEDDING_MODEL)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info("Chroma Service initialized successfully")
            return self.collection

        except Exception as e:
            logger.error(f"Failed to initialize Chroma Service: {e}")
            raise

    async def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a single text entry to the vector store."""
        try:
            if not self.collection:
                await self.initialize()

            # Generate embedding
            embedding = self.embeddings.encode(text).tolist()

            # Generate a unique ID
            memory_id = str(uuid.uuid4())

            # Add to collection
            self.collection.add(
                ids=[memory_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata] if metadata else None
            )
            logger.info(f"Added memory with ID: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise

    async def add_memories_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add multiple text entries to the vector store in batch."""
        try:
            if not self.collection:
                await self.initialize()

            # Generate embeddings for all texts
            embeddings = [self.embeddings.encode(
                text).tolist() for text in texts]

            # Generate unique IDs for each text
            memory_ids = [str(uuid.uuid4()) for _ in texts]

            # Add to collection
            self.collection.add(
                ids=memory_ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Added {len(texts)} memories in batch")
            return memory_ids

        except Exception as e:
            logger.error(f"Failed to add memories in batch: {e}")
            raise

    async def retrieve_memories(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on a query."""
        try:
            if not self.collection:
                await self.initialize()

            # Generate query embedding
            query_embedding = self.embeddings.encode(query).tolist()

            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            memories = []
            for i in range(len(results["documents"][0])):
                memories.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                    "relevance_score": 1 - results["distances"][0][i]
                })

            logger.info(
                f"Retrieved {len(memories)} memories for query: '{query}'")
            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise

    async def retrieve_with_metadata(
        self,
        query: str,
        metadata_filter: Dict[str, Any],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with metadata filtering."""
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

            # Format results
            memories = []
            for i in range(len(results["documents"][0])):
                memories.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                    "relevance_score": 1 - results["distances"][0][i]
                })

            logger.info(
                f"Retrieved {len(memories)} memories for query: '{query}' "
                f"with metadata filter: {metadata_filter}"
            )
            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories with metadata: {e}")
            raise

    async def update_memory(self, memory_id: str, new_text: str, new_metadata: Optional[Dict[str, Any]] = None):
        """Update an existing memory by ID."""
        try:
            if not self.collection:
                await self.initialize()

            # Generate new embedding
            new_embedding = self.embeddings.encode(new_text).tolist()

            # Update in collection
            self.collection.update(
                ids=[memory_id],
                embeddings=[new_embedding],
                documents=[new_text],
                metadatas=[new_metadata] if new_metadata else None
            )
            logger.info(f"Updated memory with ID: {memory_id}")

        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            raise

    async def delete_memory(self, memory_id: str):
        """Delete a memory by ID."""
        try:
            if not self.collection:
                await self.initialize()

            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory with ID: {memory_id}")

        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            raise

    async def clear_collection(self):
        """Clear all memories from the collection."""
        try:
            if not self.collection:
                await self.initialize()

            # Get all IDs in the collection
            results = self.collection.get()
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
            logger.info("Cleared all memories from collection")

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
