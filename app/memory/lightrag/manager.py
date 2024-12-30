"""Enhanced memory manager combining LightRAG and relational database."""

import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union, Any, Tuple
from pathlib import Path
from enum import Enum
import json
import re

from lightrag import QueryParam
from sentence_transformers import SentenceTransformer
from .config import (
    LIGHTRAG_DATA_DIR,
    ENTITY_TYPE_KEYWORDS,
    EMBEDDING_SIMILARITY_THRESHOLD,
    RELATIONSHIP_TYPE_MAP,
    ENTITY_CONFIDENCE,
)
from .datastore import MemoryDatastore
from .ingestion import MemoryIngestor
from .tasks import MemoryTasks
from .manager_base import LightRAGManager
from app.models.memory import MemoryType

logger = logging.getLogger(__name__)


class MemoryAccessLevel(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class MemoryHierarchyLevel(Enum):
    CHUNK = "chunk"
    DOCUMENT = "document"
    COLLECTION = "collection"
    PROJECT = "project"


class EnhancedLightRAGManager(LightRAGManager):
    """
    Enhanced memory manager combining:
    1) LightRAG for unstructured, chunk-based retrieval
    2) Relational database for entities, relationships, and metadata
    3) Background tasks for memory maintenance and optimization
    4) Hierarchical memory organization
    5) Access control and retention policies
    """

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
        self._embedder = None

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

            # Initialize embedder
            try:
                self._embedder = SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2')
                logger.info(
                    "Using sentence-transformers for entity extraction")
            except ImportError:
                from lightrag.llm import ollama_embedding
                self._embedder = lambda texts: ollama_embedding(
                    texts, embed_model="nomic-embed-text")
                logger.info("Using Ollama embeddings for entity extraction")

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

    # Memory organization and access control
    def create_memory_hierarchy(self, name: str, level: MemoryHierarchyLevel, parent_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Create a new memory hierarchy level with metadata."""
        entity_id = str(uuid.uuid4())
        self.datastore.create_entity(
            entity_id,
            name,
            level.value,
            f"Memory hierarchy level: {level.value}"
        )

        # Set hierarchy level and parent
        self.datastore.set_entity_hierarchy(entity_id, level.value, parent_id)

        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                self.datastore.add_entity_metadata(entity_id, key, value)

        if parent_id:
            self.create_relationship(
                parent_id,
                entity_id,
                "contains",
                confidence=1.0
            )

        logger.debug(
            f"Created memory hierarchy {name} ({level.value}) with {len(metadata or {})} metadata items")
        return entity_id

    def get_memory_hierarchy(self, entity_id: str) -> Dict:
        """Get the full hierarchy information for an entity."""
        hierarchy = self.datastore.get_entity_hierarchy(entity_id)
        if not hierarchy:
            return {}

        result = {
            "entity": self.datastore.get_entity(entity_id),
            "hierarchy_level": hierarchy["hierarchy_level"],
            "parent_id": hierarchy["parent_id"],
            "children": self.datastore.get_child_entities(entity_id),
            "metadata": {}
        }

        # Get metadata if available
        metadata = self.datastore.get_entity_metadata(entity_id)
        if metadata:
            result["metadata"] = metadata

        return result

    def check_access(self, entity_id: str, user_id: str, required_level: MemoryAccessLevel) -> bool:
        """Check if user has required access level for an entity."""
        access_controls = self.datastore.get_access_controls(entity_id)
        for control in access_controls:
            if control["granted_to"] == user_id:
                if MemoryAccessLevel(control["access_level"]) >= required_level:
                    return True
        return False

    def grant_access(self, entity_id: str, user_id: str, level: MemoryAccessLevel):
        """Grant access to an entity for a user."""
        self.datastore.add_access_control(entity_id, level.value, user_id)
        logger.debug(
            f"Granted {level.value} access to {user_id} for entity {entity_id}")

    # Enhanced memory operations
    async def chunk_exists(self, chunk_hash: str) -> bool:
        """Check if a chunk with the given hash already exists in memory.

        Args:
            chunk_hash: The hash of the chunk to check

        Returns:
            bool: True if chunk exists, False otherwise
        """
        existing = self.datastore.search_entities(chunk_hash, limit=1)
        return bool(existing)

    async def insert_entity(self, metadata: Dict) -> str:
        """Insert entity metadata into the memory system without storing text content.

        Args:
            metadata: Dictionary containing entity metadata including:
                - entity_id: Unique identifier for the entity
                - hierarchy_level: The level in the memory hierarchy
                - parent_id: Optional parent entity ID
                - Other metadata fields as needed

        Returns:
            str: The entity ID
        """
        if not metadata.get('entity_id'):
            metadata['entity_id'] = str(uuid.uuid4())

        # Validate required fields
        if not metadata.get('hierarchy_level'):
            raise ValueError("hierarchy_level is required")

        # Store entity in datastore
        self.datastore.create_entity(
            metadata['entity_id'],
            metadata.get('title', 'Untitled Entity'),
            metadata['hierarchy_level'],
            metadata.get('description', '')
        )

        # Add metadata fields
        for key, value in metadata.items():
            if key not in ['entity_id', 'hierarchy_level', 'title', 'description']:
                self.datastore.add_entity_metadata(
                    metadata['entity_id'], key, value)

        # Handle parent relationship if specified
        if metadata.get('parent_id'):
            self.create_relationship(
                metadata['parent_id'],
                metadata['entity_id'],
                'contains',
                confidence=1.0
            )

        logger.debug(
            f"Inserted entity {metadata['entity_id']} with {len(metadata)} metadata items")
        return metadata['entity_id']

    async def insert_text(self, text: str, metadata: Optional[Dict] = None, user_id: Optional[str] = None):
        """Add unstructured text to memory with access control and deduplication."""
        if not text.strip():
            return

        # Check access if user_id is provided
        if user_id and not self.check_access("global", user_id, MemoryAccessLevel.WRITE):
            raise PermissionError(f"User {user_id} does not have write access")

        # Generate content hash for deduplication
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

        # Check for existing content
        if await self.chunk_exists(content_hash):
            logger.debug(
                f"Skipping duplicate content with hash {content_hash}")
            return content_hash

        # Prepare metadata
        full_metadata = metadata or {}
        full_metadata.update({
            "content_hash": content_hash,
            "timestamp": datetime.now().isoformat()
        })

        # Insert into RAG system
        logger.debug(f"Inserting text (size={len(text)}) into memory")
        await self.rag.insert(text)

        # Ingest with metadata
        if full_metadata:
            await self.ingestor.ingest_text(text, full_metadata)

        # Return the content hash as ID
        return content_hash

    async def query_memory(self, query_str: str, user_id: str, mode: str = "mix", top_k: int = 5) -> Dict:
        """
        Enhanced memory query with:
        - Access control
        - Hybrid search (vector + keyword)
        - Relational context
        - Chunk-based document reconstruction
        """
        # Check read access
        if not self.check_access("global", user_id, MemoryAccessLevel.READ):
            raise PermissionError(f"User {user_id} does not have read access")

        # Step 1: Named Entity Extraction
        extracted_entities = await self.extract_entities(query_str)

        # Step 2: Link Entities
        linked_ids = []
        for ent in extracted_entities:
            entity_id = self.link_entity(ent["text"], ent["label"])
            linked_ids.append(entity_id)

            # Step 3: Infer and Create Relationships
            relation_type = self.interpret_relation(ent["label"], query_str)
            if relation_type:
                self.create_relationship(
                    user_id,
                    entity_id,
                    relation_type,
                    confidence=1.0
                )

        # Step 4: Hybrid Search with chunk handling
        initial_results = await self.rag.query(query_str, param=QueryParam(mode=mode, top_k=top_k * 3))

        # Process results to handle chunks and reconstruct documents
        processed_results = []
        seen_docs = set()

        if isinstance(initial_results, dict) and 'matches' in initial_results:
            for match in initial_results['matches']:
                metadata = match.get('metadata', {})

                # Skip if this is a chunk and we've already processed its parent document
                if metadata.get('is_chunk') == 'true':
                    parent_doc = metadata.get('parent_doc')
                    if parent_doc and parent_doc in seen_docs:
                        continue
                    seen_docs.add(parent_doc)

                processed_results.append(match)

                # Stop when we have enough unique results
                if len(processed_results) >= top_k:
                    break

        # Step 5: Fetch Relational Information
        relational_info = []
        for eid in linked_ids:
            rels = self.get_entity_relations(eid)
            entity = self.datastore.get_entity(eid)
            relational_info.append({
                "entity_id": eid,
                "entity": entity,
                "relations": rels
            })

        # Step 6: Build System Prompt with chunk context
        system_prompt = self.build_system_prompt(relational_info)

        return {
            "results": processed_results[:top_k],
            "relational_context": relational_info,
            "system_prompt": system_prompt
        }

    # Memory maintenance and optimization
    async def apply_retention_policy(self, policy_name: str = "default"):
        """Apply retention policy to memory content."""
        cutoff = datetime.now() - timedelta(days=30)  # Default 30-day retention
        if policy_name == "short_term":
            cutoff = datetime.now() - timedelta(days=7)
        elif policy_name == "long_term":
            cutoff = datetime.now() - timedelta(days=365)

        deleted_count = self.datastore.cleanup_old_entities(cutoff)
        logger.info(
            f"Applied retention policy {policy_name}, deleted {deleted_count} entities")

    async def optimize_memory(self) -> Dict:
        """Optimize memory storage and retrieval performance."""
        # Run standard optimization
        result = await self.tasks.optimize_memory()

        # Additional optimization steps
        await self.apply_retention_policy()
        await self.tasks.reindex_memory()

        return result

    # Entity and relationship management
    def create_entity(self, name: str, entity_type: str, description: str = "", retention_policy: str = "default") -> str:
        """Create a new entity with retention policy."""
        entity_id = str(uuid.uuid4())
        self.datastore.create_entity(
            entity_id,
            name,
            entity_type,
            description,
            retention_policy
        )
        logger.debug(
            f"Created entity {name} ({entity_type}) with retention policy {retention_policy}")
        return entity_id

    def create_relationship(self, src_id: str, dst_id: str, relation_type: str, confidence: float = 1.0):
        """Create a relationship between two entities with version tracking."""
        rel_id = str(uuid.uuid4())
        self.datastore.create_relationship(
            rel_id,
            src_id,
            dst_id,
            relation_type,
            confidence
        )
        logger.debug(
            f"Created versioned relationship {rel_id}: {src_id} --{relation_type}--> {dst_id}")

    def get_entity_relations(self, entity_id: str) -> List[Dict]:
        """Get all relationships for a given entity with version history."""
        return self.datastore.get_entity_relations(entity_id)

    # Helper methods
    async def extract_entities(self, text):
        """Extract entities from text using multiple techniques."""
        if not self._initialized:
            await self.initialize()

        # Initialize results
        entities = []
        seen_spans = set()

        # Technique 1: Embedding-based similarity (original approach)
        try:
            # Split text into potential entity spans
            words = text.split()
            spans = []

            # Create spans of 1-5 words for potential entities
            for i in range(len(words)):
                for j in range(1, 5 + 1):  # Increased max span size
                    if i + j <= len(words):
                        span = ' '.join(words[i:i+j])
                        spans.append(span)

            if spans:
                # Get embeddings for spans
                if isinstance(self._embedder, SentenceTransformer):
                    span_embeddings = self._embedder.encode(spans)
                else:
                    span_embeddings = await self._embedder(spans)

                # Get embeddings for type keywords
                type_embeddings = {}
                for etype, keywords in ENTITY_TYPE_KEYWORDS.items():
                    if isinstance(self._embedder, SentenceTransformer):
                        type_embeddings[etype] = self._embedder.encode(
                            keywords)
                    else:
                        type_embeddings[etype] = await self._embedder(keywords)

                # Find entities by comparing embeddings
                for i, span in enumerate(spans):
                    if span.lower() in seen_spans:
                        continue

                    # Compare with type keywords
                    for etype, type_embs in type_embeddings.items():
                        similarity = span_embeddings[i] @ type_embs.T
                        if max(similarity) > EMBEDDING_SIMILARITY_THRESHOLD:
                            seen_spans.add(span.lower())
                            entity = self._create_entity(
                                span, etype, text, similarity)
                            entities.append(entity)
                            break
        except Exception as e:
            logger.warning(f"Embedding-based entity extraction failed: {e}")

        # Technique 2: Pattern-based extraction
        try:
            # Extract dates, numbers, and other patterns
            date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
            number_pattern = r'\b\d+\b'
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

            for pattern, label in [(date_pattern, 'DATE'),
                                   (number_pattern, 'NUMBER'),
                                   (email_pattern, 'EMAIL')]:
                matches = re.finditer(pattern, text)
                for match in matches:
                    span = match.group()
                    if span.lower() not in seen_spans:
                        seen_spans.add(span.lower())
                        entity = self._create_entity(span, label, text, 1.0)
                        entities.append(entity)
        except Exception as e:
            logger.warning(f"Pattern-based entity extraction failed: {e}")

        # Technique 3: Keyword-based extraction
        try:
            # Look for specific keywords/phrases
            keywords = {
                'birthday': 'EVENT',
                'anniversary': 'EVENT',
                'meeting': 'EVENT',
                'deadline': 'DATE'
            }

            for keyword, label in keywords.items():
                if keyword in text.lower():
                    # Find all occurrences
                    matches = re.finditer(
                        r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE)
                    for match in matches:
                        span = match.group()
                        if span.lower() not in seen_spans:
                            seen_spans.add(span.lower())
                            entity = self._create_entity(
                                span, label, text, 1.0)
                            entities.append(entity)
        except Exception as e:
            logger.warning(f"Keyword-based entity extraction failed: {e}")

        return entities

    def _create_entity(self, span, label, text, confidence):
        """Helper method to create entity dictionary."""
        span_start = text.find(span)
        context_start = max(0, span_start - 50)
        context_end = min(len(text), span_start + len(span) + 50)
        context = text[context_start:context_end]

        # Get relation type and confidence
        relation_type, relation_confidence = self.interpret_relation(
            label, context)

        return {
            "text": span,
            "label": label,
            "start": span_start,
            "end": span_start + len(span),
            "context": context,
            "metadata": {
                "source": "multi_technique",
                "confidence": float(confidence),
                "relation_type": relation_type,
                "relation_confidence": relation_confidence
            }
        }

    def build_system_prompt(self, relational_info: List[Dict]) -> str:
        """Build a system prompt string from relational information."""
        prompt = "You are an AI with an enhanced memory system. Here are some relationships and entities you should be aware of:\n\n"
        for info in relational_info:
            entity = info["entity"]
            if entity:
                prompt += f"- {entity['name']} ({entity['entity_type']}):\n"
                for rel in info["relations"]:
                    dst_entity = self.datastore.get_entity(rel["dst_entity"])
                    if dst_entity:
                        prompt += f"  - {rel['relation_type']} -> {dst_entity['name']}\n"
        prompt += "\nUse this information to provide contextual and accurate responses."
        return prompt

    def link_entity(self, text: str, label: str) -> str:
        """Link or create an entity based on text and label.

        Args:
            text: The entity text
            label: The entity type/label from spaCy

        Returns:
            str: The entity ID
        """
        # First try to find existing entity
        existing = self.datastore.search_entities(text)
        for entity in existing:
            if entity["name"].lower() == text.lower() and entity["entity_type"] == label:
                return entity["id"]

        # Create new entity if not found
        entity_id = str(uuid.uuid4())
        self.datastore.create_entity(
            entity_id,
            text,
            label,
            f"Entity extracted from text: {text} ({label})"
        )
        return entity_id

    def interpret_relation(self, entity_label: str, context: str) -> Tuple[Optional[str], float]:
        """Interpret the relationship type based on entity label and context."""
        # Use configured relationship type map
        relation_type = RELATIONSHIP_TYPE_MAP.get(entity_label)
        if relation_type:
            return relation_type, ENTITY_CONFIDENCE["MEDIUM"]

        return None, ENTITY_CONFIDENCE["LOW"]

    def _sanitize_id(self, text: str) -> str:
        """Sanitize text for use as an ID in GraphML."""
        # Remove quotes and spaces
        sanitized = text.replace('"', '').replace(' ', '_')
        # Replace other potentially problematic characters
        sanitized = sanitized.replace('&', 'and').replace(
            '<', 'lt').replace('>', 'gt')
        # Ensure valid NMTOKEN format
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c in '_-.')
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'n' + sanitized
        return sanitized

    async def add_memory(
        self,
        text: str,
        collection: MemoryType = MemoryType.EPHEMERAL,
        metadata: Optional[Dict] = None,
        max_chunk_tokens: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> str:
        """Add a new memory with background entity extraction and relationship tracking."""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

            # Check for existing memory with same content
            existing = self.datastore.search_entities(content_hash, limit=1)
            if existing:
                logger.info(
                    f"Found existing memory for content hash {content_hash}")
                return existing[0]["id"]

            # Generate memory ID
            memory_id = str(uuid.uuid4())

            # Format metadata
            full_metadata = metadata or {}
            full_metadata.update({
                "timestamp": datetime.now().isoformat(),
                "collection": collection.value,
                "memory_id": memory_id,
                "content_hash": content_hash
            })

            # Queue for processing - this will handle both storage and entity extraction
            await self.tasks.queue_memory_processing(text, full_metadata)
            logger.info(f"Queued memory {memory_id} for processing")

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
            param.mode = "hybrid"  # Use hybrid search mode
            param.k = top_k * 3  # Get more results for deduplication

            # Add collection to metadata filter if specified
            if collection:
                if metadata_filter is None:
                    metadata_filter = {}
                metadata_filter["collection"] = collection.value
            if metadata_filter:
                param.filter = metadata_filter

            # Extract entities from query for better matching
            query_entities = await self.extract_entities(query)
            enhanced_query = query
            for entity in query_entities:
                if entity.get("relation_type"):
                    enhanced_query += f" {entity['relation_type']} {entity['text']}"

            # Query using the enhanced query
            result = await self.query(enhanced_query, param)
            matches = result.get("matches", [])

            # Process matches to extract metadata and content
            processed_matches = []
            seen_hashes = set()

            for match in matches:
                content = match.get("content", "")
                try:
                    # Split into metadata and content sections
                    parts = content.split("\nCONTENT: ", 1)
                    if len(parts) == 2:
                        metadata_str = parts[0].replace("METADATA: ", "")
                        try:
                            metadata = json.loads(metadata_str)
                            actual_content = parts[1]

                            # Clean up any potential JSON formatting issues
                            if isinstance(metadata, str):
                                metadata = json.loads(metadata)

                            # Calculate content hash for deduplication
                            content_hash = hashlib.md5(
                                actual_content.encode('utf-8')).hexdigest()
                            if content_hash in seen_hashes:
                                continue
                            seen_hashes.add(content_hash)

                            # Extract entities from content for better relevance scoring
                            content_entities = await self.extract_entities(actual_content)
                            relevance_score = match.get("score", 0.0)

                            # Calculate semantic similarity using embeddings
                            if isinstance(self._embedder, SentenceTransformer):
                                query_embedding = self._embedder.encode([query])[
                                    0]
                                content_embedding = self._embedder.encode([actual_content])[
                                    0]
                                semantic_similarity = query_embedding @ content_embedding.T
                                relevance_score *= (1 + semantic_similarity)

                            # Boost score based on entity matches
                            for q_entity in query_entities:
                                for c_entity in content_entities:
                                    if (q_entity["label"] == c_entity["label"] and
                                            q_entity.get("relation_type") == c_entity.get("relation_type")):
                                        relevance_score *= 1.5

                            processed_matches.append({
                                "content": actual_content,
                                "metadata": metadata,
                                "score": relevance_score,
                                "entities": content_entities,
                                "hash": content_hash
                            })
                        except json.JSONDecodeError as je:
                            logger.warning(
                                f"Failed to parse metadata JSON: {je}")
                            # Try to salvage the content even if metadata parsing fails
                            processed_matches.append({
                                "content": content,
                                "metadata": {},
                                "score": match.get("score", 0.0)
                            })
                except Exception as e:
                    logger.warning(f"Failed to process match: {e}")
                    continue

            # Sort by score and apply top_k
            processed_matches.sort(key=lambda x: x["score"], reverse=True)
            return processed_matches[:top_k]
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}", exc_info=True)
            return []

    async def retrieve_text(self, chunk_id: str) -> str:
        """Retrieve text content for a specific chunk.

        Args:
            chunk_id: The ID of the chunk to retrieve

        Returns:
            str: The retrieved text content
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Query the RAG system for this specific chunk
            result = await self.rag.aquery(f"id:{chunk_id}", QueryParam(mode="exact", top_k=1))

            if isinstance(result, str):
                return result

            if isinstance(result, dict) and "matches" in result:
                if result["matches"]:
                    return result["matches"][0].get("content", "")

            return ""
        except Exception as e:
            logger.error(
                f"Failed to retrieve text for chunk {chunk_id}: {str(e)}")
            return ""

    async def query(self, query: str, param: QueryParam) -> Dict:
        """Query the memory system with enhanced filtering, cache validation, and response caching."""
        try:
            if not self._initialized:
                await self.initialize()

            # Generate cache key
            cache_key = hashlib.md5(
                (query + str(param)).encode('utf-8')).hexdigest()

            # Check cache first
            cached = self.datastore.get_cache_entry(cache_key)
            if cached:
                logger.debug(f"Returning cached response for query: {query}")
                return cached

            # Get metadata filter and collection
            metadata_filter = getattr(param, "filter", {}) or {}
            collection = metadata_filter.get("collection")

            # Enhance query with semantic context
            query_entities = await self.extract_entities(query)
            enhanced_query = query
            for entity in query_entities:
                if entity.get("relation_type"):
                    enhanced_query += f" {entity['relation_type']} {entity['text']}"

            # Query the base system with enhanced query
            result = await self.rag.aquery(enhanced_query, param)

            # Validate and clean the result
            def validate_match(match: Dict) -> Optional[Dict]:
                """Validate and clean a single match result."""
                try:
                    content = match.get("content", "")
                    if not content or not isinstance(content, str):
                        return None

                    # Extract and validate metadata
                    metadata = {}
                    if "\nCONTENT: " in content:
                        meta_part, content_part = content.split(
                            "\nCONTENT: ", 1)
                        if meta_part.startswith("METADATA: "):
                            try:
                                metadata = json.loads(
                                    meta_part[len("METADATA: "):])
                                if isinstance(metadata, str):
                                    metadata = json.loads(metadata)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Invalid metadata format, using empty metadata")
                                metadata = {}
                        content = content_part

                    # Validate content
                    if not content.strip():
                        return None

                    # Apply collection filter
                    if collection and metadata.get("collection") != collection:
                        return None

                    # Apply metadata filters
                    if metadata_filter:
                        for k, v in metadata_filter.items():
                            if k != "collection" and metadata.get(k) != v:
                                return None

                    # Calculate semantic similarity with query
                    if isinstance(self._embedder, SentenceTransformer):
                        query_embedding = self._embedder.encode([query])[0]
                        content_embedding = self._embedder.encode([content])[0]
                        semantic_similarity = query_embedding @ content_embedding.T
                    else:
                        semantic_similarity = 1.0

                    return {
                        "content": content,
                        "metadata": metadata,
                        "score": float(match.get("score", 1.0)) * (1 + semantic_similarity)
                    }
                except Exception as e:
                    logger.warning(f"Failed to validate match: {e}")
                    return None

            # Process results based on type
            final_result = {"matches": []}

            if isinstance(result, str):
                if not result.strip() or "ERROR:" in result:
                    final_result = {"matches": []}
                else:
                    final_result = {
                        "matches": [{
                            "content": result,
                            "metadata": metadata_filter,
                            "score": 1.0
                        }]
                    }

            elif isinstance(result, dict) and "matches" in result:
                # Validate and clean all matches
                valid_matches = []
                for match in result["matches"]:
                    validated = validate_match(match)
                    if validated:
                        valid_matches.append(validated)

                # Clean cache if too many invalid matches
                if len(valid_matches) < len(result["matches"]) * 0.5:
                    logger.info(
                        "Cleaning cache due to high invalid match rate")
                    await self.rag.clean_cache()

                # Sort matches by combined score
                valid_matches.sort(key=lambda x: x["score"], reverse=True)
                final_result = {"matches": valid_matches}

            # Cache the final result if it's valid
            if final_result["matches"]:
                self.datastore.set_cache_entry(
                    cache_key,
                    final_result,
                    expiration=timedelta(hours=1)  # Cache for 1 hour
                )

            return final_result

        except Exception as e:
            logger.error(
                f"Error querying memory system: {str(e)}", exc_info=True)
            return {"matches": []}

    def format_entity(self, entity: str) -> str:
        """Format entity ID to be valid for storage."""
        # Remove quotes and spaces
        entity = entity.strip('"').strip()
        # Replace spaces and special chars with underscores
        entity = re.sub(r'[^a-zA-Z0-9]', '_', entity)
        # Ensure it starts with a letter
        if not entity[0].isalpha():
            entity = 'e_' + entity
        return entity

    def format_relationship(self, rel: Dict[str, Any]) -> Dict[str, Any]:
        """Format relationship for storage."""
        return {
            "source": self.format_entity(rel["source"]),
            "target": self.format_entity(rel["target"]),
            "type": rel.get("type", "default"),
            "weight": float(rel.get("weight", 1.0)),
            "metadata": rel.get("metadata", {}),
            "source_id": rel.get("source_id") or str(uuid.uuid4())
        }

    async def extract_entities_and_relationships(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract entities and relationships from text using configured rules."""
        try:
            if not self._initialized:
                await self.initialize()

            # Extract entities using embeddings and keywords
            entities = await self.extract_entities(text)

            # Build relationships between entities
            relationships = []
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i == j:
                        continue

                    # Determine relationship type based on entity types and context
                    rel_type = self.interpret_relation(
                        entity1["label"],
                        f"{entity1['text']} {entity2['text']}"
                    )[0]

                    if rel_type:
                        relationships.append({
                            "source": entity1["text"],
                            "target": entity2["text"],
                            "type": rel_type,
                            "weight": 1.0,
                            "metadata": {
                                "source_entity": entity1,
                                "target_entity": entity2,
                                "context": text
                            }
                        })

            # Format entities for storage
            formatted_entities = []
            for entity in entities:
                formatted_entities.append({
                    "id": self.format_entity(entity["text"]),
                    "type": entity["label"],
                    "metadata": {
                        "original_text": entity["text"],
                        "context": entity["context"],
                        "confidence": entity["metadata"]["confidence"]
                    }
                })

            # Format relationships for storage
            formatted_relationships = [
                self.format_relationship(rel) for rel in relationships
            ]

            return formatted_entities, formatted_relationships

        except Exception as e:
            logger.error(f"Error extracting entities and relationships: {e}")
            return [], []
