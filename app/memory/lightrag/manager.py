"""Enhanced memory manager combining LightRAG and relational database."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union
from pathlib import Path
from enum import Enum
import json

from lightrag import QueryParam
from sentence_transformers import SentenceTransformer
from .config import (
    LIGHTRAG_DATA_DIR,
    ENTITY_SPAN_MAX_WORDS,
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

    def __init__(self, working_dir: Union[str, Path] = LIGHTRAG_DATA_DIR):
        """Initialize the enhanced memory manager."""
        # Convert to Path and ensure it exists
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initializing enhanced LightRAGManager with working directory: {working_dir}")
        super().__init__(str(working_dir))

        # Initialize components
        self.datastore = None
        self.ingestor = None
        self.tasks = None
        self._initialized = False
        self._embedder = None

    async def initialize(self):
        """Initialize the memory system and all components."""
        if self._initialized:
            return

        try:
            # First initialize our datastore
            self.datastore = MemoryDatastore()

            # Initialize base class with our datastore
            await super().initialize(self.datastore)

            # Initialize remaining components
            self.ingestor = MemoryIngestor(self, self.datastore)
            self.tasks = MemoryTasks(self, self.datastore)

            # Initialize embedder
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2')
                logger.info(
                    "Using sentence-transformers for entity extraction")
            except ImportError:
                from lightrag.llm import ollama_embedding
                self._embedder = lambda texts: ollama_embedding(
                    texts, embed_model="nomic-embed-text")
                logger.info("Using Ollama embeddings for entity extraction")

            # Start background tasks
            await self.start()

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
    async def insert_text(self, text: str, metadata: Optional[Dict] = None, user_id: Optional[str] = None):
        """Add unstructured text to memory with access control."""
        if not text.strip():
            return

        # Check access if user_id is provided
        if user_id and not self.check_access("global", user_id, MemoryAccessLevel.WRITE):
            raise PermissionError(f"User {user_id} does not have write access")

        logger.debug(f"Inserting text (size={len(text)}) into memory")
        await self.rag.insert(text)

        if metadata:
            await self.ingestor.ingest_text(text, metadata)

    async def query_memory(self, query_str: str, user_id: str, mode: str = "mix", top_k: int = 5) -> Dict:
        """
        Enhanced memory query with:
        - Access control
        - Hybrid search (vector + keyword)
        - Relational context
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

        # Step 4: Hybrid Search
        rag_text = await self.rag.query(query_str, param=QueryParam(mode=mode, top_k=top_k))
        keyword_results = self.datastore.search_entities(
            query_str, limit=top_k)

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

        # Step 6: Build System Prompt
        system_prompt = self.build_system_prompt(relational_info)

        return {
            "rag_text": rag_text,
            "keyword_results": keyword_results,
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
        """Extract entities from text using embeddings-based similarity."""
        if not self._initialized:
            await self.initialize()

        # Split text into potential entity spans
        words = text.split()
        spans = []

        # Create spans of 1-3 words for potential entities
        for i in range(len(words)):
            for j in range(1, ENTITY_SPAN_MAX_WORDS + 1):
                if i + j <= len(words):
                    span = ' '.join(words[i:i+j])
                    spans.append(span)

        if not spans:
            return []

        # Get embeddings for spans
        if isinstance(self._embedder, SentenceTransformer):
            span_embeddings = self._embedder.encode(spans)
        else:
            span_embeddings = await self._embedder(spans)

        # Get embeddings for type keywords
        type_embeddings = {}
        for etype, keywords in ENTITY_TYPE_KEYWORDS.items():
            if isinstance(self._embedder, SentenceTransformer):
                type_embeddings[etype] = self._embedder.encode(keywords)
            else:
                type_embeddings[etype] = await self._embedder(keywords)

        entities = []
        seen_spans = set()

        # Find entities by comparing embeddings
        for i, span in enumerate(spans):
            if span.lower() in seen_spans:
                continue

            # Compare with type keywords
            for etype, type_embs in type_embeddings.items():
                similarity = span_embeddings[i] @ type_embs.T
                if max(similarity) > EMBEDDING_SIMILARITY_THRESHOLD:
                    seen_spans.add(span.lower())

                    # Find span context
                    span_start = text.find(span)
                    context_start = max(0, span_start - 50)
                    context_end = min(len(text), span_start + len(span) + 50)
                    context = text[context_start:context_end]

                    # Get relation type and confidence
                    relation_type, relation_confidence = self.interpret_relation(
                        etype, context)

                    entity = {
                        "text": span,
                        "label": etype,
                        "start": span_start,
                        "end": span_start + len(span),
                        "context": context,
                        "metadata": {
                            "source": "embedding_similarity",
                            "confidence": float(max(similarity)),
                            "relation_type": relation_type,
                            "relation_confidence": relation_confidence
                        }
                    }

                    entities.append(entity)
                    break

        return entities

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

    def interpret_relation(self, entity_label: str, context: str):
        """Interpret the relationship type based on entity label and context."""
        # Special handling for dates with context
        if entity_label == "DATE":
            context_lower = context.lower()
            if "birthday" in context_lower:
                return "birthday_date", ENTITY_CONFIDENCE["EXACT_MATCH"]
            elif any(word in context_lower for word in ["born", "birth"]):
                return "birth_date", ENTITY_CONFIDENCE["HIGH"]
            else:
                return "date_reference", ENTITY_CONFIDENCE["MEDIUM"]

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
            # Format memory text with metadata
            full_metadata = metadata or {}
            full_metadata.update({
                "timestamp": datetime.now().isoformat(),
                "collection": collection.value
            })

            memory_text = f"METADATA: {json.dumps(full_metadata)}\nCONTENT: {text}"

            # Store in LightRAG using ainsert for immediate text storage
            memory_id = str(uuid.uuid4())
            await self.rag.ainsert([memory_text])
            logger.info(f"Added memory text with ID: {memory_id}")

            # Queue the memory for background processing
            await self.tasks.queue_memory_processing(text, full_metadata)
            logger.info("Queued memory for background processing")

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
            param.k = top_k * 2  # Double the results to account for filtering

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

                            # Extract entities from content for better relevance scoring
                            content_entities = await self.extract_entities(actual_content)
                            relevance_score = match.get("score", 0.0)

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
                                "entities": content_entities
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

    async def query(self, query: str, param: QueryParam) -> Dict:
        """Query the memory system with enhanced filtering."""
        try:
            if not self._initialized:
                await self.initialize()

            # Get metadata filter and collection
            metadata_filter = getattr(param, "filter", {}) or {}
            collection = metadata_filter.get("collection")

            # Query the base system
            result = await self.rag.aquery(query, param)

            # Convert result to dict if it's a string
            if isinstance(result, str):
                # If result is empty or error, return empty matches
                if not result.strip() or "ERROR:" in result:
                    return {"matches": []}

                # Otherwise, create a single match from the result
                return {
                    "matches": [{
                        "content": result,
                        "metadata": metadata_filter,
                        "score": 1.0
                    }]
                }

            # Process matches to ensure metadata is properly handled
            if isinstance(result, dict) and "matches" in result:
                matches = []
                for match in result["matches"]:
                    try:
                        content = match.get("content", "")
                        if not content:
                            continue

                        # Extract metadata if present
                        parts = content.split("\nCONTENT: ", 1)
                        if len(parts) == 2:
                            metadata_str = parts[0].replace("METADATA: ", "")
                            try:
                                match_metadata = json.loads(metadata_str)
                                if isinstance(match_metadata, str):
                                    match_metadata = json.loads(match_metadata)

                                # Apply collection filter if specified
                                if collection and match_metadata.get("collection") != collection:
                                    continue

                                # Apply other metadata filters
                                if metadata_filter:
                                    matches_filter = True
                                    for k, v in metadata_filter.items():
                                        if k != "collection" and match_metadata.get(k) != v:
                                            matches_filter = False
                                            break
                                    if not matches_filter:
                                        continue

                                matches.append({
                                    # Use only the content part
                                    "content": parts[1],
                                    "metadata": match_metadata,
                                    "score": match.get("score", 1.0)
                                })
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Failed to parse metadata: {metadata_str}")
                                # Include the match even if metadata parsing fails
                                matches.append({
                                    "content": content,
                                    "metadata": {},
                                    "score": match.get("score", 1.0)
                                })
                    except Exception as e:
                        logger.warning(f"Failed to process match: {e}")
                        continue
                result["matches"] = matches

            return result

        except Exception as e:
            logger.error(
                f"Error querying memory system: {str(e)}", exc_info=True)
            return {"matches": []}
