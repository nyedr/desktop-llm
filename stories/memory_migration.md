# Memory System Migration Guide

## Overview

This guide outlines the step-by-step process to migrate from the current memory system to the new unified, context-aware memory architecture. The migration will be performed in phases to ensure system stability and data integrity.

## Current System Analysis

### Existing Components

1. ChromaService (`app/services/chroma_service.py`):

   - Uses ChromaDB with SQLite persistence
   - Maintains separate collections for ephemeral and model memory
   - Implements chunking, deduplication, and retrieval
   - Uses SentenceTransformer for embeddings
   - Current methods:
     ```python
     async def add_memory(text, collection, metadata, max_chunk_tokens)
     async def retrieve_memories(query, collection, top_k, score_threshold)
     async def update_memory(memory_id, collection, new_text, new_metadata)
     async def delete_memory(memory_id, collection)
     ```

2. LangChainService (`app/services/langchain_service.py`):

   - Integrates with ChromaService for vector operations
   - Handles conversation processing and summarization
   - Uses HuggingFaceEmbeddings and Ollama
   - Key functionalities:
     ```python
     async def query_memory(query, context)
     async def process_conversation(messages, memory_type)
     async def summarize_text(text)
     ```

3. Memory Models (`app/models/memory.py`):
   - Simple enum-based memory types:
     ```python
     class MemoryType(str, Enum):
         EPHEMERAL = "ephemeral"
         MODEL_MEMORY = "model_memory"
     ```

## Phase 1: Database and Dependencies Setup

### 1.1. Update Dependencies

Add to `requirements.txt`:

```
spacy>=3.7.2
psycopg2-binary>=2.9.9  # For PostgreSQL
sqlalchemy>=2.0.23      # For ORM
alembic>=1.12.1        # For database migrations
sentence-transformers>=2.2.2  # Keep existing
chromadb>=0.4.18       # Keep existing
langchain>=0.1.0       # Keep existing
```

### 1.2. Install spaCy Model

```bash
python -m spacy download en_core_web_lg
```

### 1.3. Create Database Migration Scripts

Location: `migrations/`

1. Create Initial Migration (`migrations/versions/001_initial.py`):

```python
"""Initial database setup

Revision ID: 001
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

def upgrade():
    # Create entities table
    op.create_table(
        'entities',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('text', sa.Text, nullable=False),
        sa.Column('attributes', JSONB, nullable=False),
        sa.Column('metadata', JSONB, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now())
    )

    # Create relationships table
    op.create_table(
        'relationships',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('source_entity_id', sa.String(36),
                 sa.ForeignKey('entities.id'), nullable=False),
        sa.Column('target_entity_id', sa.String(36),
                 sa.ForeignKey('entities.id'), nullable=False),
        sa.Column('relation_type', sa.String(50), nullable=False),
        sa.Column('metadata', JSONB, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now())
    )

    # Create indexes
    op.create_index('idx_entities_type', 'entities', ['type'])
    op.create_index('idx_relationships_source', 'relationships', ['source_entity_id'])
    op.create_index('idx_relationships_target', 'relationships', ['target_entity_id'])
    op.create_index('idx_relationships_type', 'relationships', ['relation_type'])

def downgrade():
    op.drop_table('relationships')
    op.drop_table('entities')
```

2. Create Configuration (`migrations/env.py`):

```python
from alembic import context
from sqlalchemy import engine_from_config, pool
from logging.config import fileConfig
import os

config = context.config
fileConfig(config.config_file_name)

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=None
        )

        with context.begin_transaction():
            context.run_migrations()
```

## Phase 2: Core Infrastructure Changes

### 2.1. Update Entity Models

1. Create Base Models (`app/models/entity_models.py`):

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4

class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    text: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "type": "person",
                "text": "John Doe",
                "attributes": {"age": 30, "occupation": "developer"},
                "metadata": {"source": "chat", "confidence": 0.95}
            }
        }

class Relationship(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "source_entity_id": "uuid1",
                "target_entity_id": "uuid2",
                "relation_type": "works_for",
                "metadata": {"confidence": 0.85}
            }
        }
```

2. Update Memory Models (`app/models/memory.py`):

```python
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class MemoryType(str, Enum):
    EPHEMERAL = "ephemeral"
    MODEL_MEMORY = "model_memory"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"

class Memory(BaseModel):
    id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any]
    entities: List[str] = []  # Entity IDs
    relationships: List[str] = []  # Relationship IDs
    created_at: datetime
    updated_at: Optional[datetime] = None

class MemoryQueryResult(BaseModel):
    memory: Memory
    score: float
    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
```

### 2.2. Implement Entity Extraction System

1. Create Base Extractor (`app/entities/extractor.py`):

```python
import spacy
from typing import List, Dict, Any, Tuple
from app.models.entity_models import Entity, Relationship

class EntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def extract_entities(self, text: str) -> List[Entity]:
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entity = Entity(
                type=ent.label_,
                text=ent.text,
                attributes={
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_
                },
                metadata={
                    "confidence": ent._.confidence
                    if hasattr(ent._, "confidence") else 1.0
                }
            )
            entities.append(entity)

        return entities

    def extract_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        doc = self.nlp(text)
        relationships = []

        # Entity ID mapping for quick lookup
        entity_map = {e.text: e.id for e in entities}

        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                if token.head.text in entity_map and token.text in entity_map:
                    relationship = Relationship(
                        source_entity_id=entity_map[token.head.text],
                        target_entity_id=entity_map[token.text],
                        relation_type=token.dep_,
                        metadata={
                            "confidence": 0.8,
                            "sentence": token.sent.text
                        }
                    )
                    relationships.append(relationship)

        return relationships
```

2. Create Relation Detection Rules (`app/entities/relations.py`):

```python
from typing import List, Dict, Any
from spacy.tokens import Doc, Token

class RelationDetector:
    @staticmethod
    def get_relation_patterns() -> List[Dict[str, Any]]:
        return [
            {
                "name": "works_for",
                "pattern": [
                    {"DEP": "nsubj"},
                    {"LEMMA": {"IN": ["work", "employ"]}},
                    {"DEP": "prep", "OP": "?"},
                    {"DEP": "pobj"}
                ]
            },
            # Add more patterns as needed
        ]

    @staticmethod
    def extract_custom_relations(doc: Doc) -> List[Dict[str, Any]]:
        relations = []
        patterns = RelationDetector.get_relation_patterns()

        # Implementation of custom relation extraction logic
        # based on patterns

        return relations
```

### 2.3. Setup New Memory Services

1. Create Vector Store Interface (`app/memory/vector_store.py`):

```python
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from app.core.config import config
from app.models.memory import Memory, MemoryType

class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.chroma_client = ChromaClient()

    async def add(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> str:
        embedding = self.embedder.encode(text)

        # Add to ChromaDB
        doc_id = await self.chroma_client.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

        return doc_id[0]

    async def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.encode(query)

        results = await self.chroma_client.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return self._format_results(results)

    def _format_results(
        self,
        results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        formatted = []

        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        return formatted
```

2. Create Relational Store Interface (`app/memory/relational_store.py`):

```python
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.entity_models import Entity, Relationship

class RelationalStore:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def store_entity(self, entity: Entity) -> str:
        stmt = insert(entities).values(**entity.dict())
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.inserted_primary_key[0]

    async def store_relationship(
        self,
        relationship: Relationship
    ) -> str:
        stmt = insert(relationships).values(**relationship.dict())
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.inserted_primary_key[0]

    async def get_related_entities(
        self,
        entity_id: str
    ) -> List[Dict[str, Any]]:
        stmt = select(relationships).where(
            or_(
                relationships.c.source_entity_id == entity_id,
                relationships.c.target_entity_id == entity_id
            )
        )
        result = await self.session.execute(stmt)
        return result.mappings().all()
```

3. Implement Memory Manager (`app/memory/manager.py`):

```python
from typing import List, Dict, Any, Optional
from app.memory.vector_store import VectorStore
from app.memory.relational_store import RelationalStore
from app.entities.extractor import EntityExtractor
from app.models.memory import Memory, MemoryQueryResult

class MemoryManager:
    def __init__(
        self,
        vector_store: VectorStore,
        relational_store: RelationalStore,
        entity_extractor: EntityExtractor
    ):
        self.vector_store = vector_store
        self.relational_store = relational_store
        self.entity_extractor = entity_extractor

    async def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        # Extract entities and relationships
        entities = self.entity_extractor.extract_entities(text)
        entity_ids = []

        # Store entities
        for entity in entities:
            entity_id = await self.relational_store.store_entity(entity)
            entity_ids.append(entity_id)

        # Extract and store relationships
        relationships = self.entity_extractor.extract_relationships(
            text,
            entities
        )
        relationship_ids = []

        for rel in relationships:
            rel_id = await self.relational_store.store_relationship(rel)
            relationship_ids.append(rel_id)

        # Update metadata with entity and relationship IDs
        full_metadata = metadata or {}
        full_metadata.update({
            "entity_ids": entity_ids,
            "relationship_ids": relationship_ids
        })

        # Store in vector database
        memory_id = await self.vector_store.add(text, full_metadata)

        return memory_id

    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 5
    ) -> List[MemoryQueryResult]:
        # Get vector search results
        vector_results = await self.vector_store.search(
            query,
            top_k=top_k
        )

        # Enhance results with entity and relationship data
        enhanced_results = []

        for result in vector_results:
            # Get related entities and relationships
            entities = []
            relationships = []

            if "entity_ids" in result["metadata"]:
                for entity_id in result["metadata"]["entity_ids"]:
                    entity = await self.relational_store.get_entity(entity_id)
                    if entity:
                        entities.append(entity)

            if "relationship_ids" in result["metadata"]:
                for rel_id in result["metadata"]["relationship_ids"]:
                    rel = await self.relational_store.get_relationship(rel_id)
                    if rel:
                        relationships.append(rel)

            # Create enhanced result
            memory = Memory(
                id=result["id"],
                content=result["text"],
                metadata=result["metadata"],
                entities=[e.id for e in entities],
                relationships=[r.id for r in relationships]
            )

            query_result = MemoryQueryResult(
                memory=memory,
                score=1 - result["distance"],
                entities=[e.dict() for e in entities],
                relationships=[r.dict() for r in relationships]
            )

            enhanced_results.append(query_result)

        return enhanced_results
```

## Phase 3: Service Layer Implementation

### 3.1. Update ChromaService

Location: `app/services/chroma_service.py`

Key changes:

1. Modify collection structure to support entity linking
2. Update metadata schema for entity and relationship IDs
3. Implement new chunking logic with entity preservation
4. Add metadata enrichment with entity information

```python
class ChromaService:
    async def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        preserve_entities: bool = True
    ) -> str:
        # Implementation details...
        pass

    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        include_entities: bool = True
    ) -> List[Dict[str, Any]]:
        # Implementation details...
        pass
```

### 3.2. Create EntityService

Location: `app/services/entity_service.py`

```python
from typing import List, Dict, Any, Optional
from app.entities.extractor import EntityExtractor
from app.models.entity_models import Entity, Relationship

class EntityService:
    def __init__(self):
        self.extractor = EntityExtractor()

    async def process_text(
        self,
        text: str
    ) -> Dict[str, Any]:
        entities = self.extractor.extract_entities(text)
        relationships = self.extractor.extract_relationships(text, entities)

        return {
            "entities": [e.dict() for e in entities],
            "relationships": [r.dict() for r in relationships]
        }

    async def enrich_memory(
        self,
        memory_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Extract and store entities
        extracted = await self.process_text(text)

        # Update metadata
        enriched_metadata = {
            **metadata,
            "entities": extracted["entities"],
            "relationships": extracted["relationships"]
        }

        return enriched_metadata
```

### 3.3. Update LangChainService

Location: `app/services/langchain_service.py`

Key changes:

1. Integrate entity awareness in memory retrieval
2. Update conversation processing with entity extraction
3. Enhance summarization with entity preservation

```python
class LangChainService:
    async def process_conversation(
        self,
        messages: List[Dict[str, Any]],
        memory_type: MemoryType = MemoryType.EPHEMERAL,
        include_entities: bool = True
    ) -> List[Dict[str, Any]]:
        # Implementation details...
        pass

    async def summarize_text(
        self,
        text: str,
        preserve_entities: bool = True
    ) -> Optional[str]:
        # Implementation details...
        pass
```

## Phase 4: Integration and API Updates

### 4.1. Update API Endpoints

1. Update Chat Router (`app/routers/chat.py`):

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    entity_service: EntityService = Depends(get_entity_service)
):
    # Process message with entity extraction
    entities = await entity_service.process_text(request.message)

    # Store in memory with entity information
    memory_id = await memory_manager.add_memory(
        request.message,
        metadata={"entities": entities}
    )

    # Generate response
    response = await generate_response(request, entities)

    return {
        "response": response,
        "memory_id": memory_id,
        "entities": entities
    }
```

2. Add Entity Management Endpoints (`app/routers/entities.py`):

```python
@router.get("/entities/{entity_id}")
async def get_entity(
    entity_id: str,
    relational_store: RelationalStore = Depends(get_relational_store)
):
    entity = await relational_store.get_entity(entity_id)
    return entity

@router.get("/entities/{entity_id}/relationships")
async def get_entity_relationships(
    entity_id: str,
    relational_store: RelationalStore = Depends(get_relational_store)
):
    relationships = await relational_store.get_related_entities(entity_id)
    return relationships
```

### 4.2. Update Service Locator

Location: `app/core/service_locator.py`

```python
class ServiceLocator:
    def __init__(self):
        self.services = {}

    async def initialize(self):
        # Initialize vector store
        vector_store = VectorStore()
        await vector_store.initialize()

        # Initialize relational store
        engine = create_async_engine(config.DATABASE_URL)
        session = sessionmaker(engine, class_=AsyncSession)
        relational_store = RelationalStore(session())

        # Initialize entity extractor
        entity_extractor = EntityExtractor()

        # Initialize memory manager
        memory_manager = MemoryManager(
            vector_store,
            relational_store,
            entity_extractor
        )

        # Register services
        self.services.update({
            "vector_store": vector_store,
            "relational_store": relational_store,
            "entity_extractor": entity_extractor,
            "memory_manager": memory_manager
        })
```

### 4.3. Update Configuration

Location: `app/core/config.py`

```python
class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/db"

    # Vector Store
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

    # Entity Extraction
    SPACY_MODEL: str = "en_core_web_lg"
    ENTITY_CONFIDENCE_THRESHOLD: float = 0.5

    # Memory Management
    MAX_CHUNK_TOKENS: int = 512
    VECTOR_SEARCH_TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.7
```

## Phase 5: Testing and Validation

### 5.1. Create Test Suite

1. Entity Extraction Tests (`tests/test_entity_extraction.py`):

```python
async def test_entity_extraction():
    extractor = EntityExtractor()
    text = "John works for Microsoft in Seattle"

    entities = extractor.extract_entities(text)
    assert len(entities) == 3
    assert entities[0].text == "John"
    assert entities[1].text == "Microsoft"
    assert entities[2].text == "Seattle"
```

2. Memory Manager Tests (`tests/test_memory_manager.py`):

```python
async def test_memory_storage_and_retrieval():
    manager = MemoryManager(
        vector_store=MockVectorStore(),
        relational_store=MockRelationalStore(),
        entity_extractor=MockEntityExtractor()
    )

    text = "Test memory with entities"
    memory_id = await manager.add_memory(text)

    results = await manager.retrieve_memories("test memory")
    assert len(results) > 0
    assert results[0].memory.id == memory_id
```

### 5.2. Data Migration

1. Create Migration Script (`scripts/migrate_memories.py`):

```python
async def migrate_memories():
    # Initialize services
    old_chroma = ChromaService()
    new_memory_manager = MemoryManager()

    # Get all existing memories
    memories = await old_chroma.get_all_memories()

    # Migrate each memory
    for memory in memories:
        # Extract entities and relationships
        entities = await entity_service.process_text(memory.text)

        # Store in new system
        await new_memory_manager.add_memory(
            memory.text,
            metadata={
                **memory.metadata,
                "migrated_from": memory.id,
                "migrated_at": datetime.utcnow().isoformat()
            }
        )
```

## Phase 6: Deployment and Monitoring

### 6.1. Deployment Steps

1. Database Setup:

```bash
# Create database
createdb memory_system

# Run migrations
alembic upgrade head
```

2. Environment Configuration:

```bash
# Set environment variables
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost/memory_system"
export CHROMA_PERSIST_DIRECTORY="/path/to/chroma"
export SPACY_MODEL="en_core_web_lg"
```

3. Service Deployment:

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

# Run migration script
python scripts/migrate_memories.py

# Start application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 6.2. Monitoring Setup

1. Add Metrics Collection (`app/monitoring/metrics.py`):

```python
from prometheus_client import Counter, Histogram

# Define metrics
memory_operations = Counter(
    'memory_operations_total',
    'Total memory operations',
    ['operation_type']
)

entity_extraction_time = Histogram(
    'entity_extraction_seconds',
    'Time spent on entity extraction'
)

memory_retrieval_time = Histogram(
    'memory_retrieval_seconds',
    'Time spent on memory retrieval'
)
```

2. Add Logging (`app/monitoring/logging.py`):

```python
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryLogger:
    @staticmethod
    def log_operation(
        operation: str,
        details: Dict[str, Any],
        duration: float
    ):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "duration": duration,
            "details": details
        }
        logger.info(json.dumps(log_entry))
```

## Success Criteria Validation

1. Functionality Verification:

   - All existing memory operations work with new system
   - Entity extraction and relationship detection working
   - Memory retrieval includes entity context

2. Performance Metrics:

   - Response times within acceptable range
   - Entity extraction overhead minimal
   - Memory usage within limits

3. Data Integrity:

   - All existing memories migrated successfully
   - Entity relationships preserved
   - No data loss during migration

4. Monitoring and Maintenance:
   - Metrics collection operational
   - Logging system capturing all operations
   - Backup systems in place
