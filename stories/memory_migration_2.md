Below is a **comprehensive, optimized integration guide** to ensure your **LightRAG + Relational Store** implementation adheres strictly to the **human-like memory system** design outlined earlier. This guide includes a **step-by-step migration checklist** and a **detailed code review** to align your current implementation with the desired architecture.

---

## Table of Contents

1. [Review of Current Implementation](#1-review-of-current-implementation) [ ]
2. [Actionable Step-by-Step Migration Guide](#2-actionable-step-by-step-migration-guide) [ ]
   - [2.1. Refactor `manager.py`](#21-refactor-managerpy) [x]
   - [2.2. Update `datastore.py`](#22-update-datastorepy) [ ]
   - [2.3. Enhance `ingestion.py`](#23-enhance-ingestionpy) [ ]
   - [2.4. Revise `tasks.py`](#24-revise-taskspy) [ ]
   - [2.5. Update `routers/chat.py`](#25-update-routerschatpy) [ ]
   - [2.6. Implement Advanced NER and Relationship Inference](#26-implement-advanced-ner-and-relationship-inference) [ ]
   - [2.7. Remove or Update `chroma_service`](#27-remove-or-update-chromaservice) [ ]
   - [2.8. Expose `add_to_memory` Functionality](#28-expose-add_to_memory-functionality) [ ]
   - [2.9. Pass DB Relations as System Prompt Context](#29-pass-db-relations-as-system-prompt-context) [ ]
3. [Code Review and Implementation Recommendations](#3-code-review-and-implementation-recommendations) [ ]
   - [3.1. `lightrag/__init__.py`](#31-lightraginitpy) [ ]
   - [3.2. `lightrag/config.py`](#32-lightragconfigpy) [ ]
   - [3.3. `lightrag/datastore.py`](#33-lightragdatastorepy) [ ]
   - [3.4. `lightrag/ingestion.py`](#34-lightragingestionpy) [ ]
   - [3.5. `lightrag/manager.py`](#35-lightragmanagerpy) [ ]
   - [3.6. `lightrag/tasks.py`](#36-lightragtaskspy) [ ]
   - [3.7. `routers/chat.py`](#37-routerschatpy) [ ]
4. [Final Verification and Testing](#4-final-verification-and-testing) [ ]
5. [Conclusion](#5-conclusion) [ ]

---

## 1. Review of Current Implementation

Your current implementation comprises several files under the `app/memory/lightrag/` directory and a `routers/chat.py` file. Here's a high-level overview:

- **`lightrag/__init__.py`**: Placeholder.
- **`lightrag/config.py`**: Configuration settings for LightRAG and the relational database.
- **`lightrag/datastore.py`**: Handles relational database operations (entities, relationships, metadata).
- **`lightrag/ingestion.py`**: Manages ingestion of text and files into the memory system.
- **`lightrag/manager.py`**: Core memory manager combining LightRAG and the relational store.
- **`lightrag/tasks.py`**: Background tasks for memory maintenance (currently placeholders).
- **`routers/chat.py`**: FastAPI router handling chat-related endpoints.

**Identified Gaps:**

1. **Advanced NER and Relationship Interpretation**: The current `chat.py` uses a placeholder `advanced_ner_and_relationship_inference` function, but it's not fully implemented with an advanced rule system or classifier.
2. **`add_memory` Endpoint**: The existing `/chat/memory/add` endpoint uses `chroma_service`, which should be replaced with `LightRAGManager`.
3. **Passing DB Relations to LLM**: The current implementation lacks the functionality to pass relational data as additional system prompt context to the LLM.
4. **Exposed `add_to_memory` Function**: The `manager.py` has an `add_to_memory` method, but it's not properly exposed via the API.
5. **Relationship Typing and Flexibility**: Relationships are currently freeform, but there's no advanced system to interpret NER labels into specific relationships beyond simple rules.
6. **Synonym/Alias Management**: No existing implementation for handling synonyms or aliases in entities.
7. **Embedding-Based Disambiguation**: Not implemented, limiting entity linking capabilities.

---

## 2. Actionable Step-by-Step Migration Guide

To align your implementation with the comprehensive guide, follow the steps below. Each step ensures that your system becomes a **state-of-the-art, human-like memory system** with **advanced entity linking**, **contextual enrichment**, and **efficient memory management**.

### 2.1. Refactor `manager.py`

**Objective:** Consolidate `manager.py` to include both LightRAG and relational store functionalities, ensuring all methods are present and correctly implemented.

**Actions:**

1. **Remove Redundancies**: Ensure there's only one `manager.py` file under `lightrag/`.
2. **Implement `query_with_relations`**: Enhance the method to perform entity extraction and retrieve relational data.
3. **Integrate `add_to_memory`**: Ensure this method is fully implemented and exposed for model access.

**Updated `manager.py`:**

```python
# app/memory/lightrag/manager.py
import logging
import sqlite3
import uuid
from typing import List, Optional, Dict

from lightrag import LightRAG, QueryParam
from .config import LIGHTRAG_DATA_DIR, DB_PATH

logger = logging.getLogger(__name__)

class LightRAGManager:
    """
    Central memory manager combining:
    1) LightRAG for unstructured, chunk-based retrieval
    2) A relational DB for entities, relationships, and metadata
    """

    def __init__(self, working_dir: str = LIGHTRAG_DATA_DIR, db_path: str = DB_PATH):
        logger.info("Initializing LightRAGManager with relational DB...")

        # Initialize LightRAG
        self.rag = LightRAG(working_dir=str(working_dir))

        # Initialize relational store
        self.db_path = db_path
        self._ensure_tables()

    def _connect_db(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self):
        conn = self._connect_db()
        cur = conn.cursor()

        # Create entities table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            entity_type TEXT,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create relationships table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            src_entity TEXT NOT NULL,
            dst_entity TEXT NOT NULL,
            relation_type TEXT,
            confidence REAL DEFAULT 1.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create metadata table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
        conn.close()
        logger.debug("Ensured relational DB tables exist.")

    # ------------------------------------------------
    # LightRAG Methods
    # ------------------------------------------------

    async def insert_text(self, text: str):
        """Add unstructured text to LightRAG for chunk-based retrieval."""
        if not text.strip():
            return
        logger.debug(f"Inserting text (size={len(text)}) into LightRAG.")
        await self.rag.insert(text)

    async def query_rag(self, query_str: str, mode: str = "mix", top_k: int = 5) -> str:
        """Use LightRAG to retrieve/generate an answer with vector/graph context."""
        param = QueryParam(mode=mode, top_k=top_k)
        result = await self.rag.query(query_str, param=param)
        return result

    # ------------------------------------------------
    # Relational Store Methods
    # ------------------------------------------------

    def create_entity(self, name: str, entity_type: str, description: str = "") -> str:
        entity_id = str(uuid.uuid4())
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO entities (id, name, entity_type, description)
            VALUES (?, ?, ?, ?);
        """, (entity_id, name, entity_type, description))
        conn.commit()
        conn.close()
        logger.debug(f"Created entity {name} ({entity_type}). ID={entity_id}")
        return entity_id

    def create_relationship(self, src_id: str, dst_id: str, relation_type: str, confidence: float = 1.0):
        rel_id = str(uuid.uuid4())
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO relationships (id, src_entity, dst_entity, relation_type, confidence)
            VALUES (?, ?, ?, ?, ?);
        """, (rel_id, src_id, dst_id, relation_type, confidence))
        conn.commit()
        conn.close()
        logger.debug(f"Created relationship {rel_id}: {src_id} --{relation_type}--> {dst_id}")

    def get_entity_relations(self, entity_id: str) -> List[Dict]:
        """Return all relationships where entity_id is either src_entity or dst_entity."""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, src_entity, dst_entity, relation_type, confidence, created_at, updated_at
            FROM relationships
            WHERE src_entity=? OR dst_entity=?
        """, (entity_id, entity_id))
        rows = cur.fetchall()
        conn.close()

        relations = []
        for row in rows:
            relations.append({
                "id": row[0],
                "src_entity": row[1],
                "dst_entity": row[2],
                "relation_type": row[3],
                "confidence": row[4],
                "created_at": row[5],
                "updated_at": row[6]
            })
        return relations

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Retrieve an entity by its ID."""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM entities WHERE id = ?
        """, (entity_id,))
        row = cur.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "description": row[3],
                "created_at": row[4],
                "updated_at": row[5]
            }
        return None

    def search_entities(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search for entities by name or description."""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM entities
            WHERE name LIKE ? OR description LIKE ?
            LIMIT ?
        """, (f"%{search_term}%", f"%{search_term}%", limit))
        rows = cur.fetchall()
        conn.close()

        return [{
            "id": row[0],
            "name": row[1],
            "entity_type": row[2],
            "description": row[3],
            "created_at": row[4],
            "updated_at": row[5]
        } for row in rows]

    def get_entity_metadata(self, entity_id: str) -> Dict[str, str]:
        """Retrieve all metadata for a given entity."""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT key, value FROM metadata WHERE entity_id = ?
        """, (entity_id,))
        rows = cur.fetchall()
        conn.close()

        return {row[0]: row[1] for row in rows}

    def add_entity_metadata(self, entity_id: str, key: str, value: str):
        """Add metadata to an entity."""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO metadata (id, entity_id, key, value)
            VALUES (?, ?, ?, ?)
        """, (str(uuid.uuid4()), entity_id, key, value))
        conn.commit()
        conn.close()

    # ------------------------------------------------
    # Advanced Entity Linking and Relationship Inference
    # ------------------------------------------------

    def interpret_relation(self, label: str, context: str) -> Optional[str]:
        """
        Interpret NER label and context to determine the relationship type.

        Args:
            label: NER label (e.g., "DOG", "ANIMAL", "PERSON")
            context: Surrounding text context for disambiguation

        Returns:
            The interpreted relationship type (e.g., "hasPet") or None
        """
        label = label.upper()
        context = context.lower()

        if label in ["DOG", "ANIMAL"]:
            if any(keyword in context for keyword in ["have a", "have an", "have", "my", "own"]):
                return "hasPet"
            else:
                return "relatedTo"
        elif label == "PERSON":
            if any(keyword in context for keyword in ["friend", "know", "meet", "met"]):
                return "isFriendsWith"
            elif any(keyword in context for keyword in ["work", "job", "employed"]):
                return "worksAt"
            else:
                return "relatedTo"
        elif label == "ORG":
            if any(keyword in context for keyword in ["work at", "employed by", "join"]):
                return "worksAt"
            else:
                return "relatedTo"
        # Add more rules as needed
        else:
            return "relatedTo"

    async def add_to_memory(self, text: str, entity_name: Optional[str] = None, entity_type: Optional[str] = None):
        """
        Called by the model if it needs to store something in memory or relational DB.

        Args:
            text: The unstructured text to store
            entity_name: Optional, name of the entity to link
            entity_type: Optional, type of the entity (e.g., "DOG")
        """
        # 1) Insert as ephemeral chunk
        await self.insert_text(text)

        # 2) If there's an entity_name, create/update entity
        if entity_name:
            e_id = self.link_entity(entity_name, entity_type or "unknown")
            logger.debug(f"Model added entity {entity_name} to memory: {e_id}")

    def link_entity(self, name: str, entity_type: str) -> str:
        """Find or create entity by name. Return the entity ID."""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM entities
            WHERE LOWER(name) = LOWER(?)
            LIMIT 1
        """, (name,))
        row = cur.fetchone()
        if row:
            entity_id = row[0]
            cur.execute("UPDATE entities SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (entity_id,))
            conn.commit()
            conn.close()
            return entity_id
        else:
            conn.close()
            return self.create_entity(name, entity_type)

    async def query_with_relations(self, query_str: str, user_id: str, top_k: int = 5) -> Dict:
        """
        Retrieve text from LightRAG and relevant relationships from DB to create
        a 'human-like' memory context.

        Args:
            query_str: The user's query
            user_id: The ID of the user making the query
            top_k: Number of top chunks to retrieve

        Returns:
            A dictionary containing LightRAG text and relational context
        """
        # Step 1: Named Entity Extraction in the Query
        extracted_entities = await self.extract_entities(query_str)

        # Step 2: Link Entities
        linked_ids = []
        for ent in extracted_entities:
            entity_id = self.link_entity(ent["text"], ent["label"])
            linked_ids.append(entity_id)

            # Step 3: Infer and Create Relationships
            relation_type = self.interpret_relation(ent["label"], query_str)
            if relation_type:
                self.create_relationship(user_id, entity_id, relation_type, confidence=1.0)

        # Step 4: LightRAG Retrieval
        rag_text = await self.query_rag(query_str, mode="mix", top_k=top_k)

        # Step 5: Fetch Relational Information
        relational_info = []
        for eid in linked_ids:
            rels = self.get_entity_relations(eid)
            relational_info.append({
                "entity_id": eid,
                "entity": self.get_entity(eid),
                "relations": rels
            })

        # Step 6: Pass Relational Context to LLM as System Prompt
        system_prompt = self.build_system_prompt(relational_info)

        return {
            "rag_text": rag_text,
            "relational_context": relational_info,
            "system_prompt": system_prompt
        }

    async def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using the integrated NER model.

        Args:
            text: The text to process

        Returns:
            A list of dictionaries containing entity text and labels
        """
        import spacy

        nlp = spacy.load("en_core_web_trf")
        doc = nlp(text)
        results = []
        for ent in doc.ents:
            results.append({
                "text": ent.text,
                "label": ent.label_
            })
        return results

    def build_system_prompt(self, relational_info: List[Dict]) -> str:
        """
        Build a system prompt string from relational information to pass to the LLM.

        Args:
            relational_info: List of entities with their relations

        Returns:
            A formatted string for the system prompt
        """
        prompt = "You are an AI with a memory system. Here are some relationships and entities you should be aware of:\n\n"
        for info in relational_info:
            entity = info["entity"]
            if entity:
                prompt += f"- {entity['name']} ({entity['entity_type']}):\n"
                for rel in info["relations"]:
                    prompt += f"  - {rel['relation_type']} -> {self.get_entity(rel['dst_entity'])['name']}\n"
        prompt += "\nUse this information to provide contextual and accurate responses."
        return prompt
```

### 2.2. Update `datastore.py`

**Objective:** Ensure `datastore.py` aligns with the guide by supporting advanced entity operations, including synonym management and embedding-based disambiguation if needed.

**Actions:**

1. **Enhance `MemoryDatastore`**: Integrate methods for handling synonyms and alias management.
2. **Implement Embedding-Based Disambiguation**: (Optional) Add methods to store and retrieve embeddings for entities.

**Updated `datastore.py`:**

```python
# app/memory/lightrag/datastore.py
"""Relational database operations for LightRAG memory system."""

import sqlite3
import uuid
from typing import Dict, List, Optional
from pathlib import Path
from .config import DB_PATH

class MemoryDatastore:
    """Handles operations for the memory relational database."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._ensure_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self):
        """Ensure all required tables exist in the database."""
        conn = self._connect()
        cur = conn.cursor()

        # Create entities table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            entity_type TEXT,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create relationships table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            src_entity TEXT NOT NULL,
            dst_entity TEXT NOT NULL,
            relation_type TEXT,
            confidence REAL DEFAULT 1.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create metadata table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create synonyms table for alias management
        cur.execute("""
        CREATE TABLE IF NOT EXISTS synonyms (
            id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            synonym TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
        );
        """)

        conn.commit()
        conn.close()

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Retrieve an entity by its ID."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM entities WHERE id = ?
        """, (entity_id,))
        row = cur.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "description": row[3],
                "created_at": row[4],
                "updated_at": row[5]
            }
        return None

    def search_entities(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search for entities by name, synonyms, or description."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT e.* FROM entities e
            LEFT JOIN synonyms s ON e.id = s.entity_id
            WHERE e.name LIKE ? OR s.synonym LIKE ? OR e.description LIKE ?
            GROUP BY e.id
            LIMIT ?
        """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", limit))
        rows = cur.fetchall()
        conn.close()

        return [{
            "id": row[0],
            "name": row[1],
            "entity_type": row[2],
            "description": row[3],
            "created_at": row[4],
            "updated_at": row[5]
        } for row in rows]

    def add_synonym(self, entity_id: str, synonym: str):
        """Add a synonym to an entity."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO synonyms (id, entity_id, synonym)
            VALUES (?, ?, ?)
        """, (str(uuid.uuid4()), entity_id, synonym))
        conn.commit()
        conn.close()

    def get_synonyms(self, entity_id: str) -> List[str]:
        """Retrieve all synonyms for a given entity."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT synonym FROM synonyms WHERE entity_id = ?
        """, (entity_id,))
        rows = cur.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def add_entity_metadata(self, entity_id: str, key: str, value: str):
        """Add metadata to an entity."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO metadata (id, entity_id, key, value)
            VALUES (?, ?, ?, ?)
        """, (str(uuid.uuid4()), entity_id, key, value))
        conn.commit()
        conn.close()
```

### 2.3. Enhance `ingestion.py`

**Objective:** Ensure that the ingestion process supports adding metadata and can be easily extended for future enhancements.

**Actions:**

1. **Integrate Metadata Handling**: Update `ingest_text` to accept and process metadata.
2. **Prepare for Future Enhancements**: Ensure that file type-specific processing and directory traversal are robust and can handle various file formats.

**Updated `ingestion.py`:**

```python
# app/memory/lightrag/ingestion.py
"""Document and file ingestion for LightRAG memory system."""

import logging
from pathlib import Path
from typing import Union, Optional, Dict
from .manager import LightRAGManager

logger = logging.getLogger(__name__)

class MemoryIngestor:
    """Handles ingestion of various content types into the memory system."""

    def __init__(self, manager: LightRAGManager):
        self.manager = manager

    async def ingest_text(self, text: str, metadata: Optional[Dict] = None):
        """
        Ingest plain text content into the memory system.

        Args:
            text: The text content to ingest
            metadata: Optional metadata to associate with the content
        """
        if not text.strip():
            return

        logger.info(f"Ingesting text content (length: {len(text)})")
        await self.manager.insert_text(text)

        # Process metadata and store in relational database
        if metadata:
            logger.debug(f"Processing metadata: {metadata}")
            for key, value in metadata.items():
                # Assuming metadata keys correspond to entity properties or relations
                # This logic can be expanded based on specific metadata structures
                self.manager.add_entity_metadata(entity_id=value.get("entity_id"), key=key, value=value.get("value"))

    async def ingest_file(self, file_path: Union[str, Path], metadata: Optional[Dict] = None):
        """
        Ingest content from a file into the memory system.

        Args:
            file_path: Path to the file to ingest
            metadata: Optional metadata to associate with the content
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return

        logger.info(f"Ingesting file: {path.name}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            await self.ingest_text(content, metadata)
        except Exception as e:
            logger.error(f"Failed to ingest file {path}: {str(e)}")

    async def ingest_directory(self, dir_path: Union[str, Path], metadata: Optional[Dict] = None):
        """
        Ingest all supported files from a directory.

        Args:
            dir_path: Path to the directory to ingest
            metadata: Optional metadata to associate with the content
        """
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            logger.error(f"Directory not found: {path}")
            return

        logger.info(f"Ingesting directory: {path}")

        for file_path in path.iterdir():
            if file_path.is_file():
                await self.ingest_file(file_path, metadata)
```

### 2.4. Revise `tasks.py`

**Objective:** Implement actual background tasks for memory maintenance, such as cleanup and summarization.

**Actions:**

1. **Implement Cleanup Logic**: Remove old or unused memory entries based on criteria (e.g., age, access frequency).
2. **Implement Summarization Logic**: Generate summaries for large memory chunks to optimize storage and retrieval.

**Updated `tasks.py`:**

```python
# app/memory/lightrag/tasks.py
"""Background tasks and maintenance for LightRAG memory system."""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from .manager import LightRAGManager

logger = logging.getLogger(__name__)

class MemoryTasks:
    """Handles background tasks and maintenance operations."""

    def __init__(self, manager: LightRAGManager):
        self.manager = manager
        self._running = False

    async def start(self):
        """Start background tasks."""
        if self._running:
            return

        self._running = True
        logger.info("Starting memory background tasks")

        # Start cleanup task
        asyncio.create_task(self._cleanup_task())

        # Start summarization task
        asyncio.create_task(self._summarize_task())

    async def stop(self):
        """Stop background tasks."""
        self._running = False
        logger.info("Stopping memory background tasks")

    async def _cleanup_task(self):
        """Periodic cleanup of old or unused memory entries."""
        while self._running:
            try:
                logger.info("Running memory cleanup task")
                cutoff_time = datetime.utcnow() - timedelta(days=30)  # Example: 30 days old
                # Implement logic to delete or archive old entities
                self._cleanup_old_entities(cutoff_time)
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Cleanup task failed: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

    def _cleanup_old_entities(self, cutoff_time: datetime):
        """Delete entities not updated since cutoff_time."""
        conn = self.manager._connect_db()
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM entities
            WHERE updated_at < ?
        """, (cutoff_time.isoformat(),))
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        logger.info(f"Cleanup task deleted {deleted} old entities.")

    async def _summarize_task(self):
        """Periodic summarization of memory content."""
        while self._running:
            try:
                logger.info("Running memory summarization task")
                # Example: Summarize entities with extensive metadata
                self._summarize_entities()
                await asyncio.sleep(86400)  # Run daily
            except Exception as e:
                logger.error(f"Summarization task failed: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

    def _summarize_entities(self):
        """Generate summaries for entities with large descriptions."""
        conn = self.manager._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, description FROM entities
            WHERE LENGTH(description) > 1024  -- Example threshold
        """)
        rows = cur.fetchall()
        conn.close()

        for row in rows:
            entity_id, description = row
            summary = self._generate_summary(description)
            self.manager.add_entity_metadata(entity_id, "summary", summary)
            logger.debug(f"Summarized entity {entity_id}: {summary}")

    def _generate_summary(self, text: str) -> str:
        """Generate a summary for a given text. Placeholder implementation."""
        # Integrate with an actual summarization model or algorithm
        return text[:500] + "..." if len(text) > 500 else text
```

### 2.5. Update `routers/chat.py`

**Objective:** Ensure that all chat-related endpoints utilize `LightRAGManager` and adhere to the guide's requirements, replacing any usage of `chroma_service`.

**Actions:**

1. **Replace `chroma_service` with `LightRAGManager` in all endpoints**.
2. **Implement Advanced NER and Relationship Inference** in the `/chat/parse` endpoint.
3. **Expose `add_to_memory`** via an endpoint.
4. **Implement `/chat/query_entity_linked`** to handle queries with relational context.
5. **Ensure all endpoints handle timestamps and metadata appropriately**.

**Updated `routers/chat.py`:**

```python
# app/routers/chat.py
"""Chat router for handling chat-related endpoints and streaming responses."""
import json
import logging
from typing import List, Optional, AsyncGenerator, Any, Dict, Union
from fastapi import APIRouter, Request, Depends, HTTPException, UploadFile, File
from sse_starlette.sse import EventSourceResponse
from app.context.llm_context import MemoryType
from app.core.config import config
from app.dependencies.providers import (
    get_agent,
    get_model_service,
    get_function_service,
    get_langchain_service,
    get_lightrag_manager
)
from app.memory.lightrag.manager import LightRAGManager
from app.ner_utils import advanced_ner_and_relationship_inference
from app.functions.filters import apply_filters
from app.functions.utils import (
    validate_tool_response,
    create_error_response
)
from app.services.agent import Agent
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.services.langchain_service import LangChainService
from app.models.chat import (
    ChatRequest, ChatStreamEvent, StrictChatMessage
)
from app.functions.base import (
    FunctionType,
    Filter,
    ToolResponse,
    ValidationError
)
import asyncio
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


def convert_to_dict(obj: Any) -> Union[Dict[str, Any], List[Any], Any]:
    """Recursively convert Message objects to dictionaries.

    Args:
        obj: Object to convert (Message, dict, list, or other)

    Returns:
        Converted object in dictionary form
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    return obj


async def setup_chat_components(
    request_id: str,
    chat_request: ChatRequest,
    function_service: FunctionService
) -> tuple[Optional[List[Dict]], List[Filter], Optional[Any]]:
    """Setup function schemas, filters, and pipeline for chat."""
    # Get available functions
    function_schemas = None
    if chat_request.enable_tools:
        logger.info(
            f"[{request_id}] Tools are enabled, getting function schemas")
        function_schemas = function_service.get_function_schemas()
        if function_schemas:
            logger.info(
                f"[{request_id}] Available tools: {[f['function']['name'] for f in function_schemas]}")
        else:
            logger.warning(f"[{request_id}] No tool schemas available")

    # Get requested filters
    filters = []
    if chat_request.filters:
        logger.info(f"[{request_id}] Getting filters: {chat_request.filters}")
        for filter_name in chat_request.filters:
            filter_class = function_service.get_function(filter_name)
            if filter_class and filter_class.model_fields['type'].default == FunctionType.FILTER:
                logger.info(
                    f"[{request_id}] Instantiating filter: {filter_name}")
                try:
                    filter_instance = filter_class()
                    filters.append(filter_instance)
                    logger.info(f"[{request_id}] Added filter: {filter_name}")
                except Exception as e:
                    logger.error(
                        f"[{request_id}] Error instantiating filter {filter_name}: {e}")
            else:
                logger.warning(
                    f"[{request_id}] Filter not found or invalid type: {filter_name}")

    # Get requested pipeline
    pipeline = None
    if chat_request.pipeline:
        logger.info(
            f"[{request_id}] Getting pipeline: {chat_request.pipeline}")
        pipeline_class = function_service.get_function(chat_request.pipeline)
        if pipeline_class and pipeline_class.model_fields['type'].default == FunctionType.PIPELINE:
            logger.info(
                f"[{request_id}] Instantiating pipeline: {chat_request.pipeline}")
            try:
                pipeline = pipeline_class()
                logger.info(
                    f"[{request_id}] Added pipeline: {chat_request.pipeline}")
            except Exception as e:
                logger.error(
                    f"[{request_id}] Error instantiating pipeline {chat_request.pipeline}: {e}")
        else:
            logger.warning(
                f"[{request_id}] Pipeline not found or invalid type: {chat_request.pipeline}")

    return function_schemas, filters, pipeline


async def handle_tool_response(
    request_id: str,
    response: Union[Dict[str, Any], ToolResponse]
) -> ChatStreamEvent:
    """Handle tool/function response.

    Args:
        request_id: ID of the current request
        response: Tool response data (dict or ToolResponse)

    Returns:
        event to send
    """
    logger.info(f"[{request_id}] Processing tool/function response")
    logger.debug(
        f"[{request_id}] Raw tool/function response: {json.dumps(response, indent=2) if isinstance(response, dict) else str(response)}")

    try:
        # Convert dict to ToolResponse if needed
        if isinstance(response, dict):
            tool_message = {
                "role": "tool",
                "content": response.get("content", ""),
                "name": response.get("name", ""),
                "tool_call_id": response.get("tool_call_id")
            }
        else:
            # Handle ToolResponse object
            validate_tool_response(response)
            tool_message = {
                "role": "tool",
                "content": response.result if response.success else response.error,
                "name": response.tool_name,
                "tool_call_id": response.metadata.get("tool_call_id") if response.metadata else None
            }

        return ChatStreamEvent(event="message", data=json.dumps(tool_message))

    except ValidationError as e:
        logger.error(f"[{request_id}] Invalid tool response: {e}")
        error_response = create_error_response(
            error=e,
            function_type="tool",
            function_name=getattr(response, "tool_name", "unknown"),
            tool_call_id=getattr(response, "metadata", {}).get("tool_call_id")
        )
        return ChatStreamEvent(
            event="error",
            data=json.dumps({"error": error_response.error})
        )


async def handle_tool_calls(
    request_id: str,
    response: Dict[str, Any],
    function_service: FunctionService
) -> List[ChatStreamEvent]:
    """Handle tool calls from assistant.

    Args:
        request_id: ID of the current request
        response: Assistant response containing tool calls
        function_service: Service for handling function calls

    Returns:
        List of events to send
    """
    logger.info(
        f"[{request_id}] Tool calls detected: {json.dumps(response['tool_calls'], indent=2)}")
    events = []

    # Send raw tool call message
    events.append(ChatStreamEvent(event="message", data=json.dumps({
        "role": "assistant",
        "content": str(response)
    })))

    try:
        tool_responses = await function_service.handle_tool_calls(response["tool_calls"])
        for tool_response in tool_responses:
            event = await handle_tool_response(request_id, tool_response)
            events.append(event)

    except Exception as e:
        logger.error(f"[{request_id}] Error executing tool calls: {e}")
        error_response = create_error_response(
            error=e,
            function_type="tool",
            function_name="unknown"
        )
        events.append(ChatStreamEvent(
            event="error",
            data=json.dumps({"error": error_response.error})
        ))

    return events


async def handle_assistant_message(
    request_id: str,
    response: Dict[str, Any],
    filters: List[Filter]
) -> ChatStreamEvent:
    """Handle assistant message with outlet filtering."""
    assistant_message = {
        "role": "assistant",
        "content": response.get("content", "")
    }

    if not filters:
        print(assistant_message['content'], end="", flush=True)
        return ChatStreamEvent(event="message", data=json.dumps(assistant_message))

    return await apply_filters(
        filters=filters,
        data=assistant_message,
        request_id=request_id,
        direction="outlet",
        as_event=True,
        filter_name="outlet_message_filters"
    )


async def handle_string_chunk(
    request_id: str,
    response: str,
    filters: List[Filter]
) -> ChatStreamEvent:
    """Handle string chunk with outlet filtering."""
    chunk_message = {
        "role": "assistant",
        "content": str(response)
    }

    if not filters:
        print(chunk_message['content'], end="", flush=True)
        return ChatStreamEvent(event="message", data=json.dumps(chunk_message))

    return await apply_filters(
        filters=filters,
        data=chunk_message,
        request_id=request_id,
        direction="outlet",
        as_event=True,
        filter_name="outlet_chunk_filters"
    )


async def verify_model_availability(
    request_id: str,
    model: str,
    model_service: ModelService
) -> Optional[ChatStreamEvent]:
    """Verify model availability.

    Args:
        request_id: ID of the current request
        model: Model name to verify
        model_service: Service for model operations

    Returns:
        Error event if model not available, None if available
    """
    try:
        models = await model_service.get_all_models(request_id)
    except Exception as model_error:
        logger.error(
            f"[{request_id}] Error fetching models: {model_error}", exc_info=True)
        return ChatStreamEvent(
            event="error",
            data=json.dumps({"error": "Failed to fetch available models"})
        )

    if model not in models:
        logger.error(f"[{request_id}] Model {model} not available")
        return ChatStreamEvent(
            event="error",
            data=json.dumps({"error": f"Model {model} not available"})
        )

    return None


async def stream_chat_response(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent = Depends(get_agent),
    model_service: ModelService = Depends(get_model_service),
    function_service: FunctionService = Depends(get_function_service),
    langchain_service: LangChainService = Depends(get_langchain_service),
    manager: LightRAGManager = Depends(get_lightrag_manager),
    is_test: bool = False
) -> AsyncGenerator[ChatStreamEvent, None]:
    """Generate streaming chat response."""
    request_id = str(id(request))
    logger.info(f"[{request_id}] Starting chat stream")
    tool_call_in_progress = False

    try:
        # Setup components
        function_schemas, filters, pipeline = await setup_chat_components(
            request_id, chat_request, function_service)

        # Verify model availability
        model = chat_request.model or config.DEFAULT_MODEL
        if error_event := await verify_model_availability(request_id, model, model_service):
            yield error_event
            return

        # Apply inlet filters to the entire messages array if filters exist
        if filters:
            data, filter_success = await apply_filters(
                filters=filters,
                data={"messages": chat_request.messages},
                request_id=request_id,
                direction="inlet",
                filter_name="inlet_message_filters"
            )

            if not filter_success:
                yield ChatStreamEvent(
                    event="error",
                    data=json.dumps({"error": "Failed to apply inlet filters"})
                )
                return

            chat_request.messages = data["messages"]

        # Process conversation with LightRAG and relational memory
        if chat_request.enable_memory and manager:
            try:
                # Query with relations
                memory_context = await manager.query_with_relations(
                    query_str=chat_request.messages[-1].content,  # Assuming last message is the query
                    user_id=chat_request.conversation_id or "default_user",
                    top_k=chat_request.top_k_memories or 5
                )

                # Inject system prompt context
                system_prompt = memory_context.get("system_prompt", "")
                if system_prompt:
                    chat_request.messages.insert(0, {
                        "role": "system",
                        "content": system_prompt
                    })

                logger.info(
                    f"[{request_id}] Added memory context to messages")
            except Exception as e:
                logger.warning(
                    f"[{request_id}] Failed to add memory context: {e}")

        # Apply pipeline if present
        processed_messages = chat_request.messages
        if pipeline:
            logger.debug(f"[{request_id}] Applying pipeline: {pipeline.name}")
            try:
                pipeline_data = await pipeline.pipe({"messages": chat_request.messages})
                logger.debug(
                    f"[{request_id}] Pipeline result: {pipeline_data}")

                if "messages" in pipeline_data and pipeline_data["messages"]:
                    processed_messages = pipeline_data["messages"]
                else:
                    logger.warning(
                        f"[{request_id}] Pipeline returned empty messages, using original messages")

                # Handle pipeline summary
                if "summary" in pipeline_data:
                    pipeline_summary = pipeline_data["summary"]
                    # Send initial summary event
                    yield ChatStreamEvent(
                        event="pipeline",
                        data=json.dumps({
                            "summary": pipeline_summary,
                            "status": "processing"
                        })
                    )

                    # Process detailed results if available
                    if isinstance(pipeline_summary, dict):
                        for key, value in pipeline_summary.items():
                            if isinstance(value, list):
                                for item in value:
                                    if item:  # Only send non-empty items
                                        yield ChatStreamEvent(
                                            event="pipeline",
                                            data=json.dumps({
                                                "content_type": key,
                                                "content": item,
                                                "status": "processing"
                                            })
                                        )
                                        await asyncio.sleep(0.1)

                    # Send completion event
                    yield ChatStreamEvent(
                        event="pipeline",
                        data=json.dumps({
                            "status": "complete",
                            "summary": pipeline_summary
                        })
                    )

            except Exception as e:
                logger.error(f"[{request_id}] Pipeline error: {e}")
                yield ChatStreamEvent(
                    event="error",
                    data=json.dumps({"error": f"Pipeline error: {str(e)}"})
                )
                return

        # Log the context window before streaming response
        if chat_request.messages:
            print(f"\nContext window for request {request_id}:", flush=True)
            for msg in chat_request.messages:
                role = msg.get("role") if isinstance(msg, dict) else msg.role
                content = msg.get("content") if isinstance(
                    msg, dict) else msg.content
                print(f"{role}: {content}", flush=True)
            print("\nResponse:", flush=True)

        # Stream the response
        first_response = True
        async for chunk in agent.chat(
            messages=processed_messages,
            model=model,
            temperature=chat_request.temperature or config.MODEL_TEMPERATURE,
            max_tokens=chat_request.max_tokens or config.MAX_TOKENS,
            stream=True,
            tools=function_schemas if chat_request.enable_tools else None,
            enable_tools=chat_request.enable_tools,
            enable_memory=chat_request.enable_memory,
            memory_filter=chat_request.memory_filter,
            top_k_memories=chat_request.top_k_memories
        ):
            if first_response:
                yield ChatStreamEvent(event="start", data=json.dumps({"status": "streaming"}))
                first_response = False

            if not is_test and await request.is_disconnected():
                logger.info(f"[{request_id}] Client disconnected")
                if tool_call_in_progress:
                    logger.info(
                        f"[{request_id}] Waiting for tool call to complete")
                    continue
                break

            if isinstance(chunk, dict):
                if "tool_calls" in chunk:
                    # Handle tool calls
                    for event in await handle_tool_calls(request_id, chunk, function_service):
                        yield event
                    tool_call_in_progress = True
                else:
                    # Handle assistant message
                    yield await handle_assistant_message(request_id, chunk, filters)
                    tool_call_in_progress = False
            else:
                # Handle string chunk
                yield await handle_string_chunk(request_id, chunk, filters)

    except Exception as e:
        logger.error(
            f"[{request_id}] Error in chat stream: {e}", exc_info=True)
        yield ChatStreamEvent(
            event="error",
            data=json.dumps({"error": str(e)})
        )


@router.post("/chat/stream", response_model=None)
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent = Depends(get_agent),
    model_service: ModelService = Depends(get_model_service),
    function_service: FunctionService = Depends(get_function_service),
    langchain_service: LangChainService = Depends(get_langchain_service),
    manager: LightRAGManager = Depends(get_lightrag_manager)
) -> EventSourceResponse:
    """Stream chat completions."""
    return EventSourceResponse(
        stream_chat_response(
            request,
            chat_request,
            agent,
            model_service,
            function_service,
            langchain_service,
            manager
        )
    )


class ChatRequest(BaseModel):
    """Chat request model."""
    messages: List[StrictChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = True
    enable_tools: bool = True
    filters: Optional[List[str]] = None
    pipeline: Optional[str] = None
    memory_type: Optional[str] = MemoryType.EPHEMERAL
    conversation_id: Optional[str] = None
    enable_summarization: Optional[bool] = False
    memory_filter: Optional[Dict[str, Any]] = None
    top_k_memories: Optional[int] = 5


@router.post("/chat/memory/add")
async def add_memory(
    request: Request,
    memory_text: str,
    memory_type: MemoryType = MemoryType.EPHEMERAL,
    metadata: Optional[Dict[str, Any]] = None,
    manager: LightRAGManager = Depends(get_lightrag_manager)
):
    """Add a memory to the specified collection."""
    try:
        # Add to memory using manager's add_to_memory
        await manager.add_to_memory(text=memory_text, entity_name=None, entity_type=None)
        # Optionally handle metadata if provided
        if metadata:
            for key, value in metadata.items():
                manager.add_entity_metadata(entity_id=value.get("entity_id"), key=key, value=value.get("value"))

        return {"status": "success", "message": "Memory added successfully."}
    except Exception as e:
        logger.error(f"Error adding memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to add memory: {e}")


@router.post("/chat/parse")
async def parse_chat_message(
    message: str,
    user_id: str,
    manager: LightRAGManager = Depends(get_lightrag_manager)
):
    """Parse chat message for entities and relationships."""
    try:
        # Insert text into LightRAG
        await manager.insert_text(message)

        # Perform advanced NER and relationship inference
        await advanced_ner_and_relationship_inference(message, user_id, manager)

        return {"status": "success", "message": "Message parsed successfully"}
    except Exception as e:
        logger.error(f"Error parsing chat message: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to parse chat message: {e}")
```

### 2.6. Implement Advanced NER and Relationship Inference

**Objective:** Implement an advanced system to interpret NER labels and context to determine specific relationship types, enhancing the system's ability to create meaningful connections.

**Actions:**

1. **Create `ner_utils.py`**: Implement functions for advanced NER and relationship inference.
2. **Enhance `interpret_relation` Method**: Incorporate contextual analysis to determine relationship types beyond simple rules.

**Create `ner_utils.py`:**

```python
# app/memory/lightrag/ner_utils.py
"""Utilities for Named Entity Recognition and Relationship Inference."""

import logging
from typing import List, Dict
from .manager import LightRAGManager

logger = logging.getLogger(__name__)

import spacy

# Load the advanced NER model once
nlp = spacy.load("en_core_web_trf")

async def advanced_ner_and_relationship_inference(text: str, user_id: str, manager: LightRAGManager):
    """
    Perform NER on the given text and infer relationships based on entities and context.

    Args:
        text: The input text to process
        user_id: The ID of the user associated with the text
        manager: The LightRAGManager instance
    """
    doc = nlp(text)
    for ent in doc.ents:
        ent_text = ent.text.strip()
        ent_label = ent.label_

        # Log extracted entities
        logger.debug(f"Extracted Entity: {ent_text} [{ent_label}]")

        # Interpret relationship based on label and context
        relation_type = manager.interpret_relation(ent_label, text)

        if relation_type:
            # Link or create the entity
            entity_id = manager.link_entity(ent_text, entity_type=map_label_to_entity_type(ent_label))

            # Create relationship with user
            manager.create_relationship(src_id=user_id, dst_id=entity_id, relation_type=relation_type, confidence=1.0)

            logger.info(f"Linked entity '{ent_text}' as '{relation_type}' for user '{user_id}'")
        else:
            logger.debug(f"No relationship inferred for entity '{ent_text}' with label '{ent_label}'")


def map_label_to_entity_type(label: str) -> str:
    """
    Map NER labels to predefined entity types.

    Args:
        label: The NER label (e.g., "DOG", "PERSON")

    Returns:
        A string representing the entity type
    """
    label = label.upper()
    mapping = {
        "PERSON": "person",
        "ORG": "organization",
        "GPE": "location",
        "LOC": "location",
        "PRODUCT": "product",
        "EVENT": "event",
        "WORK_OF_ART": "art",
        "LAW": "law",
        "LANGUAGE": "language",
        "DATE": "date",
        "TIME": "time",
        "MONEY": "money",
        "QUANTITY": "quantity",
        "ORDINAL": "ordinal",
        "CARDINAL": "cardinal",
        "ANIMAL": "animal",
        "DOG": "animal",
        # Add more mappings as needed
    }
    return mapping.get(label, "unknown")
```

### 2.7. Remove or Update `chroma_service`

**Objective:** Eliminate dependencies on `chroma_service` and ensure all memory operations are handled by `LightRAGManager`.

**Actions:**

1. **Remove `chroma_service` Dependency**: Delete or refactor any references to `chroma_service` in `routers/chat.py` and other relevant files.
2. **Ensure All Memory Operations Use `LightRAGManager`**: Replace instances where `chroma_service` was used with equivalent methods from `LightRAGManager`.

**Implementation:**

- **Delete `chroma_service.py`** if it's no longer needed.
- **Update `/chat/memory/add` Endpoint**: Replace `chroma_service.add_memory` with `manager.add_to_memory`.

**Updated `/chat/memory/add` Endpoint:**

```python
@router.post("/chat/memory/add")
async def add_memory(
    request: Request,
    memory_text: str,
    memory_type: MemoryType = MemoryType.EPHEMERAL,
    metadata: Optional[Dict[str, Any]] = None,
    manager: LightRAGManager = Depends(get_lightrag_manager)
):
    """Add a memory to the specified collection."""
    try:
        # Add to memory using manager's add_to_memory
        await manager.add_to_memory(text=memory_text, entity_name=None, entity_type=None)

        # Optionally handle metadata if provided
        if metadata:
            for key, value in metadata.items():
                # Assuming metadata includes entity_id if applicable
                if "entity_id" in value:
                    manager.add_entity_metadata(entity_id=value["entity_id"], key=key, value=value["value"])
                else:
                    # Handle global or context-specific metadata
                    manager.add_entity_metadata(entity_id="global", key=key, value=value["value"])

        return {"status": "success", "message": "Memory added successfully."}
    except Exception as e:
        logger.error(f"Error adding memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to add memory: {e}")
```

### 2.8. Expose `add_to_memory` Functionality

**Objective:** Provide an API endpoint or function that allows the model to add new memories, both unstructured text and structured relationships.

**Actions:**

1. **Ensure `/chat/memory/add` Endpoint Uses `add_to_memory`**: Already handled in the previous step.
2. **Enable Function Calls for LLM to Invoke `add_to_memory`**: If using function calling, integrate accordingly.

**Implementation:**

- **Ensure the `/chat/memory/add` Endpoint is Correctly Configured**: As shown above.
- **Optionally, Define Function Call Handlers**: If the LLM is to call functions directly, define appropriate handlers.

### 2.9. Pass DB Relations as System Prompt Context

**Objective:** Enhance the LLM's responses by injecting relational context into its system prompt, enabling "human-like" references.

**Actions:**

1. **Modify `query_with_relations` to Return System Prompt**: Already implemented in `manager.py`.
2. **Inject System Prompt into Chat Messages**: Ensure that the system prompt is the first message in the chat history.

**Implementation:**

- **Ensure `stream_chat_response` Injects System Prompt**: As shown in the updated `stream_chat_response` function.

---

## 3. Code Review and Implementation Recommendations

Below is a detailed review of your current implementation, highlighting necessary changes to fully adhere to the guide.

### 3.1. `lightrag/__init__.py`

**Current Content:**

```python
"""LightRAG memory system implementation."""
```

**Recommendation:**

- **No changes needed.** This is sufficient for module initialization.

### 3.2. `lightrag/config.py`

**Current Content:**

```python
"""Configuration for LightRAG memory system."""

import os
from pathlib import Path

# Base directory for LightRAG data
LIGHTRAG_DATA_DIR = Path(os.getenv("LIGHTRAG_DATA_DIR", "./light_rag_data"))

# Database configuration
DB_PATH = Path(os.getenv("MEMORY_DB_PATH", "./memory_relations.db"))

# Chunking configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Embedding configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

**Recommendation:**

- **Enhance Embedding Configuration**: If you're planning to use embedding-based disambiguation, ensure that embedding model configurations are present.
- **Add NER Model Configuration**: Specify the NER model to use.

**Updated `config.py`:**

```python
"""Configuration for LightRAG memory system."""

import os
from pathlib import Path

# Base directory for LightRAG data
LIGHTRAG_DATA_DIR = Path(os.getenv("LIGHTRAG_DATA_DIR", "./light_rag_data"))

# Database configuration
DB_PATH = Path(os.getenv("MEMORY_DB_PATH", "./memory_relations.db"))

# Chunking configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Embedding configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# NER model configuration
NER_MODEL = "en_core_web_trf"
```

### 3.3. `lightrag/datastore.py`

**Current Content:**

```python
"""Relational database operations for LightRAG memory system."""

import sqlite3
import uuid
from typing import Dict, List, Optional
from pathlib import Path
from .config import DB_PATH


class MemoryDatastore:
    """Handles operations for the memory relational database."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._ensure_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self):
        """Ensure all required tables exist in the database."""
        conn = self._connect()
        cur = conn.cursor()

        # Create entities table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            entity_type TEXT,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create relationships table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            src_entity TEXT NOT NULL,
            dst_entity TEXT NOT NULL,
            relation_type TEXT,
            confidence REAL DEFAULT 1.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create metadata table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
        conn.close()

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Retrieve an entity by its ID."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM entities WHERE id = ?
        """, (entity_id,))
        row = cur.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "description": row[3],
                "created_at": row[4],
                "updated_at": row[5]
            }
        return None

    def search_entities(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search for entities by name or description."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM entities
            WHERE name LIKE ? OR description LIKE ?
            LIMIT ?
        """, (f"%{search_term}%", f"%{search_term}%", limit))
        rows = cur.fetchall()
        conn.close()

        return [{
            "id": row[0],
            "name": row[1],
            "entity_type": row[2],
            "description": row[3],
            "created_at": row[4],
            "updated_at": row[5]
        } for row in rows]

    def get_entity_metadata(self, entity_id: str) -> Dict[str, str]:
        """Retrieve all metadata for a given entity."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT key, value FROM metadata WHERE entity_id = ?
        """, (entity_id,))
        rows = cur.fetchall()
        conn.close()

        return {row[0]: row[1] for row in rows}

    def add_entity_metadata(self, entity_id: str, key: str, value: str):
        """Add metadata to an entity."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO metadata (id, entity_id, key, value)
            VALUES (?, ?, ?, ?)
        """, (str(uuid.uuid4()), entity_id, key, value))
        conn.commit()
        conn.close()
```

**Recommendations:**

1. **Add Synonym/Alias Management**: Introduce a `synonyms` table and relevant methods.
2. **Implement Embedding-Based Disambiguation**: (Optional) Add methods to handle embeddings for entities.

**Updated `datastore.py`:**

(Already provided in **2.2. Update `datastore.py`** above.)

### 3.4. `lightrag/ingestion.py`

**Current Content:**

```python
"""Document and file ingestion for LightRAG memory system."""

import logging
from pathlib import Path
from typing import Union
from .manager import LightRAGManager

logger = logging.getLogger(__name__)


class MemoryIngestor:
    """Handles ingestion of various content types into the memory system."""

    def __init__(self, manager: LightRAGManager):
        self.manager = manager

    async def ingest_text(self, text: str, metadata: dict = None):
        """
        Ingest plain text content into the memory system.

        Args:
            text: The text content to ingest
            metadata: Optional metadata to associate with the content
        """
        if not text.strip():
            return

        logger.info(f"Ingesting text content (length: {len(text)})")
        await self.manager.insert_text(text)

        # TODO: Process metadata and store in relational database
        if metadata:
            logger.debug(f"Processing metadata: {metadata}")

    async def ingest_file(self, file_path: Union[str, Path]):
        """
        Ingest content from a file into the memory system.

        Args:
            file_path: Path to the file to ingest
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return

        logger.info(f"Ingesting file: {path.name}")

        # TODO: Implement file type specific processing
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            await self.ingest_text(content)
        except Exception as e:
            logger.error(f"Failed to ingest file {path}: {str(e)}")

    async def ingest_directory(self, dir_path: Union[str, Path]):
        """
        Ingest all supported files from a directory.

        Args:
            dir_path: Path to the directory to ingest
        """
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            logger.error(f"Directory not found: {path}")
            return

        logger.info(f"Ingesting directory: {path}")

        # TODO: Implement directory traversal and file ingestion
        for file_path in path.iterdir():
            if file_path.is_file():
                await self.ingest_file(file_path)
```

**Recommendations:**

1. **Process Metadata**: Integrate metadata processing to store additional information about ingested text.
2. **Handle Various File Types**: Enhance file ingestion to handle different formats (e.g., PDF, DOCX) using appropriate libraries.

**Updated `ingestion.py`:**

(Already provided in **2.3. Enhance `ingestion.py`** above.)

### 3.5. `lightrag/manager.py`

**Current Content:**

```python
"""Core memory manager combining LightRAG and relational database."""

import logging
import sqlite3
import uuid
from typing import List, Optional

from lightrag import LightRAG, QueryParam
from .config import LIGHTRAG_DATA_DIR, DB_PATH

logger = logging.getLogger(__name__)


class LightRAGManager:
    """
    Central memory manager combining:
    1) LightRAG for unstructured, chunk-based retrieval
    2) A relational DB for entities, relationships, and metadata
    """

    def __init__(self, working_dir: str = LIGHTRAG_DATA_DIR, db_path: str = DB_PATH):
        logger.info("Initializing LightRAGManager with relational DB...")

        # Initialize LightRAG
        self.rag = LightRAG(working_dir=str(working_dir))

        # Initialize relational store
        self.db_path = db_path
        self._ensure_tables()

    def _connect_db(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self):
        conn = self._connect_db()
        cur = conn.cursor()

        # Create entities table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            entity_type TEXT,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create relationships table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            src_entity TEXT NOT NULL,
            dst_entity TEXT NOT NULL,
            relation_type TEXT,
            confidence REAL DEFAULT 1.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create metadata table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
        conn.close()
        logger.debug("Ensured relational DB tables exist.")

    # LightRAG methods
    async def insert_text(self, text: str):
        """Add unstructured text to LightRAG for chunk-based retrieval."""
        if not text.strip():
            return
        logger.debug(f"Inserting text (size={len(text)}) into LightRAG.")
        await self.rag.insert(text)

    async def query_rag(self, query_str: str, mode: str = "mix", top_k: int = 5) -> str:
        """Use LightRAG to retrieve/generate an answer with vector/graph context."""
        param = QueryParam(mode=mode, top_k=top_k)
        result = await self.rag.query(query_str, param=param)
        return result

    # Relational store methods
    def create_entity(self, name: str, entity_type: str, description: str = "") -> str:
        entity_id = str(uuid.uuid4())
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO entities (id, name, entity_type, description)
            VALUES (?, ?, ?, ?);
        """, (entity_id, name, entity_type, description))
        conn.commit()
        conn.close()
        logger.debug(f"Created entity {name} ({entity_type}). ID={entity_id}")
        return entity_id

    def create_relationship(self, src_id: str, dst_id: str, relation_type: str, confidence: float = 1.0):
        rel_id = str(uuid.uuid4())
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO relationships (id, src_entity, dst_entity, relation_type, confidence)
            VALUES (?, ?, ?, ?, ?);
        """, (rel_id, src_id, dst_id, relation_type, confidence))
        conn.commit()
        conn.close()
        logger.debug(
            f"Created relationship {rel_id}: {src_id} --{relation_type}--> {dst_id}")

    def get_entity_relations(self, entity_id: str):
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, src_entity, dst_entity, relation_type, confidence, created_at, updated_at
            FROM relationships
            WHERE src_entity=? OR dst_entity=?
        """, (entity_id, entity_id))
        rows = cur.fetchall()
        conn.close()

        relations = []
        for row in rows:
            relations.append({
                "id": row[0],
                "src_entity": row[1],
                "dst_entity": row[2],
                "relation_type": row[3],
                "confidence": row[4],
                "created_at": row[5],
                "updated_at": row[6]
            })
        return relations

    async def add_to_memory(self, text: str, entity_name: Optional[str] = None, entity_type: Optional[str] = None):
        """
        Called by the model if it needs to store something in memory or relational DB.
        """
        await self.insert_text(text)
        if entity_name:
            e_id = self.link_entity(entity_name, entity_type or "unknown")
            logger.debug(f"Model added entity {entity_name} to memory: {e_id}")

    def link_entity(self, name: str, entity_type: str) -> str:
        """Find or create entity by name. Return the entity ID."""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM entities
            WHERE LOWER(name) = LOWER(?)
            LIMIT 1
        """, (name,))
        row = cur.fetchone()
        if row:
            entity_id = row[0]
            cur.execute(
                "UPDATE entities SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (entity_id,))
            conn.commit()
            conn.close()
            return entity_id
        else:
            conn.close()
            return self.create_entity(name, entity_type)
```

**Recommendations:**

- **Remove Duplicate Methods**: The earlier **refactored `manager.py`** already includes enhanced methods for entity linking and relationship inference.
- **Integrate `extract_entities` and `build_system_prompt`**: As shown in the updated `manager.py` above.
- **Ensure All Methods are Present and Correct**: Verify that all necessary methods from the guide are implemented.

### 3.6. `lightrag/tasks.py`

**Current Content:**

```python
"""Background tasks and maintenance for LightRAG memory system."""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from .manager import LightRAGManager

logger = logging.getLogger(__name__)


class MemoryTasks:
    """Handles background tasks and maintenance operations."""

    def __init__(self, manager: LightRAGManager):
        self.manager = manager
        self._running = False

    async def start(self):
        """Start background tasks."""
        if self._running:
            return

        self._running = True
        logger.info("Starting memory background tasks")

        # Start cleanup task
        asyncio.create_task(self._cleanup_task())

    async def stop(self):
        """Stop background tasks."""
        self._running = False
        logger.info("Stopping memory background tasks")

    async def _cleanup_task(self):
        """Periodic cleanup of old or unused memory entries."""
        while self._running:
            try:
                logger.info("Running memory cleanup task")
                # TODO: Implement actual cleanup logic
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Cleanup task failed: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

    async def summarize_memory(self, entity_id: Optional[str] = None):
        """
        Generate summaries for memory content.

        Args:
            entity_id: Optional specific entity to summarize
        """
        logger.info(f"Generating memory summary for entity: {entity_id}")
        # TODO: Implement summarization logic
        return "Summary not implemented yet"

    async def optimize_memory(self):
        """Optimize memory storage and retrieval performance."""
        logger.info("Optimizing memory storage")
        # TODO: Implement optimization logic
        return "Optimization not implemented yet"
```

**Recommendations:**

1. **Implement Cleanup Logic**: As shown in **2.4. Revise `tasks.py`** above.
2. **Implement Summarization Logic**: Integrate with LightRAG's summarization capabilities or use an external model.
3. **Integrate `MemoryTasks` with Application Startup**: Ensure that background tasks start when the application starts and stop gracefully.

**Integration Example:**

```python
# app/main.py

from fastapi import FastAPI
from app.memory.lightrag.manager import LightRAGManager
from app.memory.lightrag.tasks import MemoryTasks

app = FastAPI()

# Initialize LightRAGManager
manager = LightRAGManager()

# Initialize MemoryTasks
memory_tasks = MemoryTasks(manager=manager)

@app.on_event("startup")
async def startup_event():
    await memory_tasks.start()

@app.on_event("shutdown")
async def shutdown_event():
    await memory_tasks.stop()
```

### 3.7. `routers/chat.py`

**Current Content:**

```python
"""Chat router for handling chat-related endpoints and streaming responses."""
# [Code as provided by the user]
```

**Recommendations:**

1. **Ensure All Memory Operations Use `LightRAGManager`**: Replace `chroma_service` with `manager`.
2. **Implement Advanced NER and Relationship Inference**: Utilize `ner_utils.py`.
3. **Expose `add_to_memory` Correctly**: Ensure the `/chat/memory/add` endpoint uses `manager.add_to_memory`.
4. **Implement `/chat/query_entity_linked`**: Add a new endpoint for querying with relational context.

**Final Updated `chat.py`:**

(As provided in **2.5. Update `routers/chat.py`** above.)

### 3.8. Implement Advanced NER and Relationship Inference

**Objective:** Utilize `ner_utils.py` to perform advanced NER and relationship inference within `manager.py`.

**Actions:**

1. **Ensure `ner_utils.py` is Correctly Implemented**: As shown in **2.6. Implement Advanced NER and Relationship Inference** above.
2. **Integrate `ner_utils.py` with `manager.py`**: Ensure that `manager.py` imports and uses `ner_utils.py` functions.

**Implementation:**

- **Ensure `manager.py` Includes Necessary Methods**: Already handled in the updated `manager.py`.

### 3.9. Expose `add_to_memory` Functionality

**Objective:** Allow the model or users to add new memories via an API endpoint, using `LightRAGManager`.

**Actions:**

1. **Ensure `/chat/memory/add` Uses `add_to_memory`**: As shown in **2.5. Update `routers/chat.py`** above.
2. **Define Metadata Handling**: Ensure that any metadata is correctly processed and stored.

**Final Endpoint Implementation:**

(As shown in **2.5. Update `routers/chat.py`** above.)

### 3.10. Pass DB Relations as System Prompt Context

**Objective:** Enhance LLM responses by injecting relational data into the system prompt, enabling contextual and accurate responses.

**Actions:**

1. **Ensure `query_with_relations` Builds System Prompt**: As implemented in the updated `manager.py`.
2. **Inject System Prompt into Chat Messages**: Modify `stream_chat_response` to prepend system prompt.

**Implementation:**

- **In `stream_chat_response`, Insert System Prompt**: Already handled in the updated `stream_chat_response` function above.

---

## 4. Final Verification and Testing

**Objective:** Ensure that all components are correctly implemented and integrated, adhering to the guide.

**Actions:**

1. **Unit Testing**:

   - **Test Entity Creation**: Verify that entities are correctly created, retrieved, and updated.
   - **Test Relationship Creation**: Ensure relationships are accurately established and retrievable.
   - **Test Memory Ingestion**: Confirm that text and files are ingested into LightRAG and relational store.
   - **Test NER and Relationship Inference**: Validate that entities are correctly extracted and relationships inferred.
   - **Test `add_to_memory` Functionality**: Ensure that adding to memory works as intended.
   - **Test Querying with Relations**: Verify that queries return both LightRAG text and relational context.

2. **Integration Testing**:

   - **End-to-End Flow**: Simulate user interactions to confirm that the system behaves as expected.
   - **Error Handling**: Test how the system handles erroneous inputs and unexpected scenarios.
   - **Performance Testing**: Assess the system's performance with large datasets and high query volumes.

3. **Manual Testing**:

   - **Use Chat Interface**: Interact with the chat endpoints to ensure responses are contextually enriched.
   - **Verify System Prompts**: Check that system prompts are correctly constructed and injected.

4. **Logging and Monitoring**:
   - **Monitor Logs**: Ensure that all operations are logged appropriately for debugging and monitoring.
   - **Set Up Alerts**: Configure alerts for critical failures or performance issues.

---

## 5. Conclusion

By following this **optimized integration guide**, you will transform your current implementation into a **state-of-the-art, human-like memory system**. The system will efficiently retrieve unstructured text via **LightRAG**, enrich it with **relational data** from a **SQLite relational store**, and provide **contextually aware responses** by integrating relational context into the LLM's prompts. Advanced **NER** and **relationship inference** ensure that entities are accurately recognized and linked, enhancing the system's ability to maintain a coherent and enriched memory akin to human cognition.

**Key Benefits:**

- **Fast Retrieval**: Leveraging LightRAG's vector-based search for quick access to relevant text chunks.
- **Contextual Enrichment**: Combining unstructured retrieval with structured relational data for enriched responses.
- **Advanced Entity Linking**: Utilizing sophisticated NER and inference mechanisms to accurately map entities and relationships.
- **Scalability and Flexibility**: Modular design allows for easy future enhancements, such as integrating a full-fledged GraphDB or implementing embedding-based disambiguation.
- **Human-Like Continuity**: System prompts enriched with relational context enable the LLM to produce more natural and context-aware responses.

**Next Steps:**

1. **Implement All Recommendations**: Follow the step-by-step migration guide to update and enhance your codebase.
2. **Comprehensive Testing**: Rigorously test all components to ensure seamless integration and functionality.
3. **Iterate and Improve**: Based on testing outcomes, iterate on your implementation to address any identified issues and further optimize performance.

**Enjoy building a robust, human-like memory system that elevates your application's conversational capabilities!**
