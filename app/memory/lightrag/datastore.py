"""Enhanced relational database operations for LightRAG memory system."""

import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, TypeVar
from pathlib import Path
from .config import DB_PATH, DEFAULT_RETENTION_DAYS, LONG_TERM_RETENTION_DAYS, SHORT_TERM_RETENTION_DAYS

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic type for database operations


class MemoryDatastore:
    """Handles operations for the enhanced memory relational database."""

    _COLUMN_MAPPINGS = {
        'entity': [
            'id', 'name', 'entity_type', 'description', 'version',
            'created_at', 'updated_at', 'retention_policy', 'hierarchy_level', 'parent_id'
        ],
        'relationship': [
            'id', 'src_entity', 'dst_entity', 'relation_type', 'confidence',
            'version', 'created_at', 'updated_at'
        ],
        'entity_history': [
            'id', 'entity_id', 'version', 'name', 'entity_type',
            'description', 'changed_at'
        ],
        'access_control': [
            'id', 'entity_id', 'access_level', 'granted_to',
            'created_at', 'updated_at'
        ],
        'collection': [
            'id', 'name', 'description', 'created_at', 'updated_at'
        ]
    }

    def __init__(self):
        """Initialize the memory datastore."""
        # Ensure database directory exists
        self.db_path = Path(DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database structure
        self._ensure_tables()
        self._setup_indexes()

        logger.debug("Memory datastore initialized")

    def _connect(self) -> sqlite3.Connection:
        """Create a new database connection.

        Returns:
            sqlite3.Connection: A new connection to the database
        """
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self):
        """Ensure all required tables exist in the database."""
        def operation(cur: sqlite3.Cursor) -> None:
            # Create entities table with enhanced hierarchy support
            cur.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT,
                description TEXT,
                version INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                retention_policy TEXT DEFAULT 'default',
                hierarchy_level TEXT DEFAULT 'chunk',
                parent_id TEXT,
                FOREIGN KEY (parent_id) REFERENCES entities(id) ON DELETE CASCADE
            );
            """)

            # Create entity history table for version tracking
            cur.execute("""
            CREATE TABLE IF NOT EXISTS entity_history (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                name TEXT,
                entity_type TEXT,
                description TEXT,
                hierarchy_level TEXT,
                parent_id TEXT,
                changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );
            """)

            # Create relationships table with versioning
            cur.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                src_entity TEXT NOT NULL,
                dst_entity TEXT NOT NULL,
                relation_type TEXT,
                confidence REAL DEFAULT 1.0,
                version INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (src_entity) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (dst_entity) REFERENCES entities(id) ON DELETE CASCADE
            );
            """)

            # Create relationship history table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS relationship_history (
                id TEXT PRIMARY KEY,
                relationship_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                src_entity TEXT,
                dst_entity TEXT,
                relation_type TEXT,
                confidence REAL,
                changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (relationship_id) REFERENCES relationships(id) ON DELETE CASCADE
            );
            """)

            # Create metadata table with versioning
            cur.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                version INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );
            """)

            # Create synonyms table with versioning
            cur.execute("""
            CREATE TABLE IF NOT EXISTS synonyms (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                synonym TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );
            """)

            # Create access control table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS access_controls (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                access_level TEXT NOT NULL,
                granted_to TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );
            """)

            # Create collections table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Create collection members table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS collection_members (
                id TEXT PRIMARY KEY,
                collection_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );
            """)

        self._execute_with_connection(
            operation,
            "Failed to create database tables"
        )

    def _setup_indexes(self):
        """Create necessary indexes for optimized queries."""
        def operation(cur: sqlite3.Cursor) -> None:
            # Indexes for entities
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_retention ON entities(retention_policy)")

            # Indexes for relationships
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_src ON relationships(src_entity)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_dst ON relationships(dst_entity)")

            # Indexes for metadata
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_key ON metadata(key)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_entity ON metadata(entity_id)")

        self._execute_with_connection(
            operation,
            "Failed to create database indexes"
        )

    def _execute_with_connection(self, operation: Callable[[sqlite3.Cursor], T], error_msg: str) -> T:
        """Execute a database operation with proper connection management and error handling.

        Args:
            operation: Callable that takes a cursor and returns a value
            error_msg: Error message to log if operation fails

        Returns:
            The result of the operation

        Raises:
            sqlite3.Error: If there is a database error
        """
        conn = self._connect()
        cur = conn.cursor()
        try:
            result = operation(cur)
            conn.commit()
            return result
        except sqlite3.Error as e:
            logger.error(f"{error_msg}: {e}")
            raise
        finally:
            conn.close()

    def _row_to_dict(self, row: Optional[Tuple], mapping_type: str) -> Optional[Dict]:
        """Convert a database row to a dictionary using predefined column mappings.

        Args:
            row: Database row tuple
            mapping_type: Type of mapping to use from _COLUMN_MAPPINGS

        Returns:
            Dict containing mapped data or None if row is None
        """
        if not row:
            return None
        return {
            col: row[i] for i, col in enumerate(self._COLUMN_MAPPINGS[mapping_type])
        }

    def _rows_to_dicts(self, rows: List[Tuple], mapping_type: str) -> List[Dict]:
        """Convert multiple database rows to dictionaries.

        Args:
            rows: List of database row tuples
            mapping_type: Type of mapping to use from _COLUMN_MAPPINGS

        Returns:
            List of dicts containing mapped data
        """
        return [self._row_to_dict(row, mapping_type) for row in rows]

    def _row_to_entity(self, row: Optional[Tuple]) -> Optional[Dict]:
        """Convert a database row to an entity dictionary."""
        return self._row_to_dict(row, 'entity')

    def _row_to_relationship(self, row: Optional[Tuple]) -> Optional[Dict]:
        """Convert a database row to a relationship dictionary."""
        return self._row_to_dict(row, 'relationship')

    def _create_history_record(self, cur: sqlite3.Cursor, table: str, record_id: str, version: int, **values) -> None:
        """Create a history record for an entity or relationship.

        Args:
            cur: Database cursor
            table: The type of record ('entity' or 'relationship')
            record_id: The ID of the record to create history for
            version: The version number
            values: Additional values to insert
        """
        if table == 'entity':
            cur.execute("""
                INSERT INTO entity_history (id, entity_id, version, name, entity_type, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), record_id, version, values['name'], values['entity_type'], values['description']))
        elif table == 'relationship':
            cur.execute("""
                INSERT INTO relationship_history (id, relationship_id, version, src_entity, dst_entity, relation_type, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), record_id, version, values['src_id'], values['dst_id'], values['relation_type'], values['confidence']))

    def create_entity(self, entity_id: str, name: str, entity_type: str, description: str = "") -> Dict:
        """Create a new entity with version tracking."""
        def operation(cur: sqlite3.Cursor) -> Dict:
            # Insert new entity
            cur.execute("""
                INSERT INTO entities (id, name, entity_type, description)
                VALUES (?, ?, ?, ?)
            """, (entity_id, name, entity_type, description))

            # Create initial history record
            self._create_history_record(cur, 'entity', entity_id, 1,
                                        name=name, entity_type=entity_type, description=description)

            return self.get_entity(entity_id)

        return self._execute_with_connection(
            operation,
            f"Failed to create entity: {entity_id}"
        )

    def update_entity(self, entity_id: str, **kwargs) -> Dict:
        """Update an entity with version tracking."""
        def operation(cur: sqlite3.Cursor) -> Dict:
            # Get current version
            cur.execute(
                "SELECT version FROM entities WHERE id = ?", (entity_id,))
            current_version = cur.fetchone()[0]
            new_version = current_version + 1

            # Update entity
            set_clause = ", ".join(f"{k} = ?" for k in kwargs)
            values = list(kwargs.values()) + [entity_id]
            cur.execute(f"""
                UPDATE entities 
                SET {set_clause}, version = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (*values, new_version, entity_id))

            # Create history record
            self._create_history_record(
                cur, 'entity', entity_id, new_version, **kwargs)

            return self.get_entity(entity_id)

        return self._execute_with_connection(
            operation,
            f"Failed to update entity: {entity_id}"
        )

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Retrieve an entity by its ID."""
        def operation(cur: sqlite3.Cursor) -> Optional[Dict]:
            cur.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
            return self._row_to_entity(cur.fetchone())

        return self._execute_with_connection(
            operation,
            f"Failed to get entity: {entity_id}"
        )

    def get_entity_by_name(self, name: str) -> Optional[Dict]:
        """Retrieve an entity by its name (case-insensitive)."""
        def operation(cur: sqlite3.Cursor) -> Optional[Dict]:
            cur.execute("""
                SELECT * FROM entities
                WHERE LOWER(name) = LOWER(?)
                LIMIT 1
            """, (name,))
            return self._row_to_entity(cur.fetchone())

        return self._execute_with_connection(
            operation,
            f"Failed to get entity by name: {name}"
        )

    def add_entity_metadata(self, entity_id: str, key: str, value: str, version: int = 1) -> str:
        """Add metadata to an entity with version tracking.

        Args:
            entity_id: The ID of the entity to add metadata to
            key: The metadata key
            value: The metadata value
            version: The version number for this metadata

        Returns:
            str: The ID of the created metadata entry

        Raises:
            sqlite3.Error: If there is a database error
            ValueError: If the entity does not exist
        """
        if not self.entity_exists(entity_id):
            raise ValueError(f"Entity {entity_id} does not exist")

        metadata_id = str(uuid.uuid4())

        def operation(cur: sqlite3.Cursor) -> str:
            cur.execute("""
                INSERT INTO metadata (id, entity_id, key, value, version)
                VALUES (?, ?, ?, ?, ?)
            """, (metadata_id, entity_id, key, value, version))
            logger.debug(
                f"Added metadata {key}: {value} to entity {entity_id}")
            return metadata_id

        return self._execute_with_connection(
            operation,
            f"Failed to add metadata to entity {entity_id}"
        )

    def cleanup_entities_by_policy(self, policy_name: str) -> int:
        """Remove entities based on their retention policy.

        Args:
            policy_name: The name of the retention policy to apply

        Returns:
            int: Number of entities removed

        Raises:
            sqlite3.Error: If there is a database error
            ValueError: If the policy name is invalid
        """
        if policy_name not in ["short_term", "long_term", "default"]:
            raise ValueError(f"Invalid retention policy: {policy_name}")

        cutoff = datetime.now()
        if policy_name == "short_term":
            cutoff -= timedelta(days=SHORT_TERM_RETENTION_DAYS)
        elif policy_name == "long_term":
            cutoff -= timedelta(days=LONG_TERM_RETENTION_DAYS)
        else:  # default
            cutoff -= timedelta(days=DEFAULT_RETENTION_DAYS)

        def operation(cur: sqlite3.Cursor) -> int:
            cur.execute("""
                DELETE FROM entities
                WHERE created_at < ? AND retention_policy = ?
            """, (cutoff, policy_name))
            count = cur.rowcount
            logger.info(
                f"Cleaned up {count} entities with policy {policy_name}")
            return count

        return self._execute_with_connection(
            operation,
            f"Failed to cleanup entities with policy {policy_name}"
        )

    def get_entity_history(self, entity_id: str) -> List[Dict]:
        """Get version history for an entity."""
        def operation(cur: sqlite3.Cursor) -> List[Dict]:
            cur.execute("""
                SELECT * FROM entity_history
                WHERE entity_id = ?
                ORDER BY changed_at DESC
            """, (entity_id,))
            return self._rows_to_dicts(cur.fetchall(), 'entity_history')

        return self._execute_with_connection(
            operation,
            f"Failed to get history for entity {entity_id}"
        )

    def create_relationship(self, rel_id: str, src_id: str, dst_id: str, relation_type: str, confidence: float = 1.0) -> Dict:
        """Create a new relationship with version tracking."""
        def operation(cur: sqlite3.Cursor) -> Dict:
            # Insert new relationship
            cur.execute("""
                INSERT INTO relationships (id, src_entity, dst_entity, relation_type, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (rel_id, src_id, dst_id, relation_type, confidence))

            # Create initial history record
            self._create_history_record(cur, 'relationship', rel_id, 1,
                                        src_id=src_id, dst_id=dst_id,
                                        relation_type=relation_type, confidence=confidence)

            return self.get_relationship(rel_id)

        return self._execute_with_connection(
            operation,
            f"Failed to create relationship between {src_id} and {dst_id}"
        )

    def get_relationship(self, rel_id: str) -> Optional[Dict]:
        """Retrieve a relationship by its ID."""
        def operation(cur: sqlite3.Cursor) -> Optional[Dict]:
            cur.execute("SELECT * FROM relationships WHERE id = ?", (rel_id,))
            return self._row_to_relationship(cur.fetchone())

        return self._execute_with_connection(
            operation,
            f"Failed to get relationship: {rel_id}"
        )

    def get_entity_relations(self, entity_id: str) -> List[Dict]:
        """Get all relationships for a given entity."""
        def operation(cur: sqlite3.Cursor) -> List[Dict]:
            cur.execute("""
                SELECT * FROM relationships
                WHERE src_entity = ? OR dst_entity = ?
            """, (entity_id, entity_id))
            return [self._row_to_relationship(row) for row in cur.fetchall()]

        return self._execute_with_connection(
            operation,
            f"Failed to get relations for entity: {entity_id}"
        )

    def search_entities(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search for entities by name, description, or synonyms."""
        def operation(cur: sqlite3.Cursor) -> List[Dict]:
            cur.execute("""
                SELECT e.* FROM entities e
                LEFT JOIN synonyms s ON e.id = s.entity_id
                WHERE e.name LIKE ? OR e.description LIKE ? OR s.synonym LIKE ?
                GROUP BY e.id
                LIMIT ?
            """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", limit))
            return self._rows_to_dicts(cur.fetchall(), 'entity')

        return self._execute_with_connection(
            operation,
            f"Failed to search entities with term: {search_term}"
        )

    def add_access_control(self, entity_id: str, access_level: str, granted_to: str) -> str:
        """Add access control to an entity."""
        access_id = str(uuid.uuid4())
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO access_controls (id, entity_id, access_level, granted_to)
            VALUES (?, ?, ?, ?)
        """, (access_id, entity_id, access_level, granted_to))
        conn.commit()
        conn.close()
        return access_id

    def get_access_controls(self, entity_id: str) -> List[Dict]:
        """Get access controls for an entity."""
        def operation(cur: sqlite3.Cursor) -> List[Dict]:
            cur.execute("""
                SELECT * FROM access_controls
                WHERE entity_id = ?
            """, (entity_id,))
            return self._rows_to_dicts(cur.fetchall(), 'access_control')

        return self._execute_with_connection(
            operation,
            f"Failed to get access controls for entity: {entity_id}"
        )

    def get_entities_older_than(self, cutoff: datetime) -> List[Dict]:
        """Get entities older than the specified cutoff."""
        def operation(cur: sqlite3.Cursor) -> List[Dict]:
            cur.execute("""
                SELECT * FROM entities
                WHERE created_at < ?
            """, (cutoff,))
            return self._rows_to_dicts(cur.fetchall(), 'entity')

        return self._execute_with_connection(
            operation,
            f"Failed to get entities older than {cutoff}"
        )

    def get_entity_parent(self, entity_id: str) -> Optional[Dict]:
        """Get the parent entity of a given entity."""
        def operation(cur: sqlite3.Cursor) -> Optional[Dict]:
            cur.execute("""
                SELECT p.* FROM entities e
                JOIN entities p ON e.parent_id = p.id
                WHERE e.id = ?
            """, (entity_id,))
            return self._row_to_entity(cur.fetchone())

        return self._execute_with_connection(
            operation,
            f"Failed to get parent for entity: {entity_id}"
        )

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its associated data."""
        def operation(cur: sqlite3.Cursor) -> bool:
            cur.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            return cur.rowcount > 0

        return self._execute_with_connection(
            operation,
            f"Failed to delete entity: {entity_id}"
        )

    def delete_entity_metadata(self, entity_id: str) -> int:
        """Delete all metadata associated with an entity."""
        def operation(cur: sqlite3.Cursor) -> int:
            cur.execute(
                "DELETE FROM metadata WHERE entity_id = ?", (entity_id,))
            return cur.rowcount

        return self._execute_with_connection(
            operation,
            f"Failed to delete metadata for entity: {entity_id}"
        )

    def get_all_relationships(self) -> List[Dict]:
        """Get all relationships in the database."""
        def operation(cur: sqlite3.Cursor) -> List[Dict]:
            cur.execute("SELECT * FROM relationships")
            return self._rows_to_dicts(cur.fetchall(), 'relationship')

        return self._execute_with_connection(
            operation,
            "Failed to get all relationships"
        )

    def entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists."""
        def operation(cur: sqlite3.Cursor) -> bool:
            cur.execute("SELECT 1 FROM entities WHERE id = ?", (entity_id,))
            return cur.fetchone() is not None

        return self._execute_with_connection(
            operation,
            f"Failed to check if entity exists: {entity_id}"
        )

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        def operation(cur: sqlite3.Cursor) -> bool:
            cur.execute("DELETE FROM relationships WHERE id = ?",
                        (relationship_id,))
            return cur.rowcount > 0

        return self._execute_with_connection(
            operation,
            f"Failed to delete relationship: {relationship_id}"
        )

    def cleanup_old_entities(self, cutoff: datetime) -> int:
        """Remove entities older than the specified cutoff."""
        def operation(cur: sqlite3.Cursor) -> int:
            cur.execute("""
                DELETE FROM entities
                WHERE created_at < ? AND retention_policy = 'default'
            """, (cutoff,))
            return cur.rowcount

        return self._execute_with_connection(
            operation,
            f"Failed to cleanup old entities before {cutoff}"
        )

    # Memory hierarchy methods
    def set_entity_hierarchy(self, entity_id: str, level: str, parent_id: Optional[str] = None) -> Dict:
        """Set the hierarchy level and parent for an entity."""
        def operation(cur: sqlite3.Cursor) -> Dict:
            cur.execute("""
                UPDATE entities
                SET hierarchy_level = ?, parent_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (level, parent_id, entity_id))

            # Create history record
            self._create_history_record(cur, 'entity', entity_id, 1,
                                        name=level, entity_type='hierarchy',
                                        description=f"Parent: {parent_id}")

            # Return the full entity including hierarchy_level
            entity = self.get_entity(entity_id)
            if entity:
                entity["hierarchy_level"] = level
                if parent_id:
                    entity["parent_id"] = parent_id
            return entity

        return self._execute_with_connection(
            operation,
            f"Failed to set hierarchy for entity: {entity_id}"
        )

    def get_entity_hierarchy(self, entity_id: str) -> Dict:
        """Get the hierarchy information for an entity."""
        def operation(cur: sqlite3.Cursor) -> Dict:
            cur.execute("""
                SELECT hierarchy_level, parent_id FROM entities WHERE id = ?
            """, (entity_id,))
            row = cur.fetchone()
            return {
                "hierarchy_level": row[0] if row else None,
                "parent_id": row[1] if row else None
            }

        return self._execute_with_connection(
            operation,
            f"Failed to get hierarchy for entity: {entity_id}"
        )

    def get_child_entities(self, parent_id: str) -> List[Dict]:
        """Get all child entities for a given parent."""
        def operation(cur: sqlite3.Cursor) -> List[Dict]:
            cur.execute("""
                SELECT * FROM entities WHERE parent_id = ?
            """, (parent_id,))
            return self._rows_to_dicts(cur.fetchall(), 'entity')

        return self._execute_with_connection(
            operation,
            f"Failed to get child entities for parent: {parent_id}"
        )

    # Collection management methods
    def create_collection(self, name: str, description: str = "") -> Dict:
        """Create a new memory collection."""
        collection_id = str(uuid.uuid4())

        def operation(cur: sqlite3.Cursor) -> Dict:
            cur.execute("""
                INSERT INTO collections (id, name, description)
                VALUES (?, ?, ?)
            """, (collection_id, name, description))
            return self.get_collection(collection_id)

        return self._execute_with_connection(
            operation,
            f"Failed to create collection: {name}"
        )

    def get_collection(self, collection_id: str) -> Optional[Dict]:
        """Retrieve a collection by its ID."""
        def operation(cur: sqlite3.Cursor) -> Optional[Dict]:
            cur.execute("SELECT * FROM collections WHERE id = ?",
                        (collection_id,))
            return self._row_to_dict(cur.fetchone(), 'collection')

        return self._execute_with_connection(
            operation,
            f"Failed to get collection: {collection_id}"
        )

    def add_to_collection(self, collection_id: str, entity_id: str) -> str:
        """Add an entity to a collection."""
        member_id = str(uuid.uuid4())

        def operation(cur: sqlite3.Cursor) -> str:
            cur.execute("""
                INSERT INTO collection_members (id, collection_id, entity_id)
                VALUES (?, ?, ?)
            """, (member_id, collection_id, entity_id))
            return member_id

        return self._execute_with_connection(
            operation,
            f"Failed to add entity {entity_id} to collection {collection_id}"
        )

    def remove_from_collection(self, collection_id: str, entity_id: str) -> int:
        """Remove an entity from a collection."""
        def operation(cur: sqlite3.Cursor) -> int:
            cur.execute("""
                DELETE FROM collection_members
                WHERE collection_id = ? AND entity_id = ?
            """, (collection_id, entity_id))
            return cur.rowcount

        return self._execute_with_connection(
            operation,
            f"Failed to remove entity {entity_id} from collection {collection_id}"
        )

    def get_collection_members(self, collection_id: str) -> List[Dict]:
        """Get all entities in a collection."""
        def operation(cur: sqlite3.Cursor) -> List[Dict]:
            cur.execute("""
                SELECT e.* FROM entities e
                JOIN collection_members cm ON e.id = cm.entity_id
                WHERE cm.collection_id = ?
            """, (collection_id,))
            return self._rows_to_dicts(cur.fetchall(), 'entity')

        return self._execute_with_connection(
            operation,
            f"Failed to get members of collection {collection_id}"
        )

    def search_collections(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search for collections by name or description."""
        def operation(cur: sqlite3.Cursor) -> List[Dict]:
            cur.execute("""
                SELECT * FROM collections
                WHERE name LIKE ? OR description LIKE ?
                LIMIT ?
            """, (f"%{search_term}%", f"%{search_term}%", limit))
            return self._rows_to_dicts(cur.fetchall(), 'collection')

        return self._execute_with_connection(
            operation,
            f"Failed to search collections with term: {search_term}"
        )

    def enhanced_search(self, search_term: str, limit: int = 10) -> Dict:
        """Perform an enhanced search across entities and collections."""
        def operation(cur: sqlite3.Cursor) -> Dict:
            # Search entities
            cur.execute("""
                SELECT e.* FROM entities e
                LEFT JOIN synonyms s ON e.id = s.entity_id
                WHERE e.name LIKE ? OR e.description LIKE ? OR s.synonym LIKE ?
                GROUP BY e.id
                LIMIT ?
            """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", limit))
            entity_results = self._rows_to_dicts(cur.fetchall(), 'entity')

            # Search collections
            cur.execute("""
                SELECT * FROM collections
                WHERE name LIKE ? OR description LIKE ?
                LIMIT ?
            """, (f"%{search_term}%", f"%{search_term}%", limit))
            collection_results = self._rows_to_dicts(
                cur.fetchall(), 'collection')

            return {
                "entities": entity_results,
                "collections": collection_results
            }

        return self._execute_with_connection(
            operation,
            f"Failed to perform enhanced search with term: {search_term}"
        )
