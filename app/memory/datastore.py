"""Enhanced storage operations for LightRAG memory system."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypeVar
from pathlib import Path
from app.core.config import config

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MemoryDatastore:
    """Handles auxiliary storage operations for the LightRAG memory system."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the memory datastore.

        Args:
            db_path: Optional custom path for the SQLite database.
                    Defaults to LIGHTRAG_DATA_DIR/memory.db
        """
        # Set up database path
        self.db_path = Path(db_path) if db_path else Path(
            config.LIGHTRAG_DATA_DIR) / "memory.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._ensure_tables()
        logger.debug(f"Memory datastore initialized at {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        """Create a database connection with proper configuration."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _execute_with_connection(self, operation: callable, error_msg: str) -> T:
        """Execute a database operation with proper connection handling."""
        try:
            conn = self._connect()
            try:
                with conn:
                    return operation(conn.cursor())
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"{error_msg}: {str(e)}")
            raise

    def _ensure_tables(self):
        """Create necessary database tables if they don't exist."""
        def operation(cursor):
            # Create entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    source_id TEXT,
                    target_id TEXT,
                    relationship_type TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES entities (id),
                    FOREIGN KEY (target_id) REFERENCES entities (id)
                )
            """)

            # Create embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entity_id) REFERENCES entities (id)
                )
            """)

            # Create indices
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_hash ON entities (content_hash)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships ON relationships (source_id, target_id)")

        self._execute_with_connection(
            operation,
            "Failed to create database tables")

    def get_metadata(self, entity_id: str) -> Dict[str, str]:
        """Get all metadata for an entity."""
        def operation(cur: sqlite3.Cursor) -> Dict[str, str]:
            cur.execute("""
                SELECT key, value FROM metadata
                WHERE entity_id = ?
            """, (entity_id,))
            return {row['key']: row['value'] for row in cur.fetchall()}

        return self._execute_with_connection(
            operation,
            f"Failed to get metadata for entity: {entity_id}"
        )

    def search_metadata(self, key: str, value: Optional[str] = None, limit: int = 10) -> List[Dict[str, str]]:
        """Search for entities by metadata."""
        def operation(cur: sqlite3.Cursor) -> List[Dict[str, str]]:
            if value:
                cur.execute("""
                    SELECT entity_id, key, value FROM metadata
                    WHERE key = ? AND value = ?
                    LIMIT ?
                """, (key, value, limit))
            else:
                cur.execute("""
                    SELECT entity_id, key, value FROM metadata
                    WHERE key = ?
                    LIMIT ?
                """, (key, limit))

            return [dict(row) for row in cur.fetchall()]

        return self._execute_with_connection(
            operation,
            f"Failed to search metadata with key: {key}"
        )

    def set_metadata(self, entity_id: str, key: str, value: str) -> None:
        """Set metadata for an entity."""
        def operation(cur: sqlite3.Cursor) -> None:
            cur.execute("""
                INSERT OR REPLACE INTO metadata (entity_id, key, value)
                VALUES (?, ?, ?)
            """, (entity_id, key, value))

        self._execute_with_connection(
            operation,
            f"Failed to set metadata for entity {entity_id}"
        )

    def get_cache(self, key: str) -> Optional[Dict]:
        """Retrieve a cache entry if it hasn't expired."""
        def operation(cur: sqlite3.Cursor) -> Optional[Dict]:
            cur.execute("""
                SELECT value FROM cache
                WHERE key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """, (key,))
            result = cur.fetchone()
            return json.loads(result['value']) if result else None

        return self._execute_with_connection(
            operation,
            f"Failed to get cache entry: {key}"
        )

    def set_cache(self, key: str, value: Dict, expiration: Optional[timedelta] = None) -> None:
        """Store a value in the cache with optional expiration."""
        def operation(cur: sqlite3.Cursor) -> None:
            expires_at = (datetime.now() +
                          expiration).isoformat() if expiration else None
            cur.execute("""
                INSERT OR REPLACE INTO cache (key, value, expires_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), expires_at))

        self._execute_with_connection(
            operation,
            f"Failed to set cache entry: {key}"
        )

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries."""
        def operation(cur: sqlite3.Cursor) -> int:
            cur.execute("""
                DELETE FROM cache
                WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP
            """)
            return cur.rowcount

        return self._execute_with_connection(
            operation,
            "Failed to cleanup expired cache entries"
        )

    def search_entities(self, content_hash: str, limit: int = 1) -> List[Dict]:
        """Search for entities by content hash."""
        def operation(cursor):
            cursor.execute(
                """
                SELECT * FROM entities 
                WHERE content_hash = ? 
                LIMIT ?
                """,
                (content_hash, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

        return self._execute_with_connection(
            operation,
            f"Failed to search entities with hash {content_hash}"
        )
