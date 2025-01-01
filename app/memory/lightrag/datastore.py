"""Enhanced storage operations for LightRAG memory system."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypeVar
from pathlib import Path
from .config import LIGHTRAG_DATA_DIR

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
            LIGHTRAG_DATA_DIR) / "memory.db"
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
        """Create necessary tables if they don't exist."""
        def operation(cur: sqlite3.Cursor) -> None:
            # Cache table for storing computed results
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME
                );
            """)

            # Metadata table for storing additional information
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    entity_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (entity_id, key)
                );
            """)

            # Create indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_entity ON metadata(entity_id);")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_key ON metadata(key);")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_key_value ON metadata(key, value);")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache(expires_at);")

        self._execute_with_connection(
            operation, "Failed to create database tables")

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

    def search_entities(self, content_hash: str, limit: int = 1) -> List[Dict[str, str]]:
        """Search for entities by content hash.

        Args:
            content_hash: Hash of the content to search for
            limit: Maximum number of results to return

        Returns:
            List of matching entities with their metadata
        """
        def operation(cur: sqlite3.Cursor) -> List[Dict[str, str]]:
            # Get all metadata for entities with matching content hash in a single query
            cur.execute("""
                WITH matching_entities AS (
                    SELECT DISTINCT entity_id
                    FROM metadata
                    WHERE key = 'content_hash' AND value = ?
                    LIMIT ?
                )
                SELECT m.entity_id, m.key, m.value, m.created_at
                FROM metadata m
                INNER JOIN matching_entities me ON m.entity_id = me.entity_id
                ORDER BY m.entity_id, m.key
            """, (content_hash, limit))

            entities = {}
            for row in cur.fetchall():
                entity_id = row['entity_id']
                if entity_id not in entities:
                    entities[entity_id] = {
                        "id": entity_id,
                        "created_at": row['created_at']
                    }
                entities[entity_id][row['key']] = row['value']

            return list(entities.values())

        return self._execute_with_connection(
            operation,
            f"Failed to search entities with content hash: {content_hash}"
        )
