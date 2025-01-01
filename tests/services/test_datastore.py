"""Tests for the MemoryDatastore class."""

import pytest
import sqlite3
from datetime import datetime, timedelta
from app.memory.lightrag.datastore import MemoryDatastore


@pytest.fixture
def datastore():
    """Create a test datastore instance."""
    ds = MemoryDatastore(":memory:")  # Use in-memory SQLite for testing
    return ds


def test_metadata_operations(datastore):
    """Test metadata storage and retrieval."""
    memory_id = "test_memory"

    # Test setting metadata
    datastore.set_metadata(memory_id, "key1", "value1")
    datastore.set_metadata(memory_id, "key2", "value2")

    # Test getting metadata
    assert datastore.get_metadata(memory_id) == {
        "key1": "value1",
        "key2": "value2"
    }

    # Test updating metadata
    datastore.set_metadata(memory_id, "key1", "updated_value")
    assert datastore.get_metadata(memory_id)["key1"] == "updated_value"


def test_cache_operations(datastore):
    """Test cache storage and retrieval."""
    cache_key = "test_cache"
    cache_data = {"test": "data"}

    # Test setting cache without expiration
    datastore.set_cache(cache_key, cache_data)
    assert datastore.get_cache(cache_key) == cache_data

    # Test setting cache with expiration
    future_cache = {"future": "data"}
    datastore.set_cache("future_key", future_cache,
                        expiration=datetime.now() + timedelta(minutes=5))
    assert datastore.get_cache("future_key") == future_cache

    # Test expired cache
    past_cache = {"past": "data"}
    datastore.set_cache("past_key", past_cache,
                        expiration=datetime.now() - timedelta(minutes=5))
    assert datastore.get_cache("past_key") is None


def test_cache_cleanup(datastore):
    """Test automatic cache cleanup."""
    # Add some expired entries
    past = datetime.now() - timedelta(minutes=10)
    datastore.set_cache("expired1", {"data": 1}, expiration=past)
    datastore.set_cache("expired2", {"data": 2}, expiration=past)

    # Add some valid entries
    future = datetime.now() + timedelta(minutes=10)
    datastore.set_cache("valid1", {"data": 3}, expiration=future)
    datastore.set_cache("valid2", {"data": 4}, expiration=future)

    # Run cleanup
    datastore.cleanup_expired_cache()

    # Check results
    assert datastore.get_cache("expired1") is None
    assert datastore.get_cache("expired2") is None
    assert datastore.get_cache("valid1") is not None
    assert datastore.get_cache("valid2") is not None


def test_database_initialization(datastore):
    """Test database initialization and table creation."""
    # Check that tables exist
    with sqlite3.connect(datastore.db_path) as conn:
        cursor = conn.cursor()

        # Check metadata table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='metadata'
        """)
        assert cursor.fetchone() is not None

        # Check cache table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='cache'
        """)
        assert cursor.fetchone() is not None
