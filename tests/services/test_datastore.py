"""Unit tests for the MemoryDatastore service."""

import pytest
from datetime import datetime, timedelta
from app.memory.lightrag.datastore import MemoryDatastore
from app.memory.lightrag.config import DB_PATH
from pathlib import Path


@pytest.fixture
def datastore(tmp_path):
    """Fixture providing a clean MemoryDatastore instance for each test."""
    test_db = tmp_path / "test.db"
    return MemoryDatastore(test_db)


class TestMemoryHierarchy:
    """Test suite for memory hierarchy functionality."""

    def test_set_entity_hierarchy(self, datastore):
        """Test setting hierarchy level and parent for an entity."""
        entity_id = "test-entity"
        datastore.create_entity(entity_id, "Test Entity", "test")

        # Set hierarchy level
        entity = datastore.set_entity_hierarchy(entity_id, "chunk")
        assert entity["hierarchy_level"] == "chunk"

        # Set parent
        parent_id = "parent-entity"
        datastore.create_entity(parent_id, "Parent Entity", "test")
        result = datastore.set_entity_hierarchy(entity_id, "chunk", parent_id)
        assert result["parent_id"] == parent_id

    def test_get_entity_hierarchy(self, datastore):
        """Test retrieving hierarchy information for an entity."""
        entity_id = "test-entity"
        datastore.create_entity(entity_id, "Test Entity", "test")
        datastore.set_entity_hierarchy(entity_id, "chunk")

        hierarchy = datastore.get_entity_hierarchy(entity_id)
        assert hierarchy["hierarchy_level"] == "chunk"

    def test_get_child_entities(self, datastore):
        """Test retrieving child entities for a parent."""
        parent_id = "parent-entity"
        datastore.create_entity(parent_id, "Parent Entity", "test")

        # Create child entities
        for i in range(3):
            entity_id = f"child-entity-{i}"
            datastore.create_entity(entity_id, f"Child Entity {i}", "test")
            datastore.set_entity_hierarchy(entity_id, "chunk", parent_id)

        children = datastore.get_child_entities(parent_id)
        assert len(children) == 3
        for child in children:
            assert child["parent_id"] == parent_id


class TestCollectionManagement:
    """Test suite for collection management functionality."""

    def test_create_collection(self, datastore):
        """Test creating a new collection."""
        collection = datastore.create_collection(
            "Test Collection", "Test Description")
        assert collection["name"] == "Test Collection"
        assert collection["description"] == "Test Description"

    def test_add_to_collection(self, datastore):
        """Test adding entities to a collection."""
        collection = datastore.create_collection("Test Collection")
        entity_id = "test-entity"
        datastore.create_entity(entity_id, "Test Entity", "test")

        member_id = datastore.add_to_collection(collection["id"], entity_id)
        assert member_id is not None

    def test_remove_from_collection(self, datastore):
        """Test removing entities from a collection."""
        collection = datastore.create_collection("Test Collection")
        entity_id = "test-entity"
        datastore.create_entity(entity_id, "Test Entity", "test")
        datastore.add_to_collection(collection["id"], entity_id)

        removed_count = datastore.remove_from_collection(
            collection["id"], entity_id)
        assert removed_count == 1

    def test_get_collection_members(self, datastore):
        """Test retrieving members of a collection."""
        collection = datastore.create_collection("Test Collection")

        # Add multiple entities
        for i in range(3):
            entity_id = f"test-entity-{i}"
            datastore.create_entity(entity_id, f"Test Entity {i}", "test")
            datastore.add_to_collection(collection["id"], entity_id)

        members = datastore.get_collection_members(collection["id"])
        assert len(members) == 3


class TestEnhancedSearch:
    """Test suite for enhanced search functionality."""

    def test_enhanced_search(self, datastore):
        """Test the enhanced search across entities and collections."""
        # Create test entities
        for i in range(3):
            entity_id = f"test-entity-{i}"
            datastore.create_entity(entity_id, f"Test Entity {i}", "test")

        # Create test collection
        collection = datastore.create_collection(
            "Test Collection", "Test Description")

        # Perform search
        results = datastore.enhanced_search("Test")
        assert len(results["entities"]) == 3
        assert len(results["collections"]) == 1

    def test_search_collections(self, datastore):
        """Test searching for collections by name and description."""
        datastore.create_collection(
            "Test Collection 1", "First test collection")
        datastore.create_collection(
            "Test Collection 2", "Second test collection")

        results = datastore.search_collections("test")
        assert len(results) == 2


@pytest.mark.integration
class TestIntegration:
    """Integration tests for combined functionality."""

    def test_hierarchy_with_collections(self, datastore):
        """Test combining hierarchy and collection features."""
        # Create parent entity
        parent_id = "parent-entity"
        datastore.create_entity(parent_id, "Parent Entity", "test")

        # Create child entities
        for i in range(3):
            entity_id = f"child-entity-{i}"
            datastore.create_entity(entity_id, f"Child Entity {i}", "test")
            datastore.set_entity_hierarchy(entity_id, "chunk", parent_id)

        # Create collection and add parent
        collection = datastore.create_collection("Test Collection")
        datastore.add_to_collection(collection["id"], parent_id)

        # Verify collection members include parent
        members = datastore.get_collection_members(collection["id"])
        assert len(members) == 1
        assert members[0]["id"] == parent_id

        # Verify hierarchy is maintained
        children = datastore.get_child_entities(parent_id)
        assert len(children) == 3
