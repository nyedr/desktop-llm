import pytest
import os
from unittest.mock import patch
from app.services.chroma_service import ChromaService

# Test data
TEST_TEXT = "This is a test memory"
TEST_METADATA = {"source": "test", "type": "unit_test"}
TEST_BATCH_TEXTS = [
    "First test memory",
    "Second test memory",
    "Third test memory"
]
TEST_BATCH_METADATA = [
    {"source": "test1", "type": "unit_test"},
    {"source": "test2", "type": "unit_test"},
    {"source": "test3", "type": "unit_test"}
]


@pytest.fixture
async def chroma_service(mock_config):
    """Create a ChromaService instance for testing."""
    with patch("app.services.chroma_service.config", mock_config):
        service = ChromaService()
        await service.initialize()
        yield service
        # Clean up
        try:
            await service.clear_collection()
        except Exception:
            pass  # Ignore cleanup errors


@pytest.mark.asyncio
async def test_initialize(chroma_service):
    """Test service initialization."""
    assert chroma_service.client is not None
    assert chroma_service.collection is not None
    assert chroma_service.embeddings is not None


@pytest.mark.asyncio
async def test_add_memory(chroma_service):
    """Test adding a single memory."""
    memory_id = await chroma_service.add_memory(TEST_TEXT, TEST_METADATA)
    assert memory_id is not None

    # Verify the memory was added
    results = await chroma_service.retrieve_memories(TEST_TEXT)
    assert len(results) > 0
    assert results[0]["document"] == TEST_TEXT
    assert results[0]["metadata"] == TEST_METADATA


@pytest.mark.asyncio
async def test_add_memories_batch(chroma_service):
    """Test adding multiple memories in batch."""
    memory_ids = await chroma_service.add_memories_batch(
        TEST_BATCH_TEXTS,
        TEST_BATCH_METADATA
    )
    assert len(memory_ids) == len(TEST_BATCH_TEXTS)

    # Verify all memories were added
    for text, metadata in zip(TEST_BATCH_TEXTS, TEST_BATCH_METADATA):
        results = await chroma_service.retrieve_memories(text)
        assert len(results) > 0
        assert any(r["document"] == text for r in results)
        assert any(r["metadata"] == metadata for r in results)


@pytest.mark.asyncio
async def test_retrieve_memories(chroma_service):
    """Test retrieving memories by similarity."""
    # Add test memories
    await chroma_service.add_memories_batch(TEST_BATCH_TEXTS, TEST_BATCH_METADATA)

    # Test retrieval
    results = await chroma_service.retrieve_memories("first memory", top_k=2)
    assert len(results) == 2
    assert any("First test memory" in r["document"] for r in results)
    # Check that relevance scores are between 0 and 1
    assert all(0 <= r["relevance_score"] <= 1 for r in results)


@pytest.mark.asyncio
async def test_retrieve_with_metadata(chroma_service):
    """Test retrieving memories with metadata filtering."""
    # Add test memories
    await chroma_service.add_memories_batch(TEST_BATCH_TEXTS, TEST_BATCH_METADATA)

    # Test retrieval with metadata filter
    metadata_filter = {"source": "test1"}
    results = await chroma_service.retrieve_with_metadata(
        "test memory",
        metadata_filter
    )
    assert len(results) > 0
    assert all(r["metadata"]["source"] == "test1" for r in results)


@pytest.mark.asyncio
async def test_update_memory(chroma_service):
    """Test updating an existing memory."""
    # Add initial memory
    memory_id = await chroma_service.add_memory(TEST_TEXT, TEST_METADATA)

    # Update the memory
    new_text = "Updated test memory"
    new_metadata = {"source": "test_updated", "type": "unit_test"}
    await chroma_service.update_memory(memory_id, new_text, new_metadata)

    # Verify the update
    results = await chroma_service.retrieve_memories(new_text)
    assert len(results) > 0
    assert results[0]["document"] == new_text
    assert results[0]["metadata"] == new_metadata


@pytest.mark.asyncio
async def test_delete_memory(chroma_service):
    """Test deleting a memory."""
    # Add a memory
    memory_id = await chroma_service.add_memory(TEST_TEXT, TEST_METADATA)

    # Delete the memory
    await chroma_service.delete_memory(memory_id)

    # Verify the memory was deleted
    results = await chroma_service.retrieve_memories(TEST_TEXT)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_clear_collection(chroma_service):
    """Test clearing all memories from the collection."""
    # Add test memories
    await chroma_service.add_memories_batch(TEST_BATCH_TEXTS, TEST_BATCH_METADATA)

    # Clear the collection
    await chroma_service.clear_collection()

    # Verify all memories were deleted
    results = await chroma_service.retrieve_memories("test")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_error_handling(chroma_service):
    """Test error handling in various scenarios."""
    # Test with invalid metadata
    with pytest.raises(Exception):
        await chroma_service.add_memory(TEST_TEXT, {"invalid": lambda x: x})

    # Test with invalid query
    with pytest.raises(Exception):
        await chroma_service.retrieve_memories(None)

    # Test with invalid metadata filter
    with pytest.raises(Exception):
        await chroma_service.retrieve_with_metadata("test", {"invalid": lambda x: x})


@pytest.mark.asyncio
async def test_memory_relevance_scores(chroma_service):
    """Test that retrieved memories include relevance scores."""
    # Add test memories
    await chroma_service.add_memories_batch(TEST_BATCH_TEXTS, TEST_BATCH_METADATA)

    # Retrieve memories and check scores
    results = await chroma_service.retrieve_memories("first memory")
    assert len(results) > 0
    assert all("relevance_score" in result for result in results)
    assert all(isinstance(result["relevance_score"], float)
               for result in results)
    assert all(0 <= result["relevance_score"] <= 1 for result in results)

    # Verify scores are in descending order (most relevant first)
    scores = [result["relevance_score"] for result in results]
    assert scores == sorted(scores, reverse=True)
