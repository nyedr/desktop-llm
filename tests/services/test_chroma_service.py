import pytest
import os
from unittest.mock import patch
from datetime import datetime
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

    # Verify metadata with timestamp
    metadata = results[0]["metadata"]
    assert metadata["source"] == TEST_METADATA["source"]
    assert metadata["type"] == TEST_METADATA["type"]
    assert "timestamp" in metadata
    assert isinstance(metadata["timestamp"], str)
    # Verify timestamp is in ISO format
    datetime.fromisoformat(metadata["timestamp"])


@pytest.mark.asyncio
async def test_add_batch_memories(chroma_service):
    """Test adding multiple memories in batch."""
    memory_ids = await chroma_service.add_memory(
        TEST_BATCH_TEXTS,
        TEST_BATCH_METADATA
    )
    assert isinstance(memory_ids, list)
    assert len(memory_ids) == len(TEST_BATCH_TEXTS)

    # Verify all memories were added
    for text, metadata in zip(TEST_BATCH_TEXTS, TEST_BATCH_METADATA):
        results = await chroma_service.retrieve_memories(text)
        assert len(results) > 0
        assert any(r["document"] == text for r in results)
        # Verify metadata with timestamp
        found = False
        for r in results:
            r_metadata = r["metadata"]
            if (r_metadata["source"] == metadata["source"] and
                r_metadata["type"] == metadata["type"] and
                "timestamp" in r_metadata and
                    "batch_index" in r_metadata):
                found = True
                break
        assert found


@pytest.mark.asyncio
async def test_duplicate_prevention(chroma_service):
    """Test that duplicate memories are not added."""
    # Add initial memory
    memory_id1 = await chroma_service.add_memory(TEST_TEXT, TEST_METADATA)
    assert memory_id1 is not None

    # Try to add the same memory again
    memory_id2 = await chroma_service.add_memory(TEST_TEXT, TEST_METADATA)
    assert memory_id2 is None  # Should return None for duplicates

    # Verify only one memory exists
    results = await chroma_service.retrieve_memories(TEST_TEXT)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_retrieve_memories(chroma_service):
    """Test retrieving memories by similarity."""
    # Add test memories in batch
    await chroma_service.add_memory(TEST_BATCH_TEXTS, TEST_BATCH_METADATA)

    # Test retrieval
    results = await chroma_service.retrieve_memories("first memory", top_k=2)
    assert len(results) == 2
    assert any("First test memory" in r["document"] for r in results)
    # Check that relevance scores are between 0 and 1
    assert all(0 <= r["relevance_score"] <= 1 for r in results)


@pytest.mark.asyncio
async def test_error_handling(chroma_service):
    """Test error handling in various scenarios."""
    # Test with empty text
    with pytest.raises(ValueError, match="Text must be a non-empty string"):
        await chroma_service.add_memory("", TEST_METADATA)

    # Test with invalid text type
    with pytest.raises(ValueError, match="Text must be a non-empty string"):
        await chroma_service.add_memory(123, TEST_METADATA)

    # Test with text containing only special characters
    with pytest.raises(ValueError, match="Text contains no valid content after sanitization"):
        await chroma_service.add_memory("@#$%^&*", TEST_METADATA)


@pytest.mark.asyncio
async def test_memory_relevance_scores(chroma_service):
    """Test that retrieved memories include relevance scores."""
    # Add test memories in batch
    await chroma_service.add_memory(TEST_BATCH_TEXTS, TEST_BATCH_METADATA)

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
