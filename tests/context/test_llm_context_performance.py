"""Performance tests for the LLM Context Manager."""
import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from app.context.llm_context import LLMContextManager
from app.models.chat import ChatMessage


@pytest.fixture
def mock_chroma_service():
    """Create a mock ChromaService."""
    service = Mock()
    service.retrieve_memories = AsyncMock()
    service.add_memory = AsyncMock()
    return service


@pytest.fixture
def mock_langchain_service():
    """Create a mock LangChainService."""
    service = Mock()
    service.query_memory = AsyncMock()
    return service


@pytest.fixture
def large_conversation():
    """Create a large conversation history."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}" * 100}
        for i in range(50)
    ]


@pytest.fixture
def large_memories():
    """Create a large set of memory documents."""
    return [
        {
            "document": f"Memory document {i}" * 100,
            "metadata": {"type": "memory", "timestamp": f"2024-01-{i:02d}T00:00:00"},
            "relevance_score": 0.9 - (i * 0.01)
        }
        for i in range(50)
    ]


@pytest.mark.asyncio
async def test_truncation_performance(mock_chroma_service, mock_langchain_service, large_conversation):
    """Test performance of context truncation with large conversations."""
    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        large_conversation,
        max_context_tokens=1000
    )

    start_time = time.time()
    async with context_manager:
        messages = context_manager.get_context_messages()
    end_time = time.time()

    truncation_time = end_time - start_time
    print(f"Truncation time: {truncation_time:.2f} seconds")

    # Verify truncation was effective
    total_tokens = sum(
        context_manager.count_message_tokens(msg)
        for msg in messages
    )
    assert total_tokens <= 1000
    assert truncation_time < 1.0  # Should complete in under 1 second


@pytest.mark.asyncio
async def test_memory_retrieval_performance(mock_chroma_service, mock_langchain_service, large_memories):
    """Test performance of memory retrieval and processing."""
    mock_chroma_service.retrieve_memories.return_value = large_memories

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        [{"role": "user", "content": "Test message"}]
    )

    start_time = time.time()
    async with context_manager:
        messages = context_manager.get_context_messages()
    end_time = time.time()

    retrieval_time = end_time - start_time
    print(f"Memory retrieval time: {retrieval_time:.2f} seconds")

    # Verify memory processing was effective
    memory_messages = [msg for msg in messages if msg["role"] == "system"]
    assert len(memory_messages) > 0
    assert retrieval_time < 1.0  # Should complete in under 1 second


@pytest.mark.asyncio
async def test_concurrent_access(mock_chroma_service, mock_langchain_service):
    """Test concurrent access to the context manager."""
    async def process_conversation(conversation):
        context_manager = LLMContextManager(
            mock_chroma_service,
            mock_langchain_service,
            conversation
        )
        async with context_manager:
            return context_manager.get_context_messages()

    # Create multiple conversations
    conversations = [
        [{"role": "user", "content": f"Message {i}"}]
        for i in range(10)
    ]

    start_time = time.time()
    # Process conversations concurrently
    tasks = [process_conversation(conv) for conv in conversations]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    concurrent_time = end_time - start_time
    print(f"Concurrent processing time: {concurrent_time:.2f} seconds")

    # Verify all conversations were processed
    assert len(results) == 10
    assert concurrent_time < 2.0  # Should complete in under 2 seconds


@pytest.mark.asyncio
async def test_memory_storage_performance(mock_chroma_service, mock_langchain_service, large_conversation):
    """Test performance of memory storage during teardown."""
    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        large_conversation
    )

    start_time = time.time()
    async with context_manager:
        pass  # Let the context manager handle teardown
    end_time = time.time()

    storage_time = end_time - start_time
    print(f"Memory storage time: {storage_time:.2f} seconds")

    # Verify storage was called
    mock_chroma_service.add_memory.assert_called_once()
    assert storage_time < 1.0  # Should complete in under 1 second


@pytest.mark.asyncio
async def test_token_counting_performance(mock_chroma_service, mock_langchain_service, large_conversation):
    """Test performance of token counting operations."""
    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        []
    )

    start_time = time.time()
    for message in large_conversation:
        context_manager.count_message_tokens(message)
    end_time = time.time()

    counting_time = end_time - start_time
    print(f"Token counting time: {counting_time:.2f} seconds")

    # Token counting should be relatively fast
    assert counting_time < 1.0  # Should complete in under 1 second


@pytest.mark.asyncio
async def test_prompt_engineering_performance(mock_chroma_service, mock_langchain_service, large_memories):
    """Test performance of prompt engineering with large memory sets."""
    mock_chroma_service.retrieve_memories.return_value = large_memories

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        [{"role": "user", "content": "Test message"}]
    )

    start_time = time.time()
    async with context_manager:
        messages = context_manager.get_context_messages()
        # Force prompt engineering by accessing messages
        _ = [msg["content"] for msg in messages]
    end_time = time.time()

    engineering_time = end_time - start_time
    print(f"Prompt engineering time: {engineering_time:.2f} seconds")

    # Verify prompt engineering was effective
    assert any("[Relevant Memory]" in msg["content"] for msg in messages)
    assert engineering_time < 1.0  # Should complete in under 1 second


@pytest.mark.asyncio
async def test_memory_filter_performance(mock_chroma_service, mock_langchain_service, large_memories):
    """Test performance of memory filtering operations."""
    mock_chroma_service.retrieve_memories.return_value = large_memories

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        [{"role": "user", "content": "Test message"}]
    )

    start_time = time.time()
    async with context_manager:
        # Apply memory filter
        filtered_memories = await context_manager._filter_memories(
            large_memories,
            {"type": "memory"}
        )
    end_time = time.time()

    filtering_time = end_time - start_time
    print(f"Memory filtering time: {filtering_time:.2f} seconds")

    # Verify filtering was effective
    assert len(filtered_memories) <= len(large_memories)
    assert filtering_time < 1.0  # Should complete in under 1 second
