"""Performance tests for the LLM Context Manager."""
import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from app.context.llm_context import LLMContextManager
from app.models.chat import StrictChatMessage


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
    # Test with different token limits
    for max_tokens in [500, 1000, 2000, 4000]:
        context_manager = LLMContextManager(
            mock_chroma_service,
            mock_langchain_service,
            large_conversation,
            max_context_tokens=max_tokens
        )

        start_time = time.time()
        async with context_manager:
            messages = context_manager.get_context_messages()
        end_time = time.time()

        truncation_time = end_time - start_time
        print(
            f"Truncation time for {max_tokens} tokens: {truncation_time:.4f} seconds")

        # Verify truncation was effective
        total_tokens = sum(
            context_manager.count_message_tokens(msg)
            for msg in messages
        )
        assert total_tokens <= max_tokens
        assert truncation_time < 1.0  # Should complete in under 1 second

        # Verify message integrity after truncation
        assert len(messages) > 0
        assert all(isinstance(msg, dict) for msg in messages)
        assert all("role" in msg and "content" in msg for msg in messages)


@pytest.mark.asyncio
async def test_memory_retrieval_performance(mock_chroma_service, mock_langchain_service, large_memories):
    """Test performance of memory retrieval and processing."""
    # Test with different memory set sizes
    for memory_size in [10, 50, 100, 200]:
        mock_chroma_service.retrieve_memories.return_value = large_memories[:memory_size]

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
        print(
            f"Memory retrieval time for {memory_size} memories: {retrieval_time:.4f} seconds")

        # Verify memory processing was effective
        memory_messages = [msg for msg in messages if msg["role"] == "system"]
        assert len(memory_messages) > 0
        assert retrieval_time < 1.0  # Should complete in under 1 second

        # Verify memory content integrity
        assert all(isinstance(msg, dict) for msg in memory_messages)
        assert all("content" in msg for msg in memory_messages)


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

    # Test with different levels of concurrency
    for num_concurrent in [10, 20, 50, 100]:
        # Create multiple conversations
        conversations = [
            [{"role": "user", "content": f"Message {i}"}]
            for i in range(num_concurrent)
        ]

        start_time = time.time()
        # Process conversations concurrently
        tasks = [process_conversation(conv) for conv in conversations]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        concurrent_time = end_time - start_time
        print(
            f"Concurrent processing time for {num_concurrent} requests: {concurrent_time:.4f} seconds")

        # Verify all conversations were processed
        assert len(results) == num_concurrent
        assert concurrent_time < 5.0  # Should complete in under 5 seconds

        # Verify result integrity
        assert all(isinstance(result, list) for result in results)
        assert all(len(result) > 0 for result in results)


@pytest.mark.asyncio
async def test_memory_storage_performance(mock_chroma_service, mock_langchain_service, large_conversation):
    """Test performance of memory storage during teardown."""
    # Test with different conversation sizes
    for conv_size in [10, 50, 100, 200]:
        context_manager = LLMContextManager(
            mock_chroma_service,
            mock_langchain_service,
            large_conversation[:conv_size]
        )

        start_time = time.time()
        async with context_manager:
            pass  # Let the context manager handle teardown
        end_time = time.time()

        storage_time = end_time - start_time
        print(
            f"Memory storage time for {conv_size} messages: {storage_time:.4f} seconds")

        # Verify storage was called
        mock_chroma_service.add_memory.assert_called()
        assert storage_time < 1.0  # Should complete in under 1 second

        # Verify metadata integrity
        call_args = mock_chroma_service.add_memory.call_args
        metadata = call_args[1]["metadata"]
        assert metadata["type"] == "conversation_summary"
        assert "timestamp" in metadata
        assert metadata["message_count"] == conv_size


@pytest.mark.asyncio
async def test_token_counting_performance(mock_chroma_service, mock_langchain_service, large_conversation):
    """Test performance of token counting operations."""
    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        []
    )

    # Test with different batch sizes
    for batch_size in [10, 50, 100, 200]:
        start_time = time.time()
        for message in large_conversation[:batch_size]:
            context_manager.count_message_tokens(message)
        end_time = time.time()

        counting_time = end_time - start_time
        print(
            f"Token counting time for {batch_size} messages: {counting_time:.4f} seconds")

        # Token counting should be relatively fast
        assert counting_time < 1.0  # Should complete in under 1 second

        # Verify token counts are positive
        for message in large_conversation[:batch_size]:
            assert context_manager.count_message_tokens(message) > 0


@pytest.mark.asyncio
async def test_prompt_engineering_performance(mock_chroma_service, mock_langchain_service, large_memories):
    """Test performance of prompt engineering with large memory sets."""
    # Test with different memory set sizes
    for memory_size in [10, 50, 100, 200]:
        mock_chroma_service.retrieve_memories.return_value = large_memories[:memory_size]

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
        print(
            f"Prompt engineering time for {memory_size} memories: {engineering_time:.4f} seconds")

        # Verify prompt engineering was effective
        assert any("[Relevant Memory]" in msg["content"] for msg in messages)
        assert engineering_time < 1.0  # Should complete in under 1 second

        # Verify message structure integrity
        assert all(isinstance(msg, dict) for msg in messages)
        assert all("role" in msg and "content" in msg for msg in messages)


@pytest.mark.asyncio
async def test_memory_filter_performance(mock_chroma_service, mock_langchain_service, large_memories):
    """Test performance of memory filtering operations."""
    # Test with different filter complexities
    filters = [
        {"type": "memory"},
        {"type": "memory", "timestamp": {"$gt": "2024-01-01"}},
        {"type": {"$in": ["memory", "user_preference"]}},
        {"type": "memory", "relevance_score": {"$gt": 0.8}}
    ]

    for filter in filters:
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
                filter
            )
        end_time = time.time()

        filtering_time = end_time - start_time
        print(
            f"Memory filtering time for {filter}: {filtering_time:.4f} seconds")

        # Verify filtering was effective
        assert len(filtered_memories) <= len(large_memories)
        assert filtering_time < 1.0  # Should complete in under 1 second

        # Verify filter criteria were applied
        for memory in filtered_memories:
            metadata = memory.get("metadata", {})
            if "type" in filter:
                assert metadata.get("type") == filter["type"]
            if "timestamp" in filter:
                assert metadata.get("timestamp") > filter["timestamp"]["$gt"]
            if "$in" in filter:
                assert metadata.get("type") in filter["$in"]
            if "relevance_score" in filter:
                assert memory.get("relevance_score",
                                  0) > filter["relevance_score"]["$gt"]
