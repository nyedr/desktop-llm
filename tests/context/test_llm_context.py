"""Tests for the LLM Context Manager."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.context.llm_context import LLMContextManager
from app.models.chat import ChatMessage
from app.core.config import config


@pytest.fixture
def mock_chroma_service():
    """Create a mock ChromaService."""
    service = Mock()
    service.retrieve_memories = AsyncMock()
    service.retrieve_with_metadata = AsyncMock()
    service.add_memory = AsyncMock()
    return service


@pytest.fixture
def mock_langchain_service():
    """Create a mock LangChainService."""
    service = Mock()
    service.query_memory = AsyncMock()
    return service


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I'll check that for you."},
        {"role": "system", "content": "Using weather API..."},
        {"role": "assistant", "content": "It's sunny today!"}
    ]


@pytest.fixture
def sample_memories():
    """Create sample memory documents."""
    return [
        {
            "document": "Previous weather query: It was raining yesterday",
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "type": "memory"
            },
            "relevance_score": 0.95
        },
        {
            "document": "User preference: Prefers detailed weather reports",
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "type": "user_preference"
            },
            "relevance_score": 0.85
        }
    ]


@pytest.mark.asyncio
async def test_context_initialization(mock_chroma_service, mock_langchain_service, sample_messages):
    """Test basic initialization of the context manager."""
    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        sample_messages
    )

    assert context_manager.chroma_service == mock_chroma_service
    assert context_manager.langchain_service == mock_langchain_service
    assert context_manager.conversation_history == sample_messages
    assert context_manager.tokenizer is not None


@pytest.mark.asyncio
async def test_process_text_message():
    """Test processing of text messages."""
    context_manager = LLMContextManager(Mock(), Mock(), [])

    # Test with dict message
    dict_msg = {"role": "user", "content": "Hello", "name": "John"}
    processed = context_manager._process_text_message(dict_msg)
    assert processed["role"] == "user"
    assert processed["content"] == "Hello"
    assert processed["name"] == "John"
    assert processed["metadata"]["type"] == "text"

    # Test with ChatMessage
    chat_msg = ChatMessage(role="user", content="Hello", name="John")
    processed = context_manager._process_text_message(chat_msg)
    assert processed["role"] == "user"
    assert processed["content"] == "Hello"
    assert processed["name"] == "John"
    assert processed["metadata"]["type"] == "text"


@pytest.mark.asyncio
async def test_memory_retrieval(mock_chroma_service, mock_langchain_service, sample_messages, sample_memories):
    """Test memory retrieval and formatting."""
    mock_chroma_service.retrieve_memories.return_value = sample_memories

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        sample_messages
    )

    async with context_manager:
        messages = context_manager.get_context_messages()

        # Verify memory retrieval was called
        mock_chroma_service.retrieve_memories.assert_called_once()

        # Check that memories were properly formatted
        assert any(
            msg["role"] == "system" and "Previous weather query" in msg["content"]
            for msg in messages
        )


@pytest.mark.asyncio
async def test_context_size_management(mock_chroma_service, mock_langchain_service):
    """Test context size management and truncation."""
    # Create a conversation that exceeds token limit
    long_messages = [
        {"role": "user", "content": "Hello " * 1000},  # Very long message
        {"role": "assistant", "content": "Hi " * 1000}  # Very long message
    ]

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        long_messages,
        max_context_tokens=1000  # Set a small limit
    )

    async with context_manager:
        messages = context_manager.get_context_messages()

        # Calculate total tokens in final messages
        total_tokens = sum(
            context_manager.count_message_tokens(msg)
            for msg in messages
        )

        # Verify we're within limits
        assert total_tokens <= 1000


@pytest.mark.asyncio
async def test_error_handling(mock_chroma_service, mock_langchain_service, sample_messages):
    """Test error handling in various scenarios."""
    # Simulate Chroma service error
    mock_chroma_service.retrieve_memories.side_effect = Exception(
        "Chroma error")

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        sample_messages
    )

    async with context_manager:
        messages = context_manager.get_context_messages()

        # Verify we still get the original messages on error
        assert messages == sample_messages


@pytest.mark.asyncio
async def test_memory_storage(mock_chroma_service, mock_langchain_service, sample_messages):
    """Test memory storage during teardown."""
    mock_langchain_service.query_memory.return_value = {
        "result": "Test summary"
    }

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        sample_messages
    )

    async with context_manager:
        pass  # Context manager will handle teardown

    # Verify summary was stored
    mock_chroma_service.add_memory.assert_called_once()

    # Check metadata
    call_args = mock_chroma_service.add_memory.call_args
    metadata = call_args[1]["metadata"]
    assert metadata["type"] == "conversation_summary"
    assert "timestamp" in metadata
    assert metadata["message_count"] == len(sample_messages)


@pytest.mark.asyncio
async def test_image_message_processing(mock_chroma_service, mock_langchain_service):
    """Test processing of messages with images."""
    image_message = {
        "role": "user",
        "content": "Check this image",
        "images": ["base64_image_data"]
    }

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        [image_message]
    )

    processed = await context_manager._process_image_message(image_message)
    assert processed["metadata"]["has_image"]
    assert "[Image attached]" in processed["content"]


@pytest.mark.asyncio
async def test_file_message_processing(mock_chroma_service, mock_langchain_service):
    """Test processing of messages with file attachments."""
    file_message = {
        "role": "user",
        "content": "Check this file",
        "file_path": "/path/to/file.txt"
    }

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        [file_message]
    )

    processed = await context_manager._process_file_message(file_message)
    assert processed["metadata"]["has_file"]
    assert "file.txt" in processed["content"]


@pytest.mark.asyncio
async def test_function_call_processing(mock_chroma_service, mock_langchain_service):
    """Test processing of function calls and results."""
    conversation = [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I'll check the weather for you."},
        {"role": "function", "name": "get_weather", "content": "Sunny, 72°F"},
        {"role": "assistant", "content": "Based on the weather data..."},
        {"role": "system", "name": "function",
            "content": "Weather alert: Rain expected"}
    ]

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        conversation
    )

    async with context_manager:
        messages = context_manager.get_context_messages()

        # Verify function calls are properly formatted
        function_messages = [
            msg for msg in messages
            if msg.get("metadata", {}).get("has_function_call")
        ]

        assert len(function_messages) > 0
        # Check that function results are attributed to the assistant
        assert all(msg["role"] == "assistant" for msg in function_messages)
        # Verify function results are included in the content
        assert any(
            "get_weather function" in msg["content"] for msg in messages)
        assert any("72°F" in msg["content"] for msg in messages)


@pytest.mark.asyncio
async def test_duplicate_memory_prevention(mock_chroma_service, mock_langchain_service, sample_messages):
    """Test prevention of duplicate memory storage."""
    # Setup mock to simulate existing similar memory
    mock_chroma_service.retrieve_memories.return_value = [{
        "document": "Very similar existing content",
        "metadata": {"type": "memory"},
        "relevance": 0.98
    }]

    mock_langchain_service.query_memory.return_value = {
        "result": "Very similar new content"
    }

    context_manager = LLMContextManager(
        mock_chroma_service,
        mock_langchain_service,
        sample_messages
    )

    async with context_manager:
        pass  # Let context manager handle teardown

    # Verify memory retrieval was called to check for duplicates
    mock_chroma_service.retrieve_memories.assert_called_once()
    # Verify no new memory was stored due to similarity
    mock_chroma_service.add_memory.assert_not_called()
