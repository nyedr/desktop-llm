"""Integration tests for the chat router with memory context."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.models.chat import ChatRequest
from app.services.chroma_service import ChromaService
from app.services.langchain_service import LangChainService


@pytest.fixture
def test_client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_chroma_service():
    """Create a mock ChromaService."""
    with patch("app.dependencies.providers.Providers.get_chroma_service") as mock:
        service = AsyncMock(spec=ChromaService)
        service.retrieve_memories.return_value = [
            {
                "document": "Previous conversation about Python: User likes type hints",
                "metadata": {"type": "memory", "timestamp": "2024-01-01T00:00:00"},
                "relevance_score": 0.95
            }
        ]
        mock.return_value = service
        yield service


@pytest.fixture
def mock_langchain_service():
    """Create a mock LangChainService."""
    with patch("app.dependencies.providers.Providers.get_langchain_service") as mock:
        service = AsyncMock(spec=LangChainService)
        service.query_memory.return_value = {
            "result": "Test summary"
        }
        mock.return_value = service
        yield service


@pytest.mark.asyncio
async def test_chat_with_memory_enabled(test_client, mock_chroma_service, mock_langchain_service):
    """Test chat endpoint with memory enabled."""
    request = ChatRequest(
        messages=[
            {"role": "user", "content": "Tell me about Python type hints"}
        ],
        model="test-model",
        enable_memory=True,
        top_k_memories=3
    )

    response = test_client.post("/api/v1/chat", json=request.dict())
    assert response.status_code == 200

    # Verify memory retrieval was called
    mock_chroma_service.retrieve_memories.assert_called_once()

    # Verify memory was stored after chat
    mock_chroma_service.add_memory.assert_called_once()


@pytest.mark.asyncio
async def test_chat_with_memory_disabled(test_client, mock_chroma_service, mock_langchain_service):
    """Test chat endpoint with memory disabled."""
    request = ChatRequest(
        messages=[
            {"role": "user", "content": "Tell me about Python"}
        ],
        model="test-model",
        enable_memory=False
    )

    response = test_client.post("/api/v1/chat", json=request.dict())
    assert response.status_code == 200

    # Verify memory retrieval was not called
    mock_chroma_service.retrieve_memories.assert_not_called()


@pytest.mark.asyncio
async def test_chat_with_memory_filter(test_client, mock_chroma_service, mock_langchain_service):
    """Test chat endpoint with memory filter."""
    request = ChatRequest(
        messages=[
            {"role": "user", "content": "Tell me about Python"}
        ],
        model="test-model",
        enable_memory=True,
        memory_filter={"type": "programming"}
    )

    response = test_client.post("/api/v1/chat", json=request.dict())
    assert response.status_code == 200

    # Verify memory filter was passed
    call_args = mock_chroma_service.retrieve_memories.call_args
    assert call_args[1]["filter_dict"] == {"type": "programming"}


@pytest.mark.asyncio
async def test_chat_with_streaming(test_client, mock_chroma_service, mock_langchain_service):
    """Test chat endpoint with streaming enabled."""
    request = ChatRequest(
        messages=[
            {"role": "user", "content": "Tell me about Python"}
        ],
        model="test-model",
        enable_memory=True,
        stream=True
    )

    response = test_client.post("/api/v1/chat", json=request.dict())
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"

    # Read streaming response
    response_text = ""
    for line in response.iter_lines():
        if line:
            response_text += line.decode()

    assert response_text  # Verify we got some response


@pytest.mark.asyncio
async def test_chat_error_handling(test_client, mock_chroma_service, mock_langchain_service):
    """Test chat endpoint error handling."""
    # Simulate Chroma service error
    mock_chroma_service.retrieve_memories.side_effect = Exception(
        "Chroma error")

    request = ChatRequest(
        messages=[
            {"role": "user", "content": "Tell me about Python"}
        ],
        model="test-model",
        enable_memory=True
    )

    response = test_client.post("/api/v1/chat", json=request.dict())
    assert response.status_code == 200  # Should still work, just without memory


@pytest.mark.asyncio
async def test_chat_with_tools(test_client, mock_chroma_service, mock_langchain_service):
    """Test chat endpoint with tools enabled."""
    request = ChatRequest(
        messages=[
            {"role": "user", "content": "What's the weather like?"}
        ],
        model="test-model",
        enable_memory=True,
        tools=["weather"]
    )

    response = test_client.post("/api/v1/chat", json=request.dict())
    assert response.status_code == 200

    # Verify tool-related memory was stored
    call_args = mock_chroma_service.add_memory.call_args
    metadata = call_args[1]["metadata"]
    assert metadata["has_tools"] is True


@pytest.mark.asyncio
async def test_chat_with_filters(test_client, mock_chroma_service, mock_langchain_service):
    """Test chat endpoint with filters enabled."""
    request = ChatRequest(
        messages=[
            {"role": "user", "content": "Tell me about Python"}
        ],
        model="test-model",
        enable_memory=True,
        filters=["code"]
    )

    response = test_client.post("/api/v1/chat", json=request.dict())
    assert response.status_code == 200

    # Verify filter-related memory was stored
    call_args = mock_chroma_service.add_memory.call_args
    metadata = call_args[1]["metadata"]
    assert metadata["has_filters"] is True


@pytest.mark.asyncio
async def test_chat_performance(test_client, mock_chroma_service, mock_langchain_service):
    """Test chat endpoint performance with large context."""
    # Create a large conversation
    large_conversation = [
        {"role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}" * 100}
        for i in range(20)
    ]

    request = ChatRequest(
        messages=large_conversation,
        model="test-model",
        enable_memory=True,
        top_k_memories=10
    )

    response = test_client.post("/api/v1/chat", json=request.dict())
    assert response.status_code == 200

    # Verify context was properly managed
    assert mock_chroma_service.retrieve_memories.called
