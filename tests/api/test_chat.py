"""Tests for chat API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from app.models.chat import ChatRequest, ChatMessage
from unittest.mock import patch, AsyncMock
from app.main import app
import json
import anyio
import logging
from sse_starlette.sse import AppStatus
from async_timeout import timeout
from fastapi import HTTPException
import asyncio

logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def patch_sse_exit_event():
    """Patch SSE exit event to use current event loop."""
    AppStatus.should_exit_event = anyio.Event()

@pytest.fixture
async def async_test_client():
    """Create an asynchronous test client."""
    transport = ASGITransport(app=app, raise_app_exceptions=True)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

async def mock_chat_generator(tool_calls=None, error=None):
    """Mock chat completion generator."""
    if error:
        raise error

    # Yield the message
    if tool_calls:
        # First yield the assistant message with tool calls
        await asyncio.sleep(0)  # Allow other tasks to run
        yield {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls
        }
        # Then yield tool responses
        for tool_call in tool_calls:
            await asyncio.sleep(0)  # Allow other tasks to run
            yield {
                "role": "tool",
                "content": "Tool response",
                "tool_call_id": tool_call["id"]
            }
        # Finally yield the assistant's response
        await asyncio.sleep(0)  # Allow other tasks to run
        yield {
            "role": "assistant",
            "content": "The weather in New York is sunny."
        }
    else:
        await asyncio.sleep(0)  # Allow other tasks to run
        yield {
            "role": "assistant",
            "content": "Hello there!"
        }

@pytest.mark.asyncio
@patch('app.services.model_service.ModelService.get_all_models')
@patch('app.services.agent.Agent.chat')
async def test_chat_stream_start(mock_agent_chat, mock_get_all_models, async_test_client):
    """Test chat stream initialization."""
    mock_get_all_models.return_value = {"test-model": {}}
    mock_agent_chat.return_value = mock_chat_generator()

    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        model="test-model",
        enable_tools=False
    )

    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json=request.model_dump(exclude_none=True),
        headers={"x-test-request": "true"}
    )
    assert response.status_code == 200
    logger.info("Got response with status %d", response.status_code)

    # Collect events from the generator with timeout
    events = []
    current_event = None
    async with timeout(5):  # 5 second timeout
        async for line in response.aiter_lines():
            logger.debug("Received line: %s", line)
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
                logger.info("Found event type: %s", current_event)
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                logger.info("Found data: %s", data)
                if current_event:
                    events.append({"event": current_event, "data": data})

    # Verify we got all expected events
    assert len(events) >= 3, f"Expected at least 3 events, got {len(events)}"
    assert events[0]["event"] == "start"
    assert events[1]["event"] == "message"
    assert events[2]["event"] == "end"
    assert events[1]["data"]["content"] == "Hello there!"

@pytest.mark.asyncio
@patch('app.services.model_service.ModelService.get_all_models')
async def test_chat_invalid_request(mock_get_all_models, async_test_client):
    """Test chat endpoint with invalid request."""
    mock_get_all_models.return_value = {"test-model": {}}

    # Test missing required fields
    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json={},
        headers={"x-test-request": "true"}
    )
    assert response.status_code == 422

    # Test empty messages
    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json={
            "messages": [],
            "model": "test-model"
        },
        headers={"x-test-request": "true"}
    )
    assert response.status_code == 422

    # Test invalid message format
    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json={
            "messages": [{"invalid": "message"}],
            "model": "test-model"
        },
        headers={"x-test-request": "true"}
    )
    assert response.status_code == 422

    # Test invalid role
    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json={
            "messages": [{"role": "invalid", "content": "test"}],
            "model": "test-model"
        },
        headers={"x-test-request": "true"}
    )
    assert response.status_code == 422

    # Test empty content
    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json={
            "messages": [{"role": "user", "content": ""}],
            "model": "test-model"
        },
        headers={"x-test-request": "true"}
    )
    assert response.status_code == 422

@pytest.mark.asyncio
@patch('app.services.model_service.ModelService.get_all_models')
@patch('app.services.agent.Agent.chat')
async def test_chat_with_tools(mock_agent_chat, mock_get_all_models, async_test_client):
    """Test chat endpoint with tools enabled."""
    mock_get_all_models.return_value = {"test-model": {}}
    tool_calls = [{
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": json.dumps({"location": "New York"})
        }
    }]
    mock_agent_chat.return_value = mock_chat_generator(tool_calls=tool_calls)

    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Get the weather")],
        model="test-model",
        enable_tools=True
    )

    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json=request.model_dump(exclude_none=True),
        headers={"x-test-request": "true"}
    )
    assert response.status_code == 200
    logger.info("Got response with status %d", response.status_code)

    # Collect events from the generator with timeout
    events = []
    current_event = None
    async with timeout(5):  # 5 second timeout
        async for line in response.aiter_lines():
            logger.debug("Received line: %s", line)
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
                logger.info("Found event type: %s", current_event)
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                logger.info("Found data: %s", data)
                if current_event:
                    events.append({"event": current_event, "data": data})

    # Verify we got all expected events
    assert len(events) >= 4, f"Expected at least 4 events (including tool call), got {len(events)}"
    assert events[0]["event"] == "start"
    assert events[1]["event"] == "message"
    assert events[1]["data"]["tool_calls"] == tool_calls
    assert events[2]["event"] == "message"
    assert events[2]["data"]["role"] == "tool"
    assert events[3]["event"] == "message"
    assert events[3]["data"]["content"] == "The weather in New York is sunny."
    assert events[4]["event"] == "end"

@pytest.mark.asyncio
@patch('app.services.model_service.ModelService.get_all_models')
@patch('app.services.agent.Agent.chat')
async def test_chat_with_filters(mock_agent_chat, mock_get_all_models, async_test_client):
    """Test chat endpoint with message filters."""
    mock_get_all_models.return_value = {"test-model": {}}
    mock_agent_chat.return_value = mock_chat_generator()

    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        model="test-model",
        filters=["sanitize", "translate"]  # Example filters
    )

    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json=request.model_dump(exclude_none=True),
        headers={"x-test-request": "true"}
    )
    assert response.status_code == 200

    events = []
    current_event = None
    async with timeout(5):
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if current_event:
                    events.append({"event": current_event, "data": data})

    # Verify filter application
    message_event = next((e for e in events if e["event"] == "message"), None)
    assert message_event is not None, "Message event not received"
    # Add assertions for filtered content once filters are implemented

@pytest.mark.asyncio
@patch('app.services.model_service.ModelService.get_all_models')
@patch('app.services.agent.Agent.chat')
async def test_chat_error_handling(mock_agent_chat, mock_get_all_models, async_test_client):
    """Test chat endpoint error handling."""
    mock_get_all_models.return_value = {"test-model": {}}

    # Test model service error
    mock_agent_chat.return_value = mock_chat_generator(error=Exception("Model service error"))

    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        model="test-model"
    )

    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json=request.model_dump(exclude_none=True),
        headers={"x-test-request": "true"}
    )
    assert response.status_code == 200

    events = []
    current_event = None
    async with timeout(5):
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if current_event:
                    events.append({"event": current_event, "data": data})

    # Verify error event
    assert len(events) >= 1, "Expected at least 1 event"
    assert events[0]["event"] == "error"
    assert "Model service error" in events[0]["data"]["error"]

@pytest.mark.asyncio
@patch('app.services.model_service.ModelService.get_all_models')
@patch('app.services.agent.Agent.chat')
async def test_chat_model_selection(mock_agent_chat, mock_get_all_models, async_test_client):
    """Test model selection and parameters."""
    mock_get_all_models.return_value = {
        "model1": {},
        "model2": {},
    }
    mock_agent_chat.return_value = mock_chat_generator()

    # Test valid model selection
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        model="model2",
        temperature=0.7,
        max_tokens=100
    )

    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json=request.model_dump(exclude_none=True),
        headers={"x-test-request": "true"}
    )
    assert response.status_code == 200

    # Test invalid model
    request.model = "nonexistent-model"
    response = await async_test_client.post(
        "/api/v1/chat/stream",
        json=request.model_dump(exclude_none=True),
        headers={"x-test-request": "true"}
    )

    events = []
    current_event = None
    async with timeout(5):
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if current_event:
                    events.append({"event": current_event, "data": data})

    # Verify error for invalid model
    assert len(events) >= 1, "Expected at least 1 event"
    assert events[0]["event"] == "error"
    assert "Model nonexistent-model not available" in events[0]["data"]["error"]
