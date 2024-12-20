"""Test agent functionality."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.chat import ChatMessage
from agent import Agent, AgentConfig
from functions.function_registry import FunctionRegistry
from functions.preprocessing_layer import PreprocessingLayer
from functions.postprocessing_layer import PostprocessingLayer
from functions.function_execution_layer import FunctionExecutionLayer

@pytest.fixture
async def agent():
    """Create a test agent instance."""
    preprocessing = PreprocessingLayer()
    postprocessing = PostprocessingLayer()
    function_executor = FunctionExecutionLayer(FunctionRegistry())
    
    return Agent(
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        function_executor=function_executor,
        model="test-model"
    )

@pytest.fixture
def config():
    """Create a test configuration."""
    return AgentConfig(
        model="test-model",
        provider="ollama",
        temperature=0.7,
        stream=False
    )

@pytest.mark.asyncio
async def test_chat_ollama_no_tools(agent, config):
    """Test chat with Ollama provider without tools."""
    # Mock the Ollama client response
    mock_response = AsyncMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    
    with patch.object(agent.ollama_client, 'chat', return_value=mock_response):
        messages = [ChatMessage(role="user", content="Test message")]
        async for response in agent.chat(config, messages):
            assert response == "Test response"

@pytest.mark.asyncio
async def test_chat_ollama_with_tools(agent, config):
    """Test chat with Ollama provider with tools."""
    # Mock tool calls and responses
    mock_tool_call = {
        "function": {
            "name": "test_function",
            "arguments": '{"arg": "value"}'
        }
    }
    
    mock_response = AsyncMock()
    mock_response.message = MagicMock(
        content="Initial response",
        tool_calls=[mock_tool_call]
    )
    
    mock_final_response = AsyncMock()
    mock_final_response.choices = [
        MagicMock(message=MagicMock(content="Final response"))
    ]
    
    with patch.object(agent.ollama_client, 'chat') as mock_chat:
        mock_chat.side_effect = [mock_response, mock_final_response]
        
        messages = [ChatMessage(role="user", content="Test message")]
        async for response in agent.chat(config, messages, stream=False):
            assert response == "Final response"

@pytest.mark.asyncio
async def test_chat_streaming(agent, config):
    """Test streaming chat responses."""
    config.stream = True
    chunks = ["Hello", " world", "!"]
    
    async def mock_stream():
        for chunk in chunks:
            yield AsyncMock(message=MagicMock(content=chunk))
    
    with patch.object(agent.ollama_client, 'chat', return_value=mock_stream()):
        messages = [ChatMessage(role="user", content="Test message")]
        received_chunks = []
        async for chunk in agent.chat(config, messages, stream=True):
            received_chunks.append(chunk)
        
        assert received_chunks == chunks

@pytest.mark.asyncio
async def test_error_handling(agent, config):
    """Test error handling in chat."""
    with patch.object(agent.ollama_client, 'chat', side_effect=Exception("Test error")):
        messages = [ChatMessage(role="user", content="Test message")]
        with pytest.raises(Exception) as exc_info:
            async for _ in agent.chat(config, messages):
                pass
        assert str(exc_info.value) == "Test error"
