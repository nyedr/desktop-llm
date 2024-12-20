"""Tests for function service."""

import pytest
from app.services.function_service import FunctionService
from app.functions.base import Tool, FunctionType
from typing import Literal
from pydantic import Field

class TestTool(Tool):
    """Test tool implementation."""
    name: str = Field(default="mock_tool", description="Test tool for testing")
    description: str = Field(default="A test tool", description="Test tool description")
    type: FunctionType = Field(default=FunctionType.PIPE, description="Test tool type")

    async def execute(self, args):
        return {"result": "test"}

@pytest.mark.asyncio
async def test_register_and_execute(function_service):
    """Test registering and executing a function."""
    # Register function
    function_service.registry.register(TestTool)
    
    # Execute function
    result = await function_service.execute_function("mock_tool", {})
    assert result == {"result": "test"}

def test_list_functions(function_service):
    """Test listing registered functions."""
    # Register test function
    function_service.registry.register(TestTool)
    
    # List functions
    functions = function_service.list_functions()
    assert any(f["name"] == "mock_tool" for f in functions)

def test_get_function_schemas(function_service):
    """Test getting function schemas."""
    function_service.registry.register(TestTool)
    schemas = function_service.get_function_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "mock_tool"

@pytest.mark.asyncio
async def test_handle_tool_calls(function_service):
    """Test handling multiple tool calls."""
    # Register test function
    function_service.registry.register(TestTool)
    
    # Create tool calls
    tool_calls = [
        {
            "id": "call1",
            "name": "mock_tool",
            "arguments": {}
        }
    ]
    
    # Handle tool calls
    results = await function_service.handle_tool_calls(tool_calls)
    assert len(results) == 1
    assert results[0]["name"] == "mock_tool"
    assert results[0]["result"] == {"result": "test"}
