"""Tests for function execution system."""

import pytest
from typing import Dict, Any, Literal
from pydantic import Field
from app.functions.base import (
    Tool,
    FunctionError,
    ExecutionError,
    InputValidationError,
    FunctionNotFoundError
)
from app.functions.executor import FunctionExecutor
from app.functions.registry import FunctionRegistry

class MockTool(Tool):
    """Mock tool for testing."""
    name: str = Field(default="mock_tool", description="Mock tool for testing")
    description: str = Field(default="A mock tool for testing", description="Mock tool description")
    parameters: dict = Field(default={
        "type": "object",
        "properties": {
            "input": {"type": "string"}
        },
        "required": ["input"]
    }, description="Mock tool parameters")

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if args.get("input") == "fail":
            raise ValueError("Mock failure")
        return {"result": args.get("input", "")}

@pytest.fixture
def registry():
    """Create a function registry with mock functions."""
    registry = FunctionRegistry()
    registry.register(MockTool)
    return registry

@pytest.fixture
def executor(registry):
    """Create a function executor with mock registry."""
    executor = FunctionExecutor()
    executor.registry = registry
    return executor

@pytest.mark.asyncio
async def test_successful_execution(executor):
    """Test successful function execution."""
    result = await executor.execute("mock_tool", {"input": "test"})
    assert result == {"result": "test"}

@pytest.mark.asyncio
async def test_function_not_found(executor):
    """Test error when function is not found."""
    with pytest.raises(FunctionNotFoundError) as exc:
        await executor.execute("nonexistent", {})
    assert "Function not found" in str(exc.value)
    assert exc.value.function_name == "nonexistent"

@pytest.mark.asyncio
async def test_input_validation_error(executor):
    """Test input validation error handling."""
    with pytest.raises(InputValidationError) as exc:
        await executor.execute("mock_tool", {})  # Missing required 'input'
    assert "Invalid input parameters" in str(exc.value)
    assert exc.value.function_name == "mock_tool"

@pytest.mark.asyncio
async def test_execution_error(executor):
    """Test execution error handling."""
    with pytest.raises(ExecutionError) as exc:
        await executor.execute("mock_tool", {"input": "fail"})
    assert "Error executing function" in str(exc.value)
    assert exc.value.function_name == "mock_tool"
    assert exc.value.details.get("error_type") == "ValueError"
