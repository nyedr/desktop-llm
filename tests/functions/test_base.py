"""Tests for base function classes and error handling."""

import pytest
from app.functions.base import (
    FunctionError,
    ValidationError,
    TimeoutError,
    ExecutionError,
    FunctionNotFoundError,
    InputValidationError,
    OutputValidationError
)

def test_function_error_with_context():
    """Test FunctionError with context information."""
    error = FunctionError("Test error", function_name="test_func", details={"key": "value"})
    assert str(error) == "Test error"
    assert error.function_name == "test_func"
    assert error.details == {"key": "value"}

def test_timeout_error():
    """Test TimeoutError with timeout information."""
    error = TimeoutError("Operation timed out", timeout_seconds=30)
    assert "Operation timed out" in str(error)
    assert error.details.get("timeout_seconds") == 30

def test_execution_error_with_original():
    """Test ExecutionError with original error preservation."""
    original = ValueError("Original error")
    error = ExecutionError("Execution failed", original_error=original)
    assert "Execution failed" in str(error)
    assert error.details.get("error_type") == "ValueError"

def test_input_validation_error():
    """Test InputValidationError with invalid parameters."""
    error = InputValidationError(
        "Validation failed",
        function_name="test_func",
        invalid_params=["param1", "param2"]
    )
    assert "Validation failed" in str(error)
    assert error.function_name == "test_func"
    assert error.details.get("invalid_params") == ["param1", "param2"]

def test_error_inheritance():
    """Test error class inheritance relationships."""
    assert issubclass(ValidationError, FunctionError)
    assert issubclass(InputValidationError, ValidationError)
    assert issubclass(OutputValidationError, ValidationError)
    assert issubclass(TimeoutError, FunctionError)
    assert issubclass(ExecutionError, FunctionError)
    assert issubclass(FunctionNotFoundError, FunctionError)
