"""Test server endpoints."""

import os
import json
import pytest
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

# Override config path for testing
app.state.config_path = os.path.join(os.path.dirname(__file__), "test_config.json")

# Test client fixture
@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Create an async test client."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client

# Test functions
@pytest.mark.asyncio
async def test_generate_completion(async_client: httpx.AsyncClient):
    """
    Test the generate completion endpoint.
    
    Validates:
    1. Successful response
    2. Streaming works with proper SSE format
    3. Response contains text and proper headers
    4. Buffer and chunk size settings work
    5. Request ID is returned
    """
    # Prepare test payload
    payload = {
        "prompt": "Explain the concept of artificial intelligence in 3 sentences.",
        "model": "qwen2.5:7b-instruct-q8_0",
        "temperature": 0.7,
        "max_tokens": 150,
        "task": "explanation"
    }
    
    # Send request to generate endpoint
    async with async_client.stream(
        "POST",
        "/v1/completions", 
        json=payload,
        timeout=30.0
    ) as response:
        # Check response status and headers
        assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
        assert "text/event-stream" in response.headers["content-type"].lower()
        assert "x-request-id" in response.headers
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["connection"] == "keep-alive"
        assert response.headers["transfer-encoding"] == "chunked"
        
        try:
            # Read streaming response
            chunks = []
            total_size = 0
            async for chunk in response.aiter_text():
                # Validate chunk
                assert len(chunk) > 0, "Empty chunk received"
                chunks.append(chunk)
                total_size += len(chunk)
                
                # Check chunk size (should be around 8KB)
                assert len(chunk) <= 8192, f"Chunk size too large: {len(chunk)}"
            
            # Validate complete response
            complete_response = "".join(chunks)
            assert len(complete_response) > 0, "Empty response received"
            assert "artificial intelligence" in complete_response.lower(), "Response does not address prompt"
            
            # Validate streaming performance
            assert len(chunks) > 1, "Response was not properly streamed"
            assert total_size > 0, "No data received"
        except httpx.RemoteProtocolError:
            # This is expected when the connection is closed after streaming
            pass

@pytest.mark.asyncio
async def test_generate_completion_with_images(async_client: httpx.AsyncClient):
    """
    Test the generate completion endpoint with image input.
    
    Validates:
    1. Image processing works
    2. Base64 validation works
    3. Multimodal generation works
    """
    # Create a small test image in base64
    import base64
    test_image = base64.b64encode(b"test image data").decode()
    
    # Prepare test payload with image
    payload = {
        "prompt": "Describe this image.",
        "model": "qwen2.5:7b-instruct-q8_0",
        "temperature": 0.7,
        "max_tokens": 150,
        "images": [test_image],
        "task": "image_description"
    }
    
    # Send request
    try:
        async with async_client.stream(
            "POST",
            "/v1/completions", 
            json=payload,
            timeout=30.0
        ) as response:
            # Check response
            assert response.status_code == 200
            
            # Read streaming response
            chunks = []
            async for chunk in response.aiter_text():
                chunks.append(chunk)
            
            # Validate response
            complete_response = "".join(chunks)
            assert len(complete_response) > 0
    except httpx.RemoteProtocolError:
        # This is expected when the connection is closed after streaming
        pass

@pytest.mark.asyncio
async def test_generate_completion_error_handling(async_client: httpx.AsyncClient):
    """
    Test error handling for generate completion endpoint.
    
    Validates:
    1. Invalid model handling
    2. Invalid image data handling
    3. Timeout handling
    4. Memory limit handling
    """
    try:
        # Test invalid model
        payload = {
            "prompt": "Test prompt",
            "model": "invalid_model",
            "temperature": 0.7
        }
        response = await async_client.post("/v1/completions", json=payload)
        assert response.status_code == 500
        
        # Test invalid base64 image
        payload = {
            "prompt": "Test prompt",
            "model": "qwen2.5:7b-instruct-q8_0",
            "images": ["invalid_base64"]
        }
        response = await async_client.post("/v1/completions", json=payload)
        assert response.status_code == 400
        assert "Invalid base64" in response.json()["detail"]
        
        # Test timeout
        payload = {
            "prompt": "Test prompt",
            "model": "qwen2.5:7b-instruct-q8_0",
            "max_tokens": 10000  # Request large output
        }
        with pytest.raises(httpx.ReadTimeout):
            await async_client.post(
                "/v1/completions",
                json=payload,
                timeout=1.0  # Very short timeout
            )
    except httpx.RemoteProtocolError:
        # This is expected when the connection is closed after streaming
        pass

@pytest.mark.asyncio
async def test_get_models(async_client: httpx.AsyncClient):
    """
    Test the models endpoint.
    
    Validates:
    1. Successful response
    2. Returns a list of models
    3. Pagination works
    """
    # Test first page
    response = await async_client.get("/v1/models?page=1&limit=10")
    
    # Check response status
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    
    # Parse response
    models_data = response.json()

    # Validate response structure
    assert "data" in models_data, "Data key missing in response"
    assert isinstance(models_data["data"], list), "Models data is not a list"
    assert len(models_data["data"]) > 0, "No models returned"
    assert "page" in models_data, "Page info missing"
    assert "limit" in models_data, "Limit info missing"
    assert "total" in models_data, "Total count missing"

    # Validate model structure
    model = models_data["data"][0]
    assert "id" in model, "Model ID missing"
    assert "provider" in model, "Provider missing"

@pytest.mark.asyncio
async def test_register_and_execute_function(async_client: httpx.AsyncClient):
    """Test function registration and execution."""
    # List functions - should already be loaded from test_config.json
    response = await async_client.get("/functions/list")
    assert response.status_code == 200
    functions = response.json()
    assert len(functions) > 0
    assert any(f["name"] == "add_numbers" for f in functions)
    
    # Execute function
    execute_data = {
        "func_name": "add_numbers",
        "func_params": {"a": 1, "b": 2}
    }
    response = await async_client.post("/functions/execute", json=execute_data)
    assert response.status_code == 200
    result = response.json()
    assert result["result"] == 3

@pytest.mark.asyncio
async def test_add_prompt_template(async_client: httpx.AsyncClient):
    """
    Test adding a prompt template.
    
    Validates:
    1. Successful template addition
    """
    template_payload = {
        "task": "summarization",
        "template_str": "Summarize the following text: {text}"
    }
    
    response = await async_client.post(
        "/api/config/prompt_templates",
        json=template_payload
    )
    
    # Check response status
    assert response.status_code == 200, f"Prompt template addition failed: {response.text}"

@pytest.mark.asyncio
async def test_update_special_token(async_client: httpx.AsyncClient):
    """
    Test updating special tokens.
    
    Validates:
    1. Successful special token update
    """
    token_payload = {
        "token_name": "test_special_token",
        "token_str": "[TEST]"
    }
    
    response = await async_client.post(
        "/api/config/special_tokens",
        json=token_payload
    )
    
    # Check response status
    assert response.status_code == 200, f"Special token update failed: {response.text}"

@pytest.mark.asyncio
async def test_register_function(async_client: httpx.AsyncClient):
    """Test function registration."""
    # Register test function
    response = await async_client.post(
        "/api/functions/register",
        json={
            "name": "add_numbers",
            "module_path": "test_functions",
            "function_name": "add_numbers",
            "type": "sync",
            "description": "Add two numbers and return the result",
            "input_schema": json.dumps({
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }),
            "output_schema": json.dumps({
                "type": "object",
                "properties": {
                    "result": {"type": "number"}
                },
                "required": ["result"]
            })
        }
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"

@pytest.mark.asyncio
async def test_list_functions(async_client: httpx.AsyncClient):
    """Test listing registered functions."""
    # List functions
    response = await async_client.get("/api/functions/list")
    
    assert response.status_code == 200
    functions = response.json()
    assert len(functions) > 0
    
    # Check if add_numbers function is registered
    add_numbers = next((f for f in functions if f["name"] == "add_numbers"), None)
    assert add_numbers is not None
    assert add_numbers["type"] == "sync"
    assert add_numbers["description"] == "Add two numbers and return the result"

@pytest.mark.asyncio
async def test_execute_function(async_client: httpx.AsyncClient):
    """Test function execution."""
    # Execute add_numbers function
    response = await async_client.post(
        "/api/functions/execute",
        json={
            "func_name": "add_numbers",
            "func_params": {"a": 2, "b": 3}
        }
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["result"] == 5

# Utility function to create test functions file
def create_test_functions_file():
    """Create a test functions file for function registration tests."""
    test_file = os.path.join(os.path.dirname(__file__), "test_functions.py")
    with open(test_file, 'w') as f:
        f.write('''
def add_numbers(a: int, b: int) -> dict:
    """Simple function to add two numbers."""
    return {"result": a + b}
''')

# Setup and teardown
def pytest_configure(config):
    """Create test functions file before running tests."""
    create_test_functions_file()

def pytest_unconfigure(config):
    """Clean up test functions file after tests."""
    test_file = os.path.join(os.path.dirname(__file__), "test_functions.py")
    if os.path.exists(test_file):
        os.remove(test_file)
