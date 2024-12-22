"""Test configuration and fixtures."""

import pytest
import asyncio
import os
from unittest.mock import patch
from app.main import app
from app.services.function_service import FunctionService
from app.services.model_service import ModelService
from app.services.agent import Agent
import sys
import platform


@pytest.fixture
def app():
    """Create a FastAPI app instance for testing."""
    return app


@pytest.fixture
async def function_service():
    """Create a function service instance for testing."""
    service = FunctionService()
    yield service


@pytest.fixture
async def model_service():
    """Create a model service instance for testing."""
    service = ModelService()
    yield service


@pytest.fixture
async def agent(function_service, model_service):
    """Create an agent instance for testing."""
    agent = Agent(function_service=function_service,
                  model_service=model_service)
    yield agent


# Chroma-specific fixtures
@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    class MockConfig:
        # Chroma settings
        CHROMA_PERSIST_DIRECTORY = "test_data"
        CHROMA_COLLECTION_NAME = "test_collection"
        CHROMA_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Using a smaller model for tests

        # MCP settings
        MCP_SERVER_FILESYSTEM_PATH = "./test_data/filesystem/dist/index.js"
        MCP_SERVER_FILESYSTEM_COMMAND = "node"
        WORKSPACE_DIR = "./test_data"
    return MockConfig()


@pytest.fixture(autouse=True)
async def setup_test_environment():
    """Set up test environment with temporary directories."""
    # Create temporary test directories
    test_dir = os.path.join(os.getcwd(), "test_data")
    filesystem_dir = os.path.join(test_dir, "filesystem", "dist")
    os.makedirs(filesystem_dir, exist_ok=True)

    # Create a dummy index.js file for MCP filesystem
    index_js_path = os.path.join(filesystem_dir, "index.js")
    with open(index_js_path, "w") as f:
        f.write("// Dummy MCP filesystem server for testing")

    yield

    # Cleanup
    try:
        import shutil
        shutil.rmtree(test_dir)
    except Exception as e:
        print(f"Warning: Failed to remove test directory: {e}")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    if sys.platform == 'win32':
        # Use ProactorEventLoop on Windows for subprocess support
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    else:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
