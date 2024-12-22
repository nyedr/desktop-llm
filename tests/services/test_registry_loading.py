"""Tests for loading functions from static configuration files."""

import pytest
import logging
from pathlib import Path
from app.functions.registry import function_registry

# Configure logging for the test
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def clean_registry():
    """Ensure registry is clean before and after each test."""
    function_registry._functions.clear()
    function_registry._dependency_cache.clear()
    yield function_registry
    function_registry._functions.clear()
    function_registry._dependency_cache.clear()


@pytest.mark.asyncio
async def test_load_simple_tool(clean_registry):
    """Test loading a simple tool from a static configuration file."""
    logger.info("Starting simple tool loading test")

    try:
        # Load the test configuration
        config_path = Path(__file__).parent.parent / "fixtures" / \
            "functions" / "test_tool" / "config.json"
        logger.info(f"Loading configuration from: {config_path}")

        await function_registry.load_from_config(config_path)

        # Verify the function was registered
        functions = function_registry.list_functions()
        logger.info(f"Registered functions: {functions}")

        assert len(functions) == 1, "Expected one function to be registered"
        assert functions[0]["name"] == "simple_tool", "Expected simple_tool to be registered"
        assert functions[0]["type"] == "tool", "Expected function type to be 'tool'"

        logger.info("Simple tool loading test completed successfully")

    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}", exc_info=True)
        raise
