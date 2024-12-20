"""Test configuration and fixtures."""

import pytest
import asyncio
from app.main import app
from app.services.function_service import FunctionService
from app.services.model_service import ModelService
from app.services.agent import Agent

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
    agent = Agent(function_service=function_service, model_service=model_service)
    yield agent
