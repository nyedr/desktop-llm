"""Service providers for dependency injection."""

import logging
from typing import Optional
from pathlib import Path

from app.core.service_locator import get_service_locator
from app.services.agent import Agent
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.services.mcp_service import MCPService
from app.memory.lightrag.manager import EnhancedLightRAGManager
from app.memory.lightrag.config import LIGHTRAG_DATA_DIR

logger = logging.getLogger(__name__)


class Providers:
    """Service providers for dependency injection."""
    _agent: Optional[Agent] = None
    _model_service: Optional[ModelService] = None
    _function_service: Optional[FunctionService] = None
    _mcp_service: Optional[MCPService] = None
    _lightrag_manager: Optional[EnhancedLightRAGManager] = None

    @classmethod
    def get_agent(cls) -> Agent:
        """Get or create agent instance."""
        if cls._agent is None:
            # Get services from service locator
            service_locator = get_service_locator()

            # Check if required services are available
            if not service_locator.has_service("model_service") or not service_locator.has_service("function_service"):
                raise ValueError(
                    "Required services (model_service, function_service) must be initialized first")

            # Get the services
            model_service = service_locator.get_service("model_service")
            function_service = service_locator.get_service("function_service")

            # Create agent with required services
            cls._agent = Agent(
                model_service=model_service,
                function_service=function_service
            )
            # Register with service locator
            service_locator.register_service("agent", cls._agent)
        return cls._agent

    @classmethod
    def get_model_service(cls) -> ModelService:
        """Get or create model service instance."""
        if cls._model_service is None:
            cls._model_service = ModelService()
            # Register with service locator
            get_service_locator().register_service("model_service", cls._model_service)
        return cls._model_service

    @classmethod
    def get_function_service(cls) -> FunctionService:
        """Get or create function service instance."""
        if cls._function_service is None:
            cls._function_service = FunctionService()
            # Register with service locator
            get_service_locator().register_service(
                "function_service", cls._function_service)
        return cls._function_service

    @classmethod
    def get_mcp_service(cls) -> MCPService:
        """Get or create MCP service instance."""
        if cls._mcp_service is None:
            cls._mcp_service = MCPService()
            # Register with service locator
            get_service_locator().register_service("mcp_service", cls._mcp_service)
        return cls._mcp_service

    @classmethod
    async def get_lightrag_manager(cls) -> EnhancedLightRAGManager:
        """Get or create EnhancedLightRAG manager instance."""
        if cls._lightrag_manager is None:
            # Create working directory if it doesn't exist
            working_dir = Path(LIGHTRAG_DATA_DIR)
            working_dir.mkdir(parents=True, exist_ok=True)

            # Create manager with working directory
            cls._lightrag_manager = EnhancedLightRAGManager(working_dir)

            # Register with service locator
            get_service_locator().register_service(
                "lightrag_manager", cls._lightrag_manager)

            # Initialize the manager
            await cls._lightrag_manager.initialize()

        return cls._lightrag_manager

    @classmethod
    async def cleanup(cls):
        """Clean up all service instances."""
        if cls._lightrag_manager:
            await cls._lightrag_manager.stop()
        cls._agent = None
        cls._model_service = None
        cls._function_service = None
        cls._mcp_service = None
        cls._lightrag_manager = None
        get_service_locator().clear()
