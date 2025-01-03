"""Service providers for dependency injection."""

import logging
from typing import Optional
from pathlib import Path

from app.core.service_locator import get_service_locator
from app.services.agent import Agent
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.services.mcp_service import MCPService
from app.memory.manager import LightRAGManager
from app.core.config import config

logger = logging.getLogger(__name__)


class Providers:
    """Service providers for dependency injection."""
    _agent: Optional[Agent] = None
    _model_service: Optional[ModelService] = None
    _function_service: Optional[FunctionService] = None
    _mcp_service: Optional[MCPService] = None
    _lightrag_manager: Optional[LightRAGManager] = None

    @classmethod
    async def get_agent(cls) -> Agent:
        """Get or create agent instance."""
        if cls._agent is None:
            try:
                # Get services from service locator
                service_locator = get_service_locator()

                # Check if required services are available
                if not service_locator.has_service("model_service") or not service_locator.has_service("function_service"):
                    raise ValueError(
                        "Required services (model_service, function_service) must be initialized first")

                # Get the services
                model_service = service_locator.get_service("model_service")

                # Create and initialize agent
                cls._agent = Agent()
                await cls._agent.initialize(model_service)

                # Register with service locator
                service_locator.register_service("agent", cls._agent)

            except Exception as e:
                logger.error(f"Failed to initialize agent: {e}")
                raise

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
    async def get_lightrag_manager(cls) -> Optional[LightRAGManager]:
        """Get or create LightRAG manager instance."""
        if cls._lightrag_manager is None:
            try:
                # Create working directory if it doesn't exist
                working_dir = Path(config.memory.data_dir)
                working_dir.mkdir(parents=True, exist_ok=True)

                # Create manager with working directory
                cls._lightrag_manager = LightRAGManager(
                    working_dir=str(working_dir))

                # Register with service locator
                get_service_locator().register_service(
                    "lightrag_manager", cls._lightrag_manager)

                # Initialize and start the manager
                await cls._lightrag_manager.initialize()
                await cls._lightrag_manager.start()

                logger.info("LightRAG manager initialized successfully")
                return cls._lightrag_manager

            except Exception as e:
                logger.error(
                    f"Failed to initialize lightrag manager: {e}", exc_info=True)
                cls._lightrag_manager = None
                return None

        return cls._lightrag_manager

    @classmethod
    async def cleanup(cls):
        """Clean up all service instances."""
        try:
            logger.info("Starting service cleanup...")

            # Stop memory manager first
            if cls._lightrag_manager:
                try:
                    await cls._lightrag_manager.stop()
                    logger.info("LightRAG manager stopped")
                except Exception as e:
                    logger.error(f"Error stopping lightrag manager: {e}")

            # Clear service instances
            cls._agent = None
            cls._model_service = None
            cls._function_service = None
            cls._mcp_service = None
            cls._lightrag_manager = None

            # Clear service locator
            get_service_locator().clear()
            logger.info("Service cleanup completed")

        except Exception as e:
            logger.error(f"Error during service cleanup: {e}", exc_info=True)
            raise
