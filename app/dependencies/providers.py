"""Service providers and dependencies."""
from typing import Optional
from fastapi import Request
import logging

from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.services.agent import Agent
from app.memory.lightrag.manager import EnhancedLightRAGManager
from app.services.mcp_service import MCPService
from app.services.langchain_service import LangChainService
from app.core.service_locator import get_service_locator

logger = logging.getLogger(__name__)


class Providers:
    """Service providers and dependencies."""
    _model_service: Optional[ModelService] = None
    _function_service: Optional[FunctionService] = None
    _agent: Optional[Agent] = None
    _mcp_service: Optional[MCPService] = None
    _langchain_service: Optional[LangChainService] = None
    _lightrag_manager: Optional[EnhancedLightRAGManager] = None

    @classmethod
    def get_model_service(cls) -> ModelService:
        """Get or create model service instance."""
        if cls._model_service is None:
            cls._model_service = ModelService()
        return cls._model_service

    @classmethod
    def get_function_service(cls) -> FunctionService:
        """Get or create function service instance."""
        if cls._function_service is None:
            cls._function_service = FunctionService()
        return cls._function_service

    @classmethod
    async def get_agent(cls) -> Agent:
        """Get or create Agent instance."""
        if cls._agent is None:
            try:
                # Get required services
                model_service = cls.get_model_service()
                function_service = cls.get_function_service()
                langchain_service = await cls.get_langchain_service()

                # Create agent with all required dependencies
                agent = Agent(
                    model_service=model_service,
                    function_service=function_service,
                    langchain_service=langchain_service
                )
                await agent.initialize()
                cls._agent = agent
            except Exception as e:
                logger.warning(
                    f"Error initializing Agent with full services: {e}. Creating with basic services.")
                # Create agent with just required services
                agent = Agent(
                    model_service=cls.get_model_service(),
                    function_service=cls.get_function_service()
                )
                await agent.initialize()
                cls._agent = agent
        return cls._agent

    @classmethod
    def get_mcp_service(cls) -> MCPService:
        """Get or create MCP service instance."""
        if cls._mcp_service is None:
            cls._mcp_service = MCPService()
            # Register with service locator
            get_service_locator().register_service("mcp_service", cls._mcp_service)
        return cls._mcp_service

    @classmethod
    async def get_langchain_service(cls) -> LangChainService:
        """Get or create LangChain service instance."""
        if cls._langchain_service is None:
            logger.info("Creating new LangChain service instance")
            cls._langchain_service = LangChainService()

            try:
                # Get required dependencies
                manager = cls.get_lightrag_manager()
                mcp_service = cls.get_mcp_service()

                # Initialize the service
                logger.info("Initializing LangChain service with dependencies")
                await cls._langchain_service.initialize(manager, mcp_service)
                logger.info("LangChain service initialized successfully")
            except Exception as e:
                logger.error(
                    f"Failed to initialize LangChain service: {str(e)}", exc_info=True)
                cls._langchain_service = None
                raise
        return cls._langchain_service

    @classmethod
    def get_langchain_service_sync(cls) -> Optional[LangChainService]:
        """Get the LangChain service instance if it exists, without initialization."""
        return cls._langchain_service

    @classmethod
    def get_lightrag_manager(cls) -> EnhancedLightRAGManager:
        """Get or create EnhancedLightRAG manager instance."""
        if cls._lightrag_manager is None:
            cls._lightrag_manager = EnhancedLightRAGManager()
        return cls._lightrag_manager


# FastAPI dependency functions

def get_model_service(request: Request) -> ModelService:
    """Get model service instance."""
    return Providers.get_model_service()


def get_function_service(request: Request) -> FunctionService:
    """Get function service instance."""
    return Providers.get_function_service()


async def get_agent(request: Request) -> Agent:
    """Get Agent instance for FastAPI dependency injection."""
    return await Providers.get_agent()


def get_mcp_service(request: Request) -> MCPService:
    """Get MCP service instance."""
    return Providers.get_mcp_service()


async def get_langchain_service(request: Request) -> LangChainService:
    """Get LangChain service instance."""
    return await Providers.get_langchain_service()


def get_lightrag_manager(request: Request) -> EnhancedLightRAGManager:
    """Get EnhancedLightRAG manager instance."""
    return Providers.get_lightrag_manager()
