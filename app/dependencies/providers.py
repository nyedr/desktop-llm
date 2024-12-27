"""Service providers and dependencies."""
from typing import Optional
from fastapi import Request

from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.services.agent import Agent
from app.services.chroma_service import ChromaService
from app.services.mcp_service import MCPService
from app.services.langchain_service import LangChainService
from app.core.service_locator import get_service_locator


class Providers:
    """Service providers and dependencies."""
    _model_service: Optional[ModelService] = None
    _function_service: Optional[FunctionService] = None
    _agent: Optional[Agent] = None
    _chroma_service: Optional[ChromaService] = None
    _mcp_service: Optional[MCPService] = None
    _langchain_service: Optional[LangChainService] = None

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
    def get_agent(cls) -> Agent:
        """Get or create agent instance."""
        if cls._agent is None:
            model_service = cls.get_model_service()
            function_service = cls.get_function_service()
            cls._agent = Agent(
                model_service=model_service,
                function_service=function_service
            )
        return cls._agent

    @classmethod
    def get_chroma_service(cls) -> ChromaService:
        """Get or create Chroma service instance."""
        if cls._chroma_service is None:
            cls._chroma_service = ChromaService()
        return cls._chroma_service

    @classmethod
    def get_mcp_service(cls) -> MCPService:
        """Get or create MCP service instance."""
        if cls._mcp_service is None:
            cls._mcp_service = MCPService()
            # Register with service locator
            get_service_locator().register_service("mcp_service", cls._mcp_service)
        return cls._mcp_service

    @classmethod
    def get_langchain_service(cls) -> LangChainService:
        """Get or create LangChain service instance."""
        if cls._langchain_service is None:
            cls._langchain_service = LangChainService()
        return cls._langchain_service


# FastAPI dependency functions

def get_model_service(request: Request) -> ModelService:
    """Get model service instance."""
    return Providers.get_model_service()


def get_function_service(request: Request) -> FunctionService:
    """Get function service instance."""
    return Providers.get_function_service()


def get_agent(request: Request) -> Agent:
    """Get agent instance."""
    return Providers.get_agent()


def get_chroma_service(request: Request) -> ChromaService:
    """Get Chroma service instance."""
    return Providers.get_chroma_service()


def get_mcp_service(request: Request) -> MCPService:
    """Get MCP service instance."""
    return Providers.get_mcp_service()


def get_langchain_service(request: Request) -> LangChainService:
    """Get LangChain service instance."""
    return Providers.get_langchain_service()
