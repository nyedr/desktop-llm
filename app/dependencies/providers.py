"""Service providers and dependencies."""
from typing import Optional, Dict, Any
from fastapi import Depends, Request

from app.core.config import config
from app.services.model_service import ModelService
from app.services.function_service import FunctionService
from app.services.agent import Agent

class Providers:
    """Service providers and dependencies."""
    _model_service: Optional[ModelService] = None
    _function_service: Optional[FunctionService] = None
    _agent: Optional[Agent] = None
    
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
