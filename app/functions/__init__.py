"""Function system initialization."""

from app.functions.registry import function_registry
from app.functions.executor import FunctionExecutor
from app.functions.agent import (
    BaseAgent,
    GeneralAgent,
    SupervisorAgent,
    AgentConfig,
    RetryConfig,
    BackoffStrategy
)

# Import all function types to ensure registration
from app.functions.types import *

executor = FunctionExecutor()

__all__ = [
    'function_registry',
    'executor',
    'BaseAgent',
    'GeneralAgent',
    'SupervisorAgent',
    'AgentConfig',
    'RetryConfig',
    'BackoffStrategy'
]
