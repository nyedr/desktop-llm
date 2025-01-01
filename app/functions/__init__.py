"""Function system for desktop-llm."""

from app.functions.registry import function_registry
from app.functions.executor import executor
from app.functions.types import *

__all__ = [
    'function_registry',
    'executor'
]
