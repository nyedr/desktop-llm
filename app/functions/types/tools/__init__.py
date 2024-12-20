"""Tool functions that can be called by the LLM."""

from app.functions.types.tools.calculator import CalculatorTool
from app.functions.types.tools.image_tools import ImageEmbeddingTool, SearchQueryTool, TagGeneratorTool
from app.functions.types.tools.weather_tools import WeatherTool

__all__ = [
    'CalculatorTool',
    'ImageEmbeddingTool',
    'SearchQueryTool',
    'TagGeneratorTool',
    'WeatherTool'
]
