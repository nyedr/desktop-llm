"""Tool functions that can be called by the LLM."""

from .web_scrape_tool import WebScrapeTool
from .weather_tools import WeatherTool

__all__ = [
    "WebScrapeTool",
    "WeatherTool",
]
